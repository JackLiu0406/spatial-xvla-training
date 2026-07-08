# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from __future__ import annotations

import logging
import math
from functools import partial
from typing import Dict, Final, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry_conditioning import GeometryCrossAttentionFusion

logger = logging.getLogger("xvla.geometry")


# ------------------------------- Small utils ----------------------------------

def _to_2tuple(x) -> Tuple:
    """Minimal replacement for timm.layers.to_2tuple."""
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        t = tuple(x)
        return (t[0], t[1]) if len(t) >= 2 else (t[0], t[0])
    return (x, x)


def _has_sdp_attention() -> bool:
    """Check if we can use PyTorch fused scaled_dot_product_attention."""
    return hasattr(F, "scaled_dot_product_attention")


# ---------------------------------- MLP --------------------------------------

class Mlp(nn.Module):
    """
    MLP used in ViT-style blocks.

    Supports Linear or 1x1 Conv 'linear_layer' for token/channel mixing.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        norm_layer: type[nn.Module] | None = None,
        bias: bool | Tuple[bool, bool] = True,
        drop: float | Tuple[float, float] = 0.0,
        use_conv: bool = False,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = _to_2tuple(bias)
        drop_probs = _to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect [B, T, C] for Linear variant; caller is responsible for shapes.
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# -------------------------------- Attention ----------------------------------

class Attention(nn.Module):
    """
    Multi-Head Self-Attention with optional fused SDPA fallback.

    If PyTorch provides `scaled_dot_product_attention`, it will be used
    (usually faster and more stable); otherwise we use a manual implementation.
    """

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = _has_sdp_attention()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape [B, T, C]
            Input sequence.

        Returns
        -------
        Tensor, shape [B, T, C]
            Output sequence after MHSA + projection.
        """
        B, T, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, T, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)  # 3 x [B, H, T, Dh]
        )
        q, k, v = qkv.unbind(0)  # each: [B, H, T, Dh]
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )  # [B, H, T, Dh]
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)        # [B, H, T, T]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v                           # [B, H, T, Dh]

        x = x.transpose(1, 2).reshape(B, T, C)     # [B, T, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ------------------------------- Utilities -----------------------------------

def basic_init(module: nn.Module) -> None:
    """
    Apply a basic initialization scheme to Linear layers.

    - Weight: Xavier uniform initialization.
    - Bias: Set to zero.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 100) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Parameters
    ----------
    t : torch.Tensor
        Shape [B]. Each element is a timestep index, may be fractional.
    dim : int
        Dimensionality of the output embedding.
    max_period : int, default=100
        Controls the minimum frequency of the sinusoids.

    Returns
    -------
    torch.Tensor
        Shape [B, dim]. Sinusoidal embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=t.dtype, device=t.device)
        / half
    )
    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ------------------------------- Core Layers ----------------------------------

class DomainAwareLinear(nn.Module):
    """
    Linear layer with domain-conditioned parameters (per-sample).

    Each domain has its own weight and bias vectors, stored in embeddings.
    """

    def __init__(self, input_size: int, output_size: int, num_domains: int = 20) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Embedding(num_domains, output_size * input_size)
        self.bias = nn.Embedding(num_domains, output_size)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x: torch.Tensor, domain_id: torch.LongTensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            [B, I] or [B, T, I]
        domain_id : LongTensor
            [B], domain indices.

        Returns
        -------
        Tensor
            [B, O] or [B, T, O]
        """
        B = domain_id.shape[0]
        squeeze_T = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_T = True
        W = self.fc(domain_id).view(B, self.input_size, self.output_size)
        b = self.bias(domain_id).view(B, self.output_size)
        y = torch.matmul(x, W) + b.view(B, 1, self.output_size)
        if squeeze_T:
            y = y.squeeze(1)
        return y


class TransformerBlock(nn.Module):
    """
    Standard Transformer block (pre-LN): LN → MHSA → residual, LN → MLP → residual.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            drop=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, [B, T, H]

        Returns
        -------
        Tensor, [B, T, H]
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------- Main Model ---------------------------------------

class SoftPromptedTransformer(nn.Module):
    """
    Multi-modal, domain-aware Transformer with optional soft prompts.

    See parameter and forward I/O descriptions inside the docstrings.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        multi_modal_input_size: int = 768,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_domains: int = 20,
        dim_action: int = 20,
        dim_propio: int = 20,
        dim_time: int = 32,
        len_soft_prompts: int = 32,
        max_len_seq: int = 512,
        use_hetero_proj: bool = False,
        geometry_conditioning: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dim_action = dim_action
        self.dim_time = dim_time
        self.len_soft_prompts = len_soft_prompts
        self.use_hetero_proj = use_hetero_proj

        # --- DA3-XVLA optional geometry cross-attention fusion --------------
        gc = geometry_conditioning or {}
        self.geometry_enabled = bool(gc.get("enabled", False))
        self.geometry_fusion_position = str(gc.get("fusion_position", "before_policy"))
        self.geometry_debug_shapes = bool(gc.get("debug_shapes", False))
        self._geom_shape_logged = False
        spatial_lang_cfg = gc.get("spatial_lang", {}) or {}
        self.spatial_lang_enabled = bool(spatial_lang_cfg.get("enabled", False))
        self.spatial_lang_method = str(spatial_lang_cfg.get("method", "post_refiner")).lower()
        self.spatial_residual_scale = float(spatial_lang_cfg.get("spatial_scale_start", 0.01))
        self.last_spatial_stats: Dict[str, torch.Tensor] = {}

        # DA3-XVLA ablation: gated action-to-spatial cross-attention adapter.
        # When use_spatial_cross_attention=True, the OLD GeometryCrossAttentionFusion
        # is bypassed entirely (spatial info does NOT touch VLM tokens). Instead,
        # a GatedSpatialCrossAttention adapter is attached to each (or selected)
        # transformer block; it operates ONLY on the action-token slice with
        # geometry_tokens as K/V. See models/geometry_conditioning.py for module.
        self.use_spatial_cross_attention = bool(gc.get("use_spatial_cross_attention", False))
        # Where the cross-attention update lands: action slice only (default,
        # preserves the VLM wall) vs the entire policy sequence (spatial reaches
        # vlm/aux/soft too). See _apply_spatial_xattn() in forward.
        self.spatial_xattn_target = str(
            gc.get("spatial_cross_attention_target", "action_only")
        ).lower()
        if self.spatial_xattn_target not in ("action_only", "full_sequence"):
            raise ValueError(
                f"spatial_cross_attention_target={self.spatial_xattn_target!r} "
                "(expected 'action_only' or 'full_sequence')."
            )

        if (
            self.geometry_enabled
            and gc.get("fusion_type", "cross_attention") == "cross_attention"
            and not self.use_spatial_cross_attention
        ):
            self.geometry_fusion = GeometryCrossAttentionFusion(
                dim=hidden_size,
                num_heads=int(gc.get("cross_attention_heads", 8)),
                num_layers=int(gc.get("cross_attention_layers", 1)),
                mlp_ratio=mlp_ratio,
                dropout=float(gc.get("cross_attention_dropout", 0.0)),
            )
        else:
            self.geometry_fusion = None

        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )

        # Build gated spatial cross-attention adapters (one per block; None for
        # layers that should not apply the adapter). Only built when enabled.
        self.spatial_cross_attn_layers: Optional[nn.ModuleList] = None
        if self.geometry_enabled and self.use_spatial_cross_attention:
            from .geometry_conditioning import (
                GatedSpatialCrossAttention,
                PerTokenGatedSpatialCrossAttention,
            )

            # Resolve which layers get the adapter.
            # Accepted values:
            #   "all"               — every block (default)
            #   "middle_late" / "late_half" / "last_half"
            #                       — last 50% of blocks: indices [depth//2, depth)
            #   "early_half" / "first_half"
            #                       — first 50% of blocks: indices [0, depth//2)
            #   "late_quarter"      — last 25% of blocks: indices [3*depth//4, depth)
            #   int                 — single block index
            #   list[int]/tuple     — explicit indices (negatives tolerated, modulo depth)
            spec = gc.get("spatial_cross_attention_layers", "all")
            if spec is None or (isinstance(spec, str) and spec.lower() == "all"):
                layer_set = set(range(depth))
            elif isinstance(spec, str) and spec.lower() in (
                "middle_late", "late_half", "last_half",
            ):
                layer_set = set(range(depth // 2, depth))
            elif isinstance(spec, str) and spec.lower() in (
                "early_half", "first_half",
            ):
                layer_set = set(range(0, depth // 2))
            elif isinstance(spec, str) and spec.lower() == "late_quarter":
                layer_set = set(range((3 * depth) // 4, depth))
            elif isinstance(spec, int) and not isinstance(spec, bool):
                layer_set = {spec % depth}
            else:
                # list/tuple of ints; tolerate negative indices.
                layer_set = {int(i) % depth for i in spec}

            # Action hidden dim defaults to transformer hidden_size; allow
            # override for forward-compat (heterogeneous projection setups).
            action_hidden_dim = int(gc.get("action_hidden_dim") or hidden_size)
            spatial_token_dim_cfg = gc.get("spatial_token_dim", None)
            spatial_num_heads = int(gc.get("spatial_cross_attention_heads",
                                          gc.get("cross_attention_heads", 8)))
            spatial_dropout = float(gc.get("spatial_cross_attention_dropout",
                                           gc.get("cross_attention_dropout", 0.0)))
            gate_init = float(gc.get("spatial_gate_init", 0.0))

            # Dispatcher: scalar (legacy) vs per-token (GatedFusion paper winner).
            gate_type = str(gc.get("spatial_gate_type", "scalar")).lower()
            if gate_type == "per_token":
                gate_mlp_hidden_cfg = gc.get("spatial_gate_mlp_hidden", None)
                def _make_adapter() -> nn.Module:
                    return PerTokenGatedSpatialCrossAttention(
                        hidden_dim=action_hidden_dim,
                        num_heads=spatial_num_heads,
                        spatial_token_dim=spatial_token_dim_cfg,
                        gate_mlp_hidden=gate_mlp_hidden_cfg,
                        dropout=spatial_dropout,
                    )
            elif gate_type == "scalar":
                def _make_adapter() -> nn.Module:
                    return GatedSpatialCrossAttention(
                        hidden_dim=action_hidden_dim,
                        num_heads=spatial_num_heads,
                        spatial_token_dim=spatial_token_dim_cfg,
                        gate_init=gate_init,
                        dropout=spatial_dropout,
                    )
            else:
                raise ValueError(
                    f"spatial_gate_type={gate_type!r} not supported "
                    "(expected 'scalar' or 'per_token')."
                )

            modules = []
            for i in range(depth):
                if i in layer_set:
                    modules.append(_make_adapter())
                else:
                    modules.append(nn.Identity())  # ModuleList entries can't be None
            self.spatial_cross_attn_layers = nn.ModuleList(modules)
            # Mirror the layer_set for fast checking inside forward (avoids
            # isinstance-on-Identity for every layer).
            self._spatial_layer_active = layer_set

        if use_hetero_proj:
            self.vlm_proj = DomainAwareLinear(multi_modal_input_size, hidden_size, num_domains=num_domains)
            self.aux_visual_proj = DomainAwareLinear(multi_modal_input_size, hidden_size, num_domains=num_domains)
        else:
            self.vlm_proj = nn.Linear(multi_modal_input_size, hidden_size)
            self.aux_visual_proj = nn.Linear(multi_modal_input_size, hidden_size)

        self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size), requires_grad=True)
        nn.init.normal_(self.pos_emb, std=0.02)

        self.norm = nn.LayerNorm(hidden_size)
        self.action_encoder = DomainAwareLinear(
            dim_action + dim_time + dim_propio, hidden_size, num_domains=num_domains
        )
        self.action_decoder = DomainAwareLinear(hidden_size, dim_action, num_domains=num_domains)

        if len_soft_prompts > 0:
            self.soft_prompt_hub = nn.Embedding(num_domains, len_soft_prompts * hidden_size)
            nn.init.normal_(self.soft_prompt_hub.weight, std=0.02)

        # =========================================================================
        # GeoStack-XVLA v3 — Side stack for action-expert spatial-language conditioning
        # =========================================================================
        # Asymmetric parallel transformer: refines [DA3 deep | T5] bank through depth,
        # action expert action-tokens cross-attend at paired layers via gated adapters.
        # Identity-at-step-0 via zero-init out_proj + closed gate + alpha schedule.
        side_cfg = gc.get("side_stack", {}) or {}
        self.side_stack_enabled = bool(side_cfg.get("enabled", False))
        self.side_stack = None
        self.s_to_x_xattn = None
        self.side_alpha_schedule = None
        self._side_pairing_map: Dict[int, int] = {}
        if self.side_stack_enabled:
            from .side_stack import SideStack, GatedCrossAttn, default_pairing
            from .geostack import AlphaSchedule
            side_depth   = int(side_cfg.get("depth", 6))
            side_hidden  = int(side_cfg.get("hidden_dim", hidden_size))
            side_heads   = int(side_cfg.get("num_heads", num_heads))
            side_mlp_r   = float(side_cfg.get("mlp_ratio", 4.0))
            side_dropout = float(side_cfg.get("dropout", 0.0))
            self.side_stack = SideStack(
                n_layers=side_depth, hidden_dim=side_hidden,
                num_heads=side_heads, mlp_ratio=side_mlp_r, dropout=side_dropout,
            )
            # Pairing: which action-expert layer fires cross-attn to which side layer.
            # Default: every (depth / side_depth)-th action layer; last side layer
            # always pairs with the LAST action layer.
            self._side_pairing_map = default_pairing(depth, side_depth)
            # One cross-attn adapter per pairing point (action queries side, gated).
            n_pair = len(self._side_pairing_map)
            self.s_to_x_xattn = nn.ModuleList([
                GatedCrossAttn(
                    hidden_dim=hidden_size,
                    num_heads=int(side_cfg.get("xattn_num_heads", 8)),
                    gate_init_bias=float(side_cfg.get("gate_init_bias", -4.0)),
                    dropout=float(side_cfg.get("xattn_dropout", 0.0)),
                ) for _ in range(n_pair)
            ])
            self.side_alpha_schedule = AlphaSchedule(
                warmup_iters=int(side_cfg.get("alpha_warmup_iters", 5000)),
                ramp_to_01_iters=int(side_cfg.get("alpha_ramp_to_01_iters", 10000)),
                ramp_to_1_iters=int(side_cfg.get("alpha_ramp_to_1_iters", 20000)),
                alpha_at_warmup_end=float(side_cfg.get("alpha_at_warmup_end", 0.001)),
            )
            # Whether cross-attn Q is just the action slice (cheaper) or full seq.
            self.side_xattn_target = str(
                side_cfg.get("xattn_target", "action_only")
            ).lower()
            if self.side_xattn_target not in ("action_only", "full_sequence"):
                raise ValueError(f"side_xattn_target must be 'action_only' or 'full_sequence', got {self.side_xattn_target!r}")

            # v3.7 — BIDIRECTIONAL cross-attention.
            # When enabled, at each pairing point the side stack ALSO queries the
            # action expert state (s queries x). This makes the side stack's K/V
            # output ACTION-CONTEXT-AWARE: it knows what the action expert is
            # currently "asking" and can produce features tailored to that question.
            #
            # Motivation: in v3.6 ckpt-5000, cross-attn was only 1.83% effective
            # (action change when bank zeroed). Side stack trained via aux loss but
            # cross-attn adapters barely moved. Adding x_to_s gives the side stack
            # an ADDITIONAL gradient path back FROM the action expert, plus makes
            # its outputs more action-relevant via direct action-context conditioning.
            #
            # Identity-at-step-0 preserved by the same α schedule.
            self.side_xattn_bidirectional = bool(side_cfg.get("bidirectional", False))
            self.x_to_s_xattn = None
            if self.side_xattn_bidirectional:
                self.x_to_s_xattn = nn.ModuleList([
                    GatedCrossAttn(
                        hidden_dim=hidden_size,
                        num_heads=int(side_cfg.get("xattn_num_heads", 8)),
                        gate_init_bias=float(side_cfg.get("gate_init_bias", -4.0)),
                        dropout=float(side_cfg.get("xattn_dropout", 0.0)),
                    ) for _ in range(n_pair)
                ])

        # ------------------------------------------------------------------
        # Spatial-language Method A/B. Built only when explicitly enabled.
        # The shared DA3/T5 tokenizer lives on XVLA; these modules consume its
        # three per-view banks and update only the action-token slice.
        # ------------------------------------------------------------------
        self.spatial_refiner = None
        self.spatial_injection_layers = None
        self.spatial_injection_start = int(spatial_lang_cfg.get("method_b_start_layer", max(0, depth - 6)))
        if self.spatial_lang_enabled:
            from .spatial_language import SpatialActionInjectionLayer, SpatialActionRefiner
            if self.spatial_lang_method in ("post_refiner", "method_a", "a"):
                self.spatial_refiner = SpatialActionRefiner(
                    hidden_dim=hidden_size,
                    num_heads=int(spatial_lang_cfg.get("spatial_heads", 8)),
                    depth=int(spatial_lang_cfg.get("method_a_layers", 6)),
                    dropout=float(spatial_lang_cfg.get("dropout", 0.0)),
                )
            elif self.spatial_lang_method in ("final6_injection", "method_b", "b"):
                n_inject = max(0, depth - self.spatial_injection_start)
                self.spatial_injection_layers = nn.ModuleList([
                    SpatialActionInjectionLayer(
                        hidden_dim=hidden_size,
                        num_heads=int(spatial_lang_cfg.get("spatial_heads", 8)),
                        dropout=float(spatial_lang_cfg.get("dropout", 0.0)),
                    )
                    for _ in range(n_inject)
                ])
            else:
                raise ValueError(
                    f"spatial_lang.method={self.spatial_lang_method!r} "
                    "(expected 'post_refiner' or 'final6_injection')"
                )

        self.apply(basic_init)

    def set_side_alpha_step(self, step: int) -> None:
        """Training loop calls this each iter so the side stack alpha advances."""
        if self.side_alpha_schedule is not None:
            self.side_alpha_schedule.set_step(int(step))

    def set_spatial_residual_scale(self, scale: float) -> None:
        """Set fixed scheduled scale for Method A/B spatial residual updates."""
        self.spatial_residual_scale = float(scale)

    @staticmethod
    def _summarize_spatial_stats(stats_list) -> Dict[str, torch.Tensor]:
        if not stats_list:
            return {}
        out = {}
        for key in ("h_action_norm", "main", "left", "right", "merge"):
            vals = [s[key] for s in stats_list if key in s]
            if vals:
                out[key] = torch.stack(vals).mean()
        h = out.get("h_action_norm", None)
        if h is not None:
            denom = h.clamp_min(1e-6)
            for key in ("main", "left", "right", "merge"):
                if key in out:
                    out[f"ratio_{key}"] = out[key] / denom
        return out

    def forward(
        self,
        domain_id: torch.LongTensor,
        vlm_features: torch.Tensor,
        aux_visual_inputs: torch.Tensor,
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
        geometry_tokens: Optional[torch.Tensor] = None,
        spatial_lang_bank: Optional[torch.Tensor] = None,    # v3: side stack input
        spatial_lang_bank_pad: Optional[torch.Tensor] = None,  # v3: bank padding mask
        spatial_lang_banks: Optional[Dict[str, torch.Tensor]] = None,  # Method A/B per-view banks
        return_side_state: bool = False,                     # v3.5: return final side stack state for aux loss
    ):
        """
        Forward pass.

        Inputs
        ------
        domain_id : [B]
        vlm_features : [B, T_vlm, D]
        aux_visual_inputs : [B, T_aux, D]
        action_with_noise : [B, T_action, dim_action]
        proprio : [B, dim_propio]
        t : [B]
        geometry_tokens : optional [B, K, H]
            Compressed DA3 geometry tokens. When provided and geometry fusion
            is enabled, the policy tokens cross-attend onto these.

        Returns
        -------
        Tensor
            Predicted actions, [B, T_action, dim_action]
        """
        B, num_actions = action_with_noise.shape[:2]
        self.last_spatial_stats = {}
        do_fuse = (
            self.geometry_fusion is not None and geometry_tokens is not None
        )

        # Encode (action + proprio + time) → tokens
        time_emb = timestep_embedding(t, self.dim_time)                     # [B, dim_time]
        time_tokens = time_emb.unsqueeze(1).expand(B, num_actions, self.dim_time)
        proprio_tokens = proprio.unsqueeze(1).expand(B, num_actions, proprio.shape[-1])
        action_tokens = torch.cat([action_with_noise, proprio_tokens, time_tokens], dim=-1)
        x = self.action_encoder(action_tokens, domain_id)                   # [B, T_action, H]

        # Project visual streams and concatenate
        if self.use_hetero_proj:
            x = torch.cat(
                [x, self.vlm_proj(vlm_features, domain_id), self.aux_visual_proj(aux_visual_inputs, domain_id)],
                dim=1,
            )
        else:
            x = torch.cat([x, self.vlm_proj(vlm_features), self.aux_visual_proj(aux_visual_inputs)], dim=1)

        tokens_before_softprompt = x.shape[1]

        # Add positional embeddings (truncate if needed)
        seq_len = x.shape[1]
        if seq_len > self.pos_emb.shape[1]:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len_seq={self.pos_emb.shape[1]}."
            )
        x = x + self.pos_emb[:, :seq_len, :]

        # Append soft prompts
        soft_prompts = None
        if self.len_soft_prompts > 0:
            soft_prompts = self.soft_prompt_hub(domain_id).view(B, self.len_soft_prompts, self.hidden_size)
            x = torch.cat([x, soft_prompts], dim=1)

        # One-time shape diagnostics (gated by config.debug_shapes).
        if self.geometry_debug_shapes and not self._geom_shape_logged:
            self._geom_shape_logged = True
            logger.info(
                "[geometry][shapes] action_tokens=%s vlm=%s aux=%s "
                "soft_prompts=%s policy_tokens=%s geometry_tokens=%s "
                "fusion_position=%s",
                tuple((B, num_actions, self.hidden_size)),
                tuple(vlm_features.shape),
                tuple(aux_visual_inputs.shape),
                tuple(soft_prompts.shape) if soft_prompts is not None else None,
                tuple(x.shape),
                tuple(geometry_tokens.shape) if geometry_tokens is not None else None,
                self.geometry_fusion_position,
            )

        # --- Geometry fusion: before policy --------------------------------
        if do_fuse and self.geometry_fusion_position == "before_policy":
            x = self.geometry_fusion(x, geometry_tokens)

        # Gated action-to-spatial cross-attention runs IF (a) configured AND
        # (b) geometry_tokens were provided. Bypass is byte-identical when
        # either is false (preserves baseline behavior for old configs).
        do_spatial = (
            self.spatial_cross_attn_layers is not None
            and geometry_tokens is not None
        )

        # GeoStack v3 — side stack co-evolution state.
        # `s` starts as the input bank and gets stepped at every pairing point
        # by the corresponding side stack block, then cross-attended INTO `x`.
        do_side = (
            self.side_stack_enabled
            and self.side_stack is not None
            and spatial_lang_bank is not None
        )
        if do_side:
            s = spatial_lang_bank.to(x.dtype)
            side_alpha = self.side_alpha_schedule.value()
        else:
            s = None
            side_alpha = 0.0

        # Transformer backbone (optionally fuse once mid-stack).
        n_blocks = len(self.blocks)
        mid = n_blocks // 2
        for i, block in enumerate(self.blocks):
            x = block(x)
            if (
                do_fuse
                and self.geometry_fusion_position == "inside_policy"
                and i == mid
            ):
                x = self.geometry_fusion(x, geometry_tokens)
            # DA3-XVLA: gated cross-attention adapter from policy tokens to
            # the K=160 spatial bank. Two dispatch modes:
            #   "action_only"    → only action slice queries spatial (preserves
            #                      the VLM wall). VLM/aux/soft pass through.
            #   "full_sequence"  → the entire policy sequence queries spatial.
            #                      VLM, aux, soft all receive spatial updates.
            #                      Broader fusion at the cost of breaking the wall.
            if do_spatial and i in self._spatial_layer_active:
                adapter = self.spatial_cross_attn_layers[i]
                if self.spatial_xattn_target == "full_sequence":
                    # Entire sequence queries spatial; adapter Q-length is flexible.
                    x = adapter(x, geometry_tokens)
                else:  # "action_only"
                    action_hidden = x[:, :num_actions]
                    updated_action = adapter(action_hidden, geometry_tokens)
                    x = torch.cat([updated_action, x[:, num_actions:]], dim=1)

            # ─── GeoStack v3: side stack co-evolution ─────────────────────────
            # At each paired action layer i: first step the corresponding side
            # block (so s_{side_idx} is fresh), then cross-attend.
            # v3.7: optionally BIDIRECTIONAL — both x queries s AND s queries x.
            if do_side and i in self._side_pairing_map:
                side_idx = self._side_pairing_map[i]
                s = self.side_stack.step(s, side_idx, key_padding_mask=spatial_lang_bank_pad)

                # Compute BOTH updates in parallel using the SAME pre-update x and s.
                # This avoids order-dependence (s-updates-with-new-x vs x-updates-with-new-s).
                # If bidirectional disabled, x_to_s_xattn is None and s_new == s.
                if self.side_xattn_target == "full_sequence":
                    x_new = self.s_to_x_xattn[side_idx](
                        x, s, alpha=side_alpha, kv_padding_mask=spatial_lang_bank_pad,
                    )
                else:  # action_only
                    action_hidden = x[:, :num_actions]
                    updated_action = self.s_to_x_xattn[side_idx](
                        action_hidden, s, alpha=side_alpha, kv_padding_mask=spatial_lang_bank_pad,
                    )
                    x_new = torch.cat([updated_action, x[:, num_actions:]], dim=1)

                if self.x_to_s_xattn is not None:
                    # Side stack queries action expert state. K/V is the full action
                    # expert sequence [action | vlm | aux | soft]; no padding mask
                    # needed (no padded positions in the main action expert stream).
                    s_new = self.x_to_s_xattn[side_idx](
                        s, x, alpha=side_alpha, kv_padding_mask=None,
                    )
                else:
                    s_new = s

                # Atomic update
                x, s = x_new, s_new

            # ─── Spatial-language Method B: inject after final XVLA blocks ───
            if (
                self.spatial_injection_layers is not None
                and spatial_lang_banks is not None
                and i >= self.spatial_injection_start
            ):
                inj_idx = i - self.spatial_injection_start
                if 0 <= inj_idx < len(self.spatial_injection_layers):
                    h_action = x[:, :num_actions]
                    h_action, stats = self.spatial_injection_layers[inj_idx](
                        h_action,
                        spatial_lang_banks["main"],
                        spatial_lang_banks["left"],
                        spatial_lang_banks["right"],
                        spatial_scale=self.spatial_residual_scale,
                        return_stats=True,
                    )
                    prior = self.last_spatial_stats.get("_raw", [])
                    prior.append(stats)
                    self.last_spatial_stats["_raw"] = prior
                    x = torch.cat([h_action, x[:, num_actions:]], dim=1)

        if "_raw" in self.last_spatial_stats:
            raw = self.last_spatial_stats.pop("_raw")
            self.last_spatial_stats.update(self._summarize_spatial_stats(raw))

        # --- Geometry fusion: after policy ---------------------------------
        if do_fuse and self.geometry_fusion_position == "after_policy":
            x = self.geometry_fusion(x, geometry_tokens)

        # Decode only the action segment
        h_action = x[:, :num_actions]
        # Spatial-language Method A: 6-layer post-XVLA action-only refiner.
        if self.spatial_refiner is not None and spatial_lang_banks is not None:
            h_action, stats = self.spatial_refiner(
                h_action,
                spatial_lang_banks,
                spatial_scale=self.spatial_residual_scale,
                return_stats=True,
            )
            self.last_spatial_stats.update(self._summarize_spatial_stats(stats))
        pred_action = self.action_decoder(self.norm(h_action), domain_id)
        if return_side_state:
            # Return the final side stack state (after all blocks, post-cross-attn).
            # When side stack is disabled, returns None so caller can skip aux loss.
            return pred_action, (s if do_side else None)
        return pred_action
