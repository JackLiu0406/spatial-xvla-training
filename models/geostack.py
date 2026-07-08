# SPDX: same as DA3-XVLA repo
"""
GeoStack-XVLA v2B — frozen DA3-Large spatial injection into Florence-2-large encoder.

Modules:
    SinCos2DPositionalEmbedding   — fixed sin/cos PE for continuous normalized [0,1]^2 coords
    AlphaSchedule                 — piecewise-linear residual-alpha controller
    DA3SpatialTokenBuilder        — DA3 feat + ray + 2D PE → spatial K/V tokens (per level)
    GeoStackCrossAttention        — gated per-token cross-attn + learnable spatial-distance bias
    GeoFusedFlorence2Encoder      — Florence2Encoder subclass that injects GeoStack adapters
                                    at configured layer indices (default 6, 9, 11)
    build_pos_orig_vlm            — utility: token-grid → normalized original-image coords (Florence)
    build_pos_orig_da3            — utility: DA3 token-grid → normalized original-image coords + valid mask

Design contracts:
  * At alpha=0 (or before warmup completes), forward is bit-identical to baseline (no residual added).
  * GeoStack cross-attn output projection is zero-initialized so even at alpha=1 the very first step
    contributes zero, then learns. Gate MLP final bias starts at -4.0 (sigmoid ≈ 0.018, near-closed).
  * State-passing into the Florence encoder is done via `set_geo_context()` BEFORE calling encoder(...),
    cleared with `clear_geo_context()` in a try/finally.
  * Subclass rebind via `enc.__class__ = GeoFusedFlorence2Encoder` preserves all pretrained weights /
    optimizer state. New submodules attached AFTER rebind register under encoder.state_dict.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import BaseModelOutput

# We subclass Florence2Encoder so we depend on it; importing here keeps the file
# self-contained, and modeling_xvla.py's wiring code can `from .geostack import ...`.
from .modeling_florence2 import (
    Florence2Encoder,
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
)


# =============================================================================
# 1. Sin/cos 2D positional embedding for continuous normalized coordinates
# =============================================================================

class SinCos2DPositionalEmbedding(nn.Module):
    """Fixed (non-learned) sinusoidal 2D positional embedding.

    Input  : coords of shape [B, N, 2] in [0, 1] × [0, 1] (normalized original-image coords).
    Output : embedding of shape [B, N, d_model].

    Half of `d_model` is allocated to x (u), the other half to y (v). Within each
    axis half, half are sin and half are cos at geometrically-spaced frequencies.
    """

    def __init__(self, d_model: int, max_period: float = 10000.0):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError(f"SinCos2DPositionalEmbedding: d_model must be divisible by 4, got {d_model}")
        self.d_model = int(d_model)
        d_per_axis = self.d_model // 2  # half for x, half for y
        half = d_per_axis // 2          # within an axis, half are sin / half are cos
        # Log-spaced frequencies
        freqs = torch.exp(-torch.arange(half).float() * math.log(max_period) / max(1, half - 1))
        self.register_buffer("freqs", freqs, persistent=False)  # [half]

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [B, N, 2] in [0, 1]^2
        x = coords[..., 0:1]   # [B, N, 1]
        y = coords[..., 1:2]
        # Scale to angular freq band [0, 2π * f_k]
        x_args = x * self.freqs * (2.0 * math.pi)   # [B, N, half]
        y_args = y * self.freqs * (2.0 * math.pi)
        x_emb = torch.cat([torch.sin(x_args), torch.cos(x_args)], dim=-1)   # [B, N, 2*half = d_per_axis]
        y_emb = torch.cat([torch.sin(y_args), torch.cos(y_args)], dim=-1)
        return torch.cat([x_emb, y_emb], dim=-1)    # [B, N, d_model]


# =============================================================================
# 2. Alpha schedule (piecewise linear)
# =============================================================================

class AlphaSchedule(nn.Module):
    """Residual-alpha controller for GeoStack injection.

        step <  warmup_iters             -> alpha = 0.0           (injection silent)
        step == warmup_iters             -> alpha = alpha_at_warmup_end (small NONZERO)
        step in (warmup, ramp_to_01)     -> alpha linearly  alpha_at_warmup_end -> 0.1
        step in [ramp_to_01, ramp_to_1)  -> alpha linearly  0.1 -> 1.0
        step >= ramp_to_1                -> alpha = 1.0

    `alpha_at_warmup_end` MUST be > 0. Reason: if α==0 exactly at the moment
    the ramp begins, the gradient through the GeoStack residual is also 0, so
    the adapter weights (attn.out_proj, gate_mlp, etc.) get NO learning signal
    on that first step. Starting at a small positive value (e.g. 1e-3) gives
    nonzero gradient immediately, so the adapter can start learning the
    moment the ramp begins.

    Step is tracked via a non-persistent buffer; the training loop sets it
    every iter via XVLA.set_geostack_step(step).
    """

    def __init__(self, warmup_iters: int = 5000,
                 ramp_to_01_iters: int = 10000,
                 ramp_to_1_iters: int = 20000,
                 alpha_at_warmup_end: float = 0.001):
        super().__init__()
        if not (warmup_iters <= ramp_to_01_iters <= ramp_to_1_iters):
            raise ValueError(
                f"AlphaSchedule: require warmup ({warmup_iters}) <= ramp_to_01 ({ramp_to_01_iters}) <= ramp_to_1 ({ramp_to_1_iters})"
            )
        if not (0.0 < alpha_at_warmup_end <= 0.1):
            raise ValueError(
                f"AlphaSchedule: alpha_at_warmup_end must be in (0, 0.1]; got {alpha_at_warmup_end}"
            )
        self.warmup_iters = int(warmup_iters)
        self.ramp_to_01_iters = int(ramp_to_01_iters)
        self.ramp_to_1_iters = int(ramp_to_1_iters)
        self.alpha_at_warmup_end = float(alpha_at_warmup_end)
        # Tracked step as a buffer so it can be moved-to-device and saved
        self.register_buffer("_step", torch.zeros((), dtype=torch.long), persistent=False)

    def set_step(self, step: int) -> None:
        self._step.fill_(int(step))

    @property
    def step(self) -> int:
        return int(self._step.item())

    def value(self) -> float:
        s = self.step
        if s < self.warmup_iters:
            return 0.0
        if s < self.ramp_to_01_iters:
            denom = max(1, self.ramp_to_01_iters - self.warmup_iters)
            frac = (s - self.warmup_iters) / denom
            return self.alpha_at_warmup_end + frac * (0.1 - self.alpha_at_warmup_end)
        if s < self.ramp_to_1_iters:
            denom = max(1, self.ramp_to_1_iters - self.ramp_to_01_iters)
            return 0.1 + (s - self.ramp_to_01_iters) / denom * 0.9
        return 1.0

    def extra_repr(self) -> str:
        return (f"warmup={self.warmup_iters}, ramp_to_01={self.ramp_to_01_iters}, "
                f"ramp_to_1={self.ramp_to_1_iters}, alpha_at_warmup_end={self.alpha_at_warmup_end}, "
                f"current_step={self.step}, alpha={self.value():.4f}")


# =============================================================================
# 3. DA3 spatial token builder (per feature level)
# =============================================================================

class DA3SpatialTokenBuilder(nn.Module):
    """Project one DA3 feature level to VLM hidden dim and combine with ray + 2D PE.

    Per-level pipeline:
        feat_emb = Linear(C_in -> D) -> GELU -> Linear -> LayerNorm
        ray_emb  = Linear(ray_in -> D/2) -> GELU -> Linear -> LayerNorm  (optional)
        pe_emb   = SinCos2D(coords)                                       (optional)
        out      = feat_emb + sigmoid(g_ray) * ray_emb + sigmoid(g_pos) * pe_emb

    Gates `g_ray` and `g_pos` are learnable scalars with negative-bias init so
    ray/PE contributions start near-zero (feature term dominates initially).
    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        ray_in: int = 3,
        use_ray: bool = True,
        use_2d_pos: bool = True,
        g_ray_init: float = -4.0,
        g_pos_init: float = -2.0,
    ):
        super().__init__()
        self.c_in = int(c_in)
        self.d_model = int(d_model)
        self.use_ray = bool(use_ray)
        self.use_2d_pos = bool(use_2d_pos)

        # DA3 feat -> VLM hidden (canonical _build_projector pattern reused).
        self.feat_proj = nn.Sequential(
            nn.Linear(c_in, d_model),
            nn.GELU(approximate="tanh"),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        if self.use_ray:
            self.ray_proj = nn.Sequential(
                nn.Linear(int(ray_in), d_model // 2),
                nn.GELU(approximate="tanh"),
                nn.Linear(d_model // 2, d_model),
                nn.LayerNorm(d_model),
            )
            self.g_ray = nn.Parameter(torch.tensor(float(g_ray_init), dtype=torch.float32))
        else:
            self.register_module("ray_proj", None)
            self.register_parameter("g_ray", None)

        if self.use_2d_pos:
            self.pe2d = SinCos2DPositionalEmbedding(d_model)
            self.g_pos = nn.Parameter(torch.tensor(float(g_pos_init), dtype=torch.float32))
        else:
            self.register_module("pe2d", None)
            self.register_parameter("g_pos", None)

    def forward(
        self,
        feat: torch.Tensor,
        coords: torch.Tensor,
        ray: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            feat:   [B, N, C_in]  flattened DA3 feature tokens for this level
            coords: [B, N, 2]     normalized original-image coords in [0, 1]^2
            ray:    [B, N, ray_in] normalized ray direction (optional; required iff use_ray)

        Returns:
            [B, N, D]  spatial tokens in VLM hidden space.
        """
        emb = self.feat_proj(feat)

        if self.use_ray:
            if ray is None:
                raise ValueError("DA3SpatialTokenBuilder: use_ray=True but ray is None")
            r = self.ray_proj(ray.to(emb.dtype))
            emb = emb + torch.sigmoid(self.g_ray) * r

        if self.use_2d_pos:
            pe = self.pe2d(coords).to(emb.dtype)
            emb = emb + torch.sigmoid(self.g_pos) * pe

        return emb


# =============================================================================
# 4. GeoStack cross-attention (gated, optional spatial-distance bias)
# =============================================================================

class GeoStackCrossAttention(nn.Module):
    """Per-layer Florence ← DA3 cross-attention adapter with gated residual fusion.

    Differences vs. PerTokenGatedSpatialCrossAttention in geometry_conditioning.py:
      * Adds an OPTIONAL learnable spatial-distance attention-logit bias term
        `-softplus(lambda) * ||pos_vlm_i - pos_da3_j||^2`, added pre-softmax.
      * `attn.out_proj.weight` is explicitly zero-initialized (PerTokenGated relied
        on HF's loader zero-ing; that doesn't happen for fresh-built GeoStack
        modules), guaranteeing identity-at-step-0.
      * Per-token gate `gate_mlp` final bias init = -4.0 → sigmoid(0) ≈ 0.018,
        so even when out_proj starts learning the residual stays small initially.

    Forward:
        kv         = LayerNorm(spatial_tokens)
        q          = LayerNorm(vlm_hidden)
        attn_logits = Q · K^T / sqrt(d_head) + spatial_bias (broadcast over heads)
                     + (-inf for invalid DA3 tokens)
        attn_out   = MHA(q, kv, kv)
        gate       = sigmoid(GateMLP(LN(vlm_hidden)))      [B, T, D]
        delta      = gate * attn_out
        return vlm_hidden + alpha * delta

    `alpha` is provided externally per-step (read from an AlphaSchedule).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        spatial_token_dim: Optional[int] = None,
        use_spatial_bias: bool = True,
        spatial_bias_lambda_init: float = 1.0,
        gate_mlp_hidden: Optional[int] = None,
        gate_init_bias: float = -4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

        s_dim = self.hidden_dim if spatial_token_dim is None else int(spatial_token_dim)
        if s_dim == self.hidden_dim:
            self.spatial_proj: nn.Module = nn.Identity()
        else:
            self.spatial_proj = nn.Linear(s_dim, self.hidden_dim)

        self.q_norm = nn.LayerNorm(self.hidden_dim)
        self.kv_norm = nn.LayerNorm(self.hidden_dim)
        self.attn = nn.MultiheadAttention(
            self.hidden_dim, num_heads=self.num_heads,
            dropout=float(dropout), batch_first=True,
        )
        # Identity-at-step-0 — zero-init out_proj. (Don't depend on HF loader behavior.)
        nn.init.zeros_(self.attn.out_proj.weight)
        if self.attn.out_proj.bias is not None:
            nn.init.zeros_(self.attn.out_proj.bias)

        # Per-token gate
        gh = self.hidden_dim // 4 if gate_mlp_hidden is None else int(gate_mlp_hidden)
        self.gate_norm = nn.LayerNorm(self.hidden_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, gh),
            nn.GELU(approximate="tanh"),
            nn.Linear(gh, self.hidden_dim),
        )
        # Initialize gate-MLP final bias to a strong negative value so sigmoid(out) ≈ 0
        # at init; the model has to actively learn to open the gate.
        nn.init.constant_(self.gate_mlp[-1].bias, float(gate_init_bias))
        # Also keep final-layer weights small so initial gate is near-uniformly-closed.
        nn.init.normal_(self.gate_mlp[-1].weight, mean=0.0, std=1e-4)

        # Spatial-distance bias
        self.use_spatial_bias = bool(use_spatial_bias)
        if self.use_spatial_bias:
            # softplus(lambda) gives a positive scale; init to log(exp(1)-1) so softplus = 1.0
            inv = math.log(math.expm1(float(spatial_bias_lambda_init)))
            # fp32 dtype to preserve softplus precision under bf16 autocast.
            self.spatial_bias_lambda = nn.Parameter(torch.tensor(inv, dtype=torch.float32))
        else:
            self.register_parameter("spatial_bias_lambda", None)

    def forward(
        self,
        vlm_hidden: torch.Tensor,                       # [B, T, D]
        spatial_tokens: torch.Tensor,                   # [B, K, D_s]
        vlm_coords: Optional[torch.Tensor] = None,      # [B, T, 2] or [1, T, 2] in [0,1]^2
        spatial_coords: Optional[torch.Tensor] = None,  # [B, K, 2] or [1, K, 2] in [0,1]^2
        vision_mask: Optional[torch.Tensor] = None,     # [B, T] or [1, T] bool — True if Q is
                                                        # a vision token; spatial bias only
                                                        # applies to vision rows. Text rows get
                                                        # bias=0 (no spatial preference, full
                                                        # softmax over DA3 tokens).
        spatial_valid_mask: Optional[torch.Tensor] = None,  # [B, K] bool, True=valid
        alpha: float = 1.0,
    ) -> torch.Tensor:
        if alpha <= 0.0:
            return vlm_hidden   # short-circuit: identity-at-step-0 / pre-warmup

        B, T, D = vlm_hidden.shape
        K = spatial_tokens.shape[1]

        kv = self.kv_norm(self.spatial_proj(spatial_tokens))
        q = self.q_norm(vlm_hidden)

        # ---- Build attention bias + key padding mask ----
        attn_mask = None
        key_padding_mask = None
        if self.use_spatial_bias and vlm_coords is not None and spatial_coords is not None:
            # Force fp32 for distance computation so bf16 autocast doesn't
            # quantize [0,1]² coords (only ~3 decimal digits in bf16).
            v = (vlm_coords if vlm_coords.shape[0] == B else vlm_coords.expand(B, -1, -1)).float()
            s = (spatial_coords if spatial_coords.shape[0] == B else spatial_coords.expand(B, -1, -1)).float()
            # squared L2 distance in normalized coord space [0,1]^2 (so dist2 in [0, 2])
            dist2 = ((v.unsqueeze(2) - s.unsqueeze(1)) ** 2).sum(-1)                # [B, T, K] fp32
            # softplus in fp32 (lambda is registered fp32; ensures precise scale)
            scale = F.softplus(self.spatial_bias_lambda.float())                    # fp32 scalar
            bias = (-scale * dist2).to(q.dtype)                                     # cast to q.dtype

            # Mask out text-token rows: their vlm_coords are sentinels with no
            # spatial meaning. Setting bias=0 for text rows means text tokens
            # apply uniform attention over DA3 K/V (no spatial preference) and
            # are not penalized for being "far" from DA3 tokens.
            if vision_mask is not None:
                vm = vision_mask
                if vm.shape[0] == 1:
                    vm = vm.expand(B, -1)
                vm = vm.to(bias.dtype).unsqueeze(-1)                                # [B, T, 1]
                bias = bias * vm

            if spatial_valid_mask is not None:
                svm = spatial_valid_mask
                if svm.shape[0] == 1:
                    svm = svm.expand(B, -1)
                invalid = (~svm.bool()).unsqueeze(1).expand(B, T, K)                # [B, T, K]
                # Use a dtype-correct -inf to avoid bf16 NaN propagation in
                # downstream softmax (literal float('-inf') has fp32 dtype).
                neg_inf = torch.tensor(float("-inf"), dtype=bias.dtype, device=bias.device)
                bias = bias.masked_fill(invalid, neg_inf)

            # nn.MultiheadAttention expects [B*nh, T, K] when attn_mask is per-batch.
            # .contiguous() before reshape forces materialization of the expand()'d
            # broadcast — otherwise reshape() triggers an implicit copy each forward.
            attn_mask = bias.unsqueeze(1).expand(B, self.num_heads, T, K).contiguous().reshape(
                B * self.num_heads, T, K
            )
        elif spatial_valid_mask is not None:
            # No spatial bias: use simpler key_padding_mask path (True=ignore)
            svm = spatial_valid_mask if spatial_valid_mask.shape[0] == B else spatial_valid_mask.expand(B, -1)
            key_padding_mask = ~svm.bool()   # [B, K]

        attn_out, _ = self.attn(
            q, kv, kv,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        # Per-token per-channel gate over action_hidden context
        gate = torch.sigmoid(self.gate_mlp(self.gate_norm(vlm_hidden)))   # [B, T, D]
        delta = gate * attn_out                                            # [B, T, D]
        return vlm_hidden + float(alpha) * delta


# =============================================================================
# 5. Coordinate utilities — Florence token grid and DA3 token grid
# =============================================================================

def build_pos_orig_vlm(
    h_grid: int, w_grid: int,
    device: torch.device, dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Florence input is a non-aspect-preserved stretch to 224×224, so Florence
    token positions map LINEARLY to normalized original-image coords:

        u = (x + 0.5) / w_grid     in [0, 1]
        v = (y + 0.5) / h_grid     in [0, 1]

    Returns:
        coords: [1, h_grid*w_grid, 2]  (broadcastable over batch)
    """
    ys, xs = torch.meshgrid(
        torch.arange(h_grid, device=device, dtype=dtype),
        torch.arange(w_grid, device=device, dtype=dtype),
        indexing="ij",
    )
    u = (xs + 0.5) / w_grid
    v = (ys + 0.5) / h_grid
    coords = torch.stack([u, v], dim=-1).view(1, h_grid * w_grid, 2)
    return coords


def build_pos_orig_da3(
    h_grid: int, w_grid: int,
    device: torch.device, dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute DA3 token-center positions in normalized ORIGINAL-image coords.

    DA3 receives the image at (da3_input_h, da3_input_w) where both dims are
    chosen as ASPECT-CORRECT multiples of patch_size. There is NO padding —
    every token corresponds to real image content. The DA3 frame is just a
    uniform per-axis scaling of the original, so normalized token-center coords
    in the DA3 frame EQUAL normalized coords in the original image.

    Per-token center in normalized coords:
        u = (x_grid + 0.5) / w_grid    in [0, 1]
        v = (y_grid + 0.5) / h_grid    in [0, 1]

    Because both Florence and DA3 stretches/resizes are deterministic per-axis
    operations on the same source content, a Florence token at normalized (u, v)
    and a DA3 token at normalized (u, v) refer to the SAME pixel region of the
    original image. Hence `dist² = (u_vlm - u_da3)² + (v_vlm - v_da3)²` is a
    meaningful spatial-distance signal in original-image space.

    Returns:
        coords: [1, h_grid*w_grid, 2]   broadcastable over batch
    """
    ys, xs = torch.meshgrid(
        torch.arange(h_grid, device=device, dtype=dtype),
        torch.arange(w_grid, device=device, dtype=dtype),
        indexing="ij",
    )
    u = (xs + 0.5) / w_grid
    v = (ys + 0.5) / h_grid
    coords = torch.stack([u, v], dim=-1).view(1, h_grid * w_grid, 2)
    return coords


# =============================================================================
# 6. GeoFusedFlorence2Encoder — Florence2Encoder subclass with injection
# =============================================================================

class GeoFusedFlorence2Encoder(Florence2Encoder):
    """Florence2Encoder + GeoStack injection at configured layer indices.

    Designed to be installed via class-rebind on an existing pretrained encoder:

        enc = vlm.language_model.model.encoder        # original Florence2Encoder
        enc.__class__ = GeoFusedFlorence2Encoder      # in-place class swap
        enc.attach_geostack(
            geo_inject_layers=(6, 9, 11),
            geo_fusers=nn.ModuleDict(...),            # one GeoStackCrossAttention per layer
            alpha_schedule=AlphaSchedule(...),
        )

    Per-step usage from the outer XVLA forward:
        enc.set_geo_context(geo_kv_by_layer, geo_coords_da3_by_layer, geo_valid_by_layer, vlm_coords)
        enc_out = enc(attention_mask=..., inputs_embeds=...)[0]
        enc.clear_geo_context()

    The override copies Florence2Encoder.forward verbatim except for ONE added
    block (the injection) right after `hidden_states = layer_outputs[0]`.
    """

    # -------- Setup / teardown helpers (called from modeling_xvla.py) --------

    def attach_geostack(
        self,
        geo_inject_layers: Tuple[int, ...],
        geo_fusers: nn.ModuleDict,
        alpha_schedule: AlphaSchedule,
    ) -> None:
        """Attach the GeoStack adapters and schedule. Call AFTER class-rebind."""
        # `tuple` for fast `in` lookup
        self.geo_inject_layers: Tuple[int, ...] = tuple(int(i) for i in geo_inject_layers)
        # ModuleDict keyed by string of layer index
        if not isinstance(geo_fusers, nn.ModuleDict):
            raise TypeError("geo_fusers must be an nn.ModuleDict keyed by str(layer_idx)")
        self.geo_fusers: nn.ModuleDict = geo_fusers
        self.geo_alpha_schedule: AlphaSchedule = alpha_schedule
        self._geo_attached = True
        # Default empty context (set per-forward)
        self._geo_kv: Optional[Dict[int, torch.Tensor]] = None
        self._geo_da3_coords: Optional[Dict[int, torch.Tensor]] = None
        self._geo_da3_valid: Optional[Dict[int, torch.Tensor]] = None
        self._geo_vlm_coords: Optional[torch.Tensor] = None
        self._geo_vision_mask: Optional[torch.Tensor] = None

    def set_geo_context(
        self,
        geo_kv: Dict[int, torch.Tensor],            # {layer_idx: [B, K_l, D]}
        geo_da3_coords: Dict[int, torch.Tensor],    # {layer_idx: [B or 1, K_l, 2]}
        geo_da3_valid: Optional[Dict[int, torch.Tensor]] = None,  # {layer_idx: [B or 1, K_l] bool}
        vlm_coords: Optional[torch.Tensor] = None,  # [1, T_vlm, 2]  (shared across injections)
        vision_mask: Optional[torch.Tensor] = None, # [1 or B, T_vlm] bool, True for vision tokens
    ) -> None:
        self._geo_kv = geo_kv
        self._geo_da3_coords = geo_da3_coords
        self._geo_da3_valid = geo_da3_valid or {}
        self._geo_vlm_coords = vlm_coords
        self._geo_vision_mask = vision_mask

    def clear_geo_context(self) -> None:
        self._geo_kv = None
        self._geo_da3_coords = None
        self._geo_da3_valid = None
        self._geo_vlm_coords = None
        self._geo_vision_mask = None

    # -------- forward (verbatim copy from Florence2Encoder + ONE injection block) --------

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # ---- input prep (verbatim from Florence2Encoder.forward) ----
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_ids = input_ids.view(-1, input_ids.shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        embed_pos = self.embed_positions(input)
        embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if attention_mask is not None:
            if self._use_flash_attention_2:
                attention_mask = attention_mask if 0 in attention_mask else None
            elif self._use_sdpa and head_mask is None and not output_attentions:
                attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, inputs_embeds.dtype)
            else:
                attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        # ---- Pull GeoStack context once (if attached) ----
        geo_attached = bool(getattr(self, "_geo_attached", False))
        geo_kv = getattr(self, "_geo_kv", None) if geo_attached else None
        geo_alpha = self.geo_alpha_schedule.value() if (geo_attached and geo_kv is not None) else 0.0

        # ---- Layer loop (verbatim, with single added injection block) ----
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )
                hidden_states = layer_outputs[0]

                # ====== GEOSTACK INJECTION ======
                if (
                    geo_attached
                    and geo_kv is not None
                    and idx in self.geo_inject_layers
                    and geo_alpha > 0.0
                    and hidden_states is not None
                ):
                    fuser = self.geo_fusers[str(idx)]
                    sp_tokens = geo_kv[idx].to(hidden_states.dtype)
                    # Coords stay fp32 — they are used to compute spatial-distance
                    # bias which needs precision; the bias is cast to q.dtype only
                    # right before being added to attn logits (inside the fuser).
                    sp_coords = self._geo_da3_coords[idx] if self._geo_da3_coords is not None else None
                    sp_valid = self._geo_da3_valid.get(idx, None) if self._geo_da3_valid else None
                    vlm_coords = self._geo_vlm_coords if self._geo_vlm_coords is not None else None
                    vision_mask = self._geo_vision_mask
                    hidden_states = fuser(
                        vlm_hidden=hidden_states,
                        spatial_tokens=sp_tokens,
                        vlm_coords=vlm_coords,
                        spatial_coords=sp_coords,
                        vision_mask=vision_mask,
                        spatial_valid_mask=sp_valid,
                        alpha=geo_alpha,
                    )
                    # Re-apply fp16 inf-clamp from Florence2EncoderLayer (lines 1280-1284)
                    # in case the residual addition pushed values out of range.
                    if hidden_states.dtype in (torch.float16,):
                        clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


# =============================================================================
# 7. GeoStack-XVLA v2C — wrist-aware aux-visual cross-attention
# =============================================================================
# v2B injects only at the Florence encoder (main view × main DA3). The wrist
# camera views skip the encoder entirely — they go through Florence's vision
# tower then straight into the action expert as `aux_visual_inputs`.
# v2C adds a parallel injection point: each wrist view's aux tokens cross-attend
# with that view's own DA3 features (enriched with world-frame ray embeddings
# derived from per-frame extrinsics). This injects 3D-grounded geometry signal
# the model cannot derive from Florence alone — addressing v2B's failure mode
# where the optimizer learned to suppress main-view geometry as redundant.


class WorldRayEmbedding(nn.Module):
    """MLP that projects per-token world-frame 6D ray (origin || direction)
    into the model's hidden dim, with a learnable gate so the contribution
    starts small and grows.

    Input  : [B, V, N, 6]  where 6 = (origin_world_xyz, direction_world_xyz)
    Output : [B, V, N, D]  ready to be added to projected DA3 token features.
    """

    def __init__(self, d_model: int, hidden: int = 256, g_init: float = -2.0):
        super().__init__()
        # NO trailing LayerNorm: at identity-at-step-0 (gate near-zero,
        # out_proj=0), the chain can have zero-variance intermediates that
        # cause LN backward to produce NaN. The receiving DA3 tokens have
        # their own normalization downstream so a clean Linear-GELU-Linear
        # is sufficient here.
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, d_model),
        )
        # Scalar gate so we can let the optimizer modulate the world-ray
        # contribution. Init at sigmoid(-2) ≈ 0.12 — small but nonzero so
        # gradient flows from step 0.
        self.g = nn.Parameter(torch.tensor(float(g_init), dtype=torch.float32))

    def forward(self, world_ray: torch.Tensor) -> torch.Tensor:
        # world_ray: [..., 6]   (B, V, N, 6) or any shape ending in 6
        emb = self.mlp(world_ray.to(self.mlp[0].weight.dtype))
        return torch.sigmoid(self.g) * emb


class WristAuxGeoStack(nn.Module):
    """Per-wrist-view cross-attention adapter applied to `aux_visual_inputs`.

    Architecture mirrors v2B's GeoStack at the encoder, but here it runs
    AFTER the Florence vision tower and BEFORE the action expert. One fuser
    per wrist view; each cross-attends that view's aux tokens with that view's
    DA3 features (enriched with world-frame ray embeddings if extrinsics are
    available).

    Forward signature:
        forward(
            aux_visual_inputs: [B, T_aux, D],          # concatenated aux views
            da3_tokens_per_view: List[[B, K, D]],      # one per wrist view, ALREADY projected to D
            world_ray_per_view: Optional[List[[B, K, 6]]] = None,  # world-frame ray (o, d)
            aux_view_token_ranges: Dict[int, (int,int)],   # which slice of T_aux is this view
            spatial_coords_per_view: List[[B|1, K, 2]],    # DA3 token coords in [0,1]² per view
            aux_coords_per_view: Dict[int, [B|1, N, 2]],   # aux token coords in [0,1]² per view
            alpha: float,
        ) -> [B, T_aux, D]   (in-place-style residual; same shape as input)

    Identity-at-step-0 guaranteed by GeoStackCrossAttention's zero-init
    out_proj and closed gate (bias=-4). alpha schedule controls overall ramp.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_wrist_views: int,
        num_heads: int = 8,
        use_spatial_bias: bool = True,
        spatial_bias_lambda_init: float = 1.0,
        gate_mlp_hidden: Optional[int] = None,
        gate_init_bias: float = -4.0,
        use_world_ray: bool = True,
        ray_mlp_hidden: int = 256,
        ray_g_init: float = -2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_wrist_views = int(num_wrist_views)
        self.use_world_ray = bool(use_world_ray)

        # One fuser per wrist view. Cross-attn weights are NOT shared across
        # views (each wrist has different geometry conventions / contact bias).
        self.per_view_fusers = nn.ModuleList([
            GeoStackCrossAttention(
                hidden_dim=self.hidden_dim,
                num_heads=int(num_heads),
                spatial_token_dim=self.hidden_dim,   # DA3 tokens already projected to D
                use_spatial_bias=bool(use_spatial_bias),
                spatial_bias_lambda_init=float(spatial_bias_lambda_init),
                gate_mlp_hidden=gate_mlp_hidden,
                gate_init_bias=float(gate_init_bias),
                dropout=float(dropout),
            )
            for _ in range(self.num_wrist_views)
        ])
        # No GELU substitution — the gate_mlp from GeoStackCrossAttention is
        # Linear→GELU→Linear (no trailing LN), which is the v2B-tested config.

        if self.use_world_ray:
            self.world_ray_emb = WorldRayEmbedding(
                d_model=self.hidden_dim,
                hidden=int(ray_mlp_hidden),
                g_init=float(ray_g_init),
            )
        else:
            self.register_module("world_ray_emb", None)

    def forward(
        self,
        aux_visual_inputs: torch.Tensor,                   # [B, T_aux, D]
        da3_tokens_per_view: List[torch.Tensor],           # list[V_wrist] of [B, K, D]
        spatial_coords_per_view: List[torch.Tensor],       # list[V_wrist] of [1, K, 2]
        aux_view_token_ranges: Dict[int, Tuple[int, int]], # {view_idx_in_list: (start, end)} in T_aux
        aux_coords_per_view: Dict[int, torch.Tensor],      # {view_idx_in_list: [1, N, 2]}
        aux_vision_mask_per_view: Optional[Dict[int, torch.Tensor]] = None,  # {vi: [1, N] bool} — True for real spatial grid tokens, False for global/pooled tokens (their spatial bias is zeroed → uniform attn)
        world_ray_per_view: Optional[List[torch.Tensor]] = None,  # list[V_wrist] of [B, K, 6]
        alpha: float = 1.0,
    ) -> torch.Tensor:
        if alpha <= 0.0:
            return aux_visual_inputs
        # Functional construction (avoids in-place slice assignment, which can
        # produce NaN gradients when multiple fusers backprop through the same
        # input tensor via different slices).
        T_aux = aux_visual_inputs.shape[1]
        # Compute updated chunk for each wrist view; chunks for non-wrist slots
        # are taken directly from the input (no-op).
        # Build sorted list of (start, end, chunk) tuples covering [0, T_aux).
        view_chunks: List[Tuple[int, int, torch.Tensor]] = []
        for v in range(self.num_wrist_views):
            if v not in aux_view_token_ranges:
                continue
            start, end = aux_view_token_ranges[v]
            aux_view = aux_visual_inputs[:, start:end]            # [B, N_view, D]
            da3_v = da3_tokens_per_view[v]                         # [B, K, D]
            if self.use_world_ray and world_ray_per_view is not None:
                ray_v = world_ray_per_view[v]                      # [B, K, 6]
                da3_v = da3_v + self.world_ray_emb(ray_v)
            sp_coords = spatial_coords_per_view[v]                 # [1, K, 2]
            aux_coords = aux_coords_per_view[v]                    # [1, N_view, 2]
            vmask = aux_vision_mask_per_view.get(v) if aux_vision_mask_per_view is not None else None  # [1, N_view] bool
            updated = self.per_view_fusers[v](
                vlm_hidden=aux_view,
                spatial_tokens=da3_v,
                vlm_coords=aux_coords,
                spatial_coords=sp_coords,
                vision_mask=vmask,             # True for grid tokens (spatial bias applies), False for global pool (uniform attn over DA3)
                spatial_valid_mask=None,
                alpha=float(alpha),
            )
            view_chunks.append((start, end, updated))
        # Sort by start position, then concat with unchanged gaps.
        view_chunks.sort(key=lambda t: t[0])
        out_pieces: List[torch.Tensor] = []
        cursor = 0
        for start, end, chunk in view_chunks:
            if cursor < start:
                out_pieces.append(aux_visual_inputs[:, cursor:start])
            out_pieces.append(chunk)
            cursor = end
        if cursor < T_aux:
            out_pieces.append(aux_visual_inputs[:, cursor:T_aux])
        return torch.cat(out_pieces, dim=1)


__all__ = [
    "SinCos2DPositionalEmbedding",
    "AlphaSchedule",
    "DA3SpatialTokenBuilder",
    "GeoStackCrossAttention",
    "GeoFusedFlorence2Encoder",
    "WorldRayEmbedding",
    "WristAuxGeoStack",
    "build_pos_orig_vlm",
    "build_pos_orig_da3",
]
