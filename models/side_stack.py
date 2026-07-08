# SPDX: same as DA3-XVLA repo
"""
GeoStack-XVLA v3 — Side stack for action-expert spatial-language conditioning.

DESIGN
======
A parallel transformer stack runs in lockstep with the pretrained action expert.
It processes a bank of [DA3 deep features | T5 language tokens] via self-attention,
and the action expert's action-token slice queries the side stack via gated
cross-attention at paired layers.

  pretrained action expert (24 blocks, frozen architecture)

  x_0 = [action | vlm | aux | soft]            s_0 = bank = [da3 | t5]
   │                                            │
   ▼ block_0   (pretrained, untouched)          ▼ side_block_0   (new, fresh-init)
   │                                            │
   ▼ block_1                                    │
   │                                            │
   ▼ block_2                                    │
   │                                            │
   ▼ block_3 ─ s_to_x_xattn_0 ─◄────────────────┤  ← cross-attn at every 4th
   │                                            │     action layer
   ▼ block_4                                    ▼ side_block_1
   ...                                          ...

Identity-at-step-0:
  * `attn.out_proj` of every GatedCrossAttn is zero-initialized
  * Gate MLP final bias initialized to -4.0 → σ ≈ 0.018 (closed)
  * Alpha schedule starts at 0 (residual fully silent during warmup)
  → at step 0, action expert output is BIT-IDENTICAL to baseline X-VLA

Components
==========
  StandardTransformerBlock     pre-LN block: LN→MHA→residual→LN→FFN→residual
                               (no time conditioning — side stack doesn't do flow-matching)
  SideStack                    nn.ModuleList of N standard blocks; exposes step(s, i)
  GatedCrossAttn               per-token gated cross-attn adapter; identity at init
  SideStackBankBuilder         build [da3 | t5] bank with 2D PE + ray + modality embeddings

Reuses from models/geostack.py:
  SinCos2DPositionalEmbedding  (continuous-coord 2D sin/cos PE for DA3 tokens)
  AlphaSchedule                (piecewise-linear residual scale: 0 → 0.001 → 0.1 → 1.0)
  build_pos_orig_da3           (uniform-grid normalized [0,1]² coords for DA3 tokens)
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse battle-tested primitives from GeoStack v2B
from .geostack import (
    SinCos2DPositionalEmbedding,
    AlphaSchedule,
    build_pos_orig_da3,
)


# =============================================================================
# 1. Standard pre-LN transformer block (no time/proprio conditioning)
# =============================================================================

class StandardTransformerBlock(nn.Module):
    """Pre-LN transformer block. Used inside the side stack.

    Unlike the action expert's DiT-style blocks (which take a time embedding
    for flow-matching denoising), the side stack just processes the static
    spatial-language bank — no time conditioning needed.

    Layout: LayerNorm → MultiheadAttention → residual → LayerNorm → FFN → residual.
    """

    def __init__(self, hidden_dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_h = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_h),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_h, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, K, hidden]
            key_padding_mask: [B, K] bool, True = pad (MHA convention).
                Padded positions are ignored in attention.
        """
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


# =============================================================================
# 2. Side stack — N-layer self-attention over the spatial-language bank
# =============================================================================

class SideStack(nn.Module):
    """Stack of N standard transformer blocks that refine the spatial-language
    bank through depth.

    Exposes `step(s, layer_idx)` so the action expert's forward loop can
    interleave side stack progression with its own block-stepping (rather than
    running the side stack as one monolithic forward call).

    This explicit interleaving enables the layer-by-layer cross-attention
    pairing: at action expert layer L_action, we cross-attend to the side
    stack state AFTER it has been processed by side_block_L_side (the side
    stack layer paired with L_action).
    """

    def __init__(self, n_layers: int, hidden_dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.n_layers = int(n_layers)
        self.hidden_dim = int(hidden_dim)
        self.blocks = nn.ModuleList([
            StandardTransformerBlock(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])

    def step(self, s: torch.Tensor, layer_idx: int,
             key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run one side stack block. Returns the updated bank state."""
        return self.blocks[layer_idx](s, key_padding_mask=key_padding_mask)


# =============================================================================
# 3. Gated cross-attention adapter (identity-at-step-0)
# =============================================================================

class GatedCrossAttn(nn.Module):
    """Per-token gated cross-attention adapter from Q-side to K/V-side.

    Implements:
        attn_out = MHA(Q=q_hidden, K=V=kv_hidden)             [B, T_q, hidden]
        gate = sigmoid(gate_mlp(LayerNorm(q_hidden)))         [B, T_q, hidden]
        residual = q_hidden + alpha * gate * attn_out

    Identity-at-step-0 by construction:
      * `attn.out_proj.weight` and `attn.out_proj.bias` are zero-initialized
        → attn_out = 0 regardless of input
      * `gate_mlp[-1].bias` is initialized to gate_init_bias (default -4.0)
        → σ(bias) ≈ 0.018, so even when attn_out becomes nonzero the residual
          contribution is heavily damped initially
      * `alpha` parameter (passed externally) defaults to 0 during warmup
        → residual = 0 + 0 = 0 → identity

    Gradient flow at init: even though forward output equals q_hidden,
    gradients ∂L/∂(out_proj_weight) ≠ 0 (proportional to alpha · gate · pre_attn_out),
    so the adapter trains as soon as alpha > 0.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8,
                 gate_init_bias: float = -4.0,
                 dropout: float = 0.0,
                 gate_mlp_hidden: Optional[int] = None):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.q_norm = nn.LayerNorm(self.hidden_dim)
        self.kv_norm = nn.LayerNorm(self.hidden_dim)
        self.attn = nn.MultiheadAttention(
            self.hidden_dim, num_heads=self.num_heads,
            dropout=float(dropout), batch_first=True,
        )
        # FIX (v3.4): NO out_proj weight init override at all — let MultiheadAttention's
        # default kaiming init stand (std ≈ 1.8e-2 at hidden_dim=1024).
        #
        # Previous attempts:
        #   v3.2g — out_proj init = exact 0 → upstream gradient blocked → side stack didn't train
        #   v3.3  — out_proj init std = 1e-3 → still too small → upstream gradient ~1e-3 ×
        #            tiny_loss_signal → still didn't move upstream params after 5000 steps
        # Both failures empirically confirmed: bank builder + side_stack blocks at ratio
        # 1.00-1.02 of fresh kaiming init = essentially untrained.
        #
        # Why removing the override works:
        #   - out_proj at full kaiming (~1.8e-2 std) means upstream gradient is
        #     substantial from step 1000 (α onset). No gradient bottleneck.
        #   - Identity-at-step-0 is guaranteed by the α schedule alone:
        #     α=0 for the first 1000 steps → residual = 0×anything = 0, regardless
        #     of out_proj. out_proj=0 was REDUNDANT safety that became a training killer.
        #
        # Only the bias still zero-initialized — that's harmless (additive constant
        # absorbed into the LayerNorm bias of the next block at no cost).
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
        # Closed gate at init: large negative bias → σ ≈ 0.018
        nn.init.constant_(self.gate_mlp[-1].bias, float(gate_init_bias))
        nn.init.normal_(self.gate_mlp[-1].weight, mean=0.0, std=1e-4)

    def forward(
        self,
        q_hidden: torch.Tensor,                          # [B, T_q, hidden]
        kv_hidden: torch.Tensor,                         # [B, K, hidden]
        alpha: float = 1.0,
        kv_padding_mask: Optional[torch.Tensor] = None,  # [B, K] bool, True = pad
    ) -> torch.Tensor:
        if alpha <= 0.0:
            return q_hidden   # short-circuit; identity-at-step-0 / pre-warmup

        q = self.q_norm(q_hidden)
        kv = self.kv_norm(kv_hidden)
        attn_out, _ = self.attn(
            q, kv, kv,
            key_padding_mask=kv_padding_mask,
            need_weights=False,
        )
        gate = torch.sigmoid(self.gate_mlp(self.gate_norm(q_hidden)))   # [B, T_q, hidden]
        return q_hidden + float(alpha) * gate * attn_out


# =============================================================================
# 3b. Plücker-ray utility — convert local-frame ray directions + extrinsics
#     into world-frame Plücker rays.
# =============================================================================

def compute_world_ray_6d(
    ray_dir_local: torch.Tensor,        # [B, 3, H, W] DA3's per-pixel ray direction in cam-local frame
    extrinsic_w2c: torch.Tensor,        # [B, 4, 4] OpenCV world-to-camera transform
) -> torch.Tensor:
    """Compute per-pixel world-frame ray as [origin, direction] (6D, both in world coords).

    For our use case (transformer attention over heterogeneous tokens), explicit
    (origin, direction) is preferable to Plücker (direction, moment) because:
      - Origin is the camera position — directly tells the model "where is this
        view's camera in world space" (critical for grasping with wrist cams).
      - Plücker buries origin in moment = origin × direction, requiring the
        model to learn an obscure bilinear decoding to recover it.
      - Attention naturally computes distance / dot-product over (origin, dir);
        Plücker's intersection condition is a bilinear identity that's much
        harder to learn implicitly.

    Inputs:
        ray_dir_local: [B, 3, H, W] DA3-predicted direction in cam-local frame.
        extrinsic_w2c: [B, 4, 4] OpenCV world-to-camera transform.

    Returns:
        [B, 6, H, W] — channels are (origin_x, origin_y, origin_z,
                                     dir_x,    dir_y,    dir_z),
        all in WORLD coordinates. For pinhole cameras, origin is the camera
        center (broadcast — same for all pixels in a view).
    """
    B, _, H, W = ray_dir_local.shape

    R_w2c = extrinsic_w2c[:, :3, :3].to(ray_dir_local.dtype)   # [B, 3, 3]
    t_w2c = extrinsic_w2c[:, :3, 3].to(ray_dir_local.dtype)    # [B, 3]
    # Camera-to-world rotation
    R_c2w = R_w2c.transpose(-1, -2)                             # [B, 3, 3]
    # Camera position in world frame: pos_world = -R_c2w · t_w2c
    pos_world = -torch.einsum('bij,bj->bi', R_c2w, t_w2c)       # [B, 3]

    # Transform direction: world = R_c2w · cam
    dir_cam_flat = ray_dir_local.reshape(B, 3, H * W)            # [B, 3, H*W]
    dir_world_flat = torch.einsum('bij,bjk->bik', R_c2w, dir_cam_flat)  # [B, 3, H*W]
    dir_world = dir_world_flat.reshape(B, 3, H, W)               # [B, 3, H, W]

    # Origin broadcast: every pixel in this view shares the camera center.
    # (For non-pinhole cameras with per-pixel origin offsets, would transform
    # DA3's channels [3:6] here instead — for pinhole RoboReal this is exact.)
    origin_world = pos_world.view(B, 3, 1, 1).expand(-1, -1, H, W)  # [B, 3, H, W]

    return torch.cat([origin_world, dir_world], dim=1).contiguous()


# =============================================================================
# 4. Side-stack bank builder — [DA3 deep | T5] with 2D PE + ray + modality emb
# =============================================================================

class SideStackBankBuilder(nn.Module):
    """Build the spatial-language bank that the side stack processes.

    Bank layout: [da3_tokens (h_da3 × w_da3) | t5_tokens (L_t5)]

    Per-token assembly:
      DA3: feat_proj(deep_feat) + σ(g_ray) · ray_proj(ray) + σ(g_pos) · PE2D(coords)
           + modality_emb(0)
      T5:  t5_proj(t5_feat) + modality_emb(1)
           (T5 encoder output already has positional info baked in)

    All identity-friendly inits:
      - g_ray init = -4.0 → ray contribution starts ~1.8% strength
      - g_pos init = -2.0 → 2D PE contribution starts ~12% strength
      - modality_emb std = 0.01 (small perturbation)
      - feat_proj kaiming, LayerNorm at end

    Padding handling:
      DA3 tokens are always valid (real spatial content).
      T5 tokens may include padding — returns a bank_padding_mask the
      side stack and cross-attn use to ignore padded positions.
    """

    def __init__(
        self,
        da3_in_dim: int,
        t5_in_dim: int,
        hidden_dim: int,
        use_ray: bool = True,
        use_2d_pos: bool = True,
        use_modality_emb: bool = True,
        g_ray_init: float = -4.0,
        g_pos_init: float = -2.0,
        ray_in_dim: int = 6,            # 3 = DA3-local direction only; 6 = world-frame Plücker
        # Multi-view DA3 (v3.1: wrist cameras get their own DA3 forward at lower
        # resolution; view_emb distinguishes main/wrist1/wrist2 in the bank)
        num_views: int = 1,
        use_view_emb: bool = False,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.use_ray = bool(use_ray)
        self.use_2d_pos = bool(use_2d_pos)
        self.use_modality_emb = bool(use_modality_emb)
        self.num_views = int(num_views)
        self.use_view_emb = bool(use_view_emb)

        # ---- DA3 feature projector (canonical MLP + LN pattern) ----
        self.da3_proj = nn.Sequential(
            nn.Linear(int(da3_in_dim), hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # ---- T5 feature projector ----
        self.t5_proj = nn.Sequential(
            nn.Linear(int(t5_in_dim), hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # ---- Ray (DA3's DualDPT ray-head direction; see da3_for_geostack.py) ----
        if self.use_ray:
            self.ray_proj = nn.Sequential(
                nn.Linear(int(ray_in_dim), hidden_dim // 2),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            # fp32 dtype so softplus/sigmoid don't lose precision under bf16 autocast
            self.g_ray = nn.Parameter(
                torch.tensor(float(g_ray_init), dtype=torch.float32)
            )
        else:
            self.register_module("ray_proj", None)
            self.register_parameter("g_ray", None)

        # ---- 2D sin/cos PE over normalized [0,1]² original-image coords ----
        if self.use_2d_pos:
            self.pe2d = SinCos2DPositionalEmbedding(hidden_dim)
            self.g_pos = nn.Parameter(
                torch.tensor(float(g_pos_init), dtype=torch.float32)
            )
        else:
            self.register_module("pe2d", None)
            self.register_parameter("g_pos", None)

        # ---- Modality embedding (distinguishes da3 vs t5 in self-attention) ----
        if self.use_modality_emb:
            self.modality_emb = nn.Embedding(2, hidden_dim)
            # Small init so it doesn't dominate the feat at step 0.
            nn.init.normal_(self.modality_emb.weight, mean=0.0, std=0.01)
        else:
            self.register_module("modality_emb", None)

        # ---- View embedding (distinguishes main / wrist1 / wrist2 / ... DA3 tokens) ----
        if self.use_view_emb:
            self.view_emb = nn.Embedding(self.num_views, hidden_dim)
            # Small init so view distinction is a NUDGE, not an overriding signal
            nn.init.normal_(self.view_emb.weight, mean=0.0, std=0.01)
        else:
            self.register_module("view_emb", None)

    def forward(
        self,
        da3_feat: torch.Tensor,                          # [B, C_da3, h_grid, w_grid]
        da3_coords: torch.Tensor,                        # [B or 1, N_da3, 2] in [0,1]²
        ray: Optional[torch.Tensor] = None,              # [B, 3, h_grid, w_grid]
        t5_feat: Optional[torch.Tensor] = None,          # [B, L_t5, t5_dim]
        t5_mask: Optional[torch.Tensor] = None,          # [B, L_t5] bool, True = real token
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (bank, bank_padding_mask).
            bank:                [B, N_da3 + L_t5, hidden_dim]
            bank_padding_mask:   [B, N_da3 + L_t5] bool, True = pad (MHA convention)
        """
        B, C, H, W = da3_feat.shape
        N_da3 = H * W
        device = da3_feat.device

        # ---- DA3 path ----
        # [B, C, H, W] → [B, H, W, C] → [B, N_da3, C]
        da3_flat = da3_feat.permute(0, 2, 3, 1).reshape(B, N_da3, C).contiguous()
        da3_emb = self.da3_proj(da3_flat)                                  # [B, N_da3, hidden]

        if self.use_ray:
            if ray is None:
                raise ValueError("SideStackBankBuilder: use_ray=True but ray is None")
            ray_C = ray.shape[1]
            ray_flat = ray.permute(0, 2, 3, 1).reshape(B, N_da3, ray_C).contiguous()
            r = self.ray_proj(ray_flat.to(da3_emb.dtype))                   # [B, N_da3, hidden]
            da3_emb = da3_emb + torch.sigmoid(self.g_ray.to(da3_emb.dtype)) * r

        if self.use_2d_pos:
            coords_b = (da3_coords if da3_coords.shape[0] == B
                        else da3_coords.expand(B, -1, -1))
            pe = self.pe2d(coords_b.float()).to(da3_emb.dtype)              # [B, N_da3, hidden]
            da3_emb = da3_emb + torch.sigmoid(self.g_pos.to(da3_emb.dtype)) * pe

        if self.use_modality_emb:
            mod_da3 = self.modality_emb(
                torch.zeros(1, dtype=torch.long, device=device)
            ).to(da3_emb.dtype)                                             # [1, hidden]
            da3_emb = da3_emb + mod_da3.unsqueeze(0)                        # broadcast over B, N_da3

        # ---- T5 path ----
        if t5_feat is None:
            # T5 not provided — bank is DA3 only (treat as L_t5 = 0)
            t5_emb_full = torch.zeros(B, 0, self.hidden_dim,
                                       dtype=da3_emb.dtype, device=device)
            t5_pad_full = torch.zeros(B, 0, dtype=torch.bool, device=device)
        else:
            t5_emb = self.t5_proj(t5_feat.to(da3_emb.dtype))                # [B, L_t5, hidden]
            if self.use_modality_emb:
                mod_t5 = self.modality_emb(
                    torch.ones(1, dtype=torch.long, device=device)
                ).to(t5_emb.dtype)
                t5_emb = t5_emb + mod_t5.unsqueeze(0)
            t5_emb_full = t5_emb

            if t5_mask is not None:
                # T5 mask convention: True = real token.
                # MHA expects key_padding_mask: True = pad/ignore.
                t5_pad_full = ~t5_mask.bool()
            else:
                t5_pad_full = torch.zeros(B, t5_emb.shape[1],
                                          dtype=torch.bool, device=device)

        # ---- Concatenate bank + padding mask ----
        bank = torch.cat([da3_emb, t5_emb_full], dim=1)                      # [B, N_da3 + L_t5, hidden]
        da3_pad = torch.zeros(B, N_da3, dtype=torch.bool, device=device)    # DA3 always valid
        bank_pad = torch.cat([da3_pad, t5_pad_full], dim=1)                  # [B, N_da3 + L_t5]
        return bank, bank_pad

    def forward_multi_view(
        self,
        da3_views: list,                                  # list of (feat, coords, ray, view_idx) tuples
        t5_feat: Optional[torch.Tensor] = None,           # [B, L_t5, t5_dim]
        t5_mask: Optional[torch.Tensor] = None,           # [B, L_t5] bool, True = real
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-view bank build. Each view contributes its own block of DA3 tokens
        (at its own grid resolution), and all blocks are concatenated with T5 at the end.

        Args:
            da3_views: list of dicts, each with keys:
                "feat":      [B, C_da3, h_grid, w_grid]
                "coords":    [B or 1, h_grid*w_grid, 2] in normalized [0,1]² (per-view)
                "ray":       [B, 3, h_grid, w_grid]
                "view_idx":  int (0=main, 1=wrist1, 2=wrist2, ...)
            t5_feat, t5_mask: same as `forward`.

        Returns:
            bank:                [B, sum(N_view) + L_t5, hidden]
            bank_padding_mask:   [B, sum(N_view) + L_t5] (DA3 always valid, T5 may pad)
        """
        if not isinstance(da3_views, (list, tuple)) or len(da3_views) == 0:
            raise ValueError("forward_multi_view requires at least one view dict")

        # Process each view block. Reuses the same projectors, ray gate, PE gate.
        per_view_blocks = []
        for v in da3_views:
            feat = v["feat"]               # [B, C, h, w]
            coords = v["coords"]           # [B or 1, N, 2]
            ray = v.get("ray", None)       # [B, 3, h, w] or None
            view_idx = int(v["view_idx"])
            B, C, H, W = feat.shape
            N = H * W

            # Flatten + project DA3 feat
            flat = feat.permute(0, 2, 3, 1).reshape(B, N, C).contiguous()
            emb = self.da3_proj(flat)

            # Add ray (if enabled and provided). ray channels: 3 (DA3-local direction)
            # OR 6 (world-frame Plücker = direction || moment).
            if self.use_ray and ray is not None:
                ray_C = ray.shape[1]
                ray_flat = ray.permute(0, 2, 3, 1).reshape(B, N, ray_C).contiguous()
                r = self.ray_proj(ray_flat.to(emb.dtype))
                emb = emb + torch.sigmoid(self.g_ray.to(emb.dtype)) * r

            # Add 2D PE in the view's image-space normalized coords
            if self.use_2d_pos:
                coords_b = coords if coords.shape[0] == B else coords.expand(B, -1, -1)
                pe = self.pe2d(coords_b.float()).to(emb.dtype)
                emb = emb + torch.sigmoid(self.g_pos.to(emb.dtype)) * pe

            # Add modality emb (0 = da3)
            if self.use_modality_emb:
                mod = self.modality_emb(
                    torch.zeros(1, dtype=torch.long, device=emb.device)
                ).to(emb.dtype)
                emb = emb + mod.unsqueeze(0)

            # Add view emb (NEW: distinguishes main/wrist1/wrist2)
            if self.use_view_emb:
                ve = self.view_emb(
                    torch.tensor([view_idx], dtype=torch.long, device=emb.device)
                ).to(emb.dtype)
                emb = emb + ve.unsqueeze(0)

            per_view_blocks.append(emb)

        # Concatenate all DA3 view blocks
        da3_concat = torch.cat(per_view_blocks, dim=1)        # [B, sum(N_v), hidden]
        N_da3_total = da3_concat.shape[1]
        B = da3_concat.shape[0]
        device = da3_concat.device

        # T5 path (identical to single-view forward)
        if t5_feat is None:
            t5_emb_full = torch.zeros(B, 0, self.hidden_dim,
                                       dtype=da3_concat.dtype, device=device)
            t5_pad_full = torch.zeros(B, 0, dtype=torch.bool, device=device)
        else:
            t5_emb = self.t5_proj(t5_feat.to(da3_concat.dtype))
            if self.use_modality_emb:
                mod_t5 = self.modality_emb(
                    torch.ones(1, dtype=torch.long, device=device)
                ).to(t5_emb.dtype)
                t5_emb = t5_emb + mod_t5.unsqueeze(0)
            t5_emb_full = t5_emb
            if t5_mask is not None:
                t5_pad_full = ~t5_mask.bool()
            else:
                t5_pad_full = torch.zeros(B, t5_emb.shape[1],
                                          dtype=torch.bool, device=device)

        bank = torch.cat([da3_concat, t5_emb_full], dim=1)
        da3_pad = torch.zeros(B, N_da3_total, dtype=torch.bool, device=device)
        bank_pad = torch.cat([da3_pad, t5_pad_full], dim=1)
        return bank, bank_pad


# =============================================================================
# 5. Default pairing utility (action expert layer → side stack layer)
# =============================================================================

def default_pairing(action_depth: int, side_depth: int) -> dict[int, int]:
    """Map action expert layer indices to side stack layer indices.

    With action_depth=24 and side_depth=6:
        Pairings = {3: 0, 7: 1, 11: 2, 15: 3, 19: 4, 23: 5}
    (i.e., cross-attention at every (action_depth / side_depth)-th action layer)

    Side stack layer `i` is paired with action layer `((i + 1) * action_depth /
    side_depth) - 1`. The last side layer always pairs with the LAST action layer.
    """
    if side_depth <= 0:
        return {}
    pairing = {}
    for side_i in range(side_depth):
        action_l = int((side_i + 1) * action_depth / side_depth) - 1
        pairing[action_l] = side_i
    return pairing


# =============================================================================
# 6. Side-stack auxiliary loss head (v3.5 — fixes the gradient-utility problem)
# =============================================================================
#
# WHY THIS EXISTS:
#   v3.2g/v3.3/v3.4 all empirically failed: side stack params stayed at fresh
#   kaiming init even after 50k/5k training steps. Root cause was not init
#   magnitude (we tried 0 → 1e-3 → kaiming-default, all stuck) but gradient
#   UTILITY: cross-attn from action expert to side stack produced random noise
#   because the side stack is fresh-init, so ∂loss/∂(side_stack_output) was
#   essentially zero-mean noise → AdamW updates partially cancelled → side
#   stack stayed random → cycle repeats forever (chicken-and-egg failure).
#
# THIS FIX:
#   Add a parallel action-prediction head that takes the side stack's FINAL
#   layer output and predicts the ground-truth action chunk directly. This
#   gives the side stack a DIRECT supervised gradient signal that is:
#     (a) consistent across batches (always the same training signal: predict
#         actions from spatial-language features)
#     (b) action-aligned (forces side stack to encode action-relevant features)
#     (c) gradient-bypass-able (independent of whether main action expert uses
#         the cross-attn or not)
#
# ARCHITECTURE:
#   side_stack_output [B, K_bank, hidden]   ← final side stack state after all blocks
#        │
#        ▼ attention-pool with single learnable query (respects bank padding mask)
#   pooled [B, hidden]
#        │
#        ▼ 2-layer MLP
#   aux_action_pred [B, num_actions, dim_action]
#        │
#        ▼ action_space.compute_loss vs ground-truth action chunk
#   aux_loss (added to total loss with weight ~0.1)
#
# LOSS WEIGHT:
#   Start at 0.1 — small enough that main flow-matching loss dominates the
#   action expert's optimization, but big enough that side stack receives
#   useful gradient. Tunable via geometry_conditioning.side_stack.aux_loss.weight.

class SideStackAuxHead(nn.Module):
    """Aux action-prediction head for the side stack.

    Takes the side stack's FINAL output state [B, K_bank, hidden] and produces
    a predicted action chunk [B, num_actions, dim_action] via attention pooling +
    MLP. Trained via supervised loss against ground-truth action chunk to give
    the side stack direct gradient signal independent of the cross-attn path.

    v3.6: optional `proprio_dim` enables proprio fusion — the current robot state
    is projected and concatenated with the pooled side stack features before the
    MLP. Without proprio, the aux head must predict actions from JUST spatial
    features (essentially impossible for some tasks like trajectory smoothness).
    With proprio, the aux task becomes "given scene+language+current_state,
    predict next action chunk" — a tractable behavior cloning signal that still
    forces the side stack to encode action-relevant spatial features (otherwise
    the proprio path would have to do all the work, and the side stack would be
    underutilized — the fusion concat ensures both paths contribute).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_actions: int,
        dim_action: int,
        mlp_ratio: float = 2.0,
        pool_init_std: float = 0.02,
        proprio_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_actions = int(num_actions)
        self.dim_action = int(dim_action)
        self.proprio_dim = int(proprio_dim) if proprio_dim else None

        # Attention-pool: single learnable query selects the most useful tokens.
        # Cheap (~1k params vs MHA's ~hidden^2). Gives flexible weighting across
        # bank positions; respects bank_padding_mask so padded T5 tokens are ignored.
        self.pool_query = nn.Parameter(torch.empty(1, hidden_dim))
        nn.init.normal_(self.pool_query, mean=0.0, std=pool_init_std)
        self.pool_norm = nn.LayerNorm(hidden_dim)

        # v3.6: optional proprio projector + fusion
        if self.proprio_dim is not None:
            self.proprio_proj = nn.Sequential(
                nn.LayerNorm(self.proprio_dim),
                nn.Linear(self.proprio_dim, hidden_dim),
                nn.GELU(approximate="tanh"),
                nn.LayerNorm(hidden_dim),
            )
            # Fusion: concat [pooled, proprio_emb] → hidden_dim
            # This lets the model learn relative weighting between spatial and proprio
            # paths (vs additive, which would have fixed magnitude blending).
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(approximate="tanh"),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.proprio_proj = None
            self.fusion = None

        # 2-layer MLP head: hidden → 2*hidden → num_actions*dim_action
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, num_actions * dim_action),
        )

    def forward(
        self,
        side_state: torch.Tensor,                        # [B, K, hidden]
        bank_padding_mask: Optional[torch.Tensor] = None,  # [B, K] bool, True = pad
        proprio: Optional[torch.Tensor] = None,            # [B, proprio_dim] — only if built with proprio_dim
    ) -> torch.Tensor:
        """Returns predicted action chunk [B, num_actions, dim_action]."""
        B, K, H = side_state.shape
        if H != self.hidden_dim:
            raise ValueError(f"side_state hidden_dim mismatch: got {H}, expected {self.hidden_dim}")

        # Pool: weighted avg over K dim using a learnable query
        normed = self.pool_norm(side_state)                                # [B, K, H]
        scores = (normed @ self.pool_query.to(normed.dtype).t()).squeeze(-1)  # [B, K]
        if bank_padding_mask is not None:
            # mask padded positions to -inf so softmax assigns ~0 weight
            scores = scores.masked_fill(bank_padding_mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)                                # [B, K]
        pooled = (weights.unsqueeze(-1) * side_state).sum(dim=1)           # [B, H]

        # v3.6: fuse in proprio if available
        if self.proprio_proj is not None:
            if proprio is None:
                raise ValueError(
                    "SideStackAuxHead built with proprio_dim but forward() got proprio=None"
                )
            p_emb = self.proprio_proj(proprio.to(pooled.dtype))            # [B, H]
            pooled = self.fusion(torch.cat([pooled, p_emb], dim=-1))       # [B, H]

        # Predict action chunk
        out = self.head(pooled)                                            # [B, T*D]
        return out.reshape(B, self.num_actions, self.dim_action)


__all__ = [
    "StandardTransformerBlock",
    "SideStack",
    "GatedCrossAttn",
    "SideStackBankBuilder",
    "SideStackAuxHead",
    "default_pairing",
    "compute_world_ray_6d",
]
