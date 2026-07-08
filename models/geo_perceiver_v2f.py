# SPDX: same as DA3-XVLA repo
"""
GeoPerceiver v2F — k320-style perceiver fusion with GeoStack-style K/V enrichment.

DESIGN
======
Combines lessons from two architectures we've tried:
  * k320 (the published RoboPRO model): DA3-BASE → projector → K=320 perceiver
    → 320 geometry tokens → action-expert cross-attn (before_policy fusion).
    Worked well end-to-end but used DA3-BASE at 504×504 square stretch and
    raw DA3 features (no spatial signature in K/V → memory rule violation).

  * GeoStack v2C/v2D/v2E: DA3-Large at aspect-preserved 252×336, with
    DA3SpatialTokenBuilder enrichment (SinCos 2D PE + ray) added to each K/V
    token. Strong positional signal but the cross-attn fused INSIDE Florence's
    encoder (inside_policy / encoder layers 6/9/11), not at the action expert.

v2F is the hybrid:
  DA3-Large @ 252×336 aspect (from DA3LargeForGeoStack)
    ↓ enrich each token with SinCos 2D PE + ray via DA3SpatialTokenBuilder
    ↓ K/V = [B, h*w, D]  (e.g., 18*24=432 tokens per view at 252×336)
  Perceiver Resampler with K=160 learnable queries (single shared perceiver)
    ↓ [B, 160, D]
  → returned as geometry_tokens for the action expert (before_policy fusion
    via the existing transformer.geometry_fusion module — unchanged from k320).

WHY PERCEIVER DOESN'T DESTROY SPATIAL INFO
==========================================
The 160 learnable queries cross-attend with enriched K/V tokens. Each K/V token
carries (DA3-feat + 2D PE + ray) — so queries can selectively pull "tokens at
position (u, v) with ray pointing direction d." Output tokens encode learned
summaries that preserve spatial selectivity. The compression 432 → 160 is real
(~37% retention) but is fine because most raw DA3 tokens are redundant for
action prediction.

The SinCos PE is FIXED (not learned) — see geostack.py::SinCos2DPositionalEmbedding.
The ray projection uses DA3's own ray-head output (camera-frame direction).

POSITIONAL CONTRACT
===================
Florence sees pixel_values at 224×224; DA3 internally resizes to (h_in, w_in)
that is aspect-preserved (default 252×336). The DA3 token grid (h_in/14, w_in/14)
lives in normalized [0,1]² coords of the ORIGINAL image — same coord system
Florence's vision tower uses (linear bijection on each axis). This keeps the
K/V positional encoding meaningful even though Florence and DA3 process the
same content at different resolutions.

MULTI-VIEW (V > 1)
==================
Default: process FIRST view only (matches k320's "main camera DA3" pattern).
The wrist views provide RGB context to Florence's vision tower → action expert,
but don't go through the geometry path. For multi-view DA3, set
`geo_perceiver_v2f_views = "all"` and stack views; queries pool across V*N
tokens.

WORLD-FRAME RAY (OPTIONAL, requires extrinsics)
===============================================
When `geo_perceiver_v2f_use_world_ray=True` AND extrinsics are provided,
compute world-frame ray (origin + direction) and inject as an additional
gated additive embedding on the K/V tokens. Same WorldRayEmbedding module as
v2C/v2D wrist path.

IDENTITY-AT-STEP-0
==================
The cross-attn output projection inside the perceiver is zero-initialized at
build time so geometry contribution starts at 0 and grows only as the model
learns. Override via `geo_perceiver_v2f_out_proj_init_std > 0` to inject
non-zero from step 0 (v2D-aggressive style).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .da3_for_geostack import DA3LargeForGeoStack
from .geostack import (
    DA3SpatialTokenBuilder,
    WorldRayEmbedding,
    build_pos_orig_da3,
)
from .geometry_conditioning import GeometryTokenResampler


class GeoPerceiverV2F(nn.Module):
    """v2F: DA3-Large + aspect-preserved input + GeoStack-style K/V enrichment
    + K=160 (default) perceiver resampler.

    Used as a drop-in replacement for `SegmentedDA3GeometryConditioner` when
    `use_geo_perceiver_v2f=True` in the model config. Plugged into
    `model.geometry_conditioner` so the action expert's existing
    `geometry_fusion` cross-attn path (before_policy) consumes its output
    unchanged.

    Forward signature matches the in-model call site: takes pixel_values
    (the FIRST view by default) and optionally per-view extrinsics, returns
    [B, K, D] geometry tokens.
    """

    def __init__(
        self,
        geometry_cfg: dict,
        hidden_dim: int,
    ):
        super().__init__()
        gc = geometry_cfg

        # --- DA3-Large frozen backbone (aspect-preserved) ---
        self.da3 = DA3LargeForGeoStack(
            model_name=str(gc.get("geo_perceiver_v2f_da3_model", "depth-anything/DA3-Large-1.1")),
            out_layers=tuple(gc.get("geo_perceiver_v2f_da3_out_layers", [11, 19, 23])),
            da3_input_h=int(gc.get("geo_perceiver_v2f_da3_input_h", 252)),
            da3_input_w=int(gc.get("geo_perceiver_v2f_da3_input_w", 336)),
            patch_size=int(gc.get("geo_perceiver_v2f_da3_patch_size", 14)),
            use_bf16=bool(gc.get("geo_perceiver_v2f_da3_use_bf16", True)),
        )
        # Which DA3 feature level to use as K/V. Default "deep" (last out_layer).
        self._feat_tap_idx = int(gc.get("geo_perceiver_v2f_da3_tap_idx", -1))

        # --- K/V enrichment: feat_proj + 2D PE + ray ---
        # Same builder used by GeoStack — adds SinCos PE and gated ray contribution.
        self.token_builder = DA3SpatialTokenBuilder(
            c_in=int(self.da3.embed_dim),
            d_model=int(hidden_dim),
            ray_in=3,
            use_ray=bool(gc.get("geo_perceiver_v2f_use_ray", True)),
            use_2d_pos=bool(gc.get("geo_perceiver_v2f_use_2d_pos", True)),
            g_ray_init=float(gc.get("geo_perceiver_v2f_g_ray_init", -3.0)),
            g_pos_init=float(gc.get("geo_perceiver_v2f_g_pos_init", -1.0)),
        )

        # --- Optional world-frame ray (requires extrinsics in forward) ---
        self.use_world_ray = bool(gc.get("geo_perceiver_v2f_use_world_ray", False))
        if self.use_world_ray:
            self.world_ray_emb = WorldRayEmbedding(
                d_model=int(hidden_dim),
                hidden=int(gc.get("geo_perceiver_v2f_world_ray_mlp_hidden", 256)),
                g_init=float(gc.get("geo_perceiver_v2f_world_ray_g_init", -1.0)),
            )
        else:
            self.register_module("world_ray_emb", None)

        # --- Perceiver resampler (K=160 by default) ---
        self.num_geometry_tokens = int(gc.get("geo_perceiver_v2f_num_queries", 160))
        self.resampler = GeometryTokenResampler(
            num_geometry_tokens=self.num_geometry_tokens,
            hidden_dim=int(hidden_dim),
            num_heads=int(gc.get("geo_perceiver_v2f_num_heads", 8)),
            dropout=float(gc.get("geo_perceiver_v2f_dropout", 0.0)),
            num_layers=int(gc.get("geo_perceiver_v2f_num_layers", 1)),
        )

        # Identity-at-step-0 (zero out the perceiver's final cross-attn projection
        # so initial geometry contribution is zero). Override via init_std > 0.
        self._out_proj_init_std = float(gc.get("geo_perceiver_v2f_out_proj_init_std", 0.0))
        self._views_mode = str(gc.get("geo_perceiver_v2f_views", "first")).lower()
        if self._views_mode not in ("first", "all"):
            raise ValueError(f"geo_perceiver_v2f_views must be 'first' or 'all'; got {self._views_mode}")

    # ---- post-from_pretrained hook (called by train.py / _reinit_geometry_modules) ----

    def reload_pretrained_weights(self) -> None:
        """Reload DA3-Large frozen weights — HF's no_init_weights zeroes them
        during model.from_pretrained. Call this AFTER loading the checkpoint."""
        self.da3.reload_pretrained_weights()

    def reinit_identity_at_step_0(self) -> None:
        """Zero out the perceiver's cross-attn output projection so the residual
        contribution is exactly 0 at step 0 (DiT-style identity init). Or, if
        out_proj_init_std > 0, use that std for non-zero aggressive init."""
        for m in self.resampler.modules():
            # GeometryTokenResampler uses nn.MultiheadAttention internally.
            if isinstance(m, nn.MultiheadAttention):
                with torch.no_grad():
                    if self._out_proj_init_std > 0.0:
                        nn.init.normal_(m.out_proj.weight, mean=0.0, std=self._out_proj_init_std)
                    else:
                        nn.init.zeros_(m.out_proj.weight)
                    if m.out_proj.bias is not None:
                        nn.init.zeros_(m.out_proj.bias)

    # ---- forward ----

    def forward(
        self,
        pixel_values: torch.Tensor,            # [B, V, 3, H, W] or [B, 3, H, W]
        extrinsics: Optional[torch.Tensor] = None,  # [B, V, 4, 4] world-to-camera (cv) or None
    ) -> torch.Tensor:
        """Returns [B, K, D] geometry tokens (K = num_geometry_tokens, default 160)."""
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(1)   # [B, 1, 3, H, W]
        B, V = pixel_values.shape[:2]

        # Pick views to process
        if self._views_mode == "first":
            views = [0]
        else:
            views = list(range(V))

        # Run DA3 + build enriched K/V for each selected view
        kv_chunks = []
        for v_idx in views:
            view_pixels = pixel_values[:, v_idx]            # [B, 3, H, W]
            out = self.da3(view_pixels)
            # Pick the configured feature level (default: last out_layer = "deep")
            feat = out["feats"][self._feat_tap_idx]         # [B, C, h, w]
            ray_cam = out["ray"]                            # [B, 3, h, w] camera-frame ray
            h_grid = int(out["h_grid"])
            w_grid = int(out["w_grid"])
            Cd = feat.shape[1]

            # Flatten spatial dims → tokens
            feat_flat = feat.permute(0, 2, 3, 1).reshape(B, h_grid * w_grid, Cd).contiguous()
            ray_flat = ray_cam.permute(0, 2, 3, 1).reshape(B, h_grid * w_grid, 3).contiguous()

            # 2D coords in normalized original-image space (DA3 frame == VLM frame)
            coords = build_pos_orig_da3(
                h_grid=h_grid, w_grid=w_grid,
                device=feat.device, dtype=torch.float32,
            )                                                # [1, h*w, 2]
            coords_b = coords.expand(B, -1, -1).to(feat_flat.dtype)

            # Enriched K/V token = feat_proj(feat) + σ(g_ray)·ray_proj(ray) + σ(g_pos)·PE2D(coords)
            tokens = self.token_builder(
                feat=feat_flat,
                coords=coords_b,
                ray=ray_flat,
            )                                                # [B, h*w, D]

            # Optional world-frame ray enrichment (additive, gated by σ(world_ray_emb.g))
            if self.use_world_ray and extrinsics is not None and self.world_ray_emb is not None:
                from .side_stack import compute_world_ray_6d
                ext_v = extrinsics[:, v_idx]                # [B, 4, 4]
                world_ray_6d = compute_world_ray_6d(ray_cam, ext_v)            # [B, 6, h, w]
                world_ray_flat = world_ray_6d.permute(0, 2, 3, 1).reshape(B, h_grid * w_grid, 6).contiguous()
                tokens = tokens + self.world_ray_emb(world_ray_flat)

            kv_chunks.append(tokens)

        # Concat across views (shared perceiver pools across all of them)
        kv = torch.cat(kv_chunks, dim=1) if len(kv_chunks) > 1 else kv_chunks[0]  # [B, V*N, D]

        # Perceiver: K=160 learnable queries cross-attend with enriched K/V
        out_tokens = self.resampler(kv)                      # [B, K, D]
        return out_tokens


__all__ = ["GeoPerceiverV2F"]
