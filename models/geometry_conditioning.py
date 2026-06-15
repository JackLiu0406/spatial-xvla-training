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

"""
Segmented DA3 Geometry Conditioning for X-VLA.

Optional, config-gated extension. Pipeline (all on the DA3 *latent* feature
map — never a point cloud):

    DA3 latent + object masks
        -> DA3LatentSegmenter            (soft/hard mask on the latent)
        -> flatten/tokenize + MLP project
        -> GeometryTokenResampler        (Perceiver: variable N -> fixed K)
        -> geometry tokens [B, K, D]

    X-VLA policy tokens
        -> GeometryCrossAttentionFusion  (Q=policy, K/V=geometry)
        -> fused policy tokens

Nothing here runs unless ``config.geometry_conditioning["enabled"] is True``.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("xvla.geometry")


# -----------------------------------------------------------------------------
# Small building blocks
# -----------------------------------------------------------------------------

class _FeedForward(nn.Module):
    """GELU MLP matching the repo's `Mlp` style (tanh-approx GELU)."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class _QueryResampler(nn.Module):
    """
    Shared learnable-query cross-attention resampler core.

    queries [1,K,D] (learnable) cross-attend to kv [B,N,D] -> [B,K,D],
    pre-LN residual + FFN. Used by both ``PerceiverResampler`` (back-compat
    name) and ``GeometryTokenResampler`` (spec name).
    """

    def __init__(
        self,
        dim: int,
        num_queries: int,
        num_heads: int = 8,
        depth: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.query = nn.Parameter(torch.zeros(1, num_queries, dim))
        # Init scale: was std=0.02 — too small. With tiny queries the residual
        # path (x = x + attn_out) keeps x near zero through all layers, and
        # softmax-over-uniform-scores returns the mean of V (~0 after LayerNorm),
        # collapsing geometry_conditioner output to denormals. std=1.0 gives the
        # queries unit-scale signal to start with, matching typical pos-emb init.
        nn.init.normal_(self.query, std=1.0)
        self.layers = nn.ModuleList()
        for _ in range(max(1, depth)):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "q_norm": nn.LayerNorm(dim),
                        "kv_norm": nn.LayerNorm(dim),
                        "attn": nn.MultiheadAttention(
                            dim, num_heads, dropout=dropout, batch_first=True
                        ),
                        "ffn_norm": nn.LayerNorm(dim),
                        "ffn": _FeedForward(dim, mlp_ratio, dropout),
                    }
                )
            )
        self.out_norm = nn.LayerNorm(dim)

    def forward(
        self, kv: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B = kv.shape[0]
        x = self.query.expand(B, -1, -1).contiguous()
        for layer in self.layers:
            q = layer["q_norm"](x)
            k = layer["kv_norm"](kv)
            attn_out, _ = layer["attn"](
                q, k, k, key_padding_mask=key_padding_mask, need_weights=False
            )
            x = x + attn_out
            x = x + layer["ffn"](layer["ffn_norm"](x))
        return self.out_norm(x)


class PerceiverResampler(_QueryResampler):
    """Back-compat alias (kept for existing imports/tests)."""


class GeometryTokenResampler(_QueryResampler):
    """
    Perceiver-style geometry-token resampler (spec name / signature).

        Q = K learnable geometry queries
        K = V = projected DA3 tokens
        -> residual + LayerNorm + FFN
        -> geometry_tokens [B, K, D]
    """

    def __init__(
        self,
        num_geometry_tokens: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        num_layers: int = 1,
    ) -> None:
        super().__init__(
            dim=hidden_dim,
            num_queries=num_geometry_tokens,
            num_heads=num_heads,
            depth=num_layers,
            dropout=dropout,
        )

    def forward(self, da3_tokens: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return super().forward(da3_tokens)


class ViewBiasedGeometryResampler(nn.Module):
    """
    Option A — single global perceiver with learnable per-view attention bias.

    All K queries cross-attend over ALL views' tokens (preserves cross-view
    binding the baseline K=160 single perceiver has), but a learnable scalar
    `view_bias[v]` is added to attention logits before softmax to push the
    attention budget toward more-informative views (head vs wrists).

    Forward: kv [B, V*N, D] with V chunks of N tokens each (contiguous per view).
    """

    def __init__(
        self,
        num_geometry_tokens: int,
        hidden_dim: int,
        num_views: int,
        num_heads: int = 8,
        num_layers: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_geometry_tokens = int(num_geometry_tokens)
        self.num_views = int(num_views)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.query = nn.Parameter(torch.zeros(1, num_geometry_tokens, hidden_dim))
        nn.init.normal_(self.query, std=1.0)
        self.view_bias = nn.Parameter(torch.zeros(num_views))  # log-additive bias per view
        self.layers = nn.ModuleList()
        for _ in range(max(1, num_layers)):
            self.layers.append(nn.ModuleDict({
                "q_norm": nn.LayerNorm(hidden_dim),
                "kv_norm": nn.LayerNorm(hidden_dim),
                "attn": nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True),
                "ffn_norm": nn.LayerNorm(hidden_dim),
                "ffn": _FeedForward(hidden_dim, mlp_ratio, dropout),
            }))
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, kv: torch.Tensor, n_per_view: int) -> torch.Tensor:
        """kv: [B, V*N, D] flattened per-view; n_per_view = N (per-view token count)."""
        B, total_N, D = kv.shape
        V = self.num_views
        assert total_N == V * n_per_view, f"kv has {total_N} tokens, expected V({V})*N({n_per_view})"
        K = self.num_geometry_tokens
        # Build (K, V*N) attn_mask where mask[:, j] = view_bias[view_of(j)]
        # Same bias for every query, additive to attention logits before softmax.
        view_bias_per_token = self.view_bias.repeat_interleave(n_per_view)  # [V*N]
        # MHA accepts attn_mask of shape (L, S) — broadcasts over batch & heads
        attn_mask = view_bias_per_token.unsqueeze(0).expand(K, -1).contiguous().to(kv.dtype)
        x = self.query.expand(B, -1, -1).contiguous()
        for layer in self.layers:
            q = layer["q_norm"](x)
            k_in = layer["kv_norm"](kv)
            attn_out, _ = layer["attn"](
                q, k_in, k_in, attn_mask=attn_mask, need_weights=False
            )
            x = x + attn_out
            x = x + layer["ffn"](layer["ffn_norm"](x))
        return self.out_norm(x)


class HierarchicalGeometryResampler(nn.Module):
    """
    Option B — two-stage perceiver: per-view stage-1 perceivers compress each
    view independently (asymmetric capacity allocation), then a stage-2 global
    perceiver fuses across views.

    Stage 1: each view -> own perceiver with K_v output tokens.
    Stage 2: concat stage-1 outputs -> single global perceiver with K_global tokens.

    Output K_global tokens have both per-view capacity guarantees (stage 1) AND
    full cross-view binding (stage 2). Output token count can match baseline so
    downstream cross-attn fusion isn't disrupted.
    """

    def __init__(
        self,
        tokens_per_view: List[int],
        num_geometry_tokens: int,
        hidden_dim: int,
        num_heads: int = 8,
        stage1_layers: int = 1,
        stage2_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.tokens_per_view = list(tokens_per_view)
        self.num_geometry_tokens = int(num_geometry_tokens)
        self.stage1 = nn.ModuleList([
            GeometryTokenResampler(
                num_geometry_tokens=int(k),
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=stage1_layers,
                dropout=dropout,
            )
            for k in tokens_per_view
        ])
        self.stage2 = GeometryTokenResampler(
            num_geometry_tokens=int(num_geometry_tokens),
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=stage2_layers,
            dropout=dropout,
        )

    def forward(self, projected: torch.Tensor) -> torch.Tensor:
        """projected: [B, V, N, D]  per-view projected tokens"""
        V = projected.shape[1]
        assert V == len(self.stage1), f"input V={V} != len(tokens_per_view)={len(self.stage1)}"
        # Stage 1 per-view
        stage1_chunks = [self.stage1[v](projected[:, v]) for v in range(V)]
        stage1_concat = torch.cat(stage1_chunks, dim=1)  # [B, sum(K_v), D]
        # Stage 2 global fusion
        return self.stage2(stage1_concat)


# -----------------------------------------------------------------------------
# DA3 latent segmenter
# -----------------------------------------------------------------------------

class DA3LatentSegmenter(nn.Module):
    """
    Segment a DA3 *latent feature map* with language-conditioned object masks.

    The segmentation happens on the DA3 latent (NOT a point cloud, NOT the
    RGB) before geometry-token compression and cross-attention.

    Accepted ``da3_features`` shapes
        [B, C, H, W] | [B, N, C] | [B, V, C, H, W] | [B, V, N, C]
    Accepted ``object_masks`` shapes
        [B, M, Hi, Wi] | [B, 1, Hi, Wi] | [B, V, M, Hi, Wi] | [B, V, 1, Hi, Wi]
        (also tolerates [B, Hi, Wi] / [B, V, Hi, Wi])

    Behavior
      * missing masks: raise (or, if ``allow_missing_masks``, pass-through + warn)
      * masks clamped to [0, 1]
      * masks bilinearly resized to the DA3 feature grid (batch/view preserved)
      * M objects reduced via ``mask_reduce`` (max | sum_clamp | separate_objects)
      * ``segmentation_mode``:
          soft_mask: feat*fg + bw*feat*(1-fg)   (default, bw=0.2)
          hard_mask: feat*fg
          none:      feat
    Output has the **same shape** as the input ``da3_features``.
    """

    def __init__(
        self,
        background_weight: float = 0.2,
        segmentation_mode: str = "soft_mask",
        mask_reduce: str = "max",
        allow_missing_masks: bool = False,
        debug_shapes: bool = False,
    ) -> None:
        super().__init__()
        self.background_weight = float(background_weight)
        self.segmentation_mode = str(segmentation_mode)
        self.mask_reduce = str(mask_reduce)
        self.allow_missing_masks = bool(allow_missing_masks)
        self.debug_shapes = bool(debug_shapes)
        self._logged = False
        if self.segmentation_mode not in ("soft_mask", "hard_mask", "none"):
            raise ValueError(
                f"segmentation_mode={self.segmentation_mode!r} not in "
                "{'soft_mask','hard_mask','none'}"
            )
        if self.mask_reduce not in ("max", "sum_clamp", "separate_objects"):
            raise ValueError(
                f"mask_reduce={self.mask_reduce!r} not in "
                "{'max','sum_clamp','separate_objects'}"
            )

    # ---- mask reduction -------------------------------------------------- #
    def _reduce(self, masks: torch.Tensor, obj_dim: int) -> torch.Tensor:
        if masks.shape[obj_dim] == 1:
            return masks.squeeze(obj_dim)
        if self.mask_reduce == "max":
            return masks.max(dim=obj_dim).values
        if self.mask_reduce == "sum_clamp":
            return masks.sum(dim=obj_dim).clamp(0.0, 1.0)
        raise NotImplementedError(
            "mask_reduce='separate_objects' is not implemented: preserving "
            "per-object features would change the downstream token count and "
            "resampler contract. Use 'max' or 'sum_clamp'."
        )

    @staticmethod
    def _apply_mask(feat: torch.Tensor, fg: torch.Tensor, mode: str, bw: float) -> torch.Tensor:
        # NOTE: was named `_apply` originally, which shadowed nn.Module._apply
        # (the recursive .to/.cpu/.cuda machinery). Triggered TypeError when
        # accelerator.prepare(model) ran .to(device) and torch tried to call
        # _apply(fn) on this submodule. Renamed to break the collision.
        if mode == "none":
            return feat
        if mode == "hard_mask":
            return feat * fg
        return feat * fg + bw * feat * (1.0 - fg)

    # ---- forward --------------------------------------------------------- #
    def forward(
        self,
        da3_features: torch.Tensor,
        object_masks: Optional[torch.Tensor] = None,
        da3_feature_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if object_masks is None or self.segmentation_mode == "none":
            if object_masks is None and not self.allow_missing_masks and self.segmentation_mode != "none":
                raise ValueError(
                    "DA3LatentSegmenter: object_masks is None but "
                    "allow_missing_masks=False. Provide precomputed masks "
                    "(scripts/precompute_object_masks.py) or set "
                    "geometry_conditioning.allow_missing_masks=true."
                )
            if object_masks is None and self.segmentation_mode != "none":
                logger.warning(
                    "DA3LatentSegmenter: no object_masks; returning unsegmented "
                    "DA3 features (allow_missing_masks=True)."
                )
            return da3_features

        feats = da3_features.float()
        masks = object_masks.float().clamp(0.0, 1.0)
        dim = feats.dim()

        if self.debug_shapes and not self._logged:
            self._logged = True
            logger.info(
                "[geometry][segmenter] da3=%s masks=%s mode=%s reduce=%s",
                tuple(feats.shape), tuple(masks.shape),
                self.segmentation_mode, self.mask_reduce,
            )

        # ---- multi-view dense [B,V,C,H,W] ----
        if dim == 5:
            B, V, C, H, W = feats.shape
            m = masks if masks.dim() == 5 else masks.unsqueeze(2)  # [B,V,M,Hi,Wi]
            fg = self._reduce(m, obj_dim=2)                          # [B,V,Hi,Wi]
            fg = F.interpolate(
                fg.reshape(B * V, 1, *fg.shape[-2:]), size=(H, W),
                mode="bilinear", align_corners=False,
            ).reshape(B, V, 1, H, W)
            return self._apply_mask(feats, fg, self.segmentation_mode, self.background_weight)

        # ---- single-view dense [B,C,H,W] ----
        if dim == 4 and (da3_feature_hw is None and masks.dim() >= 3
                         and masks.shape[-2] > 1 and masks.shape[-1] > 1
                         and feats.shape[1] != feats.shape[-1]):
            B, C, H, W = feats.shape
            m = masks if masks.dim() == 4 else masks.unsqueeze(1)   # [B,M,Hi,Wi]
            fg = self._reduce(m, obj_dim=1)                          # [B,Hi,Wi]
            fg = F.interpolate(
                fg.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
            )                                                       # [B,1,H,W]
            return self._apply_mask(feats, fg, self.segmentation_mode, self.background_weight)

        # ---- tokenized [B,N,C] or [B,V,N,C] ----
        if dim in (3, 4):
            multiview = dim == 4
            if multiview:
                B, V, N, C = feats.shape
            else:
                B, N, C = feats.shape
                V = 1
            hw = da3_feature_hw or _infer_hw(N)
            if hw is None:
                raise ValueError(
                    f"Tokenized DA3 features [..., N={N}, C] need da3_feature_hw "
                    "to align spatial masks (N is not a perfect square)."
                )
            H, W = hw
            mv = masks if masks.dim() in (4, 5) else masks.unsqueeze(-3)
            if multiview:
                mv = mv if mv.dim() == 5 else mv.reshape(B, V, -1, *mv.shape[-2:])
                fg = self._reduce(mv, obj_dim=2)                     # [B,V,Hi,Wi]
                fg = F.interpolate(
                    fg.reshape(B * V, 1, *fg.shape[-2:]), size=(H, W),
                    mode="bilinear", align_corners=False,
                ).reshape(B, V, H * W, 1)
                ftok = feats.reshape(B, V, N, C)
                return self._apply_mask(ftok, fg, self.segmentation_mode,
                                   self.background_weight).reshape(B, V, N, C)
            mv = mv if mv.dim() == 4 else mv.reshape(B, -1, *mv.shape[-2:])
            fg = self._reduce(mv, obj_dim=1)                         # [B,Hi,Wi]
            fg = F.interpolate(
                fg.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
            ).reshape(B, H * W, 1)
            return self._apply_mask(feats, fg, self.segmentation_mode, self.background_weight)

        raise ValueError(
            f"Unsupported da3_features shape {tuple(feats.shape)}. Expected "
            "[B,C,H,W], [B,N,C], [B,V,C,H,W] or [B,V,N,C]."
        )


def _infer_hw(n: int) -> Optional[Tuple[int, int]]:
    r = int(round(n ** 0.5))
    return (r, r) if r * r == n else None


# -----------------------------------------------------------------------------
# Geometry conditioner
# -----------------------------------------------------------------------------

class SegmentedDA3GeometryConditioner(nn.Module):
    """
    DA3 latent (+ masks) -> segmented latent -> tokens -> projector ->
    Perceiver resampler -> geometry tokens ``[B, K, D]`` (D == hidden_dim).

    Flatten rules:
        [B,C,H,W]    -> [B, H*W, C]
        [B,N,C]      -> [B, N, C]
        [B,V,C,H,W]  -> [B, V*H*W, C]
        [B,V,N,C]    -> [B, V*N, C]
    """

    def __init__(self, cfg: dict, hidden_dim: int) -> None:
        super().__init__()
        self.cfg = dict(cfg)

        def g(*keys, default=None):
            for k in keys:
                if k in cfg and cfg[k] is not None:
                    return cfg[k]
            return default

        self.hidden_dim = int(g("geometry_hidden_dim") or hidden_dim)
        self.num_geometry_tokens = int(g("num_geometry_tokens", default=32))
        self.use_segmentation = bool(
            g("use_da3_latent_segmentation", "use_object_mask", default=True)
        )
        self.da3_source = str(g("da3_source", "source", default="precomputed"))
        self.resampler_type = str(
            g("geometry_resampler_type", "resampler_type", default="perceiver")
        )
        self.projector_type = str(g("geometry_projector_type", default="mlp"))
        self._da3_input_dim = g("da3_input_dim")
        self.debug_shapes = bool(g("debug_shapes", default=False))
        self._logged = False

        self.segmenter = DA3LatentSegmenter(
            background_weight=float(g("background_weight", "mask_background_weight",
                                      default=0.2)),
            segmentation_mode=str(g("segmentation_mode", default="soft_mask")),
            mask_reduce=str(g("mask_reduce", default="max")),
            allow_missing_masks=bool(g("allow_missing_masks",
                                       "allow_missing_geometry", default=False)),
            debug_shapes=self.debug_shapes,
        )

        self.input_proj: Optional[nn.Module] = None
        if self._da3_input_dim is not None:
            self._build_projector(int(self._da3_input_dim))

        # Per-view token budget (perceiver_per_view only). When set, one
        # GeometryTokenResampler is built per entry; outputs are concatenated.
        # sum(geometry_tokens_per_view) must equal num_geometry_tokens.
        self.tokens_per_view: Optional[List[int]] = (
            list(g("geometry_tokens_per_view")) if g("geometry_tokens_per_view") is not None else None
        )
        if self.resampler_type == "perceiver":
            self.resampler: nn.Module = GeometryTokenResampler(
                num_geometry_tokens=self.num_geometry_tokens,
                hidden_dim=self.hidden_dim,
                num_heads=int(g("cross_attention_heads", default=8)),
                dropout=float(g("cross_attention_dropout", default=0.0)),
                num_layers=int(g("cross_attention_layers", default=1)),
            )
        elif self.resampler_type == "perceiver_per_view":
            if self.tokens_per_view is None:
                raise ValueError(
                    "geometry_resampler_type='perceiver_per_view' requires "
                    "'geometry_tokens_per_view' (list[int], one entry per view)."
                )
            if sum(self.tokens_per_view) != self.num_geometry_tokens:
                raise ValueError(
                    f"sum(geometry_tokens_per_view)={sum(self.tokens_per_view)} "
                    f"!= num_geometry_tokens={self.num_geometry_tokens}"
                )
            self.resampler = nn.ModuleList([
                GeometryTokenResampler(
                    num_geometry_tokens=int(k),
                    hidden_dim=self.hidden_dim,
                    num_heads=int(g("cross_attention_heads", default=8)),
                    dropout=float(g("cross_attention_dropout", default=0.0)),
                    num_layers=int(g("cross_attention_layers", default=1)),
                )
                for k in self.tokens_per_view
            ])
        elif self.resampler_type == "perceiver_view_biased":
            # Option A: single global perceiver with learnable per-view bias.
            # Requires 'num_views' (default 3). num_geometry_tokens is unchanged.
            self.resampler = ViewBiasedGeometryResampler(
                num_geometry_tokens=self.num_geometry_tokens,
                hidden_dim=self.hidden_dim,
                num_views=int(g("num_views", default=3)),
                num_heads=int(g("cross_attention_heads", default=8)),
                num_layers=int(g("cross_attention_layers", default=1)),
                dropout=float(g("cross_attention_dropout", default=0.0)),
            )
        elif self.resampler_type == "perceiver_hierarchical":
            # Option B: per-view stage-1 perceivers -> global stage-2 perceiver.
            # geometry_tokens_per_view = stage-1 K per view (e.g. [64, 16, 16]).
            # num_geometry_tokens = stage-2 K (final output, e.g. 160).
            if self.tokens_per_view is None:
                raise ValueError(
                    "geometry_resampler_type='perceiver_hierarchical' requires "
                    "'geometry_tokens_per_view' (list[int], stage-1 K per view)."
                )
            self.resampler = HierarchicalGeometryResampler(
                tokens_per_view=self.tokens_per_view,
                num_geometry_tokens=self.num_geometry_tokens,
                hidden_dim=self.hidden_dim,
                num_heads=int(g("cross_attention_heads", default=8)),
                stage1_layers=int(g("hierarchical_stage1_layers", default=1)),
                stage2_layers=int(g("cross_attention_layers", default=1)),
                dropout=float(g("cross_attention_dropout", default=0.0)),
            )
        elif self.resampler_type == "adaptive_pool":
            self.resampler = nn.AdaptiveAvgPool1d(self.num_geometry_tokens)
            self.post_pool_norm = nn.LayerNorm(self.hidden_dim)
        else:
            raise ValueError(
                f"Unknown geometry_resampler_type={self.resampler_type!r}; "
                "expected 'perceiver', 'perceiver_per_view', "
                "'perceiver_view_biased', 'perceiver_hierarchical', or 'adaptive_pool'."
            )

    # ---- helpers --------------------------------------------------------- #
    def _build_projector(self, in_dim: int) -> None:
        if self.projector_type != "mlp":
            raise ValueError(
                f"geometry_projector_type={self.projector_type!r} unsupported "
                "(only 'mlp')."
            )
        proj = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        ref = next(self.parameters(), None)
        if ref is not None:
            proj = proj.to(device=ref.device, dtype=ref.dtype)
        self.input_proj = proj
        self._da3_input_dim = in_dim

    def _flatten_per_view(self, da3: torch.Tensor) -> torch.Tensor:
        """
        Tokenize while preserving the view axis -> [B, V, N, C].

        Required by perceiver_per_view. Accepts only inputs that have an
        explicit view axis: [B,V,C,H,W] or [B,V,N,C]. The view-less shapes
        ([B,C,H,W], [B,N,C]) cannot be split per view and raise here.
        """
        d = da3.dim()
        if d == 5:                                   # [B,V,C,H,W]
            B, V, C, H, W = da3.shape
            return da3.permute(0, 1, 3, 4, 2).reshape(B, V, H * W, C)
        if d == 4:
            cdim = self._da3_input_dim
            tokenized = cdim is not None and da3.shape[-1] == cdim and da3.shape[1] != cdim
            if tokenized:                            # [B,V,N,C]
                return da3
        raise ValueError(
            f"perceiver_per_view requires per-view-shaped da3_features "
            f"([B,V,C,H,W] or [B,V,N,C]); got {tuple(da3.shape)}."
        )

    def _flatten(self, da3: torch.Tensor) -> torch.Tensor:
        """
        Tokenize per the documented flatten rules -> [B, N, C].

        The 4-D case ([B,C,H,W] dense vs [B,V,N,C] tokenized) is disambiguated
        with ``da3_input_dim`` when known; otherwise a 4-D tensor is assumed
        dense [B,C,H,W] (the common DA3 feature-map layout).
        """
        d = da3.dim()
        if d == 3:                                   # [B,N,C]
            return da3
        if d == 4:
            cdim = self._da3_input_dim
            tokenized = cdim is not None and da3.shape[-1] == cdim and da3.shape[1] != cdim
            if tokenized:                            # [B,V,N,C]
                B, V, N, C = da3.shape
                return da3.reshape(B, V * N, C)
            B, C, H, W = da3.shape                    # [B,C,H,W]
            return da3.flatten(2).transpose(1, 2)
        if d == 5:                                   # [B,V,C,H,W]
            B, V, C, H, W = da3.shape
            return da3.permute(0, 1, 3, 4, 2).reshape(B, V * H * W, C)
        raise ValueError(f"Unsupported da3_features shape {tuple(da3.shape)}.")

    # ---- forward --------------------------------------------------------- #
    def forward(
        self,
        da3_features: torch.Tensor,
        object_masks: Optional[torch.Tensor] = None,
        da3_feature_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if self.da3_source == "extractor_stub":
            raise NotImplementedError(
                "geometry_conditioning.da3_source='extractor_stub': in-model "
                "DA3 encoding is not implemented. Use 'precomputed' with "
                "da3_features in the batch (scripts/precompute_da3_features.py)."
            )
        if not torch.is_tensor(da3_features):
            raise TypeError(f"da3_features must be a tensor, got {type(da3_features)!r}.")

        feats = da3_features.float()
        if self.use_segmentation:
            feats = self.segmenter(feats, object_masks, da3_feature_hw)

        if self.resampler_type == "perceiver_per_view":
            per_view = self._flatten_per_view(feats)     # [B, V, N, C]
            B, V, N, C = per_view.shape
            if V != len(self.resampler):
                raise ValueError(
                    f"input V={V} != len(geometry_tokens_per_view)={len(self.resampler)}."
                )
            if self.input_proj is None:
                self._build_projector(C)
            elif self._da3_input_dim != C:
                raise ValueError(
                    f"da3_features channel dim {C} != configured/inferred "
                    f"da3_input_dim {self._da3_input_dim}. Set "
                    "geometry_conditioning.da3_input_dim explicitly if it varies."
                )
            projected = self.input_proj(per_view.reshape(B * V, N, C))
            projected = projected.reshape(B, V, N, self.hidden_dim)
            geom = torch.cat(
                [self.resampler[v](projected[:, v]) for v in range(V)], dim=1
            )                                            # [B, sum(K_v), D]
            if self.debug_shapes and not self._logged:
                self._logged = True
                logger.info(
                    "[geometry][conditioner][per_view] in=%s per_view=%s geometry=%s",
                    tuple(da3_features.shape), tuple(per_view.shape), tuple(geom.shape),
                )
            return geom

        if self.resampler_type == "perceiver_view_biased":
            # Option A: single global perceiver, per-view bias on attention logits.
            per_view = self._flatten_per_view(feats)     # [B, V, N, C]
            B, V, N, C = per_view.shape
            if V != self.resampler.num_views:
                raise ValueError(
                    f"input V={V} != configured num_views={self.resampler.num_views}."
                )
            if self.input_proj is None:
                self._build_projector(C)
            elif self._da3_input_dim != C:
                raise ValueError(
                    f"da3_features channel dim {C} != configured/inferred "
                    f"da3_input_dim {self._da3_input_dim}."
                )
            projected = self.input_proj(per_view.reshape(B * V, N, C))
            projected = projected.reshape(B, V * N, self.hidden_dim)   # contiguous per view
            geom = self.resampler(projected, n_per_view=N)             # [B, K, D]
            if self.debug_shapes and not self._logged:
                self._logged = True
                logger.info(
                    "[geometry][conditioner][view_biased] in=%s kv=%s geometry=%s",
                    tuple(da3_features.shape), tuple(projected.shape), tuple(geom.shape),
                )
            return geom

        if self.resampler_type == "perceiver_hierarchical":
            # Option B: per-view stage-1 -> global stage-2.
            per_view = self._flatten_per_view(feats)     # [B, V, N, C]
            B, V, N, C = per_view.shape
            if V != len(self.resampler.stage1):
                raise ValueError(
                    f"input V={V} != len(stage1 perceivers)={len(self.resampler.stage1)}."
                )
            if self.input_proj is None:
                self._build_projector(C)
            elif self._da3_input_dim != C:
                raise ValueError(
                    f"da3_features channel dim {C} != configured/inferred "
                    f"da3_input_dim {self._da3_input_dim}."
                )
            projected = self.input_proj(per_view.reshape(B * V, N, C))
            projected = projected.reshape(B, V, N, self.hidden_dim)
            geom = self.resampler(projected)                            # [B, K_global, D]
            if self.debug_shapes and not self._logged:
                self._logged = True
                logger.info(
                    "[geometry][conditioner][hierarchical] in=%s per_view=%s geometry=%s",
                    tuple(da3_features.shape), tuple(per_view.shape), tuple(geom.shape),
                )
            return geom

        tokens = self._flatten(feats)                # [B, N, C]
        C = tokens.shape[-1]
        if self.input_proj is None:
            self._build_projector(C)
        elif self._da3_input_dim != C:
            raise ValueError(
                f"da3_features channel dim {C} != configured/inferred "
                f"da3_input_dim {self._da3_input_dim}. Set "
                "geometry_conditioning.da3_input_dim explicitly if it varies."
            )
        tokens = self.input_proj(tokens)             # [B, N, D]

        if self.resampler_type == "perceiver":
            geom = self.resampler(tokens)            # [B, K, D]
        else:
            pooled = self.resampler(tokens.transpose(1, 2)).transpose(1, 2)
            geom = self.post_pool_norm(pooled)

        if self.debug_shapes and not self._logged:
            self._logged = True
            logger.info(
                "[geometry][conditioner] in=%s tokens=%s geometry=%s",
                tuple(da3_features.shape), tuple(tokens.shape), tuple(geom.shape),
            )
        return geom


# -----------------------------------------------------------------------------
# Cross-attention fusion with X-VLA policy tokens
# -----------------------------------------------------------------------------

class GeometryCrossAttentionFusion(nn.Module):
    """
    Q = policy_tokens [B,T,D]; K = V = geometry_tokens [B,K,D].

        fused = CrossAttention(Q, K, V)
        out   = LayerNorm(policy_tokens + fused)
        out   = LayerNorm(out + FFN(out))   -> [B, T, D]
    """

    def __init__(
        self,
        hidden_dim: int = None,
        num_heads: int = 8,
        num_layers: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        dim: int = None,  # legacy alias for hidden_dim
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else dim
        if hidden_dim is None:
            raise ValueError("GeometryCrossAttentionFusion needs hidden_dim.")
        self.layers = nn.ModuleList()
        for _ in range(max(1, num_layers)):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "q_norm": nn.LayerNorm(hidden_dim),
                        "kv_norm": nn.LayerNorm(hidden_dim),
                        "attn": nn.MultiheadAttention(
                            hidden_dim, num_heads, dropout=dropout, batch_first=True
                        ),
                        "attn_norm": nn.LayerNorm(hidden_dim),
                        "ffn_norm": nn.LayerNorm(hidden_dim),
                        "ffn": _FeedForward(hidden_dim, mlp_ratio, dropout),
                    }
                )
            )

    def forward(
        self, policy_tokens: torch.Tensor, geometry_tokens: torch.Tensor
    ) -> torch.Tensor:
        if policy_tokens.shape[-1] != geometry_tokens.shape[-1]:
            raise ValueError(
                f"policy/geometry hidden dims differ: "
                f"{policy_tokens.shape[-1]} vs {geometry_tokens.shape[-1]}."
            )
        x = policy_tokens
        for layer in self.layers:
            q = layer["q_norm"](x)
            kv = layer["kv_norm"](geometry_tokens)
            attn_out, _ = layer["attn"](q, kv, kv, need_weights=False)
            x = layer["attn_norm"](x + attn_out)
            x = layer["ffn_norm"](x + layer["ffn"](x))
        return x


__all__ = [
    "DA3LatentSegmenter",
    "SegmentedDA3GeometryConditioner",
    "GeometryTokenResampler",
    "PerceiverResampler",
    "GeometryCrossAttentionFusion",
]
