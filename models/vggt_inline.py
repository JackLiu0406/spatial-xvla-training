# ------------------------------------------------------------------------------
# VGGT-Omega inline encoder for X-VLA — drop-in alternative to DA3 inline.
# Mirrors models/da3_inline.py: same forward signature and output shape so the
# downstream geometry projector / perceiver / cross-attn fusion are untouched.
# Selected via geometry_conditioning.geometry_backbone="vggt".
#
# Differences from DA3 path:
#   - patch_size 16 (vs DA3 14) → process_res must be multiple of 16
#   - aggregator does its own ImageNet renorm internally → we feed [0, 1]
#   - aggregator token layout: [camera, register×16, patches] → slice via
#     patch_token_start before reshaping to a dense grid
#   - cached output channels = 2 × embed_dim (concat of frame + inter-frame
#     streams) → output is 2048-D for the 1B (ViT-L) variant
#   - VGGT-Omega has no extrinsics / intrinsics input — pose is predicted, not
#     consumed. extrinsics/intrinsics kwargs are accepted for signature parity
#     with the DA3 path but ignored.
#
# Vendored source lives at ../third_party/vggt_omega/. Added to sys.path here
# so the package's absolute self-imports (`from vggt_omega...`) resolve.
# ------------------------------------------------------------------------------
from __future__ import annotations
import logging
import os
import sys
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add vendored vggt_omega to sys.path (it uses absolute imports internally).
_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
from vggt_omega.models.vggt_omega import VGGTOmega   # noqa: E402

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class VGGTInlineEncoder(nn.Module):
    """
    Inline VGGT-Omega multi-view encoder.

    Args mirror DA3InlineEncoder. Returns [B, V, C=2*embed_dim, feature_size, feature_size]
    so the downstream geometry path is shape-compatible (input dim 2048 instead
    of DA3-LARGE's 1024 — the geometry projector reshapes accordingly).

    Args:
        model_name:    HF repo id holding the .pt weights (e.g. "JackLiu0406/vggt-omega-1b").
        ckpt_filename: filename within the repo (default "vggt_omega_1b_512.pt").
        use_bf16:      wrap the aggregator forward in bf16 autocast.
        gradient_checkpointing: wrap each frame/inter-frame block with checkpoint().
        freeze:        freeze the VGGT backbone (no backward through aggregator).
        feature_size:  spatial size of the returned dense grid (interpolated up
                       from the patch grid).
        feature_layer: "last" (only mode supported in this cell). Use cached
                       layer outputs[-1] (block index 23 by default).
        process_res:   resolution VGGT sees. 224 keeps the dataset native size
                       (224 / 16 = 14 patches per side, valid). Larger e.g. 512
                       can be set if you want VGGT's pretraining-time native.
    """

    def __init__(
        self,
        model_name: str = "JackLiu0406/vggt-omega-1b",
        ckpt_filename: str = "vggt_omega_1b_512.pt",
        use_bf16: bool = True,
        gradient_checkpointing: bool = True,
        freeze: bool = True,
        feature_size: int = 64,
        feature_layer: str | int = "last",
        process_res: int = 224,
        multi_fusion_init: str = "last_only",
    ) -> None:
        super().__init__()
        self._is_multi = isinstance(feature_layer, str) and feature_layer.lower() == "multi"
        if (not self._is_multi) and feature_layer != "last":
            raise NotImplementedError(
                f"VGGT inline supports feature_layer='last' or 'multi' (got {feature_layer!r})."
            )
        if int(process_res) % 16 != 0:
            raise ValueError(
                f"process_res={process_res} is not a multiple of patch_size=16 "
                "(VGGT-Omega's patch_embed requires it)."
            )

        self.model_name = model_name
        self.ckpt_filename = ckpt_filename
        self.use_bf16 = bool(use_bf16)
        self.feature_size = int(feature_size)
        self.feature_layer = feature_layer
        self.process_res = int(process_res)
        # 1B variant. Heads are disabled — we only need the aggregator output.
        # patch_size=16 and embed_dim=1024 are VGGT-Omega's published 1B defaults.
        self.patch_size = 16
        self.embed_dim = 1024
        # Cached aggregator output is [frame_tokens ; inter_frame_tokens] on dim -1,
        # so the channel dim of the dense feature we return is 2*embed_dim = 2048.
        self.out_channels = 2 * self.embed_dim

        logger.info("[vggt_inline] building VGGTOmega(patch_size=%d, embed_dim=%d) heads OFF",
                    self.patch_size, self.embed_dim)
        self.model = VGGTOmega(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            enable_camera=False,
            enable_depth=False,
            enable_alignment=False,
        )
        self._load_pretrained_weights()

        # Skip checkpoint wrapping when the backbone is frozen — backward never
        # traverses these blocks, so checkpoint() just adds saved_tensor_hooks +
        # recompute overhead with zero memory benefit. py-spy attributed ~9.5%
        # of VGGT wall samples to _ckpt_fwd in the running training (2026-06-11).
        if gradient_checkpointing and not freeze:
            self._enable_gradient_checkpointing()

        if freeze:
            # CRITICAL: scope freeze to self.model only — never to self.parameters().
            # Any adapter added later (e.g. multi-layer fusion) must stay trainable.
            # Matches the DA3-inline convention so train.py's name.startswith("model.")
            # adapter/backbone split works without modification.
            for p in self.model.parameters():
                p.requires_grad_(False)
            logger.info("[vggt_inline] backbone frozen (no backward through aggregator)")

        # Multi-layer fusion adapter — per-tap LayerNorm → channel-concat →
        # 1×1 Conv(4·2C → 2C). Sits OUTSIDE self.model (stays trainable under freeze).
        # Output shape is shape-identical to single-layer mode, so the downstream
        # geometry projector / perceiver / cross-attn don't change.
        self._fuse_init_mode = str(multi_fusion_init)
        if self._is_multi:
            cached = sorted(self.model.aggregator.cached_layer_indices)
            if len(cached) != 4:
                raise RuntimeError(
                    f"VGGT multi-layer fusion expects 4 cached layers; "
                    f"aggregator.cached_layer_indices = {cached}"
                )
            self._resolved_layers = list(cached)
            C = self.out_channels   # 2 * embed_dim = 2048
            self._fuse_norms = nn.ModuleList([nn.LayerNorm(C) for _ in range(4)])
            self._fuse_conv = nn.Conv2d(4 * C, C, kernel_size=1, bias=True)
            self._init_fuse_conv(self._fuse_init_mode)
            logger.info(
                "[vggt_inline] multi-layer mode: layers=%s channels=%d fuse_init=%s",
                self._resolved_layers, C, self._fuse_init_mode,
            )

        # denorm buffers (X-VLA dataloader provides ImageNet-normalized images;
        # VGGT-Omega's aggregator does its own ImageNet renorm internally, so
        # we have to feed it [0, 1]).
        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def _init_fuse_conv(self, mode: str) -> None:
        """
        Initialize the 1×1 fusion Conv2d(4·2C → 2C) so step 0 is byte-identical
        to single-layer behavior (mode='last_only') or evenly-averaged across
        layers (mode='uniform'). Bias is always zero. LayerNorm params left at
        their nn.LayerNorm defaults (weight=1, bias=0).
        """
        C = self.out_channels
        w = self._fuse_conv.weight   # [C, 4C, 1, 1]
        b = self._fuse_conv.bias     # [C]
        with torch.no_grad():
            w.zero_()
            if mode == "last_only":
                # output = last-layer feature exactly (identity on the deepest tap)
                eye = torch.eye(C, dtype=w.dtype, device=w.device)
                w[:, 3 * C:4 * C, 0, 0].copy_(eye)
            elif mode == "uniform":
                eye = torch.eye(C, dtype=w.dtype, device=w.device) / 4.0
                for i in range(4):
                    w[:, i * C:(i + 1) * C, 0, 0].copy_(eye)
            else:
                raise ValueError(
                    f"unknown multi_fusion_init={mode!r}; expected 'last_only' | 'uniform'"
                )
            b.zero_()

    # ------------------------------------------------------------------ helpers
    def _resolve_ckpt_path(self) -> str:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id=self.model_name,
            filename=self.ckpt_filename,
            cache_dir=os.environ.get("HF_HOME", None),
        )

    def _load_pretrained_weights(self) -> None:
        """Load VGGT-Omega weights from the HF repo into self.model."""
        path = self._resolve_ckpt_path()
        logger.info("[vggt_inline] loading weights from %s", path)
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        # Drop the head weights (camera_head.*, dense_head.*, text_alignment_head.*) —
        # they don't exist in self.model because enable_*=False, but load_state_dict
        # would otherwise raise on unexpected keys. strict=False handles this.
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        # Filter the noise from the report.
        head_prefixes = ("camera_head", "dense_head", "text_alignment_head")
        unexpected_real = [k for k in unexpected if not any(k.startswith(p) for p in head_prefixes)]
        if missing:
            logger.warning("[vggt_inline] %d missing keys (first 5): %s", len(missing), missing[:5])
        if unexpected_real:
            logger.warning("[vggt_inline] %d unexpected non-head keys (first 5): %s",
                           len(unexpected_real), unexpected_real[:5])
        logger.info("[vggt_inline] loaded VGGT-Omega-1B aggregator weights "
                    "(skipped %d head keys)", len(unexpected) - len(unexpected_real))

    def _enable_gradient_checkpointing(self) -> None:
        """Wrap each frame_block and inter_frame_block's forward with checkpoint()."""
        try:
            import torch.utils.checkpoint as _cp
            n_wrapped = 0
            for blocks in (self.model.aggregator.frame_blocks,
                           self.model.aggregator.inter_frame_blocks):
                for blk in blocks:
                    _orig = blk.forward
                    def _ckpt_fwd(*a, _o=_orig, **kw):
                        return _cp.checkpoint(_o, *a, use_reentrant=False, **kw)
                    blk.forward = _ckpt_fwd
                    n_wrapped += 1
            logger.info("[vggt_inline] gradient_checkpointing ON: wrapped %d blocks", n_wrapped)
        except Exception as e:
            logger.warning("[vggt_inline] gradient_checkpointing setup failed: %s", e)

    def reload_pretrained_weights(self) -> None:
        """
        Restore VGGT-Omega weights AFTER ``XVLA.from_pretrained`` runs.

        HF's loader re-inits every key not present in the X-VLA checkpoint, which
        wipes the VGGT backbone we loaded in __init__. Call this once, right
        after ``XVLA.from_pretrained(...)``, mirroring the DA3 path.

        Multi-layer fusion adapter (LayerNorms + Conv) is also re-initialised
        to its `last_only`/`uniform` startpoint here when re-init isn't suppressed
        — but in the normal training entry XVLA_SKIP_GEOMETRY_REINIT=1 prevents
        reload_pretrained_weights() from running at all on resumed ckpts, so
        learned fusion weights from the resumed ckpt are preserved.
        """
        logger.info("[vggt_inline] reloading pretrained %s (HF re-init wiped it)", self.model_name)
        self._load_pretrained_weights()
        for p in self.model.parameters():
            p.requires_grad_(False)
        if getattr(self, "_is_multi", False):
            for ln in self._fuse_norms:
                nn.init.ones_(ln.weight)
                nn.init.zeros_(ln.bias)
            self._init_fuse_conv(self._fuse_init_mode)
        logger.info("[vggt_inline] reload done, backbone frozen")

    # ------------------------------------------------------------------- forward
    def _batched_call(self, images_normed: torch.Tensor) -> torch.Tensor:
        """
        One batched call into VGGT-Omega's aggregator for the whole batch.

        Pipeline:
          1. Denormalize ImageNet stats → [0, 1] (VGGT does its own renorm).
          2. Resize to a patch-clean (H, W) only if the input is not already so.
             Rectangular input is supported natively (VGGT uses RoPE pos-embed).
          3. bf16 autocast → aggregator(images) → (cached_tokens_list, patch_token_start).
          4. In last-layer mode: take outputs[-1].
             In multi-layer mode: take all 4 cached outputs, LayerNorm each,
             channel-concat, fuse via 1×1 Conv(4·2C → 2C).
             Slice off [camera, register×16] prefix, reshape patches to (h, w).
          5. F.interpolate to (feature_size, feature_size).
        Returns [B, V, 2*embed_dim, feature_size, feature_size].
        """
        B, V, Cin, H, W = images_normed.shape
        # 1. Denormalize
        x = images_normed * self._std + self._mean
        x = x.clamp_(0.0, 1.0)
        # 2. Resize only if dims aren't already patch-clean. Prefer to keep the
        # input as-is (the dual-resolution dataloader feeds us aspect-preserved
        # multiples-of-16 by construction; resize only on mis-config / fallback).
        if (H % self.patch_size) != 0 or (W % self.patch_size) != 0:
            target = self.process_res
            x = x.view(B * V, Cin, H, W)
            x = F.interpolate(x, size=(target, target), mode="bilinear", align_corners=False)
            H, W = target, target
            x = x.view(B, V, Cin, H, W)

        # 3. VGGT aggregator expects [B, S, 3, H, W]. We already have that layout.
        aggregator = self.model.aggregator
        cached_list, patch_start = aggregator(x)

        h_grid = H // self.patch_size
        w_grid = W // self.patch_size
        expected = h_grid * w_grid
        C = self.out_channels   # 2 * embed_dim = 2048

        def _patches_to_chw(tokens: torch.Tensor) -> torch.Tensor:
            """[B, V, num_tokens, C] → [B, V, C, h_grid, w_grid] after slicing prefix."""
            t = tokens[:, :, patch_start:, :]
            if t.shape[2] != expected:
                raise RuntimeError(
                    f"unexpected num_patches: got {t.shape[2]}, expected {expected} "
                    f"({h_grid}x{w_grid} for {H}x{W} / patch_size={self.patch_size})"
                )
            return t.reshape(B, V, h_grid, w_grid, C).permute(0, 1, 4, 2, 3).contiguous()

        if self._is_multi:
            # Pick all non-None cached entries in cached_layer_indices order
            # (sorted ascending by block index — matches self._resolved_layers).
            taps = [cached_list[i] for i in self._resolved_layers]
            if any(t is None for t in taps):
                missing = [i for i, t in zip(self._resolved_layers, taps) if t is None]
                raise RuntimeError(
                    f"VGGT aggregator did not cache layers {missing}; "
                    f"requested {self._resolved_layers}"
                )
            # Per-layer LayerNorm in token space (channel-last is already there),
            # then convert to [B, V, C, h, w].
            ln_dtype = self._fuse_norms[0].weight.dtype
            feats_chw = []
            for i, tok in enumerate(taps):
                tok_norm = self._fuse_norms[i](tok.to(ln_dtype))      # [B, V, N, C]
                feats_chw.append(_patches_to_chw(tok_norm))            # [B, V, C, h, w]
            # Channel-concat: 4 × [B, V, C, h, w] → [B, V, 4C, h, w]
            x_cat = torch.cat(feats_chw, dim=2)
            conv_dtype = self._fuse_conv.weight.dtype
            x_cat = x_cat.reshape(B * V, 4 * C, h_grid, w_grid).to(conv_dtype)
            feat = self._fuse_conv(x_cat)                              # [B*V, C, h, w]
            feat = feat.reshape(B, V, C, h_grid, w_grid)
        else:
            # Last-layer only: VGGT-Omega's default cached_layer_indices is
            # (4, 11, 17, 23) so cached_list[-1] = cached_list[23] (last block).
            tok = cached_list[-1]
            if tok is None:
                raise RuntimeError(
                    "VGGT aggregator did not cache the final layer; check cached_layer_indices."
                )
            feat = _patches_to_chw(tok)                                 # [B, V, C, h, w]

        # 5. Resize spatial dims to feature_size (matches DA3 path's 64×64 default).
        feat = feat.reshape(B * V, C, h_grid, w_grid)
        feat = F.interpolate(
            feat.float(), size=(self.feature_size, self.feature_size),
            mode="bilinear", align_corners=False,
        )
        feat = feat.view(B, V, C, self.feature_size, self.feature_size)
        return feat.contiguous()

    def forward(self, image_input: torch.Tensor,
                extrinsics: torch.Tensor | None = None,
                intrinsics: torch.Tensor | None = None) -> torch.Tensor:
        """
        image_input: [B, V, 3, H, W] ImageNet-normalized RGB (same as DA3 path).
        extrinsics, intrinsics: ACCEPTED for signature parity with the DA3 path
            but IGNORED. VGGT-Omega predicts camera pose internally — XVLA_POSED_DA3
            becomes a no-op when geometry_backbone='vggt'.
        Returns: [B, V, 2*embed_dim, feature_size, feature_size].
        """
        if image_input.dim() != 5:
            raise ValueError(f"image_input must be [B,V,3,H,W], got {tuple(image_input.shape)}")
        if extrinsics is not None or intrinsics is not None:
            # Warn once per process — too noisy otherwise.
            if not getattr(self, "_warned_pose_ignored", False):
                logger.info(
                    "[vggt_inline] extrinsics/intrinsics provided but VGGT-Omega does not "
                    "consume them; ignoring. XVLA_POSED_DA3 env flag is a no-op for VGGT."
                )
                self._warned_pose_ignored = True

        device = image_input.device
        out_dtype = image_input.dtype
        ctx = (
            torch.autocast(device.type, dtype=torch.bfloat16)
            if self.use_bf16 and device.type == "cuda"
            else nullcontext()
        )
        with ctx:
            out = self._batched_call(image_input)
        return out.to(device=device, dtype=out_dtype)
