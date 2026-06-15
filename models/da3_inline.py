# ------------------------------------------------------------------------------
# DA3 inline encoder for X-VLA — runs DA3 inside the training forward pass,
# parallel to Florence2. Spatialva-style efficiency tricks:
#   - bf16 autocast around the DA3 forward (~2x faster on H100, negligible
#     quality loss).
#   - Optional gradient checkpointing (memory savings).
#   - Frozen by default (`freeze=True` => requires_grad=False on all DA3 params)
#     so no backward through DA3. If a future PyTorch / DDP / grad-ckpt issue
#     forces partial-freeze, swap to the spatialva "train()+LR=0" pattern in
#     the training loop instead.
#
# Expected use: caller passes `image_input` from the X-VLA dataloader, which
# is already ImageNet-normalized at 224x224. This module denormalizes back to
# uint8 RGB before handing each sample's V views to DA3's multi-view inference,
# producing a [B, V, C, 64, 64] dense latent that drops straight into the
# existing `SegmentedDA3GeometryConditioner` 5-D branch.
# ------------------------------------------------------------------------------
from __future__ import annotations
import logging
from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ImageNet mean/std — match what `datasets/dataset.py:image_aug` applies.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# DA3 DPT depth-head taps (per Depth-Anything-3/src/depth_anything_3/configs/da3-*.yaml).
# Used by feature_layer="multi" when self.model.model.backbone.out_layers isn't readable.
_DA3_OUT_LAYERS = {
    "small": [5, 7, 9, 11],
    "base":  [5, 7, 9, 11],
    "large": [11, 15, 19, 23],
    "giant": [19, 27, 33, 39],
}
_DA3_EMBED_DIM = {
    "small": 384,
    "base":  768,
    "large": 1024,
    "giant": 1536,
}


def _detect_da3_variant(model_name: str) -> str:
    n = model_name.lower()
    for k in ("giant", "large", "base", "small"):
        if k in n:
            return k
    return "base"


class DA3InlineEncoder(nn.Module):
    """
    Inline DA3 multi-view encoder.

    Args:
        model_name: HF id, e.g. "depth-anything/DA3-BASE" or "depth-anything/DA3-SMALL".
        use_bf16: wrap the DA3 forward in `torch.autocast(bfloat16)`.
        gradient_checkpointing: try to enable on the DA3 model (best-effort).
        freeze: if True, set `requires_grad_=False` on all DA3 params (no
            backward through DA3).
        feature_size: spatial size of the returned latent (DA3's own output is
            resized to this; default 64 matches the rest of the geometry path).
        feature_layer: DINOv2 block index to export. "last" picks the deepest.
        process_res: DA3's internal processing resolution.
    """

    def __init__(
        self,
        model_name: str = "depth-anything/DA3-BASE",
        use_bf16: bool = True,
        gradient_checkpointing: bool = True,
        freeze: bool = True,
        feature_size: int = 64,
        feature_layer: str | int = "last",
        process_res: int = 504,
        multi_fusion_init: str = "last_only",
    ) -> None:
        super().__init__()
        try:
            from depth_anything_3.api import DepthAnything3
        except Exception as exc:
            raise ImportError(
                "depth_anything_3 not installed; install third_party/Depth-Anything-3"
            ) from exc
        logger.info("[da3_inline] loading %s", model_name)
        self.model = DepthAnything3.from_pretrained(model_name)
        self.model_name = model_name
        self.use_bf16 = bool(use_bf16)
        self.feature_size = int(feature_size)
        self.feature_layer = feature_layer
        self.process_res = int(process_res)

        # Gradient checkpointing on DA3's ViT (DINOv2) blocks. DA3 has no
        # gradient_checkpointing_enable() method, so we wrap each block.forward
        # in torch.utils.checkpoint.checkpoint manually. Trades ~25% backward
        # compute for ~50%+ DA3 activation memory; required at bs>=12 with
        # unfrozen DA3 on 80GB H100. Best-effort path discovery — silently no-op
        # if the backbone structure changes in a future DA3 release.
        # Skip when freeze=True: backward never traverses these blocks (no grad
        # required), so the checkpoint wrapper just adds saved_tensor_hooks +
        # recompute overhead with zero memory benefit. py-spy attributed ~19% of
        # DA3 wall samples to _ckpt_fwd in the running training (2026-06-11 audit).
        if gradient_checkpointing and not freeze:
            try:
                import torch.utils.checkpoint as _cp
                blocks = None
                # known path from safetensors layout: model.model.backbone.pretrained.blocks
                for path in (("model", "backbone", "pretrained", "blocks"),
                             ("model", "blocks"),
                             ("backbone", "blocks")):
                    cur = self.model
                    ok = True
                    for p in path:
                        cur = getattr(cur, p, None)
                        if cur is None: ok = False; break
                    if ok and hasattr(cur, "__iter__"):
                        blocks = cur; break
                if blocks is not None:
                    n_wrapped = 0
                    for blk in blocks:
                        _orig = blk.forward
                        def _ckpt_fwd(*a, _o=_orig, **kw):
                            return _cp.checkpoint(_o, *a, use_reentrant=False, **kw)
                        blk.forward = _ckpt_fwd
                        n_wrapped += 1
                    logger.info("[da3_inline] gradient_checkpointing ON: wrapped %d ViT blocks", n_wrapped)
                else:
                    logger.warning("[da3_inline] could not locate DA3 ViT blocks; checkpointing OFF")
            except Exception as e:
                logger.warning("[da3_inline] gradient_checkpointing setup failed: %s", e)

        # CRITICAL: scope the freeze to self.model (DA3 backbone) only — never
        # to self.parameters(). The multi-layer fusion adapter built below is a
        # FRESH module that must stay trainable; freezing self.* would zero its
        # requires_grad too, silently killing the new mode.
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
            logger.info("[da3_inline] backbone frozen (no backward through DA3)")

        # Resolve feature-layer spec: single int (legacy) | "last" | "multi" | "dpt_fused".
        # _resolved_layer stays for back-compat; _resolved_layers is the
        # canonical list used at inference time.
        self._fuse_init_mode = str(multi_fusion_init)
        _fl_str = feature_layer.lower() if isinstance(feature_layer, str) else None
        self._is_multi = (_fl_str == "multi")
        self._is_dpt_fused = (_fl_str == "dpt_fused")
        if self._is_multi:
            self._resolved_layers, self._embed_dim = self._resolve_multi_layers()
            self._resolved_layer = self._resolved_layers[-1]
            # Per-layer LayerNorm → channel-concat → 1×1 Conv(4C→C).
            # Output [B,V,C,h,w] is shape-identical to single-layer mode, so the
            # downstream geometry projector / perceiver / cross-attn don't change.
            C = self._embed_dim
            self._fuse_norms = nn.ModuleList([nn.LayerNorm(C) for _ in range(4)])
            self._fuse_conv = nn.Conv2d(4 * C, C, kernel_size=1, bias=True)
            self._init_fuse_conv(self._fuse_init_mode)
            logger.info(
                "[da3_inline] multi-layer mode: layers=%s embed_dim=%d fuse_init=%s",
                self._resolved_layers, self._embed_dim, self._fuse_init_mode,
            )
        elif self._is_dpt_fused:
            # Tap the latent that feeds DA3's depth + ray heads — i.e. the input
            # to head.scratch.output_conv2. This is DA3's own pretrained DPT
            # pyramid output (multi-layer fused, spatially recovered, ReLU-refined).
            # Channel dim is the DPT `features` config (BASE=128, LARGE=256,
            # SMALL=64, GIANT=256). No new trainable params — head is frozen
            # along with the backbone.
            self._resolved_layers = []   # no aux taps needed
            self._resolved_layer = None
            self._embed_dim = None
            self._dpt_capture: dict = {}
            self._dpt_hook = None         # registered lazily after weight load
            self._register_dpt_hook()
            logger.info("[da3_inline] dpt_fused mode: tapping head.scratch.output_conv2 input")
        else:
            self._resolved_layer = self._resolve_layer()
            self._resolved_layers = [self._resolved_layer]
            self._embed_dim = None

        # denorm buffers (Florence2/DINOv2 use the same ImageNet stats)
        self.register_buffer(
            "_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    def _resolve_layer(self) -> int:
        if isinstance(self.feature_layer, int):
            return int(self.feature_layer)
        # "last" -> infer from DA3's DINOv2 depth
        nb = None
        try:
            backbone = getattr(self.model, "model", self.model)
            blocks = getattr(backbone, "blocks", None)
            if blocks is not None:
                nb = len(blocks)
        except Exception:
            pass
        return int((nb - 1) if nb else 11)  # ViT-B/S default depth = 12

    def _resolve_multi_layers(self) -> tuple[list[int], int]:
        """
        Resolve the 4 DPT tap indices + embed_dim for feature_layer="multi".

        Prefer runtime introspection of the DA3 backbone over the static table:
          - self.model.model.backbone.out_layers      → list[int]
          - self.model.model.backbone.pretrained.embed_dim → int
        Fall back to _DA3_OUT_LAYERS / _DA3_EMBED_DIM keyed by variant.
        """
        layers = None
        embed_dim = None
        try:
            bb = self.model.model.backbone
            ol = getattr(bb, "out_layers", None)
            if ol is not None:
                layers = [int(i) for i in ol]
            pre = getattr(bb, "pretrained", None)
            if pre is not None:
                ed = getattr(pre, "embed_dim", None)
                if ed is not None:
                    embed_dim = int(ed)
        except Exception:
            pass
        if layers is None or embed_dim is None:
            variant = _detect_da3_variant(self.model_name)
            if layers is None:
                layers = list(_DA3_OUT_LAYERS[variant])
            if embed_dim is None:
                embed_dim = int(_DA3_EMBED_DIM[variant])
        if len(layers) != 4:
            raise ValueError(
                f"feature_layer='multi' expects 4 DPT taps, got {layers}"
            )
        return layers, embed_dim

    def _init_fuse_conv(self, mode: str) -> None:
        """
        Initialize the 1×1 fusion Conv2d(4C→C) so step 0 is byte-identical to
        single-layer behavior (mode='last_only') or evenly-averaged across
        layers (mode='uniform'). Bias is always zero.
        """
        C = self._embed_dim
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
        # LayerNorms: standard init (weight=1, bias=0). Done by nn.LayerNorm
        # constructor, but re-asserted in reload_pretrained_weights().

    def _register_dpt_hook(self) -> None:
        """
        Register a forward hook on DA3's depth-head output Conv to capture its
        INPUT, which is the fully-fused, spatially-recovered geometry latent
        used by both the depth head AND (in DualDPT) the ray head.

        The hook walks the DA3 model hierarchy:
            self.model.model            # DepthAnything3Net
                .head                   # DPT or DualDPT
                .scratch.output_conv2   # final depth-prediction Conv
        and grabs `input[0]` on each forward (overwriting any previous capture
        so we always see the most recent forward's latent).
        """
        try:
            head = self.model.model.head
            target = head.scratch.output_conv2
        except AttributeError as exc:
            raise RuntimeError(
                "dpt_fused mode requires DA3 backbone with head.scratch.output_conv2"
            ) from exc

        if self._dpt_hook is not None:
            self._dpt_hook.remove()
            self._dpt_hook = None

        def _capture(_module, _input, _output):
            if isinstance(_input, tuple) and len(_input) > 0:
                self._dpt_capture["fused_main"] = _input[0]

        self._dpt_hook = target.register_forward_hook(_capture)
        logger.info("[da3_inline] dpt_fused hook attached to head.scratch.output_conv2")

    @torch.no_grad()
    def _denorm_to_uint8(self, normed: torch.Tensor) -> np.ndarray:
        """[V, 3, H, W] normalized float -> [V, H, W, 3] uint8 ndarray."""
        x = normed.to(self._mean.device) * self._std + self._mean       # [V,3,H,W] in [0,1]
        x = (x.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        x = x.permute(0, 2, 3, 1).contiguous()                          # [V,H,W,3]
        return x.cpu().numpy()

    def _fixed_size(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x, size=(self.feature_size, self.feature_size),
            mode="bilinear", align_corners=False,
        )

    def _batched_call(self, images_normed: torch.Tensor,
                      extrinsics: torch.Tensor | None = None,
                      intrinsics: torch.Tensor | None = None) -> torch.Tensor:
        """
        Batched DA3 forward — ONE call for the whole batch [B,V,3,H,W].

        Calls DA3's underlying nn.Module directly (``self.model.model``),
        bypassing the public ``forward`` wrapper that has
        ``@torch.inference_mode()`` (which breaks autograd in the geometry
        path even though we freeze DA3 — the geometry conditioner still
        needs gradients on its own params w.r.t. DA3's output).

        Pipeline (all on GPU, no python loops):
          1. Denormalize ImageNet-stat input back to [0, 1]
          2. Resize each view to process_res × process_res (square; DA3
             handles aspect ratio internally via its input_processor in
             the high-level API, but for direct forward we feed square)
          3. bf16 autocast → ``self.model.model(image=[B,V,3,P,P], ...)``
          4. Extract feat layer from the returned aux dict
          5. Reshape to [B, V, C, h, w] and resize to feature_size

        This replaces the per-sample loop in the old code — ~B× speedup.
        """
        B, V, C, H, W = images_normed.shape
        # 1. Denormalize ImageNet stats -> [0, 1] (in autograd graph, cheap)
        x = images_normed * self._std + self._mean
        x = x.clamp_(0.0, 1.0)
        # 2. Resize. Default path: stretch to process_res × process_res.
        #    Native path (XVLA_DA3_NATIVE_INPUT=1): skip resize when input H, W
        #    are already multiples of DA3's patch_size (14). DA3's DinoV2 backbone
        #    accepts arbitrary rectangular input via interpolate_pos_encoding;
        #    no need to upscale 224×224 (16×14) to 504×504 (36×14) — that just
        #    invents pixels with bilinear and bloats the geometry feature map.
        import os as _os
        _native = _os.environ.get("XVLA_DA3_NATIVE_INPUT", "0") == "1"
        x = x.view(B * V, C, H, W)
        if _native and (H % 14 == 0) and (W % 14 == 0):
            target_H, target_W = H, W
        else:
            target_H, target_W = self.process_res, self.process_res
            if (H, W) != (target_H, target_W):
                x = F.interpolate(
                    x, size=(target_H, target_W),
                    mode="bilinear", align_corners=False,
                )
        x = x.view(B, V, C, target_H, target_W)

        # 3. Call DA3's INNER nn.Module directly (not DepthAnything3.forward,
        # which has @torch.inference_mode and disables autograd downstream).
        inner = self.model.model  # the nn.Module underneath DepthAnything3
        # Match DA3.forward()'s call signature (positional args):
        #   (image, extrinsics, intrinsics, export_feat_layers, infer_gs,
        #    use_ray_pose, ref_view_strategy)
        # Posed multi-view: when extrinsics + intrinsics are provided, DA3 uses
        # the camera encoder (`cam_enc`) to fuse pose tokens with backbone
        # features → geometrically-consistent depth without DA3 having to
        # estimate poses from appearance. Falls back to unposed when None.
        # Intrinsics rescaled to target_H/target_W (we resize images from
        # (H, W) to (target_H, target_W), so K scales by those ratios).
        if intrinsics is not None:
            sx = target_H / float(H)   # height scale
            sy = target_W / float(W)   # width scale
            K = intrinsics.clone().to(device=x.device, dtype=torch.float32)
            K[..., 0, 0] *= sy   # fx column scales with width
            K[..., 1, 1] *= sx   # fy column scales with height
            K[..., 0, 2] *= sy
            K[..., 1, 2] *= sx
        else:
            K = None
        E = extrinsics.to(device=x.device, dtype=torch.float32) if extrinsics is not None else None
        out = inner(
            x, E, K,
            list(self._resolved_layers),  # [last] in single-layer mode, 4-tap list in multi mode
            False,                        # infer_gs
            False,                        # use_ray_pose
            "saddle_balanced",            # ref_view_strategy
        )

        # 4. Extract feat layer(s) — output may be dict or dataclass with .aux
        if isinstance(out, dict):
            aux = out.get("aux") or {}
        else:
            aux = getattr(out, "aux", None) or {}

        def _to_bvchw(feat: torch.Tensor) -> torch.Tensor:
            """Normalize an aux entry to [B, V, C, h, w]."""
            if feat.dim() == 6:                # [B, V, s, h, w, C]
                feat = feat.mean(dim=2)
            if feat.dim() == 5:                # [B, V, h, w, C]
                return feat.permute(0, 1, 4, 2, 3)
            if feat.dim() == 4:                # [B, V, N, C] tokens
                B_, V_, N, C_ = feat.shape
                s = max(1, int(round(N ** 0.5)))
                return feat[:, :, : s * s].permute(0, 1, 3, 2).reshape(B_, V_, C_, s, s)
            raise RuntimeError(f"unexpected DA3 aux tensor rank: {feat.dim()}")

        if self._is_dpt_fused:
            # `fused_main` was captured by the forward hook on
            # head.scratch.output_conv2. Its shape is [B*V, features, h_out, w_out]
            # where features ∈ {64,128,256} per DA3-{SMALL,BASE,LARGE/GIANT}.
            fused = self._dpt_capture.get("fused_main")
            if fused is None:
                raise RuntimeError(
                    "dpt_fused mode: head.scratch.output_conv2 hook did not fire — "
                    "DA3 depth head was not executed. Check that the head is "
                    "constructed in the DA3 backbone."
                )
            # [B*V, C, h_out, w_out] → [B, V, C, h_out, w_out]
            feat = fused.view(B, V, *fused.shape[1:])
            self._dpt_capture.clear()
        elif self._is_multi:
            feats = []
            for li in self._resolved_layers:
                key = f"feat_layer_{li}"
                fi = aux.get(key)
                if fi is None:
                    raise RuntimeError(
                        f"DA3 inner forward returned no aux['{key}']; "
                        f"available keys: {list(aux.keys())[:8]}"
                    )
                feats.append(_to_bvchw(fi))   # [B, V, C, h, w]
            # Per-layer LayerNorm (channel-last for LN, then back to channel-first).
            # Cast inputs to LN param dtype because DA3 forward runs in bf16 autocast
            # while LN params stay fp32.
            ln_dtype = self._fuse_norms[0].weight.dtype
            normed = []
            for i, fi in enumerate(feats):
                # [B, V, C, h, w] → [B, V, h, w, C]
                xi = fi.permute(0, 1, 3, 4, 2).contiguous().to(ln_dtype)
                xi = self._fuse_norms[i](xi)
                normed.append(xi.permute(0, 1, 4, 2, 3))  # back to [B, V, C, h, w]
            # Concat on channel: 4 × [B, V, C, h, w] → [B, V, 4C, h, w]
            x_cat = torch.cat(normed, dim=2)
            B_, V_, C4_, h_, w_ = x_cat.shape
            # Conv expects [N, C, H, W]; cast to conv param dtype.
            conv_dtype = self._fuse_conv.weight.dtype
            x_cat = x_cat.reshape(B_ * V_, C4_, h_, w_).to(conv_dtype)
            feat = self._fuse_conv(x_cat)             # [B*V, C, h, w]
            feat = feat.reshape(B_, V_, self._embed_dim, h_, w_)
        else:
            key = f"feat_layer_{self._resolved_layer}"
            feat = aux.get(key)
            if feat is None:
                raise RuntimeError(
                    f"DA3 inner forward returned no aux['{key}']; "
                    f"available keys: {list(aux.keys())[:5]}"
                )
            feat = _to_bvchw(feat)

        # 5. Resize spatial dims to feature_size
        feat = feat.reshape(B * V, *feat.shape[2:])
        feat = F.interpolate(
            feat.float(), size=(self.feature_size, self.feature_size),
            mode="bilinear", align_corners=False,
        )
        feat = feat.view(B, V, feat.shape[1], self.feature_size, self.feature_size)
        return feat.contiguous()

    def reload_pretrained_weights(self) -> None:
        """
        Reload DA3 pretrained weights AFTER ``XVLA.from_pretrained`` runs.

        Why this method exists: HF's ``from_pretrained`` treats every key
        absent from the X-VLA checkpoint as a "missing" param and calls
        ``_init_weights`` on it. That overwrites the DA3 weights we loaded
        inside ``__init__``, leaving da3_inline at *random init* — which
        produces NaN through a 12-block ViT and made V1 loss=NaN from iter 0.
        Call this once, right after ``XVLA.from_pretrained(...)``, to restore
        the actual DA3 weights. Re-applies the freeze too.
        """
        try:
            from depth_anything_3.api import DepthAnything3
        except Exception as exc:
            raise ImportError("depth_anything_3 not installed") from exc
        logger.info("[da3_inline] reloading pretrained %s (HF re-init wiped it)", self.model_name)
        src = DepthAnything3.from_pretrained(self.model_name)
        missing, unexpected = self.model.load_state_dict(src.state_dict(), strict=False)
        del src
        if unexpected:
            logger.warning("[da3_inline] %d unexpected keys (first 3): %s",
                           len(unexpected), unexpected[:3])
        if missing:
            logger.warning("[da3_inline] %d missing keys (first 3): %s",
                           len(missing), missing[:3])
        # Re-apply frozen state in case finetune policy hasn't run yet.
        # Scope to self.model only so the multi-layer fusion adapter (LayerNorms
        # + Conv2d) remains trainable.
        for p in self.model.parameters():
            p.requires_grad_(False)
        # HF's recursive _init_weights also wiped the fusion adapter (it has no
        # corresponding key in the X-VLA checkpoint). Re-establish identity-init.
        if getattr(self, "_is_multi", False):
            for ln in self._fuse_norms:
                with torch.no_grad():
                    ln.weight.fill_(1.0)
                    ln.bias.zero_()
            self._init_fuse_conv(self._fuse_init_mode)
            logger.info(
                "[da3_inline] reload done, backbone frozen; fusion adapter re-init mode=%s",
                self._fuse_init_mode,
            )
        elif getattr(self, "_is_dpt_fused", False):
            # The DA3 head module (and its scratch.output_conv2) may have been
            # rebuilt by HF re-init; re-attach the capture hook to whatever the
            # current target object is.
            self._dpt_capture = {}
            self._register_dpt_hook()
            logger.info("[da3_inline] reload done, DA3 frozen; dpt_fused hook re-attached")
        else:
            logger.info("[da3_inline] reload done, DA3 frozen")

    def forward(self, image_input: torch.Tensor,
                extrinsics: torch.Tensor | None = None,
                intrinsics: torch.Tensor | None = None) -> torch.Tensor:
        """
        image_input: [B, V, 3, H, W] — the same tensor X-VLA passes to Florence2
                     (ImageNet-normalized, typically 224x224).
        extrinsics:  Optional [B, V, 4, 4] world-to-camera (OpenCV). When given,
                     DA3 uses them via cam_enc → posed multi-view mode.
        intrinsics:  Optional [B, V, 3, 3] in *original image resolution*. We
                     rescale to process_res inside `_batched_call`.
        Returns:     [B, V, C, feature_size, feature_size] dense DA3 latent.

        Spatialva-style: one batched call into DA3's inner module instead of B
        per-sample calls. Wrapped in bf16 autocast.
        """
        if image_input.dim() != 5:
            raise ValueError(f"image_input must be [B,V,3,H,W], got {tuple(image_input.shape)}")
        device = image_input.device
        out_dtype = image_input.dtype
        ctx = (
            torch.autocast(device.type, dtype=torch.bfloat16)
            if self.use_bf16 and device.type == "cuda"
            else nullcontext()
        )
        with ctx:
            out = self._batched_call(image_input, extrinsics, intrinsics)   # [B, V, C, h, w]
        return out.to(device=device, dtype=out_dtype)
