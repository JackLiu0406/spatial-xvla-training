# SPDX: same as DA3-XVLA repo
"""
DA3-Large frozen feature extractor for GeoStack-XVLA v2B.

This wrapper around `depth_anything_3.api.DepthAnything3`:
  * Loads DA3-Large-1.1 and freezes ALL parameters.
  * Receives ImageNet-normalized RGB at any input dimensions (typically the
    already-square 224×224 that the X-VLA processor produces for Florence).
  * Bilinearly resizes to the configured (da3_input_h, da3_input_w) — typically
    a 4:3-aspect rectangle like 252×336 that recovers the original aspect after
    Florence's stretch. NO PADDING — both dims must be multiples of patch_size.
  * Returns a list of N feature levels at the configured DPT taps (default
    out_layers=[11, 19, 23] for DA3-Large = shallow, mid, deep), each as
    [B, C, h_grid, w_grid] with (h_grid, w_grid) = (input_h/14, input_w/14).
  * Returns the DA3 ray-head output (3D direction unit vector per pixel),
    captured via a forward hook on `head.scratch.output_conv2_aux[-1]` and
    resampled to (h_grid, w_grid).

Spatial-bias coordinate alignment:
  * Florence sees a stretched 224×224 → its token-grid (e.g. 7×7) maps linearly
    back to the ORIGINAL image's normalized [0,1]² coords (no aspect correction
    needed; the stretch is bijective per axis).
  * DA3 sees the same content but at the aspect-correct dims (e.g. 252×336)
    → its token-grid (e.g. 18×24) ALSO maps linearly to the ORIGINAL image's
    normalized [0,1]² coords.
  * Both grids therefore live in the same normalized coord space, and the
    spatial-distance bias `-λ · ||pos_vlm - pos_da3||²` is well-defined.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# DA3-Large fallback (used if hub model has no `backbone.out_layers` attribute).
_DA3_LARGE_OUT_LAYERS = (11, 15, 19, 23)
_DA3_LARGE_EMBED_DIM = 1024


class DA3LargeForGeoStack(nn.Module):
    """Frozen DA3-Large feature + ray-head extractor for GeoStack.

    Usage:
        da3 = DA3LargeForGeoStack(
            model_name="depth-anything/DA3-Large-1.1",
            out_layers=(11, 19, 23),
            da3_input_h=252, da3_input_w=336,   # aspect-correct 4:3 (~1.05× of 240×320)
        )
        # pixel_values: [B, 3, H_in, W_in] in ImageNet-normalized space
        # (typically H_in=W_in=224 — the X-VLA processor's stretched square).
        out = da3(pixel_values)
        # out["feats"]:        list[N] of [B, C, h_grid, w_grid]
        # out["ray"]:          [B, 3, h_grid, w_grid]  (direction-only, unit-normalized)
        # out["h_grid"]:       int
        # out["w_grid"]:       int
    """

    def __init__(
        self,
        model_name: str = "depth-anything/DA3-Large-1.1",
        out_layers: Tuple[int, ...] = (11, 19, 23),
        da3_input_h: int = 252,
        da3_input_w: int = 336,
        patch_size: int = 14,
        use_bf16: bool = True,
    ):
        super().__init__()
        try:
            from depth_anything_3.api import DepthAnything3
        except Exception as exc:
            raise ImportError(
                "depth_anything_3 not installed; install third_party/Depth-Anything-3"
            ) from exc

        self.model_name = str(model_name)
        self.out_layers = tuple(int(i) for i in out_layers)
        self.patch_size = int(patch_size)
        self.use_bf16 = bool(use_bf16)
        self.da3_input_h = int(da3_input_h)
        self.da3_input_w = int(da3_input_w)
        if self.da3_input_h % self.patch_size != 0 or self.da3_input_w % self.patch_size != 0:
            raise ValueError(
                f"DA3 input dims must be multiples of patch_size={self.patch_size}; "
                f"got ({self.da3_input_h}, {self.da3_input_w})"
            )
        self.h_grid = self.da3_input_h // self.patch_size      # e.g. 252/14 = 18
        self.w_grid = self.da3_input_w // self.patch_size      # e.g. 336/14 = 24

        # Load DA3 + freeze
        self.model = DepthAnything3.from_pretrained(self.model_name)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

        # Channel dim: introspect the loaded backbone rather than assuming
        # DA3-Large. DA3-Large=1024, DA3NESTED-GIANT-LARGE-1.1=1536, etc.
        # Downstream perceiver/projection is sized from self.embed_dim, so this
        # MUST match the real feature width or K/V dims mismatch.
        self.embed_dim = self._infer_embed_dim()

        # Ray-head hook capture (after DA3's DualDPT output_conv2_aux).
        self._ray_capture: Dict[str, torch.Tensor] = {}
        self._register_ray_hook()

        # v4.1 OPT-IN: capture depth output (main head) via a second hook on
        # head.scratch.output_conv2. Off by default — only enabled when the
        # depth-distillation aux head is configured.
        self._depth_capture: Dict[str, torch.Tensor] = {}
        self._depth_hook = None
        self._capture_depth = False

    # ---------- backbone feature-dim introspection (GIANT-safe) ----------

    def _infer_embed_dim(self) -> int:
        """Read the DINOv2 backbone hidden width from the loaded DA3 model.

        Walks common attribute paths to a `.embed_dim`/`.num_features`; falls
        back to inspecting a transformer block's LayerNorm, then to the
        DA3-Large constant. DA3-Large=1024, DA3NESTED-GIANT-LARGE-1.1=1536.
        """
        obj = self.model
        for path in (("model", "da3", "backbone", "pretrained"),
                     ("da3", "backbone", "pretrained"),
                     ("backbone", "pretrained"),
                     ("backbone",)):
            cur = obj
            ok = True
            for a in path:
                if hasattr(cur, a):
                    cur = getattr(cur, a)
                else:
                    ok = False
                    break
            if ok:
                for a in ("embed_dim", "num_features", "hidden_size", "n_embd"):
                    if hasattr(cur, a):
                        try:
                            return int(getattr(cur, a))
                        except Exception:
                            pass
        # last-ditch: infer from a block's norm weight length
        for n, p in self.model.named_parameters():
            if "blocks.0." in n and n.endswith("norm1.weight") and p.ndim == 1:
                return int(p.shape[0])
        logger.warning("[da3_for_geostack] could not introspect embed_dim; "
                       "falling back to DA3-Large=%d", int(_DA3_LARGE_EMBED_DIM))
        return int(_DA3_LARGE_EMBED_DIM)

    # ---------- post-load reload of DA3 weights ----------

    def reload_pretrained_weights(self) -> None:
        """Reload DA3-Large pretrained weights AFTER ``XVLA.from_pretrained``.

        HF's ``from_pretrained`` treats every key absent from the loaded
        checkpoint as a "missing" param and overwrites it via ``_init_weights``.
        That zeroes/random-inits the DA3-Large weights we loaded inside
        ``__init__``, producing garbage geometry features as soon as alpha > 0.

        Call this once right after ``XVLA.from_pretrained(...)`` to restore the
        actual DA3 weights. Same pattern as ``DA3InlineEncoder.reload_pretrained_weights``.
        """
        try:
            from depth_anything_3.api import DepthAnything3
        except Exception as exc:
            raise ImportError("depth_anything_3 not installed") from exc
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        fresh = DepthAnything3.from_pretrained(self.model_name)
        self.model.load_state_dict(fresh.state_dict(), strict=True)
        self.model.to(device=device, dtype=dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        # Re-attach the ray hook defensively (load_state_dict copies weights
        # into existing modules so the hook target is preserved, but if any
        # downstream change re-creates head modules this re-attach saves us).
        try:
            if hasattr(self, "_ray_hook"):
                self._ray_hook.remove()
        except Exception:
            pass
        self._register_ray_hook()

    # ---------- head-path resolution (DA3-Large vs NESTED) ----------

    def _find_head_scratch(self):
        """Return the DPT ``head.scratch`` module across DA3 variants.

        DA3-Large: ``self.model.model.head.scratch``.
        DA3NESTED-GIANT-LARGE: the encoder is wrapped one level deeper as
        ``self.model.model.da3.head.scratch`` (net children = da3 / da3_metric).
        """
        for path in (("model", "head", "scratch"),
                     ("model", "da3", "head", "scratch"),
                     ("da3", "head", "scratch"),
                     ("head", "scratch")):
            cur = self.model
            ok = True
            for a in path:
                if hasattr(cur, a):
                    cur = getattr(cur, a)
                else:
                    ok = False
                    break
            if ok:
                return cur
        return None

    # ---------- ray-head hook ----------

    def _register_ray_hook(self) -> None:
        """Capture the ray-head pre-activation output via a forward hook on the
        final aux conv in DA3's DualDPT. Output shape: [B*V, 7, H_ray, W_ray]
        where channels are [dir_x, dir_y, dir_z, ori_x, ori_y, ori_z, conf].

        DA3's ``head.scratch.output_conv2_aux`` may be an nn.Sequential or a
        single Conv2d — hook whichever leaf module is at the tail.
        """
        scratch = self._find_head_scratch()
        if scratch is None or not hasattr(scratch, "output_conv2_aux"):
            raise RuntimeError(
                "DA3LargeForGeoStack: could not locate head.scratch.output_conv2_aux "
                "(this DA3 variant may not have a DualDPT head)"
            )
        aux = scratch.output_conv2_aux
        # DA3's DualDPT defines output_conv2_aux as an nn.ModuleList of length
        # aux_pyramid_levels (default 4). The forward call site uses
        # `scratch.output_conv2_aux[-1](last_aux)` (dualdpt.py:255) — i.e., only
        # the LAST module is actually called. Hook it.
        if isinstance(aux, (nn.Sequential, nn.ModuleList)):
            if len(aux) == 0:
                raise RuntimeError("DA3LargeForGeoStack: output_conv2_aux is empty")
            target = aux[-1]
        else:
            target = aux

        def _hook(module, inputs, output):
            self._ray_capture["ray_raw"] = output

        self._ray_hook = target.register_forward_hook(_hook)

    # ---------- depth-head hook (v4.1 opt-in) ----------

    def enable_depth_capture(self, enabled: bool = True) -> None:
        """Turn on capture of DA3's main depth output (from output_conv2).
        Idempotent — safe to call multiple times.
        """
        if enabled and self._depth_hook is None:
            scratch = self._find_head_scratch()
            if scratch is None or not hasattr(scratch, "output_conv2"):
                raise RuntimeError(
                    "DA3LargeForGeoStack: could not locate head.scratch.output_conv2 for depth hook"
                )
            target = scratch.output_conv2
            def _hook(module, inputs, output):
                self._depth_capture["depth_raw"] = output
            self._depth_hook = target.register_forward_hook(_hook)
        elif not enabled and self._depth_hook is not None:
            self._depth_hook.remove()
            self._depth_hook = None
        self._capture_depth = bool(enabled)

    # ---------- forward ----------

    @torch.no_grad()
    def _da3_forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run DA3 inner forward and return the aux dict.

        x: [B, V, 3, H, W] in ImageNet-normalized space (already resized to
        (da3_input_h, da3_input_w)).
        """
        inner = self.model.model
        out = inner(
            x, None, None,                          # no extrinsics/intrinsics
            list(self.out_layers),                  # export these feat layers
            False,                                  # infer_gs
            False,                                  # use_ray_pose
            "saddle_balanced",                       # ref_view_strategy
        )
        if isinstance(out, dict):
            aux = out.get("aux") or {}
        else:
            aux = getattr(out, "aux", None) or {}
        return aux

    @torch.no_grad()
    def _da3_forward_posed(
        self,
        x: torch.Tensor,          # [B, V, 3, H, W]
        extrinsics: torch.Tensor,  # [B, V, 4, 4] OpenCV world-to-camera
        intrinsics: torch.Tensor,  # [B, V, 3, 3] rescaled to (H, W)
    ) -> Dict[str, torch.Tensor]:
        """Joint posed multi-view DA3 forward: ALL views in one inner() call
        with camera extrinsics + intrinsics, so DA3's camera encoder fuses pose
        tokens into the backbone and returns per-view features that are
        geometrically consistent ACROSS views. Mirrors da3_inline's posed call
        (use_ray_pose=False; poses drive cam_enc, not the ray head)."""
        inner = self.model.model
        out = inner(
            x,
            extrinsics.to(device=x.device, dtype=torch.float32),
            intrinsics.to(device=x.device, dtype=torch.float32),
            list(self.out_layers),                  # export these feat layers
            False,                                  # infer_gs
            False,                                  # use_ray_pose
            "saddle_balanced",                       # ref_view_strategy
        )
        # Nested GIANT returns a top-level metric-depth map (da3_metric branch):
        # out["depth"] = [B,V,H,W], out["is_metric"]. Surface it alongside aux so
        # the spatial tokens can carry ray + DEPTH (scale-aware ray).
        if isinstance(out, dict):
            aux = out.get("aux") or {}
            depth = out.get("depth", None)
        else:
            aux = getattr(out, "aux", None) or {}
            depth = getattr(out, "depth", None)
        return aux, depth

    @staticmethod
    def _rescale_intrinsics(
        K: torch.Tensor, src_hw: Tuple[int, int], dst_hw: Tuple[int, int]
    ) -> torch.Tensor:
        """Rescale pixel intrinsics [..., 3, 3] from (src_H, src_W) to
        (dst_H, dst_W). fx, cx scale with the width ratio; fy, cy with height.
        Matches da3_inline._batched_call so the pose math sees K aligned to the
        actual tensor DA3 receives. Identity when src == dst."""
        sh, sw = int(src_hw[0]), int(src_hw[1])
        dh, dw = int(dst_hw[0]), int(dst_hw[1])
        if (sh, sw) == (dh, dw):
            return K.to(torch.float32)
        sx = float(dw) / float(sw)   # width scale
        sy = float(dh) / float(sh)   # height scale
        K = K.clone().to(torch.float32)
        K[..., 0, 0] *= sx   # fx
        K[..., 1, 1] *= sy   # fy
        K[..., 0, 2] *= sx   # cx
        K[..., 1, 2] *= sy   # cy
        return K

    def _to_bchw(self, feat: torch.Tensor, B: int) -> torch.Tensor:
        """Normalize an aux entry to [B*V, C, h, w]. V=1 for GeoStack."""
        if feat.dim() == 6:                  # [B, V, s, h, w, C]
            feat = feat.mean(dim=2)
        if feat.dim() == 5:                  # [B, V, h, w, C]
            return feat.permute(0, 1, 4, 2, 3).reshape(-1, feat.shape[4], feat.shape[2], feat.shape[3])
        if feat.dim() == 4:                  # [B, V, N, C] or [B*V, C, h, w]
            if feat.shape[0] == B:           # heuristic: [B, V, N, C]
                B_, V_, N, C_ = feat.shape
                s = max(1, int(round(N ** 0.5)))
                return feat[:, :, : s * s].permute(0, 1, 3, 2).reshape(B_ * V_, C_, s, s)
            return feat                       # already [B*V, C, h, w]
        raise RuntimeError(f"unexpected DA3 feat tensor rank: {feat.dim()}, shape={tuple(feat.shape)}")

    def forward(
        self,
        pixel_values: torch.Tensor,
        target_input_hw: Tuple[int, int] | None = None,
    ) -> Dict[str, object]:
        """Run DA3 forward on a batch with aspect-correct resize.

        Args:
            pixel_values: [B, 3, H_in, W_in] ImageNet-normalized float. Any input
                resolution accepted; bilinearly resized to (h, w) before DA3 forward.
            target_input_hw: optional (h, w) override for DA3 input size. When set,
                this overrides the configured (da3_input_h, da3_input_w) — useful
                for running the same DA3 at different resolutions per view
                (e.g., main camera at 252×336, wrist at 168×224). Both dims must
                be multiples of patch_size. The returned feats and ray use a
                token grid derived from the target dims, NOT the configured ones.

        Returns:
            dict with keys:
                feats:  List[torch.Tensor]  one per out_layer, each [B, C, h_grid, w_grid]
                ray:    torch.Tensor        [B, 3, h_grid, w_grid], unit-normalized direction
                h_grid: int
                w_grid: int
        """
        if pixel_values.dim() != 4:
            raise ValueError(f"pixel_values must be [B, 3, H, W]; got shape {tuple(pixel_values.shape)}")
        B = pixel_values.shape[0]
        device = pixel_values.device
        x = pixel_values.to(torch.float32)

        # Resolve input dims + corresponding token grid.
        if target_input_hw is None:
            in_h, in_w = self.da3_input_h, self.da3_input_w
            h_grid, w_grid = self.h_grid, self.w_grid
        else:
            in_h, in_w = int(target_input_hw[0]), int(target_input_hw[1])
            if in_h % self.patch_size != 0 or in_w % self.patch_size != 0:
                raise ValueError(
                    f"target_input_hw=({in_h},{in_w}) must be multiples of patch_size={self.patch_size}"
                )
            h_grid, w_grid = in_h // self.patch_size, in_w // self.patch_size

        # Aspect-correct resize (bilinear). NO PADDING — input dims chosen so
        # both are multiples of patch_size.
        x = F.interpolate(x, size=(in_h, in_w), mode="bilinear", align_corners=False)
        # Expand to V=1 view dim (DA3 expects [B, V, 3, H, W])
        x = x.unsqueeze(1)

        # Clear stale ray capture (and depth if enabled), run DA3
        self._ray_capture.clear()
        if self._capture_depth:
            self._depth_capture.clear()
        if self.use_bf16:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                aux = self._da3_forward(x)
        else:
            aux = self._da3_forward(x)

        feats, ray_at_feat, depth_at_feat = self._finalize(aux, B, h_grid, w_grid)
        return {
            "feats": feats,                # list of [B, C, h_grid, w_grid]
            "ray": ray_at_feat,            # [B, 3, h_grid, w_grid]
            "depth": depth_at_feat,        # [B, 1, h_grid, w_grid] or None
            "h_grid": int(h_grid),
            "w_grid": int(w_grid),
        }

    def _finalize(self, aux, B: int, h_grid: int, w_grid: int):
        """Pull per-layer feats + ray + optional depth out of a completed DA3
        forward. Returns tensors whose leading dim is B*V (V baked in by DA3's
        aux layout): feats as list of [B*V, C, h_grid, w_grid], ray as
        [B*V, 3, h_grid, w_grid], depth as [B*V, 1, h_grid, w_grid] or None.
        The single-view ``forward`` (V=1) gets [B, ...]; the joint multi-view
        path unflattens the leading dim back to (B, V).
        """
        # Extract per-level feats at native (h_grid, w_grid). DA3 may emit them
        # at the backbone resolution already matching (h_grid, w_grid) since
        # in_h/in_w = h_grid*patch × w_grid*patch.
        feats: List[torch.Tensor] = []
        for li in self.out_layers:
            key = f"feat_layer_{li}"
            fi = aux.get(key)
            if fi is None:
                raise RuntimeError(
                    f"DA3LargeForGeoStack: aux missing key '{key}'; available={list(aux.keys())[:8]}"
                )
            fi = self._to_bchw(fi, B=B)  # [B*V, C, h, w]
            if fi.shape[-1] != w_grid or fi.shape[-2] != h_grid:
                # Defensive resize in case DA3 produced a different grid.
                fi = F.interpolate(
                    fi.float(), size=(h_grid, w_grid),
                    mode="bilinear", align_corners=False,
                )
            feats.append(fi)

        # Extract ray-head output (3D direction only, normalized)
        ray_raw = self._ray_capture.get("ray_raw", None)
        if ray_raw is None:
            raise RuntimeError(
                "DA3LargeForGeoStack: ray-head hook did not fire; DA3 forward did not run "
                "the aux/ray branch. Check head.scratch.output_conv2_aux exists."
            )
        if ray_raw.dim() == 4:
            ray_dir = ray_raw[:, :3, :, :]
        elif ray_raw.dim() == 5:                          # [B, V, C, H, W]
            ray_dir = ray_raw[:, :, :3, :, :].reshape(-1, 3, ray_raw.shape[-2], ray_raw.shape[-1])
        else:
            raise RuntimeError(f"Unexpected ray tensor rank: {ray_raw.dim()}, shape={tuple(ray_raw.shape)}")
        ray_at_feat = F.interpolate(
            ray_dir.float(), size=(h_grid, w_grid),
            mode="bilinear", align_corners=False,
        )
        ray_at_feat = ray_at_feat / (ray_at_feat.norm(dim=1, keepdim=True) + 1e-6)
        self._ray_capture.clear()

        # Optional depth (v4.1) — extract DA3's main-head depth output at patch grid
        depth_at_feat = None
        if self._capture_depth:
            depth_raw = self._depth_capture.get("depth_raw", None)
            if depth_raw is not None:
                # depth_raw: [B*V, C_out, H, W] or [B, V, C_out, H, W]
                if depth_raw.dim() == 5:
                    depth_raw = depth_raw.reshape(-1, depth_raw.shape[-3], depth_raw.shape[-2], depth_raw.shape[-1])
                # First channel = raw depth logits (DPT convention)
                depth_scalar = depth_raw[:, :1].float()
                depth_at_feat = F.interpolate(
                    depth_scalar, size=(h_grid, w_grid),
                    mode="bilinear", align_corners=False,
                )
            self._depth_capture.clear()

        return feats, ray_at_feat, depth_at_feat


    # ---------- multi-view forward (v2C) ----------

    def forward_multi_view(
        self,
        pixel_values: torch.Tensor,                              # [B, V, 3, H_in, W_in]
        target_input_hw: Tuple[int, int] | None = None,
        extrinsics: torch.Tensor | None = None,                  # [B, V, 4, 4] OpenCV w2c
        intrinsics: torch.Tensor | None = None,                  # [B, V, 3, 3] @ (H_in, W_in)
    ) -> Dict[str, object]:
        """Process multiple views, returning per-view feats/rays.

        Two modes, same output contract:
          * Unposed (extrinsics/intrinsics=None): flatten [B,V]→[B*V] and run
            each view through DA3 INDEPENDENTLY (legacy v2C wrist-depth path).
          * Posed joint (both provided): feed ALL V views TOGETHER into one DA3
            inner() call with camera extrinsics + intrinsics, so DA3's camera
            encoder makes the per-view features geometrically consistent across
            views. Downstream per-view routing/cross-attention is unchanged —
            only the feature content becomes cross-view-aware.

        Args:
            pixel_values: [B, V, 3, H_in, W_in] ImageNet-normalized float.
            target_input_hw: optional (h, w) override (e.g., 224×224 for wrists).
            extrinsics: optional [B, V, 4, 4] OpenCV world-to-camera.
            intrinsics: optional [B, V, 3, 3] calibrated to (H_in, W_in); DA3
                resizes views to (in_h, in_w) internally so K is rescaled to match.

        Returns:
            dict with keys:
                feats:  List[torch.Tensor]  one per out_layer, each [B, V, C, h_grid, w_grid]
                ray:    torch.Tensor        [B, V, 3, h_grid, w_grid]
                depth:  torch.Tensor | None [B, V, 1, h_grid, w_grid]
                h_grid: int
                w_grid: int
        """
        if pixel_values.dim() != 5:
            raise ValueError(
                f"forward_multi_view expects [B,V,3,H,W]; got {tuple(pixel_values.shape)}"
            )
        B, V = pixel_values.shape[:2]

        # ---- Posed joint multi-view path ----
        if extrinsics is not None and intrinsics is not None:
            return self._forward_multi_view_posed(
                pixel_values, extrinsics, intrinsics, target_input_hw
            )

        # ---- Legacy unposed per-view path ----
        flat = pixel_values.flatten(0, 1)                        # [B*V, 3, H_in, W_in]
        out_flat = self.forward(flat, target_input_hw=target_input_hw)
        feats_flat = out_flat["feats"]                            # list of [B*V, C, h, w]
        ray_flat = out_flat["ray"]                                # [B*V, 3, h, w]
        h_grid = int(out_flat["h_grid"])
        w_grid = int(out_flat["w_grid"])
        feats = [f.unflatten(0, (B, V)) for f in feats_flat]      # list of [B, V, C, h, w]
        ray = ray_flat.unflatten(0, (B, V))                       # [B, V, 3, h, w]
        result = {
            "feats": feats,
            "ray": ray,
            "h_grid": h_grid,
            "w_grid": w_grid,
        }
        # v4.1: forward multi-view depth if the depth hook is on
        depth_flat = out_flat.get("depth", None)
        if depth_flat is not None:
            result["depth"] = depth_flat.unflatten(0, (B, V))    # [B, V, 1, h, w]
        else:
            result["depth"] = None
        return result

    def _forward_multi_view_posed(
        self,
        pixel_values: torch.Tensor,           # [B, V, 3, H_in, W_in]
        extrinsics: torch.Tensor,             # [B, V, 4, 4] OpenCV w2c
        intrinsics: torch.Tensor,             # [B, V, 3, 3] @ (H_in, W_in)
        target_input_hw: Tuple[int, int] | None = None,
    ) -> Dict[str, object]:
        """Joint posed multi-view DA3: one inner() call over all V views with
        camera pose conditioning → cross-view-consistent per-view features.
        Same return contract as forward_multi_view (unposed)."""
        B, V, _, H_in, W_in = pixel_values.shape
        device = pixel_values.device

        # Resolve DA3 input dims + the resulting token grid.
        if target_input_hw is None:
            in_h, in_w = self.da3_input_h, self.da3_input_w
            h_grid, w_grid = self.h_grid, self.w_grid
        else:
            in_h, in_w = int(target_input_hw[0]), int(target_input_hw[1])
            if in_h % self.patch_size != 0 or in_w % self.patch_size != 0:
                raise ValueError(
                    f"target_input_hw=({in_h},{in_w}) must be multiples of patch_size={self.patch_size}"
                )
            h_grid, w_grid = in_h // self.patch_size, in_w // self.patch_size

        # Aspect-correct resize of every view to (in_h, in_w).
        x = pixel_values.to(torch.float32).flatten(0, 1)          # [B*V, 3, H_in, W_in]
        if (H_in, W_in) != (in_h, in_w):
            x = F.interpolate(x, size=(in_h, in_w), mode="bilinear", align_corners=False)
        x = x.unflatten(0, (B, V))                                # [B, V, 3, in_h, in_w]

        # Rescale intrinsics from the incoming tensor resolution to (in_h, in_w)
        # so DA3's pose math sees K aligned with the pixels it actually gets.
        K = self._rescale_intrinsics(intrinsics, (H_in, W_in), (in_h, in_w))

        # Clear stale captures, run ONE joint posed DA3 forward.
        self._ray_capture.clear()
        if self._capture_depth:
            self._depth_capture.clear()
        if self.use_bf16:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                aux, depth_full = self._da3_forward_posed(x, extrinsics, K)
        else:
            aux, depth_full = self._da3_forward_posed(x, extrinsics, K)

        # Shared extraction — leading dim is B*V (V baked into DA3's aux layout).
        feats_flat, ray_flat, depth_hook = self._finalize(aux, B, h_grid, w_grid)
        feats = [f.unflatten(0, (B, V)) for f in feats_flat]      # list of [B, V, C, h, w]
        ray = ray_flat.unflatten(0, (B, V))                       # [B, V, 3, h, w]

        # Metric depth from the top-level output (preferred over the DPT hook):
        # [B,V,H,W] → resize to the token grid → [B,V,1,h,w].
        if depth_full is not None:
            df = depth_full.float()
            if df.dim() == 4:                                     # [B, V, H, W]
                df = df.flatten(0, 1).unsqueeze(1)                # [B*V, 1, H, W]
            elif df.dim() == 5:                                   # [B, V, 1, H, W]
                df = df.flatten(0, 1)                             # [B*V, 1, H, W]
            df = F.interpolate(df, size=(h_grid, w_grid), mode="bilinear", align_corners=False)
            depth_grid = df.unflatten(0, (B, V))                  # [B, V, 1, h, w]
        elif depth_hook is not None:
            depth_grid = depth_hook.unflatten(0, (B, V))
        else:
            depth_grid = None

        result: Dict[str, object] = {
            "feats": feats,
            "ray": ray,
            "h_grid": int(h_grid),
            "w_grid": int(w_grid),
            "depth": depth_grid,                                  # [B, V, 1, h, w] metric
        }
        return result


__all__ = ["DA3LargeForGeoStack"]
