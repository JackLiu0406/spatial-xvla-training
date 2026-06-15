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
DA3 (Depth Anything 3) feature-extractor wrapper.

This is a thin, dependency-light interface used *only* by the offline
preprocessing script (`scripts/precompute_da3_features.py`). DA3 is **not**
run inside the X-VLA model forward pass.

Two backends:
  * ``backend="dummy"``  -> deterministic random features, no DA3 install
                            needed (for plumbing/CI tests).
  * ``backend="da3"``    -> real Depth Anything 3 encoder. If the package is
                            not importable a clear installation error is raised.

All backends expose the same contract:

    extractor.extract_features(images) -> Tensor [V, C, H, W]

where ``images`` is a list/array of ``V`` RGB frames (HxWx3, uint8 or float).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, "Image.Image"]  # noqa: F821

_DA3_INSTALL_HINT = (
    "Depth Anything 3 is not installed.\n"
    "Install it before using backend='da3', e.g.:\n"
    "    pip install depth-anything-3\n"
    "  or follow https://github.com/ByteDance-Seed/Depth-Anything-3\n"
    "Alternatively run the precompute script with --backend dummy to generate\n"
    "random placeholder features for pipeline testing."
)


class DA3FeatureExtractor:
    """
    Wrapper around a DA3 image encoder.

    Parameters
    ----------
    backend : "dummy" | "da3"
    model_name : str
        DA3 checkpoint id (only used by the real backend).
    image_size : int
        Frames are resized to ``(image_size, image_size)`` before encoding so
        that every saved feature map has a fixed, batch-collatable shape.
    feature_dim : int
        Channel dim of the produced features (dummy backend; for the real
        backend it is whatever DA3 outputs).
    device : str
    dtype : torch.dtype
    """

    def __init__(
        self,
        backend: str = "dummy",
        model_name: str = "depth-anything/DA3-BASE",
        image_size: int = 224,
        feature_dim: int = 256,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
        feature_size: int = 64,
        process_res: int = 504,
        feature_kind: str = "latent",
        feature_layer: int | str = "last",
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.image_size = int(image_size)
        self.feature_dim = int(feature_dim)
        # Fixed output spatial size so every saved feature is batch-collatable.
        self.feature_size = int(feature_size)
        self.process_res = int(process_res)
        # "latent": DA3/DINOv2 encoder feature map (rich 3D-aware geometry
        #           tokens — what segmented-DA3 conditioning is designed for).
        # "depth" : per-view depth(+conf) [V,2,H,W] (lightweight fallback).
        self.feature_kind = str(feature_kind)
        # DINOv2 block index, or "last" -> resolved from the backbone depth.
        self.feature_layer = feature_layer
        self.device = device
        self.dtype = dtype
        self._seed = seed
        self._model = None
        self._n_blocks: int | None = None

        if backend == "dummy":
            pass
        elif backend == "da3":
            self._load_da3()
        else:
            raise ValueError(
                f"Unknown DA3 backend {backend!r}; expected 'dummy' or 'da3'."
            )

    # ------------------------------------------------------------------ #
    def _load_da3(self) -> None:
        """Import and build the real DA3 encoder, or fail with a clear msg."""
        try:
            from depth_anything_3.api import DepthAnything3  # type: ignore
        except Exception as exc:  # ImportError or partial install
            raise ImportError(_DA3_INSTALL_HINT) from exc

        # API ref: https://github.com/ByteDance-Seed/Depth-Anything-3
        #   from depth_anything_3.api import DepthAnything3
        #   model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE")
        #   pred  = model.inference(image=[np/PIL/path, ...])
        #   pred.depth -> (N, H, W);  pred.conf -> (N, H, W) optional
        try:
            try:
                model = DepthAnything3.from_pretrained(self.model_name)
            except Exception:
                # Also accept short ids like "da3-base".
                model = DepthAnything3(model_name=self.model_name)
            model = model.to(self.device)
            if hasattr(model, "eval"):
                model.eval()
            self._model = model
            # Introspect the DINOv2 backbone depth so feature_layer="last"
            # resolves to a valid block index across DA3 sizes.
            for m in model.modules():
                nb = getattr(m, "n_blocks", None)
                if isinstance(nb, int) and nb > 0:
                    self._n_blocks = nb
                    break
        except Exception as exc:
            raise RuntimeError(
                f"Failed to construct Depth Anything 3 model "
                f"'{self.model_name}'. Adapt DA3FeatureExtractor._load_da3 / "
                f"_encode_da3 to your installed DA3 API. Original error: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_chw_float(img: ArrayLike) -> torch.Tensor:
        """RGB array/PIL -> float CHW tensor in [0,1]."""
        if hasattr(img, "convert"):  # PIL.Image
            img = np.asarray(img.convert("RGB"))
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] != 3 and arr.shape[0] == 3:  # already CHW
            t = torch.as_tensor(arr).float()
        else:
            t = torch.as_tensor(arr).permute(2, 0, 1).float()
        if t.max() > 1.5:
            t = t / 255.0
        return t

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(
            x.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def extract_features(self, images: Sequence[ArrayLike]) -> torch.Tensor:
        """
        Encode ``V`` views into a stacked feature tensor.

        Returns
        -------
        Tensor of shape ``[V, C, H, W]`` (float, on CPU).
        """
        if len(images) == 0:
            raise ValueError("extract_features received an empty image list.")

        if self.backend == "dummy":
            chw = torch.stack(
                [self._resize(self._to_chw_float(im)) for im in images], 0
            )
            feats = self._encode_dummy(chw)
        else:
            # DA3 does its own resize/normalization; hand it the raw frames.
            np_imgs = [self._to_uint8_rgb(im) for im in images]
            feats = self._encode_da3(np_imgs)
        return feats.detach().to("cpu", torch.float32).contiguous()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_uint8_rgb(img: ArrayLike) -> np.ndarray:
        """Any RGB image -> HxWx3 uint8 ndarray (DA3's accepted input)."""
        if hasattr(img, "convert"):  # PIL.Image
            return np.asarray(img.convert("RGB"), dtype=np.uint8)
        a = np.asarray(img)
        if a.ndim == 2:
            a = np.stack([a] * 3, -1)
        if a.dtype != np.uint8:
            a = (a * 255 if a.max() <= 1.5 else a).clip(0, 255).astype(np.uint8)
        return a

    def _fixed_size(self, x: torch.Tensor) -> torch.Tensor:
        """[V,C,h,w] -> [V,C,feature_size,feature_size] (bilinear)."""
        return torch.nn.functional.interpolate(
            x, size=(self.feature_size, self.feature_size),
            mode="bilinear", align_corners=False,
        )

    def _encode_dummy(self, chw: torch.Tensor) -> torch.Tensor:
        """Deterministic random features at the fixed feature size."""
        V = chw.shape[0]
        s = max(1, self.image_size // 16)
        g = torch.Generator(device="cpu").manual_seed(
            self._seed + int(chw.float().mean().item() * 1e4) % 100000
        )
        return torch.randn(V, self.feature_dim, s, s, generator=g)

    def _resolve_layer(self) -> int:
        """feature_layer -> a valid DINOv2 block index."""
        if isinstance(self.feature_layer, int):
            n = self._n_blocks
            L = self.feature_layer
            if L < 0 and n:
                L = n + L
            if n is not None:
                L = max(0, min(L, n - 1))
            return int(L)
        # "last" / unknown -> deepest block we can infer (fallback 23 ≈ ViT-L).
        return int((self._n_blocks - 1) if self._n_blocks else 23)

    def _encode_da3(self, np_imgs) -> torch.Tensor:
        """
        Real DA3 forward via the documented inference API.

        The ``V`` views are passed as one multi-view inference call (DA3
        reconstructs joint geometry across views).

        feature_kind="latent" (default): return the DA3/DINOv2 encoder feature
        map for the requested block as a dense ``[V, C, feature_size,
        feature_size]`` tensor — the rich, 3D-aware latent the segmented-DA3
        geometry conditioner is designed to consume.

        feature_kind="depth": per-view depth(+conf) ``[V, 2, ...]`` fallback.
        """
        assert self._model is not None
        want_latent = self.feature_kind == "latent"
        layer = self._resolve_layer() if want_latent else None

        # export_dir=None (default) -> no disk export; export_format must stay
        # a string (DA3 does `"gs" in export_format` unconditionally).
        pred = self._model.inference(
            image=np_imgs,
            infer_gs=False,
            use_ray_pose=False,
            process_res=self.process_res,
            export_feat_layers=[layer] if want_latent else None,
        )

        if want_latent:
            aux = getattr(pred, "aux", None) or {}
            key = f"feat_layer_{layer}"
            feat = aux.get(key)
            if feat is not None:
                # DA3 aux layout: [V, s, h, w, C] -> reduce s, -> [V, C, h, w]
                t = torch.as_tensor(np.asarray(feat)
                                    if not torch.is_tensor(feat) else feat).float()
                if t.dim() == 5:
                    t = t.mean(dim=1)                    # [V, h, w, C]
                if t.dim() == 4:
                    t = t.permute(0, 3, 1, 2)            # [V, C, h, w]
                elif t.dim() == 3:                       # [V, N, C] tokens
                    V, N, C = t.shape
                    s = max(1, int(round(N ** 0.5)))
                    t = t[:, : s * s].transpose(1, 2).reshape(V, C, s, s)
                return self._fixed_size(t.contiguous())
            logger.warning(
                "DA3 aux['%s'] missing; falling back to depth+conf features. "
                "Set feature_layer to a valid DINOv2 block index.", key,
            )

        depth = torch.as_tensor(np.asarray(pred.depth)).float()  # [V,H,W]
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)
        chans = [depth.unsqueeze(1)]  # [V,1,H,W]
        conf = getattr(pred, "conf", None)
        if conf is not None:
            c = torch.as_tensor(np.asarray(conf)).float()
            if c.dim() == 2:
                c = c.unsqueeze(0)
            chans.append(c.unsqueeze(1))
        feat = torch.cat(chans, dim=1)  # [V, C(=1 or 2), H, W]
        return self._fixed_size(feat)


__all__ = ["DA3FeatureExtractor"]
