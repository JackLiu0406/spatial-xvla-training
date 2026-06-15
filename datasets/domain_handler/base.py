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

import io
import random
from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Optional, Sequence, Any

import numpy as np
import h5py
import torch
from mmengine import fileio
from PIL import Image
from scipy.interpolate import interp1d

class DomainHandler(ABC):
    """
    Minimal domain handler interface.

    Subclasses provide dataset-specific decoding by implementing an iterator
    that yields per-sample dictionaries compatible with the training loop.
    """
    dataset_name: str

    def __init__(self, meta: dict, num_views: int) -> None:
        self.meta = meta
        self.num_views = num_views

    @abstractmethod
    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        action_mode,
        lang_aug_map: dict | None,
        **kwargs
    ) -> Iterable[dict]:
        """Yield samples for a single episode."""
        ...


def _open_h5(path: str) -> h5py.File:
    """Open HDF5 from local FS or remote backend via mmengine.fileio."""
    try:
        return h5py.File(path, "r")
    except OSError:
        return h5py.File(io.BytesIO(fileio.get(path)), "r")


class BaseHDF5Handler(DomainHandler):
    """
    Generic HDF5 handler with resource-safe iteration.

    Subclasses only implement:
      - build_left_right(f) -> (left, right, left_time, right_time, freq, qdur)
          left/right: abs_trajectory [T, C], left_time/right_time: optional time arrays [T],
          freq (Hz), qdur (seconds of future window)
      - index_candidates(T_left, training) -> Iterable[int]

    Optionally override:
      - get_image_datasets(f): sequence of image arrays/datasets
      - read_instruction(f): string instruction
    """

    # --- Optional overrides -------------------------------------------------
    def get_image_datasets(self, f: h5py.File) -> Sequence[Any]:
        keys: Sequence[str] = self.meta["observation_key"]
        return [f[k][()] for k in keys]

    def get_camera_poses(self, f: h5py.File):
        """
        Optional posed-DA3 hook. Override in handlers that have per-frame
        camera poses. Return (extrinsics_list, intrinsics_list):
          extrinsics_list: list of [T,4,4] float32 (world-to-camera)
          intrinsics_list: list of [T,3,3] float32
        Both lists must have len == num_views in observation_key order.
        Return (None, None) to disable posed-DA3 for this dataset.
        """
        return None, None

    def read_instruction(self, f: h5py.File) -> str:
        key: str = self.meta["language_instruction_key"]
        ds = f[key]
        v = ds[()]
        return v.decode() if getattr(ds, "shape", ()) == () else v[0].decode()

    # --- Required hooks -----------------------------------------------------
    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        raise NotImplementedError

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        raise NotImplementedError
    # -----------------------------------------------------------------------

    @staticmethod
    def _pil_from_arr(arr: Any) -> Image.Image:
        from ..utils import decode_image_from_bytes
        return decode_image_from_bytes(arr) if not isinstance(arr, Image.Image) else arr

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        image_aug_da3=None,
        lang_aug_map: dict | None,
        **kwargs
    ) -> Iterable[dict]:
        """Open once, yield many samples; file is always closed on exit."""
        datapath = self.meta["datalist"][traj_idx]
        if not isinstance(datapath, str):
            datapath = datapath[0]

        with _open_h5(datapath) as f:
            # Images and mask
            images = self.get_image_datasets(f)
            # Language
            ins = self.read_instruction(f)
            # Domain-specific kinematics and timing
            left, right, lt, rt, freq, qdur = self.build_left_right(f)
            # Optional posed-DA3: per-frame extrinsics + intrinsics per camera.
            # Bulk-load with the file open; index by frame in the inner loop.
            ext_list, intr_list = self.get_camera_poses(f)
        
        
        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[:len(images)] = True
        if lt is None: lt = np.arange(left.shape[0], dtype=np.float64) / float(freq)
        if rt is None: rt = np.arange(right.shape[0], dtype=np.float64) / float(freq)

        # Candidate indices (optionally shuffled)
        idxs = list(self.index_candidates(left.shape[0], training))
        if training: random.shuffle(idxs)

        # Interpolators; clamp to endpoints
        L = interp1d(lt, left, axis=0, bounds_error=False, fill_value=(left[0], left[-1]))
        R = interp1d(rt, right, axis=0, bounds_error=False, fill_value=(right[0], right[-1]))
        ref = (lt + rt) / 2.0

        V = min(self.num_views, len(images))
        for idx in idxs:

            # Query future window
            cur = ref[idx]
            q = np.linspace(cur, min(cur + qdur, float(ref.max())), num_actions + 1, dtype=np.float32)
            lseq = torch.tensor(L(q))
            rseq = torch.tensor(R(q))

            # Skip static segments
            if (lseq[1] - lseq[0]).abs().max() < 1e-5 and (rseq[1] - rseq[0]).abs().max() < 1e-5: continue
            
            # Language augmentation
            if training and lang_aug_map and ins in lang_aug_map:
                ins = random.choice(lang_aug_map[ins])
            
            # Decode each view's PIL once so both branches share the same
            # source pixels (resize/jitter still applied independently per
            # branch via their respective Compose).
            pil_views = [self._pil_from_arr(images[v][idx]) for v in range(V)]
            imgs = [image_aug(p) for p in pil_views]
            while len(imgs) < self.num_views: imgs.append(torch.zeros_like(imgs[0]))
            image_input = torch.stack(imgs, dim=0)

            sample = {
                "language_instruction": ins,
                "image_input": image_input,
                "image_mask": image_mask,
                "abs_trajectory": torch.cat([lseq, rseq], -1).float()
            }

            # Dual-resolution: aspect-preserved tensor for the geometry encoder.
            # PIL.Image.size returns (W, H); capture before resize so we can
            # rescale intrinsics by the exact pixel ratio used here.
            da3_native_hw = None
            da3_target_hw = None
            if image_aug_da3 is not None and len(pil_views) > 0:
                src_W, src_H = pil_views[0].size
                imgs_da3 = [image_aug_da3(p) for p in pil_views]
                while len(imgs_da3) < self.num_views:
                    imgs_da3.append(torch.zeros_like(imgs_da3[0]))
                image_input_da3 = torch.stack(imgs_da3, dim=0)
                sample["image_input_da3"] = image_input_da3
                da3_native_hw = (src_H, src_W)
                da3_target_hw = (image_input_da3.shape[-2], image_input_da3.shape[-1])

            # Posed DA3 hook: attach per-frame extrinsics + intrinsics for the
            # cameras actually used (V views), at the sampled frame `idx`.
            # Stays absent from `sample` if the handler returned None, so the
            # default collate just sees the standard fields.
            if ext_list is not None and intr_list is not None:
                ext_per_view = np.stack(
                    [ext_list[v][idx] for v in range(V)], axis=0
                ).astype(np.float32)                                # [V,4,4]
                intr_per_view = np.stack(
                    [intr_list[v][idx] for v in range(V)], axis=0
                ).astype(np.float32)                                # [V,3,3]
                # Intrinsics in the HDF5 are calibrated for the *native* image
                # pixel grid. When the dual-resolution DA3 branch is active, we
                # resize each view from native (src_H, src_W) → (da3_H, da3_W)
                # for the geometry encoder, so K must be rescaled to match —
                # otherwise da3_inline's pose math operates on miscalibrated
                # rays. Note: da3_inline expects K to align with the input
                # tensor it receives, so we rescale to da3_target_hw here.
                if da3_target_hw is not None and da3_native_hw is not None:
                    src_H, src_W = da3_native_hw
                    da3_H, da3_W = da3_target_hw
                    sx = float(da3_W) / float(src_W)  # width scale
                    sy = float(da3_H) / float(src_H)  # height scale
                    intr_per_view[..., 0, 0] *= sx   # fx
                    intr_per_view[..., 1, 1] *= sy   # fy
                    intr_per_view[..., 0, 2] *= sx   # cx
                    intr_per_view[..., 1, 2] *= sy   # cy
                # Pad to num_views (zero rows for any missing view) so the
                # collated batch tensor has a stable shape.
                if V < self.num_views:
                    ext_pad = np.zeros((self.num_views, 4, 4), dtype=np.float32)
                    ext_pad[:V] = ext_per_view
                    ext_per_view = ext_pad
                    intr_pad = np.zeros((self.num_views, 3, 3), dtype=np.float32)
                    intr_pad[:V] = intr_per_view
                    intr_per_view = intr_pad
                sample["extrinsics"] = torch.from_numpy(ext_per_view)
                sample["intrinsics"] = torch.from_numpy(intr_per_view)

            # DA3-XVLA: optionally attach precomputed geometry features. Strict
            # no-op unless an attacher was injected by the dataset reader
            # (i.e. geometry_conditioning.enabled + da3_feature_root set).
            da3_attacher = getattr(self, "da3_attacher", None)
            if da3_attacher is not None:
                da3_attacher.attach(
                    sample,
                    getattr(self, "da3_dataset_name", self.dataset_name),
                    traj_idx,
                    idx,
                    self.num_views,
                )

            yield sample