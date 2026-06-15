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
Offline DA3 feature lookup for the X-VLA dataloader.

Storage layout (written by ``scripts/precompute_da3_features.py``)::

    <da3_feature_root>/
        manifest.json
        <dataset_slug>/<traj_idx>/<frame_idx>.pt

Each ``<frame_idx>.pt`` is a dict::

    {
      "da3_features": FloatTensor[V, ...],   # [V,C,H,W] or [V,N,C]
      "object_masks": FloatTensor[V,1,h,w],  # optional
      "camera_ids":   [str, ...],            # length V
    }

The key ``(dataset_name, traj_idx, frame_idx)`` is exactly the triplet
available inside ``BaseHDF5Handler.iter_episode`` (``frame_idx`` indexes the
raw HDF5 image array), so attach is unambiguous and order-independent.

Nothing here is imported unless geometry conditioning is enabled.
"""

from __future__ import annotations

import os
import re
from typing import Optional, Tuple

import torch

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def dataset_slug(dataset_name: str) -> str:
    """Filesystem-safe, stable slug for a dataset name / root path."""
    return _SLUG_RE.sub("_", str(dataset_name)).strip("_") or "dataset"


def feature_path(root: str, dataset_name: str, traj_idx: int, frame_idx: int) -> str:
    """Layout A (legacy): <root>/<dataset_slug>/<traj>/<frame>.pt"""
    return os.path.join(
        root, dataset_slug(dataset_name), str(int(traj_idx)), f"{int(frame_idx)}.pt"
    )


def episode_dir(root: str, traj_idx: int) -> str:
    return os.path.join(root, f"episode_{int(traj_idx):06d}")


def episode_camera_path(root: str, traj_idx: int, camera: str, frame_idx: int) -> str:
    """Layout B (spec scripts): <root>/episode_{:06d}/<camera>/frame_{:06d}.pt"""
    return os.path.join(
        episode_dir(root, traj_idx), str(camera), f"frame_{int(frame_idx):06d}.pt"
    )


class DA3FeatureAttacher:
    """
    Look up precomputed DA3 features and attach them to a training sample.

    Parameters
    ----------
    da3_feature_root : str
        Root directory produced by the precompute script.
    use_object_mask : bool
        If True, also surface ``object_masks`` when present in the .pt file
        (or from a parallel ``object_mask_root``).
    object_mask_root : str | None
        Optional separate root for masks (same layout). If None, masks are
        taken from the feature .pt file when available.
    allow_missing : bool
        If True, a missing feature file yields a small zero placeholder
        instead of raising (debug only — keeps batches collatable).
    placeholder_dim : int
        Channel dim used for the zero placeholder when ``allow_missing``.

    Supports two on-disk layouts (tried in order):
      A. legacy : <root>/<dataset_slug>/<traj>/<frame>.pt  (single stacked blob)
      B. spec   : <root>/episode_{idx:06d}/<camera>/frame_{t:06d}.pt
                  (per-camera; all camera dirs are stacked to a leading V axis)

    Alignment key is (episode_id~=traj_idx, timestep~=frame_idx, camera).
    """

    def __init__(
        self,
        da3_feature_root: str,
        use_object_mask: bool = True,
        object_mask_root: Optional[str] = None,
        allow_missing: bool = False,
        placeholder_dim: int = 1,
        da3_feature_key: str = "da3_features",
        object_mask_key: str = "object_masks",
        allow_dummy_da3: bool = False,
        allow_dummy_masks: bool = False,
    ) -> None:
        if not da3_feature_root:
            raise ValueError("da3_feature_root must be set to attach DA3 features.")
        self.root = da3_feature_root
        self.use_object_mask = use_object_mask
        self.mask_root = object_mask_root
        self.allow_missing = allow_missing
        self.placeholder_dim = placeholder_dim
        self.da3_feature_key = da3_feature_key
        self.object_mask_key = object_mask_key
        self.allow_dummy_da3 = allow_dummy_da3 or allow_missing
        self.allow_dummy_masks = allow_dummy_masks or allow_missing

    # ------------------------------------------------------------------ #
    def _stack_cameras(self, root: str, traj_idx: int, frame_idx: int, key: str):
        """Layout B: load every camera dir and stack -> [V, ...]."""
        ed = episode_dir(root, traj_idx)
        if not os.path.isdir(ed):
            return None, ed
        cams = sorted(d for d in os.listdir(ed) if os.path.isdir(os.path.join(ed, d)))
        tensors = []
        for cam in cams:
            p = episode_camera_path(root, traj_idx, cam, frame_idx)
            if not os.path.isfile(p):
                return None, p
            blob = torch.load(p, map_location="cpu")
            t = blob.get(key, blob.get("masks")) if isinstance(blob, dict) else blob
            if t is None:
                return None, p
            tensors.append(torch.as_tensor(t).float())
        if not tensors:
            return None, ed
        return torch.stack(tensors, 0), ed  # [V, ...]

    def _load_blob(self, root: str, dataset_name: str, traj_idx: int,
                   frame_idx: int, key: str):
        """Return (tensor_or_None, path). Tries layout A then B."""
        pa = feature_path(root, dataset_name, traj_idx, frame_idx)
        if os.path.isfile(pa):
            blob = torch.load(pa, map_location="cpu")
            t = blob.get(key, blob.get("masks")) if isinstance(blob, dict) else blob
            return (None if t is None else torch.as_tensor(t).float()), pa
        return self._stack_cameras(root, traj_idx, frame_idx, key)

    # ------------------------------------------------------------------ #
    def attach(
        self,
        sample: dict,
        dataset_name: str,
        traj_idx: int,
        frame_idx: int,
        num_views: int,
    ) -> None:
        """
        Mutate ``sample`` in-place: add ``da3_features`` and (optionally)
        ``object_masks``. Clear error if missing and dummies not allowed.
        """
        feats, fpath = self._load_blob(
            self.root, dataset_name, traj_idx, frame_idx, self.da3_feature_key
        )
        if feats is None:
            if not self.allow_dummy_da3:
                raise FileNotFoundError(
                    f"Precomputed DA3 feature not found: {fpath}\n"
                    f"Run scripts/precompute_da3_features.py for dataset "
                    f"'{dataset_name}', or set "
                    f"geometry_conditioning.allow_dummy_da3_features=true."
                )
            feats = torch.zeros(num_views, self.placeholder_dim, 1, 1)
        sample["da3_features"] = feats

        if not self.use_object_mask:
            return

        mroot = self.mask_root or self.root
        mask, mpath = self._load_blob(
            mroot, dataset_name, traj_idx, frame_idx, self.object_mask_key
        )
        if mask is None:
            if not self.allow_dummy_masks:
                raise FileNotFoundError(
                    f"Precomputed object mask not found: {mpath}\n"
                    f"Run scripts/precompute_object_masks.py, or set "
                    f"geometry_conditioning.allow_dummy_masks=true / "
                    f"allow_missing_masks=true."
                )
            # Dummy: an all-ones mask -> segmenter becomes a no-op.
            mask = torch.ones(num_views, 1, 1, 1)
        sample["object_masks"] = mask


__all__ = [
    "DA3FeatureAttacher",
    "feature_path",
    "episode_camera_path",
    "episode_dir",
    "dataset_slug",
]
