"""DA3 feature-cache reader for the pi0.5-DA3 pipeline.

Joins precached DA3 geometry (feats/ray/depth) to LeRobot `camext` frames. Two facts the
alignment gate established:
  * cache traj order (from `_meta_all_ct_3cam_clean.json` datalist) != LeRobot episode order
    -> join by HDF5 path (NOT by index).
  * frame_idx is 1:1 with the raw HDF5 frame == LeRobot frame_index, but the cache is a sparse
    subset (static-skip + ~13.5 Hz subsample) -> snap to nearest cached frame.

Decodes fp8-e4m3fn feats (via torch). Runs in the torch data workers; emits numpy arrays.
"""

import functools
import json
import os

import numpy as np
import torch

_FP8 = torch.float8_e4m3fn


@functools.lru_cache(maxsize=4)
def _episode_join(raw_root: str, meta_path: str, converter_path: str) -> tuple[int, ...]:
    """LeRobot episode index -> cache traj index (via HDF5 path); -1 if not in the cache manifest."""
    datalist = json.load(open(meta_path))["datalist"]
    path2traj = {p: i for i, p in enumerate(datalist)}
    import importlib.util

    spec = importlib.util.spec_from_file_location("_da3cv", converter_path)
    cv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cv)
    valid = [e["hdf5"] for e in cv.enumerate_episodes(cv.Path(raw_root)) if cv.read_length(e["hdf5"])[0] > 0]
    return tuple(path2traj.get(hp, -1) for hp in valid)


class DA3CacheReader:
    def __init__(
        self,
        cache_root: str,
        slug: str,
        raw_root: str,
        meta_path: str,
        converter_path: str,
        view_order: tuple[str, ...] = ("countertop", "left", "right"),
    ):
        self.cache_root = cache_root
        self.slug = slug
        self.view_order = view_order
        self.ep2traj = _episode_join(raw_root, meta_path, converter_path)
        self._frames: dict[int, list[int]] = {}

    def _traj_dir(self, ep: int) -> str | None:
        traj = self.ep2traj[ep]
        return os.path.join(self.cache_root, self.slug, str(traj)) if traj >= 0 else None

    def cached_frames(self, ep: int) -> list[int] | None:
        d = self._traj_dir(ep)
        if d is None or not os.path.isdir(d):
            return None
        if ep not in self._frames:
            self._frames[ep] = sorted(int(f[:-3]) for f in os.listdir(d) if f.endswith(".pt"))
        return self._frames[ep] or None

    def has_episode(self, ep: int) -> bool:
        return self.cached_frames(ep) is not None

    def snap(self, ep: int, frame: int) -> int | None:
        fr = self.cached_frames(ep)
        if not fr:
            return None
        i = int(np.searchsorted(fr, frame))
        cands = [fr[max(0, i - 1)], fr[min(len(fr) - 1, i)]]
        return min(cands, key=lambda c: abs(c - frame))

    def load(self, ep: int, frame: int):
        """Return (feats[4,V,C,h,w] uint8-fp8, ray[V,3,h,w] f32, depth[V,1,h,w] f32) or None if uncached.

        feats are kept as raw fp8 BYTES (uint8) — 4x smaller than f32 — so the loader/IPC/host->device
        move only 8 MB/sample instead of 32 MB. They are bitcast to float8_e4m3fn and upcast to f32 on
        the GPU (see Pi0._compute_banks), which also offloads the decode from the CPU workers.
        """
        traj = self.ep2traj[ep]
        f = self.snap(ep, frame)
        if f is None:
            return None
        blob = torch.load(
            os.path.join(self.cache_root, self.slug, str(traj), f"{f}.pt"), map_location="cpu", weights_only=False
        )
        feats = np.stack(
            [t.contiguous().view(torch.uint8).numpy() for t in blob["feats"]], axis=0
        )  # [4,V,C,h,w] uint8 (raw fp8-e4m3fn bytes; decoded on GPU)
        ray = blob["ray"].to(torch.float32).numpy()  # [V,3,h,w]
        if blob.get("depth") is not None:
            depth = blob["depth"].to(torch.float32).numpy()  # [V,1,h,w]
        else:
            depth = np.zeros((ray.shape[0], 1, ray.shape[2], ray.shape[3]), np.float32)
        return feats, ray, depth


def stack_extrinsics(raw: dict, view_order: tuple[str, ...]) -> np.ndarray:
    """Stack per-view OpenCV world->cam 4x4 from LeRobot parquet columns, in cache view order."""
    mats = []
    for v in view_order:
        e = np.asarray(raw[f"observation.{v}.extrinsic_cv"], np.float32).reshape(3, 4)
        m = np.eye(4, dtype=np.float32)
        m[:3, :4] = e
        mats.append(m)
    return np.stack(mats, axis=0)  # [V,4,4]
