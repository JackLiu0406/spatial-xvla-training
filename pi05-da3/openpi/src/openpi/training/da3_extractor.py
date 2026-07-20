"""Inline DA3-GIANT feature extractor (PyTorch, runs in the openpi venv).

Produces the SAME feats/ray/depth as the offline cache, but on-the-fly at train time — so
no precached features are needed (essential for datasets where caching is infeasible, e.g.
b1k at 30 Hz would be ~3 PB). Frozen, no_grad; the output tensors are handed to the JAX
model via dlpack.

depth_anything_3.api transitively imports rendering/SfM utils (moviepy/gsplat/pycolmap/
trimesh/evo) that (a) aren't needed for feature extraction and (b) pin numpy<2 (conflicts
with openpi's numpy 2.x). We stub those modules so nothing gets installed/downgraded.
"""

import concurrent.futures
import contextlib
import importlib.util
import os
import sys
import types

import numpy as np
import torch

# This box has 240 cores, so torch (and XLA/BLAS) default their CPU thread pools to ~240 threads
# PER POOL PER PROCESS. With 8 JAX processes each loading DA3-GIANT and booting simultaneously, that
# thread explosion (on top of co-tenant jobs) drove pthread_create into EAGAIN and crashed ranks with
# an uncatchable C++ std::system_error. Cap torch's CPU threads — DA3 runs on the GPU, so this does
# not slow training. (Also honored by forked dataloader workers via OMP_NUM_THREADS in the launcher.)
try:
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "4")))
    torch.set_num_interop_threads(int(os.environ.get("TORCH_NUM_INTEROP_THREADS", "2")))
except Exception:  # noqa: BLE001 — set_num_interop_threads throws if pool already initialized; ignore
    pass

# Absolute paths to the DA3 code, overridable by env so this runs off the training box.
# DA3_SRC: the Depth-Anything-3 python `src` dir. DA3_GEOSTACK: the da3_for_geostack.py file.
_DA3_SRC = os.environ.get("DA3_SRC", "/work/jack/projects/Depth-Anything-3/src")
_GEOSTACK = os.environ.get("DA3_GEOSTACK", "/work/jack/da3xvla_src/DA3-XVLA-cache/models/da3_for_geostack.py")
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


class _AutoStub(types.ModuleType):
    """Module stub that returns a dummy callable for any non-dunder attribute + acts as a package."""

    __path__: list = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: None


def _install_stubs():
    for name in ("moviepy", "moviepy.editor", "gsplat", "pycolmap", "trimesh",
                 "depth_anything_3.utils.export", "depth_anything_3.utils.pose_align"):
        sys.modules.setdefault(name, _AutoStub(name))
    if hasattr(sys.modules["moviepy"], "__dict__"):
        sys.modules["moviepy"].editor = sys.modules["moviepy.editor"]
    if _DA3_SRC not in sys.path:
        sys.path.insert(0, _DA3_SRC)


def _load_da3_class():
    _install_stubs()
    spec = importlib.util.spec_from_file_location("_da3_for_geostack", _GEOSTACK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DA3LargeForGeoStack


def rescale_intrinsics(intr: np.ndarray, src_hw, dst_hw) -> np.ndarray:
    """Rescale pixel K [.,3,3] from src (H,W) to dst (H,W). fx,cx by W-ratio; fy,cy by H-ratio."""
    sh, sw = src_hw
    dh, dw = dst_hw
    rw, rh = dw / sw, dh / sh
    out = np.array(intr, dtype=np.float32, copy=True)
    out[..., 0, 0] *= rw
    out[..., 0, 2] *= rw
    out[..., 1, 1] *= rh
    out[..., 1, 2] *= rh
    return out


class DA3InlineExtractor:
    """Frozen DA3-GIANT posed multi-view extractor, REPLICATED one-per-GPU.

    Each visible GPU holds its own frozen DA3-GIANT copy and runs the forward on ONLY its slice of the
    batch, all GPUs concurrently (one thread per device; CUDA kernels are async per device, and torch
    releases the GIL during them). This removes the single-GPU serial bottleneck of the old design, so
    inline extraction scales with GPU count to match the JAX data-parallel training step.
    """

    def __init__(
        self,
        model_name: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        out_layers=(19, 26, 33, 39),
        da3_hw=(252, 336),
        devices=None,
        forward_chunk: int = 16,
    ):
        DA3 = _load_da3_class()
        if devices is None:
            # Default SINGLE-GPU (cuda:0): the multi-replica path is correct but thread-based, and the
            # Python GIL serializes the DA3 forward's many kernel launches (~18% concurrency efficiency),
            # so replicating across GPUs doesn't speed it up — it only wastes memory. Pass `devices`
            # explicitly (e.g. for a future multiprocess extractor) to override.
            devices = ["cuda:0" if torch.cuda.is_available() else "cpu"]
        self.devices = list(devices)
        self.da3_hw = tuple(da3_hw)
        self.forward_chunk = int(forward_chunk)
        self.replicas = []
        for dev in self.devices:
            m = (
                DA3(model_name=model_name, out_layers=tuple(out_layers),
                    da3_input_h=da3_hw[0], da3_input_w=da3_hw[1], patch_size=14, use_bf16=True)
                .to(dev)
                .eval()
            )
            for p in m.parameters():
                p.requires_grad_(False)
            self.replicas.append(m)
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(self.devices)))

    def _preprocess(self, images: np.ndarray, device) -> torch.Tensor:
        """images: [B,V,H,W,3] uint8 (or float [0,1]) -> [B,V,3,252,336] ImageNet-normalized on `device`."""
        x = torch.as_tensor(images, device=device)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.max() > 1.5:  # already float but in [0,255]
            x = x.float() / 255.0
        x = x.permute(0, 1, 4, 2, 3)  # [B,V,3,H,W]
        x = torch.nn.functional.interpolate(
            x.flatten(0, 1), size=self.da3_hw, mode="bicubic", align_corners=False, antialias=True
        ).view(*x.shape[:2], 3, *self.da3_hw)
        x = (x - _IMAGENET_MEAN.to(x)) / _IMAGENET_STD.to(x)
        return x

    def _run_shard(self, di: int, images: np.ndarray, extrinsics: np.ndarray | None, intrinsics: np.ndarray | None):
        """Run replica `di` over its (pre-sliced) shard, in chunks to bound activation memory.

        extrinsics/intrinsics may be None (UNPOSED mode, e.g. LIBERO) — DA3 then runs each view
        independently (monocular) and returns ray from its ray head; depth is None (zeroed below)."""
        dev = self.devices[di]
        dev_idx = int(dev.split(":")[1]) if ":" in dev else None
        replica = self.replicas[di]
        chunk = self.forward_chunk
        fc, rc, dc = [], [], []
        ctx = torch.cuda.device(dev_idx) if dev_idx is not None else contextlib.nullcontext()
        with ctx, torch.no_grad():  # set current device so implicit-device tensors land on the right GPU
            for i in range(0, images.shape[0], chunk):
                x = self._preprocess(images[i : i + chunk], dev)
                if extrinsics is None or intrinsics is None:
                    e = k = None
                else:
                    e = torch.as_tensor(extrinsics[i : i + chunk], device=dev, dtype=torch.float32)
                    k = torch.as_tensor(intrinsics[i : i + chunk], device=dev, dtype=torch.float32)
                out = replica.forward_multi_view(x, extrinsics=e, intrinsics=k)
                # Ship feats as bf16 BITS (uint16): the model casts to bf16 anyway (see
                # Pi0._compute_banks), so this is numerically identical to shipping f32 while
                # halving the GPU->CPU->GPU transfer and the collate copies.
                feats = torch.stack(list(out["feats"]), dim=1).to(torch.bfloat16)  # [b,4,V,C,h,w]
                ray = out["ray"].float()  # [b,V,3,h,w]
                depth = out["depth"]
                if depth is None:
                    depth = torch.zeros(ray.shape[0], ray.shape[1], 1, ray.shape[3], ray.shape[4], device=ray.device)
                fc.append(feats.view(torch.uint16).cpu().numpy())
                rc.append(ray.cpu().numpy())
                dc.append(depth.float().cpu().numpy())
        return np.concatenate(fc, 0), np.concatenate(rc, 0), np.concatenate(dc, 0)

    def extract(self, images: np.ndarray, extrinsics: np.ndarray | None, intrinsics: np.ndarray | None):
        """images [B,V,H,W,3]; extrinsics [B,V,4,4] w2c; intrinsics [B,V,3,3] AT 252x336.

        extrinsics/intrinsics may be None (UNPOSED mode) — DA3 runs each view monocularly.
        Splits the batch across all replicas/GPUs and runs the forwards concurrently.
        Returns numpy: feats [B,4,V,1536,18,24] f32, ray [B,V,3,18,24] f32, depth [B,V,1,18,24] f32.
        """
        b = int(images.shape[0])
        nd = len(self.devices)
        bounds = [round(i * b / nd) for i in range(nd + 1)]
        futs = {}
        for di in range(nd):
            s, e = bounds[di], bounds[di + 1]
            if s >= e:
                continue
            es = None if extrinsics is None else extrinsics[s:e]
            ks = None if intrinsics is None else intrinsics[s:e]
            futs[di] = self._pool.submit(self._run_shard, di, images[s:e], es, ks)
        parts = [futs[di].result() for di in sorted(futs)]
        return (
            np.concatenate([p[0] for p in parts], axis=0),
            np.concatenate([p[1] for p in parts], axis=0),
            np.concatenate([p[2] for p in parts], axis=0),
        )
