from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import queue
import threading
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import pickle

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
from openpi.training import da3_cache as _da3_cache
from openpi.training import da3_extractor as _da3_extractor
from openpi.training import fast_video as _fast_video
import openpi.training.config as _config

# Replace LeRobot's slow decode_video_frames (fresh torchvision.io.VideoReader per call, ~250ms) with
# a fast av.open path (~15ms). Applied at import so it takes effect in spawned data-loader workers too.
_fast_video.patch_lerobot()
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        # pyav backend avoids the torchcodec/FFmpeg .so requirement on this box.
        # torchcodec's shared libs need libavutil.so.{56,57,58,59} present.
        video_backend="pyav",
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


class DA3CachedDataset(Dataset[T_co]):
    """Joins precached DA3 geometry to each frame. Reads (episode, frame, poses) from the raw
    LeRobot sample, runs the openpi transform stack, then ATTACHES the DA3 arrays to the FINAL
    dict (so no intervening transform strips them). Restricts to episodes present in the cache.
    """

    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn], da3_cfg):
        self._base = dataset
        self._transform = _transforms.compose(transforms)
        self._vo = tuple(da3_cfg.view_order)
        self._reader = _da3_cache.DA3CacheReader(
            da3_cfg.root, da3_cfg.slug, da3_cfg.raw_root, da3_cfg.meta_path, da3_cfg.converter_path, self._vo
        )
        with open(da3_cfg.lang_cache, "rb") as f:
            self._lang = pickle.load(f)
        # restrict to cached episodes via LeRobot episode_data_index
        lr = dataset
        while hasattr(lr, "_dataset"):
            lr = lr._dataset
        edi = lr.episode_data_index
        # Train on ALL 25 fps frames of every cached episode; for a frame with no exact cache entry,
        # DA3CacheReader.load snaps to the nearest cached frame (<=1-frame mismatch — negligible, esp.
        # for the static scene view). Matches the inline behavior (every observation gets DA3).
        self._indices: list[int] = []
        n_eps = 0
        for ep in range(len(edi["from"])):
            if not self._reader.has_episode(ep):
                continue
            n_eps += 1
            self._indices.extend(range(int(edi["from"][ep]), int(edi["to"][ep])))
        if not self._indices:
            raise ValueError("DA3CachedDataset: no cached episodes present for this dataset.")
        logging.info("DA3CachedDataset: %d frames (snap to nearest cached DA3) across %d episodes", len(self._indices), n_eps)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, k: SupportsIndex) -> T_co:
        raw = self._base[self._indices[int(k)]]
        ep = int(np.asarray(raw["episode_index"]))
        fr = int(np.asarray(raw["frame_index"]))
        extr = _da3_cache.stack_extrinsics(raw, self._vo)
        feats, ray, depth = self._reader.load(ep, fr)
        task = raw.get("prompt", raw.get("task", ""))
        if isinstance(task, (list, tuple, np.ndarray)):
            task = task[0] if len(task) else ""
        entry = self._lang.get(str(task)) or next(iter(self._lang.values()))
        lf, lm = entry
        out = self._transform(raw)
        out["da3_features"] = feats  # uint8 fp8 bytes — decoded to f32 on GPU (4x less host transfer)
        out["da3_ray"] = ray.astype(np.float32)
        out["da3_depth"] = depth.astype(np.float32)
        out["camera_extrinsics"] = extr
        out["lang_feat"] = np.asarray(lf, np.float32)
        out["lang_mask"] = np.asarray(lm, bool)
        return out


class DA3InlineDataset(Dataset[T_co]):
    """Prepares raw DA3 inputs per frame for INLINE extraction (no precached features).

    Attaches the 3 raw camera frames + extrinsics + intrinsics (rescaled to DA3 res) + cached
    ModernBERT lang to the final sample. The frozen DA3-GIANT forward runs later, once per BATCH,
    in the data-loader batch hook (GPU). Works on every frame (no cache / no restrict).
    """

    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn], da3_cfg):
        self._base = dataset
        self._transform = _transforms.compose(transforms)
        self._vo = tuple(da3_cfg.view_order)
        self._da3_hw = tuple(da3_cfg.da3_hw)
        # image key template: "observation.images.{v}" (aloha/robotwin) or "{v}" (LIBERO)
        self._img_tmpl = getattr(da3_cfg, "image_key_tmpl", "observation.images.{v}")
        # posed=False (UNPOSED, e.g. LIBERO 2-cam): no camera extrinsics/intrinsics in the dataset;
        # DA3 runs monocularly and the spatial bank drops the ray embedding.
        self._posed = bool(getattr(da3_cfg, "posed", True))
        with open(da3_cfg.lang_cache, "rb") as f:
            self._lang = pickle.load(f)

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: SupportsIndex) -> T_co:
        raw = self._base[int(idx)]
        imgs = []
        for v in self._vo:
            im = np.asarray(raw[self._img_tmpl.format(v=v)])
            if im.ndim == 3 and im.shape[0] in (1, 3):  # CHW -> HWC
                im = np.transpose(im, (1, 2, 0))
            im = im.astype(np.float32)
            if im.max() > 1.5:  # uint8-range -> [0,1]
                im = im / 255.0
            imgs.append(im)
        da3_images = np.stack(imgs, axis=0)  # [V,H,W,3]
        task = raw.get("prompt", raw.get("task", ""))
        if isinstance(task, (list, tuple, np.ndarray)):
            task = task[0] if len(task) else ""
        entry = self._lang.get(str(task)) or next(iter(self._lang.values()))
        lf, lm = entry
        out = self._transform(raw)
        out["da3_images"] = da3_images.astype(np.float32)
        if self._posed:
            h, w = da3_images.shape[1:3]
            extr = _da3_cache.stack_extrinsics(raw, self._vo)  # [V,4,4]
            intr = np.stack(
                [np.asarray(raw[f"observation.{v}.intrinsic_cv"], np.float32).reshape(3, 3) for v in self._vo], axis=0
            )
            intr = _da3_extractor.rescale_intrinsics(intr, (h, w), self._da3_hw)  # [V,3,3] at DA3 res
            out["camera_extrinsics"] = extr
            out["camera_intrinsics"] = intr
        out["lang_feat"] = np.asarray(lf, np.float32)
        out["lang_mask"] = np.asarray(lm, bool)
        return out


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    tlist = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]
    if data_config.da3_cache is not None:
        if data_config.da3_cache.inline:
            return DA3InlineDataset(dataset, tlist, data_config.da3_cache)
        return DA3CachedDataset(dataset, tlist, data_config.da3_cache)
    return TransformedDataset(dataset, tlist)


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Inline DA3 extraction: build the frozen GIANT extractor ONCE (main process, GPU) + a per-batch
    # hook that turns the raw images + poses into feats/ray/depth just before sharding.
    batch_transform = None
    if data_config.da3_cache is not None and data_config.da3_cache.inline:
        c = data_config.da3_cache
        logging.info("Building inline DA3-GIANT extractor (%s) ...", c.da3_model)
        _extractor = _da3_extractor.DA3InlineExtractor(
            model_name=c.da3_model, out_layers=tuple(c.da3_out_layers), da3_hw=tuple(c.da3_hw)
        )

        _posed = bool(getattr(c, "posed", True))

        def batch_transform(batch, _ex=_extractor, _posed=_posed):
            if _posed:
                feats, ray, depth = _ex.extract(
                    batch["da3_images"], batch["camera_extrinsics"], batch["camera_intrinsics"]
                )
                batch["da3_features"] = feats
                batch["da3_ray"] = ray
                batch["da3_depth"] = depth
                batch.pop("camera_intrinsics", None)
            else:
                # UNPOSED: no extrinsics/intrinsics; DA3 runs monocular. Only feats + lang feed the
                # (ray-free) spatial bank, so ray/depth/extrinsics are dropped entirely.
                feats, _ray, _depth = _ex.extract(batch["da3_images"], None, None)
                batch["da3_features"] = feats
            batch.pop("da3_images", None)
            return batch

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()
        # JAX multi-process (one process per GPU): each process must read a DISJOINT shard of the
        # data so make_array_from_process_local_data assembles a correct global batch.
        if jax.process_count() > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=jax.process_count(),
                rank=jax.process_index(),
                shuffle=shuffle,
                seed=seed,
                drop_last=True,
            )

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
        batch_transform=batch_transform,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
        batch_transform=None,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        # Multi-process (one process per GPU) IS supported: each process feeds its local shard via a
        # DistributedSampler (see create_torch_data_loader) + make_array_from_process_local_data.
        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches
        # optional per-batch hook applied to the (numpy) batch before sharding — used for inline
        # DA3 feature extraction (runs the frozen GIANT on the batch's raw images on GPU).
        self._batch_transform = batch_transform

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            # Deeper prefetch so short CPU/bandwidth-contention bursts from co-located jobs don't
            # drain the buffer and starve the GPUs (io-wait is ~0, so the limiter is worker supply).
            prefetch_factor=(4 if num_workers > 0 else None),
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def _transformed_batches(self):
        """Yields batch_transform-ed batches, looping over the dataset indefinitely.

        When a batch_transform is set (inline DA3 GPU extraction), it runs ONE BATCH AHEAD in a
        background thread, so the ~seconds-long frozen DA3 forward for batch N+1 overlaps the JAX
        train step for batch N instead of serializing with it. torch GPU kernels release the GIL,
        so the two genuinely interleave on the device.
        """
        def epochs():
            """Iterate the torch loader indefinitely, advancing the DistributedSampler epoch so each
            pass uses a fresh shuffle (without set_epoch every epoch replays the same order)."""
            epoch = 0
            sampler = getattr(self._data_loader, "sampler", None)
            while True:
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
                yield from self._data_loader
                epoch += 1

        if self._batch_transform is None:
            yield from epochs()
            return

        # maxsize=3: one batch being produced + two buffered, so the roving per-step scheduling
        # jitter between the DA3-forward (producer) and JAX-step (consumer) sharing the GPU can't
        # stall the consumer. Bigger buffers don't add GPU throughput (compute-bound) — this only
        # smooths the small bubble; costs ~1 extra transformed batch (~0.4 GB uint16 feats) in host RAM.
        q: queue.Queue = queue.Queue(maxsize=3)

        def producer():
            try:
                for raw in epochs():
                    q.put(self._batch_transform(raw))
            except BaseException as e:  # noqa: BLE001 — propagate any failure to the consumer
                q.put(e)

        threading.Thread(target=producer, daemon=True, name="batch-transform-prefetch").start()
        while True:
            item = q.get()
            if isinstance(item, BaseException):
                raise item
            yield item

    def __iter__(self):
        num_items = 0
        batches = self._transformed_batches()
        while True:
            # Check BEFORE pulling: the generator prefetches, and pulling past num_batches would
            # trigger (and discard) a whole extra batch across the epoch boundary.
            if self._num_batches is not None and num_items >= self._num_batches:
                return
            batch = next(batches)
            num_items += 1
            # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
            if self._sharding is not None:
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
            else:
                yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
