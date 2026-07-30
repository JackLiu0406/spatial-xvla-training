"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DA3CacheConfig:
    """DA3 geometry source for the pi0.5-DA3 pipeline: precached (default) or INLINE extraction.

    inline=False: load precached feats/ray/depth from `root` (path-joined by HDF5).
    inline=True:  run frozen DA3-GIANT on the raw camera frames at train time (no cache needed —
                  essential where caching is infeasible, e.g. b1k at 30 Hz ~= 3 PB). Only the
                  ModernBERT language stays cached (`lang_cache`).
    """

    root: str | None = None  # cached mode: e.g. /work/jack/da3_cache/roboreal_giant_135hz_fp8
    slug: str = "robotwin2_clean"
    raw_root: str = "/work/jack/datasets/roboreal_raw"
    meta_path: str = "/work/jack/_meta_all_ct_3cam_clean.json"
    converter_path: str = "/work/jack/spatial-starvla-103/scripts/convert_roboreal_merged_poses.py"
    view_order: tuple[str, ...] = ("countertop", "left", "right")
    # How the DA3 wrapper builds the raw image key for each view. Aloha/RoboTwin
    # LeRobot datasets store views as "observation.images.<view>"; LIBERO stores
    # them as bare "image"/"wrist_image", so set image_key_tmpl="{v}" and
    # view_order=("image","wrist_image") there.
    image_key_tmpl: str = "observation.images.{v}"
    lang_cache: str = "/work/jack/da3_cache/modernbert_camext_lang.pkl"
    lang_max_len: int = 64
    # posed=True: read per-view camera extrinsics/intrinsics and run DA3 posed multi-view.
    # posed=False (UNPOSED, e.g. LIBERO 2-cam): no camera calibration in the dataset — skip the
    # extrinsic_cv/intrinsic_cv reads, run DA3 monocularly, feed only feats + lang to the model.
    posed: bool = True
    # Correct a channel-swapped dataset AT LOAD TIME. The RoboReal LeRobot videos were built by a
    # converter that stored frames BGR-swapped (beige→blue when viewed), so the frozen DA3-GIANT and
    # SigLIP were being fed effectively-BGR images. Set True for those datasets: the raw camera frames
    # are reversed on the channel axis in DA3InlineDataset — applied ONCE, before both the DA3 and the
    # pi0.5 image paths read them, so they stay consistent. Leave False for datasets already stored as
    # correct RGB (incl. anything built with the fixed converter, which no longer swaps).
    bgr_to_rgb: bool = False
    # --- inline extraction ---
    inline: bool = False
    da3_model: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    da3_out_layers: tuple[int, ...] = (19, 26, 33, 39)
    da3_hw: tuple[int, int] = (252, 336)  # DA3 input resolution
    native_hw: tuple[int, int] = (240, 320)  # camext video resolution (for intrinsic rescale)


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # DA3 feature-cache join (None == stock pipeline).
    da3_cache: DA3CacheConfig | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    # If true, will use the LeRobot dataset task string per-episode as the prompt.
    # (Added for multi-task RoboPRO office subset — each episode has its own task
    # instruction that should be used instead of a single default_prompt.)
    prompt_from_task: bool = False

    # DA3 feature-cache join (None == stock pipeline).
    da3_cache: DA3CacheConfig | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            prompt_from_task=self.prompt_from_task,
            da3_cache=self.da3_cache,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False
    # DA3 feature-cache join (None == stock pipeline). For the pi0.5-DA3 LIBERO
    # ablation: DA3CacheConfig(inline=True, posed=False, view_order=("image","wrist_image"),
    # image_key_tmpl="{v}", da3_hw square, lang_cache=libero pkl).
    da3_cache: DA3CacheConfig | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            da3_cache=self.da3_cache,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    # Optional per-parameter-group LR schedules (maps group name -> schedule). When set, the trainer
    # builds an optax.multi_transform keyed by openpi.training.optimizer.spatial_group_labels instead of
    # the single `lr_schedule`. Used by pi0.5-DA3 (vlm / core / geom on distinct LRs).
    lr_groups: tyro.conf.Suppress[dict[str, _optimizer.LRScheduleConfig] | None] = None
    # Per-group AdamW eps. Required for the DA3 spatial branch, whose gradients sit 3-5 orders of
    # magnitude below the 1e-8 default and are therefore swallowed by Adam's epsilon floor -- see
    # create_multi_group_optimizer for the measurements. None = use the optimizer's own eps.
    lr_group_eps: tyro.conf.Suppress[dict[str, float] | None] = None
    # Per-group AdamW weight decay (FIX 3). The 1e-10 default is effectively no decay, which is what
    # allowed the spatial attention projections to run away into softmax saturation. None = default.
    lr_group_weight_decay: tyro.conf.Suppress[dict[str, float] | None] = None
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # π0.5 finetune on the RoboPRO "office" subset (3,393 episodes from the merged
    # Hoshipu-RoboPro roboreal_all_80tasks dataset — office scene only).
    # Dataset lives at $HF_LEROBOT_HOME/roboreal_office_lerobot/ built via
    # /work/jack/openpi_src/build_office_lerobot.py.
    # Camera keys in our data are observation.images.{cam_high,cam_left_wrist,cam_right_wrist}
    # (no cam_low — Aloha-Agilex has only 3 cams). The RepackTransform below
    # maps our LeRobot keys to the AlohaInputs.EXPECTED_CAMERAS names.
    TrainConfig(
        # π0.5 finetune on RoboPRO office subset. Mirrors openpi's documented
        # finetune recipe pi05_aloha_pen_uncap:
        #   num_train_steps = 20_000
        #   batch_size      = 64 (global)   ← we use 96 (24/GPU × 4) per user
        #   lr_schedule     = default CosineDecaySchedule (warmup 1k, peak 2.5e-5, decay 30k → 2.5e-6)
        #   optimizer       = default AdamW (b1=0.9, b2=0.95, wd=1e-10, clip 1.0)
        #   ema_decay       = 0.99
        #   init            = pi05_base GCS checkpoint (documented starting point)
        name="pi05_roboreal_office",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="roboreal_office_lerobot",
            assets=AssetsConfig(asset_id="roboreal_office"),
            adapt_to_pi=False,     # our data is already in a common Aloha space
            use_delta_joint_actions=True,
            prompt_from_task=True, # use each episode's LeRobot task string as the prompt
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high":         "observation.images.cam_high",
                                "cam_left_wrist":   "observation.images.cam_left_wrist",
                                "cam_right_wrist":  "observation.images.cam_right_wrist",
                            },
                            "state":   "observation.state",
                            "actions": "action",
                            "prompt":  "prompt",   # preserve per-task prompt through repack (prompt_from_task)
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=20_000,     # docs recipe (pi05_aloha_pen_uncap)
        batch_size=96,              # 24/GPU × 4 H200s — fits (~108GB/GPU); docs default is 64 but we have fewer, larger GPUs
        num_workers=16,             # default 2 starves 4 H200s on video-decode (6s/it); 16 feeds them
        wandb_enabled=False,        # headless box, no wandb key → disable (else login error kills run)
        # lr_schedule / optimizer / ema_decay all use openpi's defaults (== docs recipe)
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # π0.5 finetune on the FULL RoboPRO 16k-episode dataset (all 80 tasks, all
    # 4 scenes). Same recipe as pi05_roboreal_office; different asset_id so its
    # norm stats live at ./assets/roboreal_full/norm_stats.msgpack.
    # Dataset symlinked at $HF_LEROBOT_HOME/roboreal_full_lerobot →
    # DATASETS/Hoshipu-RoboPro/lerobot/roboreal_all_80tasks.
    TrainConfig(
        name="pi05_roboreal_full",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="roboreal_full_lerobot",
            assets=AssetsConfig(asset_id="roboreal_full_lerobot"),
            adapt_to_pi=False,
            use_delta_joint_actions=True,
            prompt_from_task=True,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                # camext dataset (roboreal_lerobot_camext) uses left/right/countertop keys.
                                # Model slots keep Aloha names: cam_high == countertop (static), wrists == left/right.
                                "cam_high":         "observation.images.countertop",
                                "cam_left_wrist":   "observation.images.left",
                                "cam_right_wrist":  "observation.images.right",
                            },
                            "state":   "observation.state",
                            "actions": "action",
                            "prompt":  "prompt",   # preserve per-task prompt through repack (prompt_from_task)
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=20_000,
        batch_size=96,              # 24/GPU × 4 H200s — fits (~108GB/GPU); docs default is 64 but we have fewer, larger GPUs
        num_workers=16,             # default 2 starves 4 H200s on video-decode (6s/it); 16 feeds them
        wandb_enabled=False,        # headless box, no wandb key → disable (else login error kills run)
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    # pi0.5 + DA3 spatial-language branch. Frozen DA3(GIANT)+ModernBERT features are precached
    # offline (/work/jack/da3_cache) so nothing PyTorch enters the JAX graph — only the JAX
    # spatial projectors/perceiver/fusion + the cross-attn injection into action-expert blocks 12-17.
    TrainConfig(
        name="pi05_roboreal_full_da3",
        model=pi0_config.Pi0Config(pi05=True, da3=pi0_config.Pi0DA3Config(enabled=True)),
        data=LeRobotAlohaDataConfig(
            repo_id="roboreal_full_lerobot",
            assets=AssetsConfig(asset_id="roboreal_full_lerobot"),
            adapt_to_pi=False,
            use_delta_joint_actions=True,
            prompt_from_task=True,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.countertop",
                                "cam_left_wrist": "observation.images.left",
                                "cam_right_wrist": "observation.images.right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            da3_cache=DA3CacheConfig(root="/work/jack/da3_cache/roboreal_giant_135hz_fp8"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=r".*lora.*|.*spatial_bank_builder.*|.*spatial_inject.*",
        ),
        # Per-group LRs: VLM+expert on pi0.5's finetune recipe; spatial branch higher (fresh modules);
        # geometry encoder highest (X-VLA-style core/geom split). DA3+ModernBERT are cached => frozen.
        lr_groups={
            "vlm": _optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=50_000, decay_lr=2.5e-6),
            "core": _optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=1e-4, decay_steps=50_000, decay_lr=1e-5),
            "geom": _optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=2e-4, decay_steps=50_000, decay_lr=2e-5),
        },
        num_train_steps=50_000,
        batch_size=128,             # try 128; fall back to 96/64 on OOM (headroom exists at bs=32=8/GPU)
        num_workers=32,             # more supply to sustain the ~2.6s/it floor under node contention (io-wait~0)
        wandb_enabled=False,
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    # Same as pi05_roboreal_full_da3 but DA3 features are extracted INLINE (no cache) — the frozen
    # GIANT runs on the raw camera frames each step. Needed where caching is infeasible (b1k ~= 3 PB).
    TrainConfig(
        name="pi05_roboreal_full_da3_inline",
        # spatial_init_std=0.01: small (non-zero) init on the injection out-projections — a tiny
        # spatial delta at step 0 instead of exact pi05_base identity, so all injection weights
        # receive gradient immediately (user-requested).
        model=pi0_config.Pi0Config(pi05=True, da3=pi0_config.Pi0DA3Config(enabled=True, spatial_init_std=0.01)),
        data=LeRobotAlohaDataConfig(
            repo_id="roboreal_full_lerobot",
            assets=AssetsConfig(asset_id="roboreal_full_lerobot"),
            adapt_to_pi=False,
            use_delta_joint_actions=True,
            prompt_from_task=True,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.countertop",
                                "cam_left_wrist": "observation.images.left",
                                "cam_right_wrist": "observation.images.right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            da3_cache=DA3CacheConfig(inline=True, bgr_to_rgb=True),  # roboreal videos are BGR-swapped -> correct at load; extract on the fly
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=r".*lora.*|.*spatial_bank_builder.*|.*spatial_inject.*",
        ),
        lr_groups={
            # vlm = openpi's documented pi0.5 finetune recipe, stretched to the 50k horizon.
            "vlm": _optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=50_000, decay_lr=2.5e-6),
            # core = the FRESH spatial modules only (bank builder + injection xattn, 421M) — the
            # pretrained VLM+action-expert are in "vlm" on the pi0.5 docs recipe. 5e-4 per user
            # (X-VLA used 1e-4 for its fresh modules; warmup+clip+small-init make 5e-4 safe to try).
            "core": _optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=5e-4, decay_steps=50_000, decay_lr=5e-5),
            # geom (ray MLP, 0.3M fresh params): X-VLA used FLAT 2e-4; we start 2.5x higher (fresh-MLP
            # mid-band, its grad is attenuated by the small injection projections early) and decay only
            # to 1e-4 so late-training stays near X-VLA's validated flat-2e-4 regime.
            "geom": _optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=5e-4, decay_steps=50_000, decay_lr=1e-4),
        },
        num_train_steps=50_000,
        batch_size=128,
        num_workers=8,
        wandb_enabled=False,
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    # v2: identical recipe to _inline but with the attention-learning fixes ON (clean A/B).
    TrainConfig(
        name="pi05_roboreal_full_da3_inline_v2",
        # spatial_init_std=0.01: small (non-zero) init on the injection out-projections — a tiny
        # spatial delta at step 0 instead of exact pi05_base identity, so all injection weights
        # receive gradient immediately (user-requested).
        model=pi0_config.Pi0Config(pi05=True, da3=pi0_config.Pi0DA3Config(
            enabled=True, spatial_init_std=0.01,
            # v2 attention-learning fixes (Q/K trained ~1000x slower than V/out in v1):
            attn_logit_gain=True, attn_logit_gain_init=32.0, bank_token_embed=True, perceiver_query_std=0.05,
        )),
        data=LeRobotAlohaDataConfig(
            repo_id="roboreal_full_lerobot",
            assets=AssetsConfig(asset_id="roboreal_full_lerobot"),
            adapt_to_pi=False,
            use_delta_joint_actions=True,
            prompt_from_task=True,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.countertop",
                                "cam_left_wrist": "observation.images.left",
                                "cam_right_wrist": "observation.images.right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            da3_cache=DA3CacheConfig(inline=True, bgr_to_rgb=True),  # roboreal videos are BGR-swapped -> correct at load; extract on the fly
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=r".*lora.*|.*spatial_bank_builder.*|.*spatial_inject.*",
        ),
        lr_groups={
            # vlm = openpi's documented pi0.5 finetune recipe, stretched to the 50k horizon.
            "vlm": _optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=50_000, decay_lr=2.5e-6),
            # core = the FRESH spatial modules only (bank builder + injection xattn, 421M) — the
            # pretrained VLM+action-expert are in "vlm" on the pi0.5 docs recipe. 5e-4 per user
            # (X-VLA used 1e-4 for its fresh modules; warmup+clip+small-init make 5e-4 safe to try).
            "core": _optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=5e-4, decay_steps=50_000, decay_lr=5e-5),
            # geom (ray MLP, 0.3M fresh params): X-VLA used FLAT 2e-4; we start 2.5x higher (fresh-MLP
            # mid-band, its grad is attenuated by the small injection projections early) and decay only
            # to 1e-4 so late-training stays near X-VLA's validated flat-2e-4 regime.
            "geom": _optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=5e-4, decay_steps=50_000, decay_lr=1e-4),
        },
        num_train_steps=50_000,
        batch_size=128,
        num_workers=8,
        wandb_enabled=False,
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    # ===== RoboTwin2.0-aloha benchmark (head=main, left/right wrists; 14-dim bimanual) =====
    # pi0.5 BASE (no DA3). Mirrors pi05_roboreal_full on the RoboTwin2 LeRobot dataset.
    # ===== RoboDyna suite: vanilla pi0.5, 16 tasks x 4 setups merged (3200 demos) =====
    # Merged from Hoshipu/robodyna-lerobot-suite by scripts/merge_robodyna.py into the canonical
    # LeRobot v2.1 layout at $HF_LEROBOT_HOME/robodyna_merged. aloha-agilex, 14-dim state/action,
    # 3 cams (head/left_wrist/right_wrist @ 240x320), 16 unique task prompts. 4-GPU, bs=256, 40k.
    TrainConfig(
        name="pi05_robodyna_full",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="robodyna_merged",
            assets=AssetsConfig(asset_id="robodyna_merged"),
            adapt_to_pi=False,
            use_delta_joint_actions=True,
            prompt_from_task=True,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.head",
                                "cam_left_wrist": "observation.images.left_wrist",
                                "cam_right_wrist": "observation.images.right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # 40k x 256 = 10.24M samples over 801k frames ~= 12.8 epochs.
        num_train_steps=40_000,
        # 256 = 64/GPU x 4. NOTE: robotwin2 measured 32/GPU (bs128) at ~123GB with NO DA3; this is
        # also DA3-free so the VLM+action-expert footprint is the same, but 64/GPU ~ 2x activations.
        # If it OOMs, drop to 192 (48/GPU) or 128 and lengthen num_train_steps to keep the epoch count.
        batch_size=256,
        num_workers=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=40_000, decay_lr=2.5e-6
        ),
        wandb_enabled=False,
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_robotwin2_full",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="robotwin2_aloha_lerobot",
            assets=AssetsConfig(asset_id="robotwin2_aloha_lerobot"),
            adapt_to_pi=False,
            use_delta_joint_actions=True,
            prompt_from_task=True,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.countertop",
                                "cam_left_wrist": "observation.images.left",
                                "cam_right_wrist": "observation.images.right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # EXTENDED 30k -> 44k for the +20k underfitting finetune (run stopped at 24k).
        # decay_steps is stretched WITH num_train_steps, which is the whole point: under
        # the old 30k schedule the LR at step 24k is 4.8e-6 (nearly floored), so 20k more
        # steps would barely move the weights. Stretching the cosine to 44k makes the LR
        # at the 24k resume point 1.25e-5 (~2.6x higher) and anneals it back to the 2.5e-6
        # floor by 44k — a proper warm restart rather than 20k steps of noise.
        # Config NAME must stay the same or --resume cannot find checkpoints/<config>/<exp>.
        num_train_steps=44_000,
        # 128 (32/GPU x 4) to MATCH pi05_robotwin2_full_da3_inline_v2 — this run is
        # that config's A/B baseline, so batch must not be a confound.
        # MEASURED: 128 uses ~123.2GB of 143.8GB/GPU. 196 (49/GPU) would need ~166GB => OOM.
        batch_size=128,
        num_workers=8,
        # Peak 2.5e-5 is the documented pi0.5 finetune recipe and equals the DA3 config's
        # "vlm" peak, so the A/B stays honest. Full finetune: freeze_filter is left at the
        # default (nnx.Nothing) => every param trains.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=44_000, decay_lr=2.5e-6
        ),
        wandb_enabled=False,
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    # pi0.5 + DA3 (tuned-v2 recipe). Mirrors pi05_roboreal_full_da3_inline_v2 (inline GIANT + logit-gain).
    TrainConfig(
        name="pi05_robotwin2_full_da3_inline_v2",
        model=pi0_config.Pi0Config(pi05=True, da3=pi0_config.Pi0DA3Config(
            enabled=True, spatial_init_std=0.01,
            attn_logit_gain=True, attn_logit_gain_init=32.0, bank_token_embed=True, perceiver_query_std=0.05,
        )),
        data=LeRobotAlohaDataConfig(
            repo_id="robotwin2_aloha_lerobot",
            assets=AssetsConfig(asset_id="robotwin2_aloha_lerobot"),
            adapt_to_pi=False,
            use_delta_joint_actions=True,
            prompt_from_task=True,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.countertop",
                                "cam_left_wrist": "observation.images.left",
                                "cam_right_wrist": "observation.images.right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            # RoboTwin2 tasks need their OWN ModernBERT lang cache (built by precompute over the
            # robotwin2 tasks.jsonl) — else inline DA3 falls back to a constant lang embedding.
            da3_cache=DA3CacheConfig(inline=True, bgr_to_rgb=True, lang_cache="/work/jack/da3_cache/modernbert_robotwin2_lang.pkl"),  # robotwin2 videos are BGR-swapped -> correct at load
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=r".*lora.*|.*spatial_bank_builder.*|.*spatial_inject.*",
        ),
        # EXTENDED 50k -> 70k for the +20k underfitting finetune (run finished at 50k).
        # All three groups' decay_steps stretch with num_train_steps. At 50k the OLD
        # schedule had every group sitting exactly ON its floor (core 5e-5 / geom 1e-4 /
        # vlm 2.5e-6), so resuming as-is would spend 20k steps at floor LR and learn almost
        # nothing. Stretching to 70k lifts the 50k resume point to core 1.37e-4, geom
        # 1.77e-4, vlm 6.85e-6 and anneals each back to its ORIGINAL floor by 70k.
        # Config NAME must stay the same or --resume cannot find checkpoints/<config>/<exp>.
        lr_groups={
            "vlm": _optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=70_000, decay_lr=2.5e-6),
            "core": _optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=5e-4, decay_steps=70_000, decay_lr=5e-5),
            "geom": _optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=5e-4, decay_steps=70_000, decay_lr=1e-4),
        },
        num_train_steps=70_000,
        # MEASURED: 128 (32/GPU) already peaks at ~140.2GB of 143.8GB/GPU with inline
        # DA3-GIANT resident - only ~3.6GB headroom. This is the CEILING; do not raise.
        batch_size=128,
        num_workers=8,
        wandb_enabled=False,
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    # pi0.5 + DA3 "v3" — v2 recipe PLUS the two perceiver-collapse fixes. FRESH RETRAIN, not a
    # resume: the fixes add params (perceiver log_gain + 2 LayerNorms per view), so v2 checkpoints
    # cannot be restored into this tree. New config NAME keeps the v2 checkpoints loadable.
    #
    # WHY (measured, not speculative):
    #   * The perceiver's latent queries NEVER trained — std still == init 0.05 with pairwise
    #     cos ~0 after 50k steps (pi05-DA3) and 75k (X-VLA-DA3). They are frozen at random init.
    #   * X-VLA-DA3's MAIN bank collapsed: its 128 tokens sit at pairwise cosine 1.0000 (left/right
    #     survived at ~0.92), so the injection attends UNIFORMLY and geometry contributes only a
    #     global constant, not "where" anything is.
    #   * Cause: the perceiver's own cross-attn has no logit gain, so with untrained queries its
    #     softmax over ~432 grid tokens is near-uniform => every query gets the same mean(V);
    #     uniform attention also scales the softmax jacobian ~1/432, starving the Q/K grads so the
    #     queries can never escape. The INJECTION got the gain fix in v2; the PERCEIVER never did.
    #   * Compounding it, the residual added RAW q: ||q||~1.6 vs ||attn_out||~500 => query identity
    #     swamped ~300:1.
    # FIXES: perceiver_logit_gain_init=32.0 (same as the injection) + perceiver_balanced_residual
    # (LayerNorm both sides of the query residual). Synthetic check: output token cosine
    # 0.350 -> 0.023 (pi05) and 0.543 -> 0.037 (X-VLA). Everything else is identical to v2.
    TrainConfig(
        name="pi05_robotwin2_full_da3_inline_v3",
        model=pi0_config.Pi0Config(pi05=True, da3=pi0_config.Pi0DA3Config(
            enabled=True, spatial_init_std=0.01,
            # ACTION-EXPERT (injection) gain: init 16, HARD-CAPPED at 32 (=2x init) by the clamp in
            # gemma.py. v2 used an uncapped 32; an unbounded exp-parameterized gain is exactly what
            # blew the perceiver's copy up to NaN at step 100, so this one is now bounded too.
            attn_logit_gain=True, attn_logit_gain_init=16.0, attn_logit_gain_max=32.0,
            bank_token_embed=True, perceiver_query_std=0.05,
            # perceiver_logit_gain_init DISABLED: init 32 on the perceiver (432 keys) drove
            # max|logit| to ~168 and the run went NaN by step 100 (reproduced twice, and with
            # DLPack off, so it is the gain -- not the data path). balanced_residual alone
            # targets the MEASURED root cause (||attn_out||~875-1200 swamping ||q||=1.6 by
            # ~550:1) without adding an unbounded exponentiated parameter. Re-enable only with
            # a much smaller init (<=4) now that the gain is clamped.
            # PERCEIVER FIXES, b1k-style (same architecture in both codebases):
            #   norm_attn_out  -> LN the ATTENTION OUTPUT before the residual (raw query preserved).
            #       Fixes the measured collapse (||attn_out||~500-1200 vs ||q||~1.6 => bank tokens at
            #       cosine 1.0000). My earlier version LN'd the QUERY too and NaN'd at step 100:
            #       LN on a std-0.05 tensor amplifies its gradient 1/std (~20x, unbounded). Never do that.
            #   logit_gain 8 (cap 16) -> sharpen the perceiver's own softmax over 432 patches so Q/K
            #       actually train. 32 was measured at max|logit| ~168 and diverges; 8 is b1k's value.
            perceiver_logit_gain=True, perceiver_logit_gain_init=8.0, perceiver_logit_gain_max=16.0,
            perceiver_norm_attn_out=True,
        )),
        data=LeRobotAlohaDataConfig(
            repo_id="robotwin2_aloha_lerobot",
            assets=AssetsConfig(asset_id="robotwin2_aloha_lerobot"),
            adapt_to_pi=False,
            use_delta_joint_actions=True,
            prompt_from_task=True,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.countertop",
                                "cam_left_wrist": "observation.images.left",
                                "cam_right_wrist": "observation.images.right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            da3_cache=DA3CacheConfig(inline=True, bgr_to_rgb=True, lang_cache="/work/jack/da3_cache/modernbert_robotwin2_lang.pkl"),  # robotwin2 videos are BGR-swapped -> correct at load
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=r".*lora.*|.*spatial_bank_builder.*|.*spatial_inject.*",
        ),
        # Fresh 50k schedule, STAGED warmup (user-tuned):
        #   vlm/core warm up fast (500 steps) — they are pretrained and stable.
        #   geom warms up 3x SLOWER (1500 steps) — the spatial modules are fresh AND, with the
        #   perceiver fixes, now emit genuinely varied geometry for the first time. Letting the
        #   core settle first before geometry reaches full strength is the same staging that
        #   rescued the X-VLA run (whose failure was geometry hitting peak LR too abruptly).
        # Peaks are slightly hotter and floors slightly cooler than v2 => wider dynamic range.
        lr_groups={
            "vlm": _optimizer.CosineDecaySchedule(warmup_steps=500, peak_lr=3e-5, decay_steps=50_000, decay_lr=2e-6),
            "core": _optimizer.CosineDecaySchedule(warmup_steps=500, peak_lr=6e-4, decay_steps=50_000, decay_lr=4e-5),
            "geom": _optimizer.CosineDecaySchedule(warmup_steps=1_500, peak_lr=5e-4, decay_steps=50_000, decay_lr=5e-5),
        },
        # THE fix for the perceiver never training. Adam's update is mu/(sqrt(nu)+eps) and is
        # scale-invariant only while sqrt(nu) >> eps. MEASURED at v3 step 6000 (and v2 step 49999,
        # identically): every spatial param sits 3-5 orders of magnitude UNDER the 1e-8 default --
        # perceiver query sqrt(nu)=2.6e-11, its q_proj 1.3e-12, the injection log_gain 1.7e-13 --
        # while the vlm backbone sits at 2.9e-07. So eps dominated the denominator and the spatial
        # branch received |update| ~1e-4..1e-6 against the vlm's 1.2e-1: frozen at init, exactly the
        # "collapse" signature (query std still == 0.05 at 50k). The tiny gradients are expected
        # (spatial_init_std=0.01 out-projections x softmax averaging over 432 patches); normalizing
        # them away is Adam's job, and it can only do it once eps stops swallowing them.
        # vlm keeps 1e-8 -- unaffected either way, and no reason to perturb a working group.
        lr_group_eps={"core": 1e-16, "geom": 1e-16},
        num_train_steps=50_000,
        # 128 = 32/GPU x 4. MEASURED ceiling: ~140.2GB of 143.8GB with inline DA3-GIANT. Do not raise.
        batch_size=128,
        num_workers=8,
        # Checkpoint every 2k; multiples of 10k are retained long-term (others GC'd once superseded).
        save_interval=2_000,
        keep_period=10_000,
        wandb_enabled=False,
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),

    TrainConfig(
        name="pi05_robotwin2_full_da3_inline_v4",
        model=pi0_config.Pi0Config(pi05=True, da3=pi0_config.Pi0DA3Config(
            enabled=True, spatial_init_std=0.01,
            # ACTION-EXPERT (injection) gain: init 16, HARD-CAPPED at 32 (=2x init) by the clamp in
            # gemma.py. v2 used an uncapped 32; an unbounded exp-parameterized gain is exactly what
            # blew the perceiver's copy up to NaN at step 100, so this one is now bounded too.
            # FIX 1: QK-norm on BOTH attentions (perceiver and injection). Bounds |logits| to
        # ~sqrt(head_dim) structurally instead of hoping the projections stay small.
        attn_qk_norm=True, perceiver_qk_norm=True,
        # FIX 7: keep per-patch magnitude through the layer projectors (see pi0_config).
        fuse_proj_norm=False,
        # FIX 2: with QK-norm the logits are O(1), so the gain is finally what it was meant to be --
        # a learned TEMPERATURE, not an amplifier on a runaway. v3's init of 16/8 was tuned for the
        # opposite regime (logits ~0, attention uniform) and is actively harmful on bounded logits.
        # Start neutral at 1.0 and let it learn its own sharpness; cap well below the old values.
        attn_logit_gain=True, attn_logit_gain_init=1.0, attn_logit_gain_max=8.0,
            bank_token_embed=True, perceiver_query_std=0.05,
            # perceiver_logit_gain_init DISABLED: init 32 on the perceiver (432 keys) drove
            # max|logit| to ~168 and the run went NaN by step 100 (reproduced twice, and with
            # DLPack off, so it is the gain -- not the data path). balanced_residual alone
            # targets the MEASURED root cause (||attn_out||~875-1200 swamping ||q||=1.6 by
            # ~550:1) without adding an unbounded exponentiated parameter. Re-enable only with
            # a much smaller init (<=4) now that the gain is clamped.
            # PERCEIVER FIXES, b1k-style (same architecture in both codebases):
            #   norm_attn_out  -> LN the ATTENTION OUTPUT before the residual (raw query preserved).
            #       Fixes the measured collapse (||attn_out||~500-1200 vs ||q||~1.6 => bank tokens at
            #       cosine 1.0000). My earlier version LN'd the QUERY too and NaN'd at step 100:
            #       LN on a std-0.05 tensor amplifies its gradient 1/std (~20x, unbounded). Never do that.
            #   logit_gain 8 (cap 16) -> sharpen the perceiver's own softmax over 432 patches so Q/K
            #       actually train. 32 was measured at max|logit| ~168 and diverges; 8 is b1k's value.
            perceiver_logit_gain=True, perceiver_logit_gain_init=1.0, perceiver_logit_gain_max=8.0,
            perceiver_norm_attn_out=True,
        )),
        data=LeRobotAlohaDataConfig(
            repo_id="robotwin2_aloha_lerobot",
            assets=AssetsConfig(asset_id="robotwin2_aloha_lerobot"),
            adapt_to_pi=False,
            use_delta_joint_actions=True,
            prompt_from_task=True,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.countertop",
                                "cam_left_wrist": "observation.images.left",
                                "cam_right_wrist": "observation.images.right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            da3_cache=DA3CacheConfig(inline=True, bgr_to_rgb=True, lang_cache="/work/jack/da3_cache/modernbert_robotwin2_lang.pkl"),  # robotwin2 videos are BGR-swapped -> correct at load
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=r".*lora.*|.*spatial_bank_builder.*|.*spatial_inject.*",
        ),
        # Fresh 50k schedule, STAGED warmup (user-tuned):
        #   vlm/core warm up fast (500 steps) — they are pretrained and stable.
        #   geom warms up 3x SLOWER (1500 steps) — the spatial modules are fresh AND, with the
        #   perceiver fixes, now emit genuinely varied geometry for the first time. Letting the
        #   core settle first before geometry reaches full strength is the same staging that
        #   rescued the X-VLA run (whose failure was geometry hitting peak LR too abruptly).
        # Peaks are slightly hotter and floors slightly cooler than v2 => wider dynamic range.
        lr_groups={
            # LR scaled by sqrt(2) for the 2x global batch (128 -> 256). Doubling the batch cuts
            # gradient noise by sqrt(2), so sqrt scaling holds the noise-to-signal ratio -- and thus
            # the training dynamics -- roughly fixed. Preferred over the linear rule here: linear is
            # derived for SGD, and Adam already normalizes by gradient magnitude, so linear tends to
            # overshoot. It is also the conservative choice given this branch has now failed twice
            # from runaway dynamics (frozen-by-eps, then saturated-by-unbounded-logits).
            # Step counts are HALVED so the run sees the same 6.4M samples as the 50k x 128 runs,
            # keeping the comparison against v2/v3 sample-matched.
            "vlm": _optimizer.CosineDecaySchedule(warmup_steps=250, peak_lr=4.2e-5, decay_steps=25_000, decay_lr=2.8e-6),
            "core": _optimizer.CosineDecaySchedule(warmup_steps=250, peak_lr=8.5e-4, decay_steps=25_000, decay_lr=5.7e-5),
            "geom": _optimizer.CosineDecaySchedule(warmup_steps=750, peak_lr=7.1e-4, decay_steps=25_000, decay_lr=7.1e-5),
        },
        # THE fix for the perceiver never training. Adam's update is mu/(sqrt(nu)+eps) and is
        # scale-invariant only while sqrt(nu) >> eps. MEASURED at v3 step 6000 (and v2 step 49999,
        # identically): every spatial param sits 3-5 orders of magnitude UNDER the 1e-8 default --
        # perceiver query sqrt(nu)=2.6e-11, its q_proj 1.3e-12, the injection log_gain 1.7e-13 --
        # while the vlm backbone sits at 2.9e-07. So eps dominated the denominator and the spatial
        # branch received |update| ~1e-4..1e-6 against the vlm's 1.2e-1: frozen at init, exactly the
        # "collapse" signature (query std still == 0.05 at 50k). The tiny gradients are expected
        # (spatial_init_std=0.01 out-projections x softmax averaging over 432 patches); normalizing
        # them away is Adam's job, and it can only do it once eps stops swallowing them.
        # vlm keeps 1e-8 -- unaffected either way, and no reason to perturb a working group.
        lr_group_eps={"core": 1e-16, "geom": 1e-16},
        # FIX 3: real weight decay on the spatial groups only. 1e-10 is no decay at all, and is what
        # let q_proj/k_proj grow until softmax saturated. Backbone ("vlm") keeps the default.
        lr_group_weight_decay={"core": 1e-4, "geom": 1e-4},
        num_train_steps=25_000,
        # 8-GPU RUN. 256 = 32/GPU x 8. Per-GPU batch is UNCHANGED from the 4-GPU run because 32 is
        # the MEASURED memory ceiling (~140.2GB of 143.8GB with inline DA3-GIANT resident); the
        # global batch doubles purely by adding devices, so per-device memory is identical.
        batch_size=256,
        num_workers=8,
        # Halved with the step count so checkpoints land at the same SAMPLE cadence as the 4-GPU runs.
        save_interval=1_000,
        keep_period=5_000,
        wandb_enabled=False,
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    # RoboTwin 2.0 pi0.5 + DA3 v5: selective port of the successful B1K "newbank"
    # spatial architecture. This is a FRESH model: K/V split, locality, query token
    # embeddings, and cross-view fusion all add/change spatial parameters.
    TrainConfig(
        name="pi05_robotwin2_full_da3_inline_v5",
        model=pi0_config.Pi0Config(
            pi05=True,
            da3=pi0_config.Pi0DA3Config(
                enabled=True,
                spatial_init_std=0.01,
                spatial_scale=1.0,
                # QK-normalized gains are temperatures; 3 attends locally without
                # approaching the hard-argmax regime. Both are bounded at 8.
                attn_qk_norm=True,
                perceiver_qk_norm=True,
                attn_logit_gain=True,
                attn_logit_gain_init=3.0,
                attn_logit_gain_max=8.0,
                perceiver_logit_gain=True,
                perceiver_logit_gain_init=3.0,
                perceiver_logit_gain_max=8.0,
                perceiver_norm_attn_out=False,
                perceiver_norm_out=True,
                fuse_proj_norm=False,
                bank_token_embed=True,
                bank_token_embed_query=True,
                perceiver_query_std=0.05,
                # B1K newbank architecture, adapted to RoboTwin's posed 18x24 grid.
                kv_split=True,
                depth_dropout=0.5,
                pos_emb_scale=0.25,
                perc_locality=True,
                locality_gamma_init=4.0,
                cross_view=True,
                cross_view_depth=2,
                # Batch centering is diagnostic-only: batch=1 serving would zero it.
                bank_center=False,
            ),
        ),
        data=LeRobotAlohaDataConfig(
            repo_id="robotwin2_aloha_lerobot",
            # Same dataset/action normalization as v3/v4; reuse the measured stats.
            assets=AssetsConfig(
                assets_dir="/work/jack/openpi_src/openpi/assets/pi05_robotwin2_full_da3_inline_v3",
                asset_id="robotwin2_aloha_lerobot",
            ),
            adapt_to_pi=False,
            use_delta_joint_actions=True,
            prompt_from_task=True,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.countertop",
                                "cam_left_wrist": "observation.images.left",
                                "cam_right_wrist": "observation.images.right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            da3_cache=DA3CacheConfig(
                inline=True,
                bgr_to_rgb=True,
                lang_cache="/work/jack/da3_cache/modernbert_robotwin2_lang.pkl",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=r".*lora.*|.*spatial_bank_builder.*|.*spatial_inject.*",
        ),
        # Same 6.4M-sample budget as v3 (50k x 128), using 8 x H200 at
        # the measured-safe per-device batch of 32.
        lr_groups={
            "vlm": _optimizer.DelayRampPlateauCosine(
                delay_steps=1_250,
                ramp_steps=3_750,
                ramp_start_lr=1e-6,
                peak_lr=1e-5,
                decay_start=12_500,
                decay_steps=25_000,
                decay_lr=1e-6,
            ),
            "core": _optimizer.DelayRampPlateauCosine(
                peak_lr=1e-4,
                decay_start=12_500,
                decay_steps=25_000,
                decay_lr=1e-5,
            ),
            "geom": _optimizer.DelayRampPlateauCosine(
                peak_lr=1e-4,
                decay_start=12_500,
                decay_steps=25_000,
                decay_lr=1e-5,
            ),
        },
        lr_group_eps={"core": 1e-16, "geom": 1e-16},
        lr_group_weight_decay={"core": 1e-4, "geom": 1e-4},
        num_train_steps=25_000,
        batch_size=256,
        num_workers=8,
        save_interval=1_000,
        keep_period=5_000,
        wandb_enabled=False,
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    # Four-GPU, sample-matched v5. Per-device batch remains 32; step and
    # schedule timings are exactly 2x the 8-GPU recipe, preserving 6.4M samples.
    TrainConfig(
        name="pi05_robotwin2_full_da3_inline_v5_4gpu",
        model=pi0_config.Pi0Config(
            pi05=True,
            da3=pi0_config.Pi0DA3Config(
                enabled=True,
                spatial_init_std=0.01,
                spatial_scale=1.0,
                attn_qk_norm=True,
                perceiver_qk_norm=True,
                attn_logit_gain=True,
                attn_logit_gain_init=3.0,
                attn_logit_gain_max=8.0,
                perceiver_logit_gain=True,
                perceiver_logit_gain_init=3.0,
                perceiver_logit_gain_max=8.0,
                perceiver_norm_attn_out=False,
                perceiver_norm_out=True,
                fuse_proj_norm=False,
                bank_token_embed=True,
                bank_token_embed_query=True,
                perceiver_query_std=0.05,
                kv_split=True,
                depth_dropout=0.5,
                pos_emb_scale=0.25,
                perc_locality=True,
                locality_gamma_init=4.0,
                cross_view=True,
                cross_view_depth=2,
                bank_center=False,
            ),
        ),
        data=LeRobotAlohaDataConfig(
            repo_id="robotwin2_aloha_lerobot",
            assets=AssetsConfig(
                assets_dir="/work/jack/openpi_src/openpi/assets/pi05_robotwin2_full_da3_inline_v3",
                asset_id="robotwin2_aloha_lerobot",
            ),
            adapt_to_pi=False,
            use_delta_joint_actions=True,
            prompt_from_task=True,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.countertop",
                                "cam_left_wrist": "observation.images.left",
                                "cam_right_wrist": "observation.images.right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            da3_cache=DA3CacheConfig(
                inline=True,
                bgr_to_rgb=True,
                lang_cache="/work/jack/da3_cache/modernbert_robotwin2_lang.pkl",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=r".*lora.*|.*spatial_bank_builder.*|.*spatial_inject.*",
        ),
        lr_groups={
            "vlm": _optimizer.DelayRampPlateauCosine(
                delay_steps=2_500,
                ramp_steps=7_500,
                ramp_start_lr=1e-6,
                peak_lr=1e-5,
                decay_start=25_000,
                decay_steps=50_000,
                decay_lr=1e-6,
            ),
            "core": _optimizer.DelayRampPlateauCosine(
                peak_lr=1e-4,
                decay_start=25_000,
                decay_steps=50_000,
                decay_lr=1e-5,
            ),
            "geom": _optimizer.DelayRampPlateauCosine(
                peak_lr=1e-4,
                decay_start=25_000,
                decay_steps=50_000,
                decay_lr=1e-5,
            ),
        },
        lr_group_eps={"core": 1e-16, "geom": 1e-16},
        lr_group_weight_decay={"core": 1e-4, "geom": 1e-4},
        num_train_steps=50_000,
        batch_size=128,
        num_workers=8,
        save_interval=2_000,
        keep_period=10_000,
        wandb_enabled=False,
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # ===== LIBERO ablation (train on CLEAN LIBERO, eval on LIBERO + LIBERO-plus) =====
    # physical-intelligence/libero: 40 tasks, 1693 eps, panda single-arm, 2 cams
    # (image + wrist_image, 256x256), 7-dim delta-EE actions. base + da3 share batch/
    # steps so the A/B is clean. pi05_base peak LR = da3 "vlm" peak (all-pretrained).
    TrainConfig(
        name="pi05_libero_base",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=30_000, decay_lr=2.5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.99,
        num_train_steps=30_000,
        batch_size=128,
        num_workers=8,
        wandb_enabled=False,
    ),
    # pi0.5 + DA3, UNPOSED (LIBERO has no camera calibration): 2-cam monocular DA3,
    # square 252x252 input (grid 18x18), inject into last 6 action-expert blocks.
    # v2 attention fixes + spatial recipe mirror the converged robopro tuned-v2.
    TrainConfig(
        name="pi05_libero_da3",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            da3=pi0_config.Pi0DA3Config(
                enabled=True, posed=False, num_views=2, grid_hw=(18, 18),
                spatial_init_std=0.01, attn_logit_gain=True, attn_logit_gain_init=32.0,
                bank_token_embed=True, perceiver_query_std=0.05,
            ),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            da3_cache=DA3CacheConfig(
                inline=True, posed=False,
                view_order=("image", "wrist_image"), image_key_tmpl="{v}",
                da3_hw=(252, 252),
                lang_cache="/work/jack/da3_cache/modernbert_libero_lang.pkl",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=r".*lora.*|.*spatial_bank_builder.*|.*spatial_inject.*",
        ),
        lr_groups={
            "vlm": _optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=30_000, decay_lr=2.5e-6),
            "core": _optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=5e-4, decay_steps=30_000, decay_lr=5e-5),
            "geom": _optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=5e-4, decay_steps=30_000, decay_lr=1e-4),
        },
        num_train_steps=30_000,
        batch_size=128,
        num_workers=8,
        wandb_enabled=False,
    ),
    # pi0 (NOT pi0.5) + DA3, same method as pi05_robotwin2_full_da3_inline_v2.
    # Byte-for-byte identical to that config EXCEPT: model is pi05=False (pi0
    # flow-matching expert: state token in prefix, no AdaRMS time conditioning,
    # max_token_len 48) and the base weights are pi0_base instead of pi05_base.
    # Both use the gemma_300m depth-18 action expert, so the DA3 injection into
    # the last 6 blocks (12-17) and every v2 flag transfer unchanged. Same
    # dataset / norm_stats / ModernBERT lang cache (all model-agnostic).
    TrainConfig(
        name="pi0_robotwin2_full_da3_inline_v2",
        model=pi0_config.Pi0Config(pi05=False, da3=pi0_config.Pi0DA3Config(
            enabled=True, spatial_init_std=0.01,
            attn_logit_gain=True, attn_logit_gain_init=32.0, bank_token_embed=True, perceiver_query_std=0.05,
        )),
        data=LeRobotAlohaDataConfig(
            repo_id="robotwin2_aloha_lerobot",
            assets=AssetsConfig(asset_id="robotwin2_aloha_lerobot"),
            adapt_to_pi=False,
            use_delta_joint_actions=True,
            prompt_from_task=True,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.countertop",
                                "cam_left_wrist": "observation.images.left",
                                "cam_right_wrist": "observation.images.right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            da3_cache=DA3CacheConfig(inline=True, bgr_to_rgb=True, lang_cache="/work/jack/da3_cache/modernbert_robotwin2_lang.pkl"),  # robotwin2 videos are BGR-swapped -> correct at load
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params",
            missing_regex=r".*lora.*|.*spatial_bank_builder.*|.*spatial_inject.*",
        ),
        lr_groups={
            "vlm": _optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=50_000, decay_lr=2.5e-6),
            "core": _optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=5e-4, decay_steps=50_000, decay_lr=5e-5),
            "geom": _optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=5e-4, decay_steps=50_000, decay_lr=1e-4),
        },
        num_train_steps=50_000,
        batch_size=128,
        num_workers=8,
        wandb_enabled=False,
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    # ===================== pi0 (NOT pi0.5) family =========================
    # Clones of the pi05_* configs with pi05=False + pi0_base weights, so the
    # ablation spans both flow-expert variants. Only the pi05 flag and base
    # checkpoint differ; dataset / repack / norm_stats / DA3 method / LRs match
    # their pi05 twins exactly. gemma_300m depth-18 is shared, so DA3 injection
    # (last 6 blocks) transfers unchanged.
    #
    # pi0 base (no DA3) — A/B baseline for pi0_robotwin2_full_da3_inline_v2.
    TrainConfig(
        name="pi0_robotwin2_full",
        model=pi0_config.Pi0Config(pi05=False),
        data=LeRobotAlohaDataConfig(
            repo_id="robotwin2_aloha_lerobot",
            assets=AssetsConfig(asset_id="robotwin2_aloha_lerobot"),
            adapt_to_pi=False,
            use_delta_joint_actions=True,
            prompt_from_task=True,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.countertop",
                                "cam_left_wrist": "observation.images.left",
                                "cam_right_wrist": "observation.images.right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params"
        ),
        num_train_steps=30_000,
        batch_size=128,
        num_workers=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=30_000, decay_lr=2.5e-6
        ),
        wandb_enabled=False,
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    # pi0 base (no DA3) on RoboPRO/roboreal — A/B baseline for the roboreal DA3 pi0 run.
    TrainConfig(
        name="pi0_roboreal_full",
        model=pi0_config.Pi0Config(pi05=False),
        data=LeRobotAlohaDataConfig(
            repo_id="roboreal_full_lerobot",
            assets=AssetsConfig(asset_id="roboreal_full_lerobot"),
            adapt_to_pi=False,
            use_delta_joint_actions=True,
            prompt_from_task=True,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high":         "observation.images.countertop",
                                "cam_left_wrist":   "observation.images.left",
                                "cam_right_wrist":  "observation.images.right",
                            },
                            "state":   "observation.state",
                            "actions": "action",
                            "prompt":  "prompt",
                        }
                    ),
                    # RoboReal LeRobot videos are BGR-swapped (beige->blue); correct to RGB here, once,
                    # right after repack. (No DA3 in this baseline, so DA3CacheConfig.bgr_to_rgb doesn't apply.)
                    _transforms.SwapImageChannels(key="images"),
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params"
        ),
        num_train_steps=30_000,
        batch_size=96,
        num_workers=8,
        wandb_enabled=False,
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    # pi0 + DA3 on RoboPRO/roboreal — same DA3 method as pi05_roboreal_full_da3_inline_v2.
    TrainConfig(
        name="pi0_roboreal_full_da3_inline_v2",
        model=pi0_config.Pi0Config(pi05=False, da3=pi0_config.Pi0DA3Config(
            enabled=True, spatial_init_std=0.01,
            attn_logit_gain=True, attn_logit_gain_init=32.0, bank_token_embed=True, perceiver_query_std=0.05,
        )),
        data=LeRobotAlohaDataConfig(
            repo_id="roboreal_full_lerobot",
            assets=AssetsConfig(asset_id="roboreal_full_lerobot"),
            adapt_to_pi=False,
            use_delta_joint_actions=True,
            prompt_from_task=True,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.countertop",
                                "cam_left_wrist": "observation.images.left",
                                "cam_right_wrist": "observation.images.right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            da3_cache=DA3CacheConfig(inline=True, bgr_to_rgb=True),  # roboreal videos are BGR-swapped -> correct at load; extract on the fly
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params",
            missing_regex=r".*lora.*|.*spatial_bank_builder.*|.*spatial_inject.*",
        ),
        lr_groups={
            "vlm": _optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=50_000, decay_lr=2.5e-6),
            "core": _optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=5e-4, decay_steps=50_000, decay_lr=5e-5),
            "geom": _optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=5e-4, decay_steps=50_000, decay_lr=1e-4),
        },
        num_train_steps=50_000,
        batch_size=128,
        num_workers=8,
        wandb_enabled=False,
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
