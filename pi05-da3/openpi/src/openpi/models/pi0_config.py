import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0DA3Config:
    """DA3 spatial-language branch config (features precached offline; only the JAX branch trains)."""

    enabled: bool = True
    num_views: int = 3
    # posed=True: multi-view DA3 with camera extrinsics/intrinsics -> world-frame scale-aware ray
    # is built and fused into each view's spatial bank (roboreal/aloha 3-cam path).
    # posed=False (UNPOSED): no camera calibration available (e.g. LIBERO 2-cam). No extrinsics/ray/
    # depth are read or fed; the spatial bank drops the ray embedding entirely and relies on
    # feats + 2D-pos + per-view embeddings only.
    posed: bool = True
    da3_channels: int = 1536  # GIANT (DA3NESTED-GIANT-LARGE-1.1) embed_dim
    da3_layers: int = 4  # len(da3_out_layers)
    grid_hw: tuple[int, int] = (18, 24)  # 252x336 / patch 14
    hidden_dim: int = 1024  # == action-expert width
    lang_dim: int = 1024  # ModernBERT-large last_hidden width
    lang_max_len: int = 64
    num_heads: int = 8
    lang_fusion_depth: int = 2
    num_inject_layers: int = 6  # inject into the LAST N action-expert blocks (12..17 for depth 18)
    spatial_scale: float = 2.0
    # Init std for the injection output projections (xattn out_projs + merge_proj).
    # 0.0 = zero-init: exact pi05_base identity at step 0, but q/k/v receive no gradient until the
    # out_projs grow off zero. >0 = small normal init: tiny spatial delta at step 0 and immediate
    # gradient flow to all injection weights.
    spatial_init_std: float = 0.0
    # --- v2 attention-learning fixes (defaults OFF to keep v1 runs reproducible) ---
    # Learnable per-head gain on the injection cross-attn logits (init 1.0). Scales dL/dQ,K and
    # lets the model sharpen its spatial lookup; without it Q/K trained ~1000x slower than V/out.
    attn_logit_gain: bool = False
    attn_logit_gain_init: float = 1.0  # >1 = sharp attention at init (restores Q/K gradient scale)
    # Learned per-token embedding added to each view's final bank tokens: guarantees persistent
    # token diversity, which is what drives softmax gradients to Q/K (common content cancels in
    # the softmax jacobian; only cross-token deviations produce attention learning signal).
    bank_token_embed: bool = False
    # Init std of the perceiver latent queries (v1 used 0.02 -> only ~2% token diversity at init).
    perceiver_query_std: float = 0.02


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    # DA3 spatial branch (None == stock pi0.5).
    da3: Pi0DA3Config | None = None

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                **self._da3_inputs_spec(batch_size),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def _da3_inputs_spec(self, batch_size: int) -> dict:
        if self.da3 is None or not self.da3.enabled:
            return {}
        d = self.da3
        v, c = d.num_views, d.da3_channels
        h, w = d.grid_hw
        f32 = jnp.float32
        spec = {
            "da3_features": jax.ShapeDtypeStruct([batch_size, d.da3_layers, v, c, h, w], f32),
            "lang_feat": jax.ShapeDtypeStruct([batch_size, d.lang_max_len, d.lang_dim], f32),
            "lang_mask": jax.ShapeDtypeStruct([batch_size, d.lang_max_len], bool),
        }
        if d.posed:
            # Geometry inputs only exist in the posed path; unposed omits ray/depth/extrinsics.
            spec["da3_ray"] = jax.ShapeDtypeStruct([batch_size, v, 3, h, w], f32)
            spec["da3_depth"] = jax.ShapeDtypeStruct([batch_size, v, 1, h, w], f32)
            spec["camera_extrinsics"] = jax.ShapeDtypeStruct([batch_size, v, 4, 4], f32)
        return spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
