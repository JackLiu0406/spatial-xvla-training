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

from .configuration_florence2 import Florence2Config
from transformers.configuration_utils import PretrainedConfig


# -----------------------------------------------------------------------------
# DA3-XVLA extension: default config blocks
#
# These are kept as plain dicts so that:
#   * They serialize cleanly through HF `PretrainedConfig.to_dict()`.
#   * Old checkpoints (config.json without these keys) still load: the model
#     falls back to the disabled defaults below.
# Nothing here changes baseline X-VLA behavior unless `enabled=True`.
# -----------------------------------------------------------------------------

def default_geometry_conditioning() -> dict:
    """
    Default (disabled) Segmented DA3 geometry-conditioning block.

    Old-style keys (``source``, ``mask_background_weight``, ``use_object_mask``,
    ``allow_missing_geometry``, ``resampler_type``) are still accepted as
    aliases for backward compatibility — see ``normalize_geometry_cfg``.
    """
    return {
        "enabled": False,

        # ---- DA3 feature input ----
        "da3_feature_root": None,
        "da3_feature_key": "da3_features",
        # "precomputed" | "extractor_stub" | "da3_encoder_inline"
        "da3_source": "precomputed",
        "freeze_da3": True,

        # ---- Inline DA3 (da3_source == "da3_encoder_inline") ----
        # Runs DA3 inside the model.forward() on the same `image_input` tensor
        # X-VLA passes to Florence2 (ImageNet-normalized, 224x224). See
        # `models/da3_inline.py::DA3InlineEncoder`. No precompute, no disk.
        "da3_model_name": "depth-anything/DA3-BASE",   # or DA3-SMALL / DA3-LARGE
        "da3_use_bf16": True,
        "da3_gradient_checkpointing": True,
        "da3_feature_size": 64,
        "da3_feature_layer": "last",         # "last" | int | "multi"
        "da3_multi_fusion_init": "last_only",  # "last_only" | "uniform" — only used when da3_feature_layer=="multi"
        "da3_process_res": 504,

        # ---- Backbone selector (DA3 vs VGGT) ----
        # When "vggt", the DA3InlineEncoder branch is skipped and a
        # VGGTInlineEncoder is instantiated instead (see modeling_xvla.py).
        # The downstream geometry projector / perceiver / cross-attn fusion are
        # backbone-agnostic — both produce [B, V, C, h, w] dense features.
        "geometry_backbone": "da3",            # "da3" | "vggt"

        # ---- Inline VGGT-Omega (geometry_backbone == "vggt") ----
        # Mirrors the DA3 keys above. Ignored when geometry_backbone=="da3".
        "vggt_model_name":              "JackLiu0406/vggt-omega-1b",
        "vggt_ckpt_filename":           "vggt_omega_1b_512.pt",
        "vggt_use_bf16":                True,
        "vggt_gradient_checkpointing":  True,
        "vggt_feature_size":            64,
        "vggt_feature_layer":           "last",     # "last" | "multi" (4-tap fusion at blocks 4/11/17/23)
        "vggt_multi_fusion_init":       "last_only",  # "last_only" | "uniform" — used only when vggt_feature_layer == "multi"
        "vggt_process_res":             224,        # dataset native; must be multiple of patch_size=16
        "vggt_input_dim":               2048,       # 2 × embed_dim (concat of frame + inter-frame streams)

        # ---- Inline Grounded-SAM (V2: language-conditioned masks) ----
        # When `object_mask_source == "gsam_inline"`, runs GroundingDINO+SAM
        # inside model.forward() to produce per-sample masks from the language
        # instruction. No precompute, no disk. See `models/gsam_inline.py`.
        "gsam_dino_id": "IDEA-Research/grounding-dino-tiny",
        "gsam_sam_id": "facebook/sam-vit-base",
        "gsam_box_threshold": 0.30,
        "gsam_text_threshold": 0.25,
        "gsam_use_bf16": True,
        "gsam_freeze": True,

        # ---- Object masks ----
        "use_da3_latent_segmentation": True,
        "object_mask_root": None,
        "object_mask_key": "object_masks",
        # "precomputed" | "batch" | "segmentation_model_stub" | "gsam_inline"
        "object_mask_source": "precomputed",
        "use_language_conditioned_masks": True,
        "allow_missing_masks": False,

        # ---- Masking behavior ----
        # "soft_mask" | "hard_mask" | "none"
        "segmentation_mode": "soft_mask",
        "background_weight": 0.2,
        # "max" | "sum_clamp" | "separate_objects"
        "mask_reduce": "max",

        # ---- Geometry tokens ----
        "num_geometry_tokens": 32,
        "da3_input_dim": None,
        "geometry_hidden_dim": None,
        "geometry_projector_type": "mlp",
        "geometry_resampler_type": "perceiver",

        # ---- Fusion ----
        "fusion_type": "cross_attention",
        # "before_policy" | "after_policy" | "inside_policy"
        "fusion_position": "before_policy",
        "cross_attention_layers": 1,
        "cross_attention_heads": 8,
        "cross_attention_dropout": 0.0,

        # ---- Param-group training (see config.finetune too) ----
        "train_geometry_projector": True,
        "train_cross_attention": True,

        # ---- Debug ----
        "debug_shapes": False,
        "allow_dummy_da3_features": False,
        "allow_dummy_masks": False,
    }


# Back-compat: old key -> new key. Old values win only if the new key is unset.
_GEOM_ALIASES = {
    "mask_background_weight": "background_weight",
    "use_object_mask": "use_da3_latent_segmentation",
    "allow_missing_geometry": "allow_missing_masks",
    "resampler_type": "geometry_resampler_type",
    # "source" used "precomputed_da3"/"da3_encoder_stub" -> map to da3_source.
}


def normalize_geometry_cfg(cfg: dict) -> dict:
    """Fold legacy keys into the current schema (idempotent)."""
    out = dict(cfg)
    for old, new in _GEOM_ALIASES.items():
        if old in out and out.get(new) in (None, default_geometry_conditioning().get(new)):
            out[new] = out[old]
    if "source" in out and "da3_source" not in cfg:
        out["da3_source"] = "precomputed" if "precomputed" in str(out["source"]) else "extractor_stub"
    return out


def default_finetune() -> dict:
    """
    Default fine-tuning recipe for Segmented-DA3 X-VLA.

    Intended scheme: freeze the existing X-VLA encoders (Florence2
    vision-language backbone + soft prompts); train ONLY the new
    cross-attention/geometry modules and fine-tune the flow-matching action
    expert (the full policy transformer + action enc/dec).

    Only applied when training is launched with ``--apply_finetune_policy``;
    otherwise baseline behaviour (everything trainable, LR schedule governs
    freezing) is unchanged.
    """
    return {
        "train_backbone": False,           # Florence2 VL encoder: FROZEN
        "train_soft_prompts": False,       # existing X-VLA encoders: FROZEN
        "train_action_head": True,         # flow-matching head: fine-tune
        "train_geometry_modules": True,    # cross-attention transformer: TRAIN
        "train_last_n_policy_layers": -1,  # full action-expert transformer
    }


def merge_config_block(user: dict | None, defaults: dict) -> dict:
    """Shallow-merge user-supplied values over a defaults dict (keeps new keys)."""
    out = dict(defaults)
    if isinstance(user, dict):
        out.update(user)
    return out


class XVLAConfig(PretrainedConfig):
    """
    Configuration class for the **XVLA (Extended Vision-Language-Action)** model.

    This configuration defines all submodules of XVLA in a single place:
      - The visual-language backbone (Florence2)
      - The temporal/action transformer
      - The action/proprio setup
    """

    model_type = "xvla"

    def __init__(
        # === Florence backbone ===
        self,
        florence_config: dict | None = None,

        # === Transformer head ===
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_domains: int = 30,
        len_soft_prompts: int = 32,
        dim_time: int = 32,
        max_len_seq: int = 512,
        use_hetero_proj: bool = False,
        soft_prompt_length: int = 32,

        # === Action & proprio ===
        max_action_dim: int = 20,  # Maximum action dimension for padding (used by "auto" action mode)
        real_action_dim: int = 20,
        num_actions: int = 30,
        action_mode: str = "ee6d",
        use_proprio: bool = True,

        # === DA3-XVLA geometry-conditioning extension (optional) ===
        geometry_conditioning: dict | None = None,
        finetune: dict | None = None,

        **kwargs,
    ):
        # Florence2 backbone configuration
        if isinstance(florence_config, dict):
            self.florence_config = Florence2Config(**florence_config)
        elif isinstance(florence_config, Florence2Config):
            self.florence_config = florence_config
        else:
            self.florence_config = Florence2Config()

        # Transformer hyperparameters
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_domains = num_domains
        self.len_soft_prompts = len_soft_prompts
        self.dim_time = dim_time
        self.max_len_seq = max_len_seq
        self.use_hetero_proj = use_hetero_proj
        self.soft_prompt_length = soft_prompt_length

        # Action/proprioception settings
        self.num_actions = num_actions
        self.action_mode = action_mode
        self.use_proprio = use_proprio
        
        self.real_action_dim = real_action_dim
        self.max_action_dim = max_action_dim

        # DA3-XVLA extension: merge user blocks over defaults so that partial
        # config.json entries (or none at all) always yield a complete block.
        self.geometry_conditioning = normalize_geometry_cfg(
            merge_config_block(geometry_conditioning, default_geometry_conditioning())
        )
        self.finetune = merge_config_block(finetune, default_finetune())

        # Initialize base HF config attributes (e.g. name_or_path)
        super().__init__(**kwargs)

    # -------------------------------------------------------------------------
    # Serialization helpers
    # -------------------------------------------------------------------------
    def to_dict(self):
        """
        Convert this configuration (and its Florence sub-config)
        into a fully serializable dictionary for HF save/load.
        """
        output = super().to_dict()
        output["florence_config"] = self.florence_config.to_dict()
        return output
