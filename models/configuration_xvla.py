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

        # ---- Gated action-to-spatial cross-attention (ablation) ----
        # When True, BYPASS the GeometryCrossAttentionFusion above and instead
        # attach a GatedSpatialCrossAttention adapter inside (selected) policy
        # transformer blocks. Adapter operates ONLY on the action-token slice
        # with geometry_tokens as K/V — VLM tokens never see spatial directly.
        #   h := h + beta * SpatialCrossAttention(q=h, k=v=spatial_tokens)
        # beta is a learned scalar gate initialized to spatial_gate_init.
        # When False (default), behavior is byte-identical to the old config.
        "use_spatial_cross_attention": False,
        "spatial_num_tokens": 160,            # record-only; actual K comes from num_geometry_tokens
        "spatial_gate_init": 0.0,             # 0.0 → byte-identical to disabled at step 0
        "spatial_token_dim": None,            # None → match transformer hidden_size
        "action_hidden_dim": None,            # None → match transformer hidden_size
        "spatial_cross_attention_heads": 8,
        "spatial_cross_attention_dropout": 0.0,
        # Accepted: "all" | "middle_late" (= last_half = last 50% of blocks)
        # | "early_half" (= first 50%) | "late_quarter" (= last 25%)
        # | int (single block index) | list[int] (explicit indices, negatives ok)
        "spatial_cross_attention_layers": "all",
        # Gate granularity for the spatial cross-attention adapter:
        #   "scalar"    -> GatedSpatialCrossAttention (one β per layer; legacy)
        #   "per_token" -> PerTokenGatedSpatialCrossAttention (per-token, per-channel
        #                  sigmoid gate from an MLP over action_hidden;
        #                  matches paper's GatedFusion winner on LIBERO)
        "spatial_gate_type": "scalar",
        # Bottleneck dim for the per-token gate MLP. None -> hidden_dim // 4.
        # Only used when spatial_gate_type == "per_token".
        "spatial_gate_mlp_hidden": None,

        # ---- T5 language tokens concatenated into the K/V bank ----
        # When enabled, runs a frozen T5 encoder over the raw language instruction
        # strings, projects per-token T5 embeddings to hidden_dim, and concatenates
        # them with the spatial tokens. The combined [spatial | T5_lang] bank
        # becomes the K/V for the gated cross-attn adapters.
        #   spatial:        [B, K=160, hidden]
        #   T5 projected:   [B, L_t5, hidden]
        #   combined:       [B, K + L_t5, hidden]
        # Pairs with spatial_cross_attention_target=full_sequence so all policy
        # tokens (action+vlm+aux+soft) can attend to both geometry AND language.
        "t5_language_enabled": False,
        "t5_model_name": "t5-base",
        "t5_use_bf16": True,
        "freeze_t5": True,
        "t5_max_length": 64,
        # Which token slice gets cross-attention updates from spatial:
        #   "action_only"   -> only the action token slice queries spatial; vlm/aux/soft
        #                      pass through unchanged (legacy, current bs=16/32 cells)
        #   "full_sequence" -> the ENTIRE policy sequence (action+vlm+aux+soft) queries
        #                      spatial. Lets geometry refine every token type. Breaks
        #                      the "VLM wall" by design — spatial reaches everything.
        "spatial_cross_attention_target": "action_only",
        # Add learnable positional embeddings to the dense DPT/multi/last features
        # BEFORE feeding them to the Perceiver resampler. Without this, the perceiver
        # queries can only distinguish spatial positions by feature content (no
        # explicit "where" signal). Standard DETR/ViT practice. ~12.6M extra params
        # for V=3, feature_size=64, hidden_dim=1024 (= 1 × 12288 × 1024).
        "perceiver_input_pos_emb": False,
        # Number of camera views for the pos_emb shape. Default 3 matches our
        # countertop + L wrist + R wrist setup.
        "perceiver_input_views": 3,

        # ---- Param-group training (see config.finetune too) ----
        "train_geometry_projector": True,
        "train_cross_attention": True,

        # ---- Debug ----
        "debug_shapes": False,
        "allow_dummy_da3_features": False,
        "allow_dummy_masks": False,

        # =====================================================================
        # GeoStack-XVLA v2B — frozen DA3-Large spatial injection into Florence
        # =====================================================================
        # All defaults are DISABLED / no-op so old X-VLA-Pt checkpoints
        # (whose config.json lacks these keys) load byte-identically via the
        # merge_config_block path. Set use_geostack=true to opt in.
        "use_geostack": False,                          # master switch
        "geostack_da3_model": "depth-anything/DA3-Large-1.1",
        "geostack_freeze_da3": True,
        "geostack_main_camera_only": True,
        "geostack_use_depth": False,                    # not implemented
        "geostack_use_ray": True,
        "geostack_use_2d_pos": True,
        "geostack_use_spatial_bias": True,
        "geostack_use_gated_fusion": True,
        "geostack_use_residual_alpha_schedule": True,
        "geostack_alpha_warmup_iters": 5000,            # alpha = 0 for first N
        "geostack_alpha_at_warmup_end": 0.001,          # alpha at step warmup_iters (small NONZERO)
        "geostack_alpha_ramp_to_01_iters": 10000,       # then linear → 0.1
        "geostack_alpha_ramp_to_1_iters": 20000,        # then linear 0.1 → 1.0
        "geostack_spatial_bias_lambda_init": 1.0,       # softplus(λ) · dist² scale
        # geostack_drop_da3_padding_tokens removed in v2B (rect input, no padding).
        "geostack_feature_levels": ["shallow", "mid", "deep"],
        # DA3-Large 4-tap layer indices are [11, 15, 19, 23]; for 3-level
        # shallow/mid/deep we use [11, 19, 23] (drop 15).
        "geostack_da3_out_layers": [11, 19, 23],
        # DA3 input dims — aspect-correct rectangle (4:3) recovered from
        # Florence's stretched 224×224 via bilinear resize. Both dims must be
        # multiples of patch_size=14. Default 252×336 ≈ 1.05× of a 240×320 source
        # → token grid 18×24 = 432 tokens per level.
        "geostack_da3_input_h": 252,
        "geostack_da3_input_w": 336,
        "geostack_da3_patch_size": 14,
        "geostack_inject_layers": [6, 9, 11],           # Florence-2 encoder layer indices
        "geostack_inject_level_map": {"6": "mid", "9": "mid", "11": "deep"},
        "geostack_num_heads": 8,
        "geostack_dropout": 0.0,
        "geostack_gate_mlp_hidden": None,               # None → hidden_dim//4
        "geostack_gate_init_bias": -4.0,                # sigmoid(-4) ≈ 0.018 (mostly closed)
        "geostack_g_ray_init": -4.0,                    # sigmoid(-4) ≈ 0.018
        "geostack_g_pos_init": -2.0,                    # sigmoid(-2) ≈ 0.119
        # Florence freeze policy
        "geostack_freeze_florence": True,
        "geostack_train_top_florence_encoder_layers": False,
        "geostack_trainable_florence_encoder_layers": [9, 10, 11],
        # Optional VLM visual-token masking (kept disabled by default for v2B)
        "geostack_enable_vlm_token_masking": False,
        # Florence visual-token grid. X-VLA processor produces 224×224 input
        # → DaViT 4-stage stride [4,2,2,2] = 32× → 7×7 = 49 vision tokens.
        # (NOT 24×24 — that would be for 768×768 input which X-VLA doesn't use.)
        "geostack_florence_visual_h": 7,
        "geostack_florence_visual_w": 7,
        # When True, GeoStack adapters get gradients (default). When False they
        # are frozen — useful for inference / ablation studies.
        "geostack_train_adapters": True,

        # v2D: override the identity-at-step-0 zero-init on attn.out_proj.weight.
        # When >0, uses N(0, std=value) instead of zeros — geometry contributes
        # immediately at step 0 (small random-direction signal) so model can't
        # learn "geometry is free to ignore" during warmup. Default 0.0 = legacy
        # zero-init = identity-at-step-0.
        "geostack_out_proj_init_std": 0.0,
        "wrist_geostack_out_proj_init_std": 0.0,

        # =====================================================================
        # v2F — k320-style perceiver fusion with GeoStack-style K/V enrichment
        # =====================================================================
        # When use_geo_perceiver_v2f=True, the model's geometry_conditioner slot
        # is filled with GeoPerceiverV2F (DA3-Large aspect-preserved + PE+ray
        # enriched K/V + K-query perceiver). Bypasses da3_inline. Action expert
        # consumes the resulting K geometry tokens via the existing
        # transformer.geometry_fusion (before_policy) — same path as k320.
        "use_geo_perceiver_v2f": False,
        "geo_perceiver_v2f_da3_model": "depth-anything/DA3-Large-1.1",
        "geo_perceiver_v2f_da3_out_layers": [11, 19, 23],     # multi-tap (we use last by default)
        "geo_perceiver_v2f_da3_tap_idx": -1,                   # which tap to feed perceiver (-1 = deepest)
        "geo_perceiver_v2f_da3_input_h": 252,                  # aspect-preserved 4:3
        "geo_perceiver_v2f_da3_input_w": 336,
        "geo_perceiver_v2f_da3_patch_size": 14,
        "geo_perceiver_v2f_da3_use_bf16": True,
        # K/V enrichment (SinCos PE + ray)
        "geo_perceiver_v2f_use_2d_pos": True,
        "geo_perceiver_v2f_use_ray": True,
        "geo_perceiver_v2f_g_ray_init": -3.0,                  # σ(-3) ≈ 0.047 ray contribution
        "geo_perceiver_v2f_g_pos_init": -1.0,                  # σ(-1) ≈ 0.269 PE contribution
        # Optional world-frame ray (requires extrinsics in batch + use_posed_da3 path)
        "geo_perceiver_v2f_use_world_ray": False,
        "geo_perceiver_v2f_world_ray_mlp_hidden": 256,
        "geo_perceiver_v2f_world_ray_g_init": -1.0,
        # Perceiver settings
        "geo_perceiver_v2f_num_queries": 160,                  # K = 160 (vs k320's 320)
        "geo_perceiver_v2f_num_heads": 8,
        "geo_perceiver_v2f_num_layers": 1,
        "geo_perceiver_v2f_dropout": 0.0,
        # Multi-view: "first" (default — main camera only) or "all" (pool across V views)
        "geo_perceiver_v2f_views": "first",
        # Identity-at-step-0 override: >0 means non-zero out_proj init (v2D-aggressive style)
        "geo_perceiver_v2f_out_proj_init_std": 0.0,

        # =====================================================================
        # GeoStack-XVLA v2C — wrist-aware aux-visual cross-attention
        # =====================================================================
        # v2B injects only at the Florence encoder (main view × main DA3 features).
        # v2C adds a parallel injection on aux_visual_inputs (the wrist views that
        # bypass Florence's encoder): per-wrist cross-attn with that view's DA3
        # features, enriched with world-frame ray embeddings derived from
        # per-frame extrinsics (sample["extrinsics"], already in the pipeline).
        #
        # All defaults are DISABLED — old checkpoints load unchanged. Requires
        # use_geostack=true (shares the main DA3 instance and alpha schedule).
        "use_wrist_geostack": False,                    # master switch
        "wrist_geostack_views": [1, 2],                 # which views are wrists in [B,V,3,H,W]
                                                         # (default: views 1,2 are left/right wrist;
                                                         # view 0 is main camera)
        "wrist_geostack_da3_input_h": 224,              # smaller DA3 input than main (224×224)
        "wrist_geostack_da3_input_w": 224,
        "wrist_geostack_num_heads": 8,
        "wrist_geostack_gate_init_bias": -4.0,          # σ(-4) ≈ 0.018 closed at init
        "wrist_geostack_use_spatial_bias": True,
        "wrist_geostack_spatial_bias_lambda_init": 1.0,
        "wrist_geostack_gate_mlp_hidden": None,         # None → hidden_dim // 4
        "wrist_geostack_dropout": 0.0,
        # World-frame ray embedding (origin || direction, 6D per token)
        "wrist_geostack_use_world_ray": True,           # add world-ray embedding to DA3 K/V tokens
        "wrist_geostack_ray_mlp_hidden": 256,
        "wrist_geostack_ray_g_init": -2.0,              # σ(-2) ≈ 0.12 initial ray weight
        # Florence aux-vision token grid (per wrist view). With X-VLA processor
        # input 224×224 + DaViT 32× downsample → 7×7 = 49 tokens per view.
        "wrist_geostack_aux_h": 7,
        "wrist_geostack_aux_w": 7,

        # =====================================================================
        # GeoStack-XVLA v3 — Side stack for action-expert spatial-language conditioning
        # =====================================================================
        # Asymmetric parallel transformer stack runs in lockstep with the pretrained
        # action expert; processes [DA3 deep | T5] bank via self-attention; action
        # expert action-tokens cross-attend at paired layers via gated identity-at-step-0
        # adapters.
        # All defaults are DISABLED — old checkpoints load unchanged. Set enabled=true
        # in side_stack to opt in.
        "side_stack": {
            "enabled": False,
            "depth": 6,                              # number of side stack blocks
            "hidden_dim": None,                      # default → match action expert hidden_size
            "num_heads": None,                       # default → match action expert num_heads
            "mlp_ratio": 4.0,
            "dropout": 0.0,
            # Cross-attention adapter config
            "xattn_num_heads": 8,
            "xattn_dropout": 0.0,
            "xattn_target": "action_only",           # 'action_only' | 'full_sequence'
            # v3.7: bidirectional cross-attn — side stack ALSO queries action expert
            # state (in addition to action expert querying side stack). Makes side
            # stack action-context-aware and gives cross-attn adapters a stronger
            # gradient path back FROM the action expert via x_to_s as well.
            # Adds 6 more GatedCrossAttn adapters (~24M params total).
            "bidirectional": False,                  # opt-in (False = backwards-compat)
            "gate_init_bias": -4.0,                  # σ(-4)≈0.018 (closed at init)
            # Alpha schedule (residual ramp)
            "alpha_warmup_iters": 5000,              # alpha=0 until step N
            "alpha_at_warmup_end": 0.001,            # alpha at warmup boundary (NONZERO)
            "alpha_ramp_to_01_iters": 10000,         # then linear → 0.1
            "alpha_ramp_to_1_iters": 20000,          # then linear → 1.0
            # Bank construction
            "da3_model": "depth-anything/DA3-Large-1.1",
            "da3_out_layers": [11, 19, 23],          # only matters if v2B not active (private DA3)
            "da3_input_h": 378,                      # aspect-correct 4:3 (27×14)
            "da3_input_w": 504,                      # (36×14)
            "da3_patch_size": 14,
            "da3_tap_idx": -1,                       # which feat level to use; -1 = deepest
            "use_ray": True,                         # add DA3 ray-head direction to bank
            "use_2d_pos": True,                      # add 2D sin/cos PE
            "use_modality_emb": True,                # distinguish da3 vs t5 in self-attn
            "g_ray_init": -4.0,                      # ray gate (σ ≈ 0.018)
            "g_pos_init": -2.0,                      # 2D PE gate (σ ≈ 0.12)
            # v3.2 — World-frame (origin, direction) 6D ray per pixel
            # When True AND extrinsics are passed to the forward, DA3's per-pixel
            # local-frame ray direction is converted to a world-frame ray encoded
            # as cat(camera_origin_world, direction_world) — 6D per pixel.
            # This gives the model an EXPLICIT 3D representation: it can read the
            # camera position (where the wrist is, where the scene cam is) directly
            # from the first 3 channels, and the world-frame direction from the last 3.
            # Strictly better than Plücker (origin, moment) for this use case because
            # attention can use origin directly via distance/dot-product ops; Plücker
            # would bury origin in moment = origin × direction.
            "use_world_ray_6d": False,               # opt-in; backwards-compat default = False
            # v3.5 — Auxiliary action-prediction head on side stack.
            # Gives side stack a DIRECT supervised gradient (vs relying on
            # cross-attn pull-through from action expert, which empirically
            # left side stack stuck at fresh init in v3.2g/v3.3/v3.4 due to
            # gradient-utility chicken-and-egg). See side_stack.py
            # SideStackAuxHead for the architecture rationale.
            "aux_loss": {
                "enabled": False,                    # opt-in (False = backwards-compat)
                "weight": 0.1,                       # aux_loss × weight added to total loss
                "mlp_ratio": 2.0,                    # MLP hidden = mlp_ratio × hidden_dim
                "pool_init_std": 0.02,               # pool query init std
                # v3.6: include preprocessed proprio in aux head input. Adds
                # ~1M params (proprio projector + fusion) but makes the aux
                # action-prediction task tractable (predicting action from
                # geometry+language ALONE is too underdetermined for many tasks).
                "use_proprio": True,
            },
            # Multi-view DA3 (v3.1) — run DA3 on main + wrist cameras at
            # different resolutions so the action expert sees gripper-relative
            # geometry too. Each view's tokens go into the bank with a learned
            # view embedding so the side stack can distinguish them.
            "multi_view_da3": {
                "enabled": False,                    # opt-in (False = backwards-compat single-view)
                "main_view": 0,                      # index of main (scene) view in image_input
                "main_da3_input_hw": [252, 336],     # near-raw 240×320 cam → 18×24 = 432 tokens
                "wrist_views": [1, 2],               # indices of wrist views
                "wrist_da3_input_hw": [168, 224],    # downscaled 240×320 → 12×16 = 192 tokens each
                "use_view_emb": True,                # learned view_emb (main/wrist1/wrist2/...)
                "num_views": 3,                      # main + 2 wrists
            },
        },

        # =====================================================================
        # Spatial-language XVLA variants — Method A / Method B
        # =====================================================================
        # Shared frozen DA3 + frozen T5 tokenizer builds three task-conditioned
        # geometry banks:
        #   main:       [B, 64, hidden]
        #   left wrist: [B, 48, hidden]
        #   right wrist:[B, 48, hidden]
        # The banks are consumed only as cross-attention memory for the action
        # token slice. Spatial tokens are NOT appended to the XVLA sequence.
        "spatial_lang": {
            "enabled": False,
            # "post_refiner" = Method A, a 6-layer action-only refiner after
            # the 24 pretrained XVLA blocks.
            # "final6_injection" = Method B, action-slice injection after each
            # of the final 6 existing XVLA blocks.
            "method": "post_refiner",

            # DA3/T5 backbones are frozen in the first implementation.
            "da3_model": "depth-anything/DA3-Large-1.1",
            "da3_freeze": True,
            "da3_use_bf16": True,
            "da3_out_layers": [11, 15, 19, 23],
            "da3_input_h": 252,
            "da3_input_w": 336,
            "da3_patch_size": 14,
            "t5_model_name": "t5-base",
            "t5_freeze": True,
            "t5_use_bf16": True,
            "t5_max_length": 64,

            # Per-view routing. View IDs match dataloader order:
            # 0=countertop/main, 1=left wrist, 2=right wrist.
            "main_view": 0,
            "left_view": 1,
            "right_view": 2,
            "num_views": 3,
            "main_tokens": 96,
            "left_tokens": 64,
            "right_tokens": 64,

            # Shared module widths.
            "pos_mlp_hidden": 256,
            "ray_mlp_hidden": 256,
            "perceiver_heads": 8,
            "lang_fusion_layers": 2,
            "lang_fusion_heads": 8,
            "spatial_heads": 8,
            "dropout": 0.0,

            # Method A / B depth placement.
            "method_a_layers": 6,
            "method_b_start_layer": 18,
            "spatial_scale_start": 0.01,
            "spatial_scale_end": 1.0,
            "spatial_scale_ramp_until": 15000,
            "method_b_conservative_geometry_lr": False,
            "aux_heads": {
                "enabled": False,
                "endpoint_weight": 0.002,
                "heatmap_weight": 0.02,
                "heatmap_sigma": 1.25,
                "patch_size": 14,
                "target_step": -1,
            },
        },
    }


def default_training_plan():
    return {
        "total_iters": 125000,
        "batch_size_per_gpu": 32,
        "num_gpus": 4,
        "global_batch_size": 128,
        "cosine_start": 50000,
        "cosine_end": 125000,
        "geometry_lr": {
            "start": 2.0e-4,
            "min": 5.0e-5,
        },
        "xvla_core_lr": {
            "stage0_until": 2000,
            "stage0_lr": 2.0e-5,
            "stage1_until": 10000,
            "stage1_lr": 5.0e-5,
            "ramp_until": 15000,
            "full_lr": 2.0e-4,
            "min_lr": 5.0e-5,
        },
        "vlm_lr": {
            "freeze_until": 5000,
            "ramp_start_lr": 1.0e-6,
            "full_until": 15000,
            "full_lr": 1.0e-5,
            "min_lr": 2.5e-6,
        },
        "spatial_scale": {
            "start": 0.01,
            "end": 1.0,
            "ramp_until": 15000,
        },
        "freeze": {
            "da3_backbone": True,
            "t5_backbone": True,
            "vlm_until_step": 5000,
        },
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
        training: dict | None = None,

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
        self.training = merge_config_block(training, default_training_plan())

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
