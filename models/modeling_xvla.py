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

import logging
import os
import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn
import json_numpy
import cv2

from transformers import PreTrainedModel
from .modeling_florence2 import Florence2ForConditionalGeneration
from .transformer import SoftPromptedTransformer
from .action_hub import build_action_space
from .configuration_xvla import XVLAConfig
from .geometry_conditioning import SegmentedDA3GeometryConditioner

logger = logging.getLogger("xvla.geometry")


class XVLA(PreTrainedModel):
    """
    XVLA: HuggingFace-compatible Vision-Language-Action policy.

    Components:
      • Florence2 encoder-only backbone (vision-language)
      • SoftPromptedTransformer (temporal/action head)
      • Action space (pre/post-processing + loss)
    """
    config_class = XVLAConfig
    base_model_prefix = "xvla"
    supports_gradient_checkpointing = True

    def __init__(self, config: XVLAConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        # Core settings
        self.num_actions: int = config.num_actions
        self.use_proprio: bool = config.use_proprio
        self.action_mode: str = config.action_mode.lower()
        # Action space (dimensions + hooks)
        if config.action_mode.lower() == "auto":
            self.action_space = build_action_space(
                config.action_mode.lower(),
                real_dim=config.real_action_dim,
                max_dim=config.max_action_dim,
            )
        else:
            self.action_space = build_action_space(config.action_mode.lower())
        dim_action = self.action_space.dim_action
        dim_proprio = getattr(self.action_space, "dim_proprio", dim_action)

        # Florence2 backbone (encoder only)
        # Note: SDPA was measured ~35% slower than eager on this model (small
        # vision/text seq lengths → SDPA dispatch overhead > kernel gain). Keep
        # eager. Opt-in to SDPA via XVLA_ATTN_IMPL=sdpa for experimentation.
        _attn_impl = os.environ.get("XVLA_ATTN_IMPL", "eager")
        if _attn_impl != "eager":
            try:
                config.florence_config._attn_implementation = _attn_impl
                if getattr(config.florence_config, "text_config", None) is not None:
                    config.florence_config.text_config._attn_implementation = _attn_impl
                if getattr(config.florence_config, "vision_config", None) is not None:
                    config.florence_config.vision_config._attn_implementation = _attn_impl
            except Exception:
                pass
        self.vlm = Florence2ForConditionalGeneration(config.florence_config).to(torch.float32)
        if hasattr(self.vlm, "language_model"):
            lm = self.vlm.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "decoder"):
                del lm.model.decoder
            if hasattr(lm, "lm_head"):
                del lm.lm_head

        projection_dim = getattr(self.vlm.config, "projection_dim", None)
        if projection_dim is None:
            raise ValueError("Florence2 config must provide `projection_dim` for multimodal fusion.")

        # DA3-XVLA optional geometry-conditioning block (disabled by default).
        self.geometry_cfg: dict = dict(getattr(config, "geometry_conditioning", {}) or {})
        self.geometry_enabled: bool = bool(self.geometry_cfg.get("enabled", False))
        self.spatial_lang_cfg: dict = dict(self.geometry_cfg.get("spatial_lang", {}) or {})
        self.spatial_lang_enabled: bool = bool(self.spatial_lang_cfg.get("enabled", False))
        self.finetune_cfg: dict = dict(getattr(config, "finetune", {}) or {})

        # Temporal/action head (geometry fusion lives inside the transformer so
        # it can be applied before/inside/after the policy stack).
        self.transformer = SoftPromptedTransformer(
            hidden_size=config.hidden_size,
            multi_modal_input_size=projection_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_domains=config.num_domains,
            dim_action=dim_action,
            dim_propio=dim_proprio,
            len_soft_prompts=config.len_soft_prompts,
            dim_time=config.dim_time,
            max_len_seq=config.max_len_seq,
            use_hetero_proj=config.use_hetero_proj,
            # v3: also forward geometry_cfg when side_stack is enabled (it lives inside that dict)
            geometry_conditioning=(
                self.geometry_cfg
                if (self.geometry_enabled
                    or bool((self.geometry_cfg.get("side_stack") or {}).get("enabled", False))
                    or self.spatial_lang_enabled)
                else None
            ),
        )

        # DA3 -> compressed geometry tokens (only built when enabled).
        # v2F: use GeoPerceiverV2F (DA3-Large aspect + GeoStack-style PE/ray
        # enrichment + K=160 perceiver) when geo_perceiver_v2f.use_geo_perceiver_v2f=True.
        # Otherwise use the legacy SegmentedDA3GeometryConditioner (k320 path).
        self.geometry_conditioner = None
        self._use_geo_perceiver_v2f: bool = bool(self.geometry_cfg.get("use_geo_perceiver_v2f", False))
        if self.geometry_enabled:
            if self._use_geo_perceiver_v2f:
                from .geo_perceiver_v2f import GeoPerceiverV2F
                self.geometry_conditioner = GeoPerceiverV2F(
                    self.geometry_cfg, hidden_dim=config.hidden_size
                )
            else:
                self.geometry_conditioner = SegmentedDA3GeometryConditioner(
                    self.geometry_cfg, hidden_dim=config.hidden_size
                )
        self._geom_shape_logged = False

        # Inline geometry encoder (da3_source == "da3_encoder_inline").
        # Runs the backbone multi-view inside model.forward; no precompute on disk.
        # Backbone is DA3 by default; set geometry_backbone="vggt" in config to
        # swap for VGGT-Omega. Both produce [B, V, C, h, w] dense features so
        # the downstream geometry projector / perceiver / cross-attn fusion are
        # backbone-agnostic (only the input dim C differs — the geometry
        # projector is fresh-init on first training step so it sizes correctly).
        self.da3_inline = None
        if self.geometry_enabled and self.geometry_cfg.get("da3_source") == "da3_encoder_inline":
            backbone = str(self.geometry_cfg.get("geometry_backbone", "da3")).lower()
            if backbone == "vggt":
                from .vggt_inline import VGGTInlineEncoder
                self.da3_inline = VGGTInlineEncoder(
                    model_name=self.geometry_cfg.get("vggt_model_name", "JackLiu0406/vggt-omega-1b"),
                    ckpt_filename=self.geometry_cfg.get("vggt_ckpt_filename", "vggt_omega_1b_512.pt"),
                    use_bf16=bool(self.geometry_cfg.get("vggt_use_bf16", True)),
                    gradient_checkpointing=bool(self.geometry_cfg.get("vggt_gradient_checkpointing", True)),
                    freeze=bool(self.geometry_cfg.get("freeze_da3", True)),
                    feature_size=int(self.geometry_cfg.get("vggt_feature_size", 64)),
                    feature_layer=self.geometry_cfg.get("vggt_feature_layer", "last"),
                    process_res=int(self.geometry_cfg.get("vggt_process_res", 224)),
                    multi_fusion_init=str(self.geometry_cfg.get("vggt_multi_fusion_init", "last_only")),
                )
            else:
                from .da3_inline import DA3InlineEncoder
                self.da3_inline = DA3InlineEncoder(
                    model_name=self.geometry_cfg.get("da3_model_name", "depth-anything/DA3-BASE"),
                    use_bf16=bool(self.geometry_cfg.get("da3_use_bf16", True)),
                    gradient_checkpointing=bool(self.geometry_cfg.get("da3_gradient_checkpointing", True)),
                    freeze=bool(self.geometry_cfg.get("freeze_da3", True)),
                    feature_size=int(self.geometry_cfg.get("da3_feature_size", 64)),
                    feature_layer=self.geometry_cfg.get("da3_feature_layer", "last"),
                    process_res=int(self.geometry_cfg.get("da3_process_res", 504)),
                    multi_fusion_init=str(self.geometry_cfg.get("da3_multi_fusion_init", "last_only")),
                )

        # Inline Grounded-SAM (V2: object_mask_source == "gsam_inline").
        # Runs GroundingDINO + SAM inside model.forward to produce per-sample
        # masks from the language instruction. No precompute on disk.
        self.gsam_inline = None
        if (self.geometry_enabled
            and self.geometry_cfg.get("use_da3_latent_segmentation", False)
            and self.geometry_cfg.get("object_mask_source") == "gsam_inline"):
            from .gsam_inline import GroundedSAMInline
            self.gsam_inline = GroundedSAMInline(
                dino_id=self.geometry_cfg.get("gsam_dino_id", "IDEA-Research/grounding-dino-tiny"),
                sam_id=self.geometry_cfg.get("gsam_sam_id", "facebook/sam-vit-base"),
                box_threshold=float(self.geometry_cfg.get("gsam_box_threshold", 0.30)),
                text_threshold=float(self.geometry_cfg.get("gsam_text_threshold", 0.25)),
                use_bf16=bool(self.geometry_cfg.get("gsam_use_bf16", True)),
                freeze=bool(self.geometry_cfg.get("gsam_freeze", True)),
            )

        # Inline T5 encoder for language-conditioning the K/V bank of the
        # spatial cross-attn adapters. When enabled, per-token T5 embeddings
        # are projected to hidden_size and CONCATENATED to the spatial tokens
        # in _compute_geometry_tokens, so the gated cross-attn adapters see
        # a combined [spatial | T5_lang] bank as K/V.
        self.t5_inline = None
        self.t5_projector = None
        # t5_language_enabled OR side_stack.enabled (v3) both require T5.
        # The legacy `geometry_enabled` gate is bypassed for v3 — v3 doesn't need
        # the old geometry_conditioner, just T5 + DA3.
        _need_t5 = (
            bool(self.geometry_cfg.get("t5_language_enabled", False))
            and (self.geometry_enabled
                 or bool((self.geometry_cfg.get("side_stack") or {}).get("enabled", False)))
        )
        if _need_t5:
            from .t5_inline import T5InlineEncoder
            self.t5_inline = T5InlineEncoder(
                model_name=str(self.geometry_cfg.get("t5_model_name", "t5-base")),
                use_bf16=bool(self.geometry_cfg.get("t5_use_bf16", True)),
                freeze=bool(self.geometry_cfg.get("freeze_t5", True)),
                max_length=int(self.geometry_cfg.get("t5_max_length", 64)),
            )
            # Project T5 d_model → hidden_size, with LN for scale matching.
            self.t5_projector = nn.Sequential(
                nn.Linear(self.t5_inline.hidden_size, config.hidden_size),
                nn.GELU(approximate="tanh"),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
            )

        # Force-set the Perceiver resampler query at unit-scale init, AFTER
        # all submodule init. (HF's recursive `apply(_init_weights)` or some
        # earlier path was leaving this parameter at near-zero; explicit
        # re-init here is bulletproof and matches typical positional-embedding
        # init scale.) See `_reinit_geometry_modules` for the full re-init.
        self._reinit_geometry_modules()

        # ======================================================================
        # GeoStack-XVLA v2B: frozen DA3-Large spatial injection into Florence
        # encoder layers. Disabled by default (use_geostack=False) -- old
        # checkpoints load byte-identically.
        # ======================================================================
        self.use_geostack: bool = bool(self.geometry_cfg.get("use_geostack", False))
        self.da3_for_geostack = None
        self.geostack_alpha_schedule = None
        if self.use_geostack:
            self._init_geostack()

        # ======================================================================
        # GeoStack-XVLA v2C: wrist-aware aux-visual cross-attention
        # Requires use_geostack=true (shares DA3 instance + alpha schedule).
        # ======================================================================
        self.use_wrist_geostack: bool = bool(self.geometry_cfg.get("use_wrist_geostack", False))
        self.wrist_geostack = None
        if self.use_wrist_geostack:
            if not self.use_geostack:
                raise ValueError("use_wrist_geostack=True requires use_geostack=True (shares DA3)")
            self._init_wrist_geostack()

        # ======================================================================
        # GeoStack-XVLA v3: side stack for action-expert spatial-language conditioning
        # ======================================================================
        # Parallel transformer stack processes [DA3 deep | T5] bank; action expert
        # action-tokens cross-attend at paired layers via gated identity-at-step-0
        # adapters. Reuses self.da3_for_geostack and self.t5_inline if available.
        side_cfg = self.geometry_cfg.get("side_stack", {}) or {}
        self.side_stack_enabled: bool = bool(side_cfg.get("enabled", False))
        self.side_stack_bank_builder = None
        if self.side_stack_enabled:
            self._init_side_stack_bank_builder()

        # v3.5: AUXILIARY action-prediction head — gives side stack a DIRECT
        # supervised gradient signal (vs relying on cross-attn pull-through from
        # the action expert, which empirically yielded ~zero useful gradient and
        # left the side stack stuck at fresh init in v3.2g/v3.3/v3.4).
        aux_cfg = (side_cfg.get("aux_loss", {}) or {}) if self.side_stack_enabled else {}
        self.side_stack_aux_enabled: bool = bool(aux_cfg.get("enabled", False))
        self.side_stack_aux_head = None
        self.side_stack_aux_weight: float = float(aux_cfg.get("weight", 0.1))
        if self.side_stack_aux_enabled:
            self._init_side_stack_aux_head(aux_cfg)

        # ======================================================================
        # Spatial-language Method A/B shared DA3/T5 tokenizer
        # ======================================================================
        self.spatial_lang = None
        self.spatial_aux_heads = None
        if self.spatial_lang_enabled:
            from .spatial_language import SpatialAuxHeads, SpatialLanguageTokenizer
            self.spatial_lang = SpatialLanguageTokenizer(
                self.spatial_lang_cfg,
                hidden_dim=config.hidden_size,
            )
            aux_cfg = dict((self.spatial_lang_cfg.get("aux_heads", {}) or {}))
            if bool(aux_cfg.get("enabled", False)):
                self.spatial_aux_heads = SpatialAuxHeads(
                    aux_cfg,
                    hidden_dim=config.hidden_size,
                    num_actions=self.config.num_actions,
                    dim_action=int(self.action_space.dim_action),
                )
                logger.info(
                    "[spatial_lang] aux heads enabled "
                    "(endpoint_weight=%.4f, heatmap_weight=%.4f)",
                    float(aux_cfg.get("endpoint_weight", 0.002)),
                    float(aux_cfg.get("heatmap_weight", 0.02)),
                )

        # v3.5 H-3 fix: SECOND reinit call after v3 modules are built. The first
        # _reinit_geometry_modules() at line ~228 ran BEFORE side stack / bank
        # builder / aux head existed (they live in transformer.__init__ and
        # _init_side_stack_*). After from_pretrained loads weights for the
        # transformer params, those v3 modules are at uninit memory if the
        # checkpoint doesn't contain them (which is the case for fresh-init from
        # X-VLA-Pt + new v3 setup). train.py also calls this after model build,
        # but eval scripts / FastAPI server / smoke harness use from_pretrained
        # directly and would NaN without this defensive second call.
        self._reinit_geometry_modules()

        # Deferred FastAPI app
        self.app: FastAPI | None = None

    def _reinit_geometry_modules(self) -> None:
        """
        Explicitly initialize geometry-path parameters at sensible scales.

        Diag #11244 showed HF's loader was wiping EVERY param inside
        geometry_conditioner / transformer.geometry_fusion (Linear weights,
        LayerNorm gamma/beta, MultiheadAttention projections) to denormal-zero,
        not just the Perceiver query. With LayerNorm weight≈0 the output is
        multiplied by 0; with Linear weight≈0 the output is 0; the whole
        Perceiver collapsed to zero regardless of query init.

        Fix: recursively call ``reset_parameters()`` on every submodule of
        the geometry conditioner and geometry_fusion that has it (Linear,
        LayerNorm, MultiheadAttention all do), and re-init the Perceiver
        query at std=1.0.
        """
        def _reset_module(mod: nn.Module):
            """Call reset_parameters on this module if it has it; recurse.

            nn.MultiheadAttention exposes its initializer as the PRIVATE
            ``_reset_parameters`` (no public ``reset_parameters``), so a
            generic ``hasattr(reset_parameters)`` check silently skips it,
            leaving ``in_proj_weight=0`` and ``in_proj_bias`` at uninitialized
            memory after HF's loader wipe. Try the private name as a fallback.
            """
            if hasattr(mod, "reset_parameters") and callable(mod.reset_parameters):
                try:
                    mod.reset_parameters()
                    return
                except Exception:
                    pass
            if hasattr(mod, "_reset_parameters") and callable(mod._reset_parameters):
                try:
                    mod._reset_parameters()
                except Exception:
                    pass

        gc = getattr(self, "geometry_conditioner", None)
        if gc is not None:
            for m in gc.modules():
                _reset_module(m)
            if hasattr(gc, "resampler") and hasattr(gc.resampler, "query"):
                nn.init.normal_(gc.resampler.query, std=1.0)
            # v2F: apply GeoStack-style custom inits on the token builder + identity-at-step-0
            # on the perceiver out_proj. Also init the world_ray_emb scalar if present.
            from .geo_perceiver_v2f import GeoPerceiverV2F
            if isinstance(gc, GeoPerceiverV2F):
                gc_cfg = self.geometry_cfg
                with torch.no_grad():
                    tb = gc.token_builder
                    if getattr(tb, "g_ray", None) is not None:
                        tb.g_ray.fill_(float(gc_cfg.get("geo_perceiver_v2f_g_ray_init", -3.0)))
                    if getattr(tb, "g_pos", None) is not None:
                        tb.g_pos.fill_(float(gc_cfg.get("geo_perceiver_v2f_g_pos_init", -1.0)))
                    if getattr(gc, "world_ray_emb", None) is not None:
                        gc.world_ray_emb.g.fill_(float(gc_cfg.get("geo_perceiver_v2f_world_ray_g_init", -1.0)))
                gc.reinit_identity_at_step_0()
        gf = getattr(self.transformer, "geometry_fusion", None) if hasattr(self, "transformer") else None
        if gf is not None:
            for m in gf.modules():
                _reset_module(m)
            # DiT-style identity init: zero the cross-attn output proj AND the
            # FFN's final linear, so the fusion contributes EXACTLY 0 at iter 0
            # (geometry path is silent). Earlier V1 runs started above baseline
            # (1.84 vs 1.37) because the random-init fusion was perturbing the
            # pretrained policy from step 0. With this init the policy
            # transformer sees an unperturbed input initially, and gradients
            # grow the output projections as geometry becomes useful.
            for layer in getattr(gf, "layers", []):
                if isinstance(layer, nn.ModuleDict):
                    if "attn" in layer and hasattr(layer["attn"], "out_proj"):
                        op = layer["attn"].out_proj
                        nn.init.zeros_(op.weight)
                        if op.bias is not None:
                            nn.init.zeros_(op.bias)
                    if "ffn" in layer and hasattr(layer["ffn"], "fc2"):
                        fc2 = layer["ffn"].fc2
                        nn.init.zeros_(fc2.weight)
                        if fc2.bias is not None:
                            nn.init.zeros_(fc2.bias)

        # Gated spatial cross-attention adapters (sibling of geometry_fusion).
        # HF's loader zeroes every key absent from the X-VLA-PT checkpoint, which
        # leaves the per-token gate's MLP weights at exactly 0. For the SCALAR
        # variant that's fine (we want gate=0 → identity by design). For the
        # PER-TOKEN variant zero-init of gate_mlp[*].{weight,bias} combined with
        # bf16+DDP triggers NaN in real training (single-GPU bf16 forward is
        # fine; the NaN only appears after the first DDP all-reduce + optim
        # step). Re-initialize the gate-side modules so weights are well-scaled.
        # Keep attn.out_proj zeroed so the adapter still emits identity at
        # step 0 (DiT-style preservation of the pretrained policy).
        sca = getattr(self.transformer, "spatial_cross_attn_layers", None) if hasattr(self, "transformer") else None
        if sca is not None:
            from .geometry_conditioning import PerTokenGatedSpatialCrossAttention
            for adapter in sca:
                if isinstance(adapter, PerTokenGatedSpatialCrossAttention):
                    # Re-init the gate path so gate_mlp doesn't have pathological zero weights.
                    _reset_module(adapter.gate_norm)
                    for m in adapter.gate_mlp.modules():
                        _reset_module(m)
                    # Keep out_proj zero (DiT identity at step 0).
                    op = adapter.attn.out_proj
                    nn.init.zeros_(op.weight)
                    if op.bias is not None:
                        nn.init.zeros_(op.bias)

        # T5 projector — HF zeros these weights too. Re-init with proper Linear/LN defaults.
        t5p = getattr(self, "t5_projector", None)
        if t5p is not None:
            for m in t5p.modules():
                _reset_module(m)

        # GeoStack-XVLA v2B adapters — same problem class as spatial_cross_attn_layers
        # above, BUT MORE SEVERE. The GeoStack modules are built inside
        # XVLA.__init__ -> _init_geostack(), which runs inside HF's no_init_weights()
        # context (transformers/modeling_utils.py:199-216) during from_pretrained().
        # That context monkey-patches every nn.init.* to a no-op, so:
        #   * nn.Linear / nn.LayerNorm constructors silently leave weights at
        #     UNINITIALIZED MEMORY (kaiming_uniform_ in reset_parameters is no-op'd).
        #   * nn.MultiheadAttention's _reset_parameters silently leaves
        #     in_proj_weight / out_proj.weight as garbage.
        #   * The explicit nn.init.zeros_(attn.out_proj.*) and
        #     nn.init.constant_(gate_mlp[-1].bias, -4.0) calls inside
        #     GeoStackCrossAttention.__init__ ALSO silently no-op.
        # Result observed in training: at step 5020 (first alpha>0 forward), one
        # of the gate_mlp Linear weights produced NaN (was raw uninit memory:
        # ±1.7e38, ±inf). _reinit_geometry_modules runs from train.py AFTER
        # from_pretrained returns — outside the no_init context — so reset_parameters
        # calls here WORK. Re-apply identity-at-step-0 zero-inits afterwards.
        import math as _math
        tb_dict = getattr(self, "geostack_token_builders", None)
        if tb_dict is not None:
            gc_cfg = self.geometry_cfg
            for tb in tb_dict.values():
                for m in tb.modules():
                    _reset_module(m)
                with torch.no_grad():
                    if getattr(tb, "g_ray", None) is not None:
                        tb.g_ray.fill_(float(gc_cfg.get("geostack_g_ray_init", -4.0)))
                    if getattr(tb, "g_pos", None) is not None:
                        tb.g_pos.fill_(float(gc_cfg.get("geostack_g_pos_init", -2.0)))
        xa_dict = getattr(self, "geostack_fusers", None)
        if xa_dict is not None:
            # v2D-aggressive: optional non-zero out_proj init to skip identity-at-step-0
            main_op_std = float(self.geometry_cfg.get("geostack_out_proj_init_std", 0.0))
            for xa in xa_dict.values():
                for m in xa.modules():
                    _reset_module(m)
                with torch.no_grad():
                    if main_op_std > 0.0:
                        nn.init.normal_(xa.attn.out_proj.weight, mean=0.0, std=main_op_std)
                    else:
                        # Identity-at-step-0: zero attn.out_proj so initial residual = 0.
                        nn.init.zeros_(xa.attn.out_proj.weight)
                    if xa.attn.out_proj.bias is not None:
                        nn.init.zeros_(xa.attn.out_proj.bias)
                    # Closed-gate at step 0: large negative bias + tiny final weight.
                    nn.init.constant_(xa.gate_mlp[-1].bias, float(
                        self.geometry_cfg.get("geostack_gate_init_bias", -4.0)))
                    nn.init.normal_(xa.gate_mlp[-1].weight, mean=0.0, std=1e-4)
                    # softplus(lambda)=1.0 at init (matches GeoStackCrossAttention's intended init).
                    if getattr(xa, "spatial_bias_lambda", None) is not None:
                        xa.spatial_bias_lambda.fill_(_math.log(_math.expm1(float(
                            self.geometry_cfg.get("geostack_spatial_bias_lambda_init", 1.0)))))

        # GeoStack-XVLA v3 — side stack modules. Same no_init_weights problem
        # class as the v2B fusers: these modules are built inside XVLA.__init__
        # (via the transformer's __init__ AND _init_side_stack_bank_builder)
        # which runs under HF's no_init_weights context during from_pretrained.
        # Re-init explicitly now (we're outside the no_init context).
        side_cfg = self.geometry_cfg.get("side_stack", {}) or {}
        side_stack_obj = getattr(getattr(self, "transformer", None), "side_stack", None)
        if side_stack_obj is not None:
            # Re-init every submodule of the side stack
            for m in side_stack_obj.modules():
                _reset_module(m)
        # Re-init BOTH directions of cross-attn (s_to_x always present when side
        # stack is enabled; x_to_s only when bidirectional flag is set).
        for xattn_list_name in ("s_to_x_xattn", "x_to_s_xattn"):
            xattn_list = getattr(getattr(self, "transformer", None), xattn_list_name, None)
            if xattn_list is None:
                continue
            for xattn in xattn_list:
                for m in xattn.modules():
                    _reset_module(m)
                with torch.no_grad():
                    # FIX (v3.4): NO out_proj weight init override — let MultiheadAttention's
                    # default kaiming stand. Identity-at-step-0 preserved by α=0 schedule.
                    if xattn.attn.out_proj.bias is not None:
                        nn.init.zeros_(xattn.attn.out_proj.bias)
                    # Closed gate: large negative bias + tiny final-layer weight
                    nn.init.constant_(xattn.gate_mlp[-1].bias,
                                      float(side_cfg.get("gate_init_bias", -4.0)))
                    nn.init.normal_(xattn.gate_mlp[-1].weight, mean=0.0, std=1e-4)
        ssbb = getattr(self, "side_stack_bank_builder", None)
        if ssbb is not None:
            for m in ssbb.modules():
                _reset_module(m)
            with torch.no_grad():
                if getattr(ssbb, "g_ray", None) is not None:
                    ssbb.g_ray.fill_(float(side_cfg.get("g_ray_init", -4.0)))
                if getattr(ssbb, "g_pos", None) is not None:
                    ssbb.g_pos.fill_(float(side_cfg.get("g_pos_init", -2.0)))
                # Small modality emb init (don't dominate at step 0)
                if getattr(ssbb, "modality_emb", None) is not None:
                    nn.init.normal_(ssbb.modality_emb.weight, mean=0.0, std=0.01)
                # Small view emb init (v3.1: distinguishes main/wrist1/wrist2)
                if getattr(ssbb, "view_emb", None) is not None:
                    nn.init.normal_(ssbb.view_emb.weight, mean=0.0, std=0.01)

        # v3.5 — Re-init aux head (same HF no_init_weights issue: built inside
        # XVLA.__init__ under the no_init context, so weights are uninit memory
        # until we explicitly reset them here.)
        aux_head = getattr(self, "side_stack_aux_head", None)
        if aux_head is not None:
            for m in aux_head.modules():
                _reset_module(m)
            with torch.no_grad():
                # Pool query: small std (matches design)
                nn.init.normal_(aux_head.pool_query, mean=0.0, std=0.02)

        # Spatial-language Method A/B — trainable tokenizer/projector/refiner
        # params need the same explicit reset when built under HF no_init.
        sl = getattr(self, "spatial_lang", None)
        if sl is not None:
            try:
                sl.reset_trainable_parameters()
                sl.freeze_backbones()
            except Exception:
                pass
        for attr in ("spatial_refiner", "spatial_injection_layers"):
            mod = getattr(getattr(self, "transformer", None), attr, None)
            if mod is not None:
                for m in mod.modules():
                    _reset_module(m)

        # GeoStack-XVLA v2C — wrist GeoStack reinit (same no_init_weights problem).
        wrist_tb = getattr(self, "wrist_da3_token_builder", None)
        if wrist_tb is not None:
            gc_cfg = self.geometry_cfg
            for m in wrist_tb.modules():
                _reset_module(m)
            with torch.no_grad():
                if getattr(wrist_tb, "g_ray", None) is not None:
                    wrist_tb.g_ray.fill_(float(gc_cfg.get("geostack_g_ray_init", -4.0)))
                if getattr(wrist_tb, "g_pos", None) is not None:
                    wrist_tb.g_pos.fill_(float(gc_cfg.get("geostack_g_pos_init", -2.0)))
        wgs = getattr(self, "wrist_geostack", None)
        if wgs is not None:
            gc_cfg = self.geometry_cfg
            # v2D-aggressive: optional non-zero wrist out_proj init
            wrist_op_std = float(gc_cfg.get("wrist_geostack_out_proj_init_std", 0.0))
            # Re-init per-view fusers (same identity-at-step-0 as main GeoStack)
            for xa in wgs.per_view_fusers:
                for m in xa.modules():
                    _reset_module(m)
                with torch.no_grad():
                    if wrist_op_std > 0.0:
                        nn.init.normal_(xa.attn.out_proj.weight, mean=0.0, std=wrist_op_std)
                    else:
                        nn.init.zeros_(xa.attn.out_proj.weight)
                    if xa.attn.out_proj.bias is not None:
                        nn.init.zeros_(xa.attn.out_proj.bias)
                    nn.init.constant_(xa.gate_mlp[-1].bias, float(
                        gc_cfg.get("wrist_geostack_gate_init_bias", -4.0)))
                    nn.init.normal_(xa.gate_mlp[-1].weight, mean=0.0, std=1e-4)
                    if getattr(xa, "spatial_bias_lambda", None) is not None:
                        xa.spatial_bias_lambda.fill_(_math.log(_math.expm1(float(
                            gc_cfg.get("wrist_geostack_spatial_bias_lambda_init", 1.0)))))
            # Re-init world_ray_emb (just reset_parameters + gate init)
            if getattr(wgs, "world_ray_emb", None) is not None:
                for m in wgs.world_ray_emb.modules():
                    _reset_module(m)
                with torch.no_grad():
                    wgs.world_ray_emb.g.fill_(float(
                        gc_cfg.get("wrist_geostack_ray_g_init", -2.0)))

    # ============================ GeoStack-XVLA v2B init ============================
    def _init_geostack(self) -> None:
        """Build the GeoStack components and rebind the Florence encoder.

        Constructs:
          self.da3_for_geostack      : DA3-Large frozen feature + ray extractor
          self.geostack_alpha_schedule: AlphaSchedule controlling residual gain
          self.geostack_token_builders: ModuleDict {level_name: DA3SpatialTokenBuilder}
          self.geostack_fusers       : ModuleDict {layer_idx_str: GeoStackCrossAttention}

        Rebinds the Florence encoder class to GeoFusedFlorence2Encoder (in place;
        preserves all pretrained weights) and attaches the schedule + fusers.

        Applies Florence freeze policy per config.
        """
        from .da3_for_geostack import DA3LargeForGeoStack
        from .geostack import (
            AlphaSchedule,
            DA3SpatialTokenBuilder,
            GeoFusedFlorence2Encoder,
            GeoStackCrossAttention,
        )

        gc = self.geometry_cfg
        vlm_hidden = int(self.config.florence_config.text_config.d_model)
        feat_levels = list(gc.get("geostack_feature_levels", ["shallow", "mid", "deep"]))
        out_layers = list(gc.get("geostack_da3_out_layers", [11, 19, 23]))
        inject_layers = list(gc.get("geostack_inject_layers", [6, 9, 11]))
        # level_map keys may be ints or strings depending on yaml/JSON load path
        raw_map = gc.get("geostack_inject_level_map", {"6": "mid", "9": "mid", "11": "deep"})
        inject_level_map: Dict[int, str] = {int(k): str(v) for k, v in raw_map.items()}

        if len(feat_levels) != len(out_layers):
            raise ValueError(
                f"GeoStack: len(feature_levels)={len(feat_levels)} != "
                f"len(da3_out_layers)={len(out_layers)}"
            )
        for li in inject_layers:
            if li not in inject_level_map:
                raise ValueError(
                    f"GeoStack: inject_layer {li} not in inject_level_map keys "
                    f"{list(inject_level_map.keys())}"
                )
            if inject_level_map[li] not in feat_levels:
                raise ValueError(
                    f"GeoStack: inject_level_map[{li}]='{inject_level_map[li]}' not in "
                    f"feature_levels {feat_levels}"
                )

        # ---------- DA3-Large frozen encoder ----------
        # GeoStack-XVLA v2B: Florence receives 224×224 stretched from X-VLA's
        # CLIPImageProcessor. DA3 receives an ASPECT-CORRECT rectangle
        # (default 252×336 = ~1.05× of 240×320 = 4:3 aspect) — both dims chosen
        # as multiples of DA3 patch_size=14 so the ViT token-grid is clean.
        # The bilinear resize from Florence's 224×224 to (da3_input_h, da3_input_w)
        # "undoes" Florence's stretch — the resulting 252×336 represents the
        # original-image content at its native aspect.
        self.da3_for_geostack = DA3LargeForGeoStack(
            model_name=str(gc.get("geostack_da3_model", "depth-anything/DA3-Large-1.1")),
            out_layers=tuple(out_layers),
            da3_input_h=int(gc.get("geostack_da3_input_h", 252)),
            da3_input_w=int(gc.get("geostack_da3_input_w", 336)),
            patch_size=int(gc.get("geostack_da3_patch_size", 14)),
            use_bf16=True,
        )
        # Save derived constants for later coord-building
        self._geostack_h_grid = int(self.da3_for_geostack.h_grid)       # e.g., 18
        self._geostack_w_grid = int(self.da3_for_geostack.w_grid)       # e.g., 24
        self._geostack_inject_layers = tuple(int(i) for i in inject_layers)
        self._geostack_inject_level_map = {int(k): str(v) for k, v in inject_level_map.items()}
        self._geostack_level_name_to_idx = {name: i for i, name in enumerate(feat_levels)}
        self._geostack_feat_levels = list(feat_levels)
        self._geostack_da3_channels = int(self.da3_for_geostack.embed_dim)
        # Florence visual-token grid (used for VLM coord computation).
        # Florence-2-large with X-VLA's 224×224 input → DaViT 4-stage 32× downsample
        # → 7×7 = 49 vision tokens. We allow override via config but default 7×7.
        self._geostack_florence_h = int(gc.get("geostack_florence_visual_h", 7))
        self._geostack_florence_w = int(gc.get("geostack_florence_visual_w", 7))

        # ---------- Alpha schedule ----------
        self.geostack_alpha_schedule = AlphaSchedule(
            warmup_iters=int(gc.get("geostack_alpha_warmup_iters", 5000)),
            ramp_to_01_iters=int(gc.get("geostack_alpha_ramp_to_01_iters", 10000)),
            ramp_to_1_iters=int(gc.get("geostack_alpha_ramp_to_1_iters", 20000)),
            alpha_at_warmup_end=float(gc.get("geostack_alpha_at_warmup_end", 0.001)),
        )

        # ---------- Per-level token builders ----------
        self.geostack_token_builders = nn.ModuleDict()
        for level_name in feat_levels:
            self.geostack_token_builders[level_name] = DA3SpatialTokenBuilder(
                c_in=self._geostack_da3_channels,
                d_model=vlm_hidden,
                ray_in=3,
                use_ray=bool(gc.get("geostack_use_ray", True)),
                use_2d_pos=bool(gc.get("geostack_use_2d_pos", True)),
                g_ray_init=float(gc.get("geostack_g_ray_init", -4.0)),
                g_pos_init=float(gc.get("geostack_g_pos_init", -2.0)),
            )

        # ---------- Per-injection-layer cross-attn fusers ----------
        self.geostack_fusers = nn.ModuleDict()
        for li in inject_layers:
            self.geostack_fusers[str(li)] = GeoStackCrossAttention(
                hidden_dim=vlm_hidden,
                num_heads=int(gc.get("geostack_num_heads", 8)),
                spatial_token_dim=vlm_hidden,   # builders project to vlm_hidden already
                use_spatial_bias=bool(gc.get("geostack_use_spatial_bias", True)),
                spatial_bias_lambda_init=float(gc.get("geostack_spatial_bias_lambda_init", 1.0)),
                gate_mlp_hidden=gc.get("geostack_gate_mlp_hidden", None),
                gate_init_bias=float(gc.get("geostack_gate_init_bias", -4.0)),
                dropout=float(gc.get("geostack_dropout", 0.0)),
            )

        # ---------- Rebind Florence encoder ----------
        enc = self.vlm.language_model.model.encoder
        enc.__class__ = GeoFusedFlorence2Encoder
        enc.attach_geostack(
            geo_inject_layers=tuple(int(i) for i in inject_layers),
            geo_fusers=self.geostack_fusers,
            alpha_schedule=self.geostack_alpha_schedule,
        )

        # ---------- Florence freeze policy ----------
        # NB: after attach_geostack, self.geostack_fusers is also a submodule of
        # self.vlm.language_model.model.encoder. A blanket "for p in self.vlm.parameters"
        # therefore freezes the GeoStack fusers too — we have to RE-enable them
        # after the blanket freeze.
        if bool(gc.get("geostack_freeze_florence", True)):
            for p in self.vlm.parameters():
                p.requires_grad_(False)
            if bool(gc.get("geostack_train_top_florence_encoder_layers", False)):
                trainable = set(int(i) for i in gc.get("geostack_trainable_florence_encoder_layers", []))
                for i, layer in enumerate(self.vlm.language_model.model.encoder.layers):
                    if i in trainable:
                        for p in layer.parameters():
                            p.requires_grad_(True)
                logger.info("[GeoStack] top Florence encoder layers unfrozen: %s", sorted(trainable))

        # Adapter trainability (re-enable after Florence freeze swept them).
        adapter_trainable = bool(gc.get("geostack_train_adapters", True))
        for mod in (self.geostack_token_builders, self.geostack_fusers):
            for p in mod.parameters():
                p.requires_grad_(adapter_trainable)

        logger.info("[GeoStack] enabled — DA3-Large frozen, main-cam only")
        logger.info(
            "[GeoStack]   inject_layers=%s, level_map=%s, alpha=%s",
            inject_layers, inject_level_map, self.geostack_alpha_schedule.extra_repr()
        )
        logger.info(
            "[GeoStack]   use_ray=%s, use_2d_pos=%s, use_spatial_bias=%s, freeze_florence=%s",
            bool(gc.get("geostack_use_ray", True)),
            bool(gc.get("geostack_use_2d_pos", True)),
            bool(gc.get("geostack_use_spatial_bias", True)),
            bool(gc.get("geostack_freeze_florence", True)),
        )

    # ============================ GeoStack training step ============================
    def set_geostack_step(self, step: int) -> None:
        """Update the residual-alpha schedule's current iteration. Call from the
        training loop each iter so alpha ramps according to schedule. No-op when
        GeoStack is disabled."""
        if self.geostack_alpha_schedule is not None:
            self.geostack_alpha_schedule.set_step(int(step))
        # v3: also forward step to the side stack alpha schedule (lives on transformer).
        if hasattr(self, "transformer") and hasattr(self.transformer, "set_side_alpha_step"):
            self.transformer.set_side_alpha_step(int(step))

    def set_spatial_residual_scale(self, scale: float) -> None:
        """Set scheduled Method A/B spatial residual scale. No-op when disabled."""
        if hasattr(self, "transformer") and hasattr(self.transformer, "set_spatial_residual_scale"):
            self.transformer.set_spatial_residual_scale(float(scale))

    # ============================ GeoStack v2C: wrist-aware aux GeoStack ============================
    def _init_wrist_geostack(self) -> None:
        """Build the WristAuxGeoStack adapter (per-wrist-view cross-attn on
        aux_visual_inputs). Shares the DA3 instance + alpha schedule with v2B.

        New trainable params (auto-collected by build_optimizer's `geometry`
        group via the `geo_fusers` walk — see modeling_xvla wrist_geostack
        attribute is intentionally named to land in the same submodule registry).
        """
        from .geostack import WristAuxGeoStack

        gc = self.geometry_cfg
        vlm_hidden = int(self.vlm.config.text_config.d_model)
        wrist_views = list(gc.get("wrist_geostack_views", [1, 2]))
        if not wrist_views:
            raise ValueError("use_wrist_geostack=True but wrist_geostack_views is empty")

        self._wrist_geostack_views = tuple(int(v) for v in wrist_views)
        # DA3 input resolution for wrists (typically 224×224, lower than main 252×336).
        self._wrist_da3_input_h = int(gc.get("wrist_geostack_da3_input_h", 224))
        self._wrist_da3_input_w = int(gc.get("wrist_geostack_da3_input_w", 224))
        patch = int(gc.get("geostack_da3_patch_size", 14))
        if self._wrist_da3_input_h % patch != 0 or self._wrist_da3_input_w % patch != 0:
            raise ValueError(
                f"wrist DA3 input ({self._wrist_da3_input_h},{self._wrist_da3_input_w}) "
                f"must be multiples of patch_size={patch}"
            )
        self._wrist_da3_h_grid = self._wrist_da3_input_h // patch
        self._wrist_da3_w_grid = self._wrist_da3_input_w // patch

        self._wrist_aux_h = int(gc.get("wrist_geostack_aux_h", 7))
        self._wrist_aux_w = int(gc.get("wrist_geostack_aux_w", 7))

        # Dedicated DA3 token builder for wrist (NOT shared with main "deep").
        # Sharing the builder caused NaN gradients in backward — torch autograd
        # complains when the same parameter is used in multiple parallel branches
        # at the same forward level. Fresh init mirrors main's "deep" config.
        from .geostack import DA3SpatialTokenBuilder
        self.wrist_da3_token_builder = DA3SpatialTokenBuilder(
            c_in=int(self.da3_for_geostack.embed_dim),
            d_model=vlm_hidden,
            ray_in=3,
            use_ray=bool(gc.get("geostack_use_ray", True)),
            use_2d_pos=bool(gc.get("geostack_use_2d_pos", True)),
            g_ray_init=float(gc.get("geostack_g_ray_init", -4.0)),
            g_pos_init=float(gc.get("geostack_g_pos_init", -2.0)),
        )

        self.wrist_geostack = WristAuxGeoStack(
            hidden_dim=vlm_hidden,
            num_wrist_views=len(self._wrist_geostack_views),
            num_heads=int(gc.get("wrist_geostack_num_heads", 8)),
            use_spatial_bias=bool(gc.get("wrist_geostack_use_spatial_bias", True)),
            spatial_bias_lambda_init=float(gc.get("wrist_geostack_spatial_bias_lambda_init", 1.0)),
            gate_mlp_hidden=gc.get("wrist_geostack_gate_mlp_hidden", None),
            gate_init_bias=float(gc.get("wrist_geostack_gate_init_bias", -4.0)),
            use_world_ray=bool(gc.get("wrist_geostack_use_world_ray", True)),
            ray_mlp_hidden=int(gc.get("wrist_geostack_ray_mlp_hidden", 256)),
            ray_g_init=float(gc.get("wrist_geostack_ray_g_init", -2.0)),
            dropout=float(gc.get("wrist_geostack_dropout", 0.0)),
        )
        logger.info(
            "[WristGeoStack v2C] enabled — views=%s, da3_input=(%d,%d), aux_grid=(%d,%d), use_world_ray=%s, gate_init=%.2f",
            self._wrist_geostack_views, self._wrist_da3_input_h, self._wrist_da3_input_w,
            self._wrist_aux_h, self._wrist_aux_w,
            bool(gc.get("wrist_geostack_use_world_ray", True)),
            float(gc.get("wrist_geostack_gate_init_bias", -4.0)),
        )

    def _apply_wrist_geostack(
        self,
        aux_visual_inputs: torch.Tensor,        # [B, T_aux, D]
        pixel_values: torch.FloatTensor,        # [B, V, 3, H, W]  (X-VLA processor output)
        extrinsics: Optional[torch.Tensor],     # [B, V, 4, 4] world-to-camera (cv) or None
    ) -> torch.Tensor:
        """v2C: enrich aux_visual_inputs by per-wrist DA3 cross-attention.

        Operates on slices of aux_visual_inputs corresponding to each wrist view.
        Florence's aux_visual_inputs is the flattened concat of views 1..V-1
        (the first view goes through the encoder); each view contributes
        N_aux = aux_h * aux_w tokens (e.g., 49 for 7×7 grid).
        """
        if self.wrist_geostack is None:
            return aux_visual_inputs
        alpha = self.geostack_alpha_schedule.value() if self.geostack_alpha_schedule is not None else 0.0
        if alpha <= 0.0:
            return aux_visual_inputs
        from .geostack import build_pos_orig_da3, build_pos_orig_vlm
        from .side_stack import compute_world_ray_6d

        B, V = pixel_values.shape[:2]
        device = pixel_values.device

        # Gather wrist-view pixels in the order specified by config
        wrist_view_indices = [v for v in self._wrist_geostack_views if v < V]
        if len(wrist_view_indices) == 0:
            return aux_visual_inputs
        wrist_pixels = torch.stack([pixel_values[:, v] for v in wrist_view_indices], dim=1)  # [B, Vw, 3, H, W]

        # Run DA3 on all wrists in one batched forward
        da3_out = self.da3_for_geostack.forward_multi_view(
            wrist_pixels,
            target_input_hw=(self._wrist_da3_input_h, self._wrist_da3_input_w),
        )
        # da3_out["feats"]: list of [B, Vw, C, h, w]   (one entry per out_layer)
        # da3_out["ray"]:   [B, Vw, 3, h, w]
        feats_per_layer = da3_out["feats"]
        ray_dir = da3_out["ray"]  # [B, Vw, 3, h, w]
        h_grid = int(da3_out["h_grid"])
        w_grid = int(da3_out["w_grid"])

        # Use the DEEPEST tap (last in list) — matches main GeoStack's "deep" level
        deep_feat = feats_per_layer[-1]  # [B, Vw, C, h, w]
        Vw, Cd = deep_feat.shape[1], deep_feat.shape[2]

        # Project DA3 features through the WRIST-DEDICATED token builder
        # (NOT shared with main's "deep" builder — sharing caused NaN gradients).
        # The builder expects [B, K, C] feat + [B, K, 2] coords + [B, K, 3] ray.
        # We process each wrist view separately to keep coord/ray semantics clean.
        builder = self.wrist_da3_token_builder
        da3_coords = build_pos_orig_da3(h_grid=h_grid, w_grid=w_grid,
                                        device=device, dtype=torch.float32)  # [1, K, 2]
        coords_b = da3_coords.expand(B, -1, -1)

        da3_tokens_per_view: List[torch.Tensor] = []
        world_ray_per_view: Optional[List[torch.Tensor]] = [] if (
            self.wrist_geostack.use_world_ray and extrinsics is not None
        ) else None
        for vi, v in enumerate(wrist_view_indices):
            feat_v = deep_feat[:, vi]                                 # [B, C, h, w]
            ray_v = ray_dir[:, vi]                                    # [B, 3, h, w]
            feat_flat = feat_v.permute(0, 2, 3, 1).reshape(B, h_grid * w_grid, Cd).contiguous()
            ray_flat = ray_v.permute(0, 2, 3, 1).reshape(B, h_grid * w_grid, 3).contiguous()
            tokens = builder(feat=feat_flat, coords=coords_b.to(feat_flat.dtype), ray=ray_flat)  # [B, K, D]
            da3_tokens_per_view.append(tokens)

            if world_ray_per_view is not None:
                # extrinsics: [B, V, 4, 4] world-to-camera (sample["extrinsics"] from base.py)
                ext_v = extrinsics[:, v]                              # [B, 4, 4]
                world_ray_6d = compute_world_ray_6d(ray_v, ext_v)     # [B, 6, h, w]
                world_ray_flat = world_ray_6d.permute(0, 2, 3, 1).reshape(B, h_grid * w_grid, 6).contiguous()
                world_ray_per_view.append(world_ray_flat)

        # Aux view token ranges. Florence aux_visual_inputs layout:
        #   [view_1_tokens (N_per_view) | view_2_tokens (N_per_view) | ...]
        # Florence DaViT emits N_per_view = grid_h*grid_w + 1 (extra global-pooled
        # token at the end). For 224×224 input → 7×7 grid + 1 global = 50.
        # We derive N_per_view from the actual aux shape divided by total aux views
        # (which is V-1, since main view 0 isn't in aux).
        T_aux = aux_visual_inputs.shape[1]
        num_aux_views = V - 1                                 # views 1..V-1 are in aux
        if T_aux % num_aux_views != 0:
            raise RuntimeError(
                f"[WristGeoStack] aux_visual_inputs shape {T_aux} not divisible by "
                f"num_aux_views={num_aux_views}; Florence layout unexpected"
            )
        N_per_view = T_aux // num_aux_views                   # 50 typically (49 spatial + 1 global)
        N_spatial = int(self._wrist_aux_h * self._wrist_aux_w)  # 49 (grid only)
        if N_per_view < N_spatial:
            raise RuntimeError(
                f"[WristGeoStack] N_per_view={N_per_view} < N_spatial={N_spatial}"
            )
        # Slot per wrist view inside aux_visual_inputs: views[v=1] is at offset 0.
        aux_view_token_ranges: Dict[int, Tuple[int, int]] = {}
        for vi, v in enumerate(wrist_view_indices):
            start = (v - 1) * N_per_view
            end = start + N_per_view
            if end > T_aux:
                continue
            aux_view_token_ranges[vi] = (start, end)

        # Aux coords per view: first N_spatial are real grid coords, the remaining
        # (typically the single global-pooled token) get sentinel coords (0.5, 0.5).
        # We ALSO build a vision_mask: True for the spatial-grid prefix (proper
        # spatial-distance bias against DA3 tokens), False for global/pooled
        # tokens (their bias is zeroed → uniform attention over DA3 K/V, which
        # is the right behavior for a global-aggregator token).
        grid_coords = build_pos_orig_vlm(h_grid=self._wrist_aux_h, w_grid=self._wrist_aux_w,
                                         device=device, dtype=torch.float32)   # [1, N_spatial, 2]
        if N_per_view > N_spatial:
            sentinel = grid_coords.new_full((1, N_per_view - N_spatial, 2), 0.5)
            aux_coords = torch.cat([grid_coords, sentinel], dim=1)             # [1, N_per_view, 2]
        else:
            aux_coords = grid_coords
        vision_mask = torch.zeros(1, N_per_view, dtype=torch.bool, device=device)
        vision_mask[:, :N_spatial] = True       # spatial bias applies only to grid tokens

        aux_coords_per_view: Dict[int, torch.Tensor] = {vi: aux_coords for vi in range(len(wrist_view_indices))}
        aux_vision_mask_per_view: Dict[int, torch.Tensor] = {vi: vision_mask for vi in range(len(wrist_view_indices))}
        spatial_coords_per_view: List[torch.Tensor] = [da3_coords for _ in wrist_view_indices]

        return self.wrist_geostack(
            aux_visual_inputs=aux_visual_inputs,
            da3_tokens_per_view=da3_tokens_per_view,
            spatial_coords_per_view=spatial_coords_per_view,
            aux_view_token_ranges=aux_view_token_ranges,
            aux_coords_per_view=aux_coords_per_view,
            aux_vision_mask_per_view=aux_vision_mask_per_view,
            world_ray_per_view=world_ray_per_view,
            alpha=alpha,
        )

    # ============================ GeoStack v3 side-stack bank ============================
    def _init_side_stack_bank_builder(self) -> None:
        """Construct the SideStackBankBuilder for v3 side stack.

        Requires:
          - self.da3_for_geostack (built in _init_geostack — needs use_geostack=True too)
          - self.t5_inline / self.t5_projector (built earlier — needs t5_language_enabled=True)
        Or builds a private DA3 extractor if v2B isn't enabled (so v3 can run standalone).
        """
        from .side_stack import SideStackBankBuilder
        gc = self.geometry_cfg
        side_cfg = gc.get("side_stack", {}) or {}
        side_hidden = int(side_cfg.get("hidden_dim", self.config.hidden_size))

        # Ensure a DA3 extractor exists (use v2B's if available, else build private)
        if self.da3_for_geostack is None:
            from .da3_for_geostack import DA3LargeForGeoStack
            logger.info("[v3] DA3-Large not built by v2B path — building private extractor")
            self.da3_for_geostack = DA3LargeForGeoStack(
                model_name=str(side_cfg.get("da3_model", "depth-anything/DA3-Large-1.1")),
                out_layers=tuple(side_cfg.get("da3_out_layers", [11, 19, 23])),
                da3_input_h=int(side_cfg.get("da3_input_h", 378)),
                da3_input_w=int(side_cfg.get("da3_input_w", 504)),
                patch_size=int(side_cfg.get("da3_patch_size", 14)),
                use_bf16=True,
            )
            # Save derived constants
            self._geostack_h_grid = int(self.da3_for_geostack.h_grid)
            self._geostack_w_grid = int(self.da3_for_geostack.w_grid)
            self._geostack_da3_channels = int(self.da3_for_geostack.embed_dim)

        # T5 must be enabled — if not, raise (the user wanted T5 in v3)
        if self.t5_inline is None:
            raise RuntimeError(
                "[v3] side_stack enabled but t5_language_enabled=False. "
                "Set geometry_conditioning.t5_language_enabled=True too."
            )

        # Multi-view DA3 config — v3.1: optional wrist cameras
        mv_cfg = side_cfg.get("multi_view_da3", {}) or {}
        self._side_multi_view_enabled = bool(mv_cfg.get("enabled", False))
        self._side_main_view = int(mv_cfg.get("main_view", 0))
        self._side_main_input_hw = tuple(mv_cfg.get("main_da3_input_hw",
                                          [self.da3_for_geostack.da3_input_h,
                                           self.da3_for_geostack.da3_input_w]))
        self._side_wrist_views = tuple(mv_cfg.get("wrist_views", []))
        self._side_wrist_input_hw = tuple(mv_cfg.get("wrist_da3_input_hw", [168, 224]))
        self._side_num_views = int(mv_cfg.get(
            "num_views", 1 + len(self._side_wrist_views)
        ))
        self._side_use_view_emb = bool(mv_cfg.get("use_view_emb", True))
        # v3.2: Plücker rays in world frame (requires extrinsics in forward call)
        # v3.2 — world-frame ray (origin || direction in world coords, 6D per pixel)
        self._side_use_world_ray = bool(side_cfg.get("use_world_ray_6d", False))
        # ray_in_dim must match what's fed in: 3 if local direction only, 6 if world ray
        side_ray_in_dim = 6 if self._side_use_world_ray else 3

        self.side_stack_bank_builder = SideStackBankBuilder(
            da3_in_dim=self._geostack_da3_channels,           # DA3-Large = 1024
            t5_in_dim=self.t5_inline.hidden_size,             # t5-base = 768
            hidden_dim=side_hidden,
            use_ray=bool(side_cfg.get("use_ray", True)),
            use_2d_pos=bool(side_cfg.get("use_2d_pos", True)),
            use_modality_emb=bool(side_cfg.get("use_modality_emb", True)),
            g_ray_init=float(side_cfg.get("g_ray_init", -4.0)),
            g_pos_init=float(side_cfg.get("g_pos_init", -2.0)),
            ray_in_dim=side_ray_in_dim,
            num_views=self._side_num_views if self._side_multi_view_enabled else 1,
            use_view_emb=self._side_use_view_emb if self._side_multi_view_enabled else False,
        )
        # Which DA3 out_layer index to tap (0=shallow, 1=mid, 2=deep)
        self._side_da3_tap_idx = int(side_cfg.get("da3_tap_idx", -1))   # default = last (deep)
        logger.info(
            "[v3] side stack bank builder ready  (da3_in=%d, t5_in=%d, hidden=%d, "
            "use_ray=%s, use_world_ray_6d=%s (ray_in_dim=%d), use_2d_pos=%s, "
            "da3_tap_idx=%d, multi_view=%s, "
            "main_input_hw=%s, wrist_views=%s, wrist_input_hw=%s)",
            self._geostack_da3_channels, self.t5_inline.hidden_size, side_hidden,
            side_cfg.get("use_ray", True), self._side_use_world_ray, side_ray_in_dim,
            side_cfg.get("use_2d_pos", True), self._side_da3_tap_idx,
            self._side_multi_view_enabled,
            self._side_main_input_hw, self._side_wrist_views, self._side_wrist_input_hw,
        )

    def _init_side_stack_aux_head(self, aux_cfg: dict) -> None:
        """v3.5: instantiate the SideStackAuxHead for the auxiliary action loss.

        Trained components: side stack final output [B, K, hidden] → attention pool →
        2-layer MLP → predicted action chunk [B, num_actions, dim_action]. Loss is
        action_space.compute_loss against ground-truth action chunk, scaled by
        `aux_loss.weight` (default 0.1) before adding to the loss dict.

        Why: gives side stack direct gradient (independent of cross-attn pull from
        action expert). Cross-attn gradient empirically went to noise in v3.2g/3/4
        because side stack started random → cross-attn output was random → action
        expert ignored it → gradient back to side stack was zero-mean noise → AdamW
        updates cancelled → side stack stayed random. Aux loss breaks this loop.
        """
        from .side_stack import SideStackAuxHead
        side_cfg = self.geometry_cfg.get("side_stack", {}) or {}
        side_hidden = int(side_cfg.get("hidden_dim", self.config.hidden_size))
        num_actions = int(self.config.num_actions)
        # Use action_space.dim_action — the actual dim the loss function expects.
        # This matches max_action_dim for ee6d but is robust to other action modes
        # (e.g., 'joint' has different dim_action vs max_action_dim).
        dim_action = int(self.action_space.dim_action)

        # v3.6: optional proprio fusion. Without proprio, aux head must predict
        # actions from ONLY pooled spatial-language features (impossible for
        # trajectory smoothness, hard for many manipulation tasks). With proprio,
        # the aux task becomes "given scene+language+current_state, predict next
        # action chunk" — tractable BC signal that still forces side stack to
        # encode useful spatial features (the proprio path alone can't carry the
        # full task).
        use_proprio_in_aux = bool(aux_cfg.get("use_proprio", True))
        proprio_dim = None
        if use_proprio_in_aux:
            proprio_dim = int(getattr(self.action_space, "dim_proprio", dim_action))

        self.side_stack_aux_head = SideStackAuxHead(
            hidden_dim=side_hidden,
            num_actions=num_actions,
            dim_action=dim_action,
            mlp_ratio=float(aux_cfg.get("mlp_ratio", 2.0)),
            pool_init_std=float(aux_cfg.get("pool_init_std", 0.02)),
            proprio_dim=proprio_dim,
        )
        logger.info(
            "[v3.5/3.6] side stack aux head ready  (hidden=%d, num_actions=%d, dim_action=%d, "
            "mlp_ratio=%.1f, loss_weight=%.3f, proprio_dim=%s)",
            side_hidden, num_actions, dim_action,
            float(aux_cfg.get("mlp_ratio", 2.0)),
            self.side_stack_aux_weight,
            proprio_dim,
        )

    def _build_spatial_lang_bank(self, image_input, language_instruction, extrinsics=None):
        """Build the v3 side stack bank from DA3 + T5.

        Args:
            image_input: either [B, 3, H, W] (single view) or [B, V, 3, H, W] (multi-view).
                If multi-view is enabled in config, this MUST be 5D so we can index views.
            language_instruction: list[str] of length B, or None.
            extrinsics: optional [B, V, 4, 4] OpenCV world-to-camera extrinsics.
                When provided AND `use_world_ray_rays` is True (v3.2), DA3's per-pixel
                ray directions are converted to world-frame Plücker rays (6D per
                pixel: direction || moment), giving the model TRUE 3D-world ray
                grounding. Without extrinsics, falls back to local-frame direction
                only (3D), which is per-view 2D only — model can't relate views in 3D.

        Returns (bank, bank_pad) or (None, None) if v3 is not enabled.

        Single-view mode: ONE DA3 forward at the configured resolution.
        Multi-view mode:  TWO DA3 forwards (main at main_input_hw, wrists batched
                          at wrist_input_hw — typically smaller).
        """
        if not self.side_stack_enabled or self.side_stack_bank_builder is None:
            return None, None
        from .geostack import build_pos_orig_da3
        from .side_stack import compute_world_ray_6d
        use_world_ray = bool(getattr(self, "_side_use_world_ray", False)) and (extrinsics is not None)

        # T5 forward (same for both single/multi view)
        t5_feats, t5_mask = self.t5_inline(language_instruction)  # [B, L, d_t5], [B, L]

        # ---------------- Single-view path ----------------
        if not getattr(self, "_side_multi_view_enabled", False):
            # Accept [B, 3, H, W] or [B, V, 3, H, W] and use first view
            if image_input.dim() == 5:
                pixels = image_input[:, self._side_main_view]
            else:
                pixels = image_input
            da3_out = self.da3_for_geostack(pixels)
            feats, ray = da3_out["feats"], da3_out["ray"]
            h_grid, w_grid = int(da3_out["h_grid"]), int(da3_out["w_grid"])
            da3_feat = feats[self._side_da3_tap_idx]
            da3_coords = build_pos_orig_da3(
                h_grid=h_grid, w_grid=w_grid,
                device=da3_feat.device, dtype=torch.float32,
            )
            # v3.2: convert local-frame ray dir → world-frame Plücker if extrinsics provided
            if use_world_ray:
                ray = compute_world_ray_6d(ray, extrinsics[:, self._side_main_view])
            return self.side_stack_bank_builder(
                da3_feat=da3_feat,
                da3_coords=da3_coords,
                ray=ray,
                t5_feat=t5_feats,
                t5_mask=t5_mask,
            )

        # ---------------- Multi-view path ----------------
        if image_input.dim() != 5:
            raise ValueError(
                f"[v3 multi-view] image_input must be 5D [B, V, 3, H, W]; "
                f"got {tuple(image_input.shape)}"
            )
        B, V = image_input.shape[:2]
        view_blocks = []

        # ----- Main camera DA3 forward at near-raw resolution -----
        main_pixels = image_input[:, self._side_main_view]              # [B, 3, H, W]
        main_out = self.da3_for_geostack(
            main_pixels, target_input_hw=self._side_main_input_hw,
        )
        main_feat = main_out["feats"][self._side_da3_tap_idx]            # [B, C, h_m, w_m]
        main_ray = main_out["ray"]                                       # [B, 3, h_m, w_m]
        main_coords = build_pos_orig_da3(
            h_grid=int(main_out["h_grid"]), w_grid=int(main_out["w_grid"]),
            device=main_feat.device, dtype=torch.float32,
        )
        # v3.2: Plücker rays in world frame
        if use_world_ray:
            main_ray = compute_world_ray_6d(main_ray, extrinsics[:, self._side_main_view])
        view_blocks.append({
            "feat": main_feat, "coords": main_coords, "ray": main_ray,
            "view_idx": self._side_main_view,
        })

        # ----- Wrist cameras: batched DA3 forward at smaller resolution -----
        if len(self._side_wrist_views) > 0:
            # Validate all wrist views are present
            valid_wrist_indices = [v for v in self._side_wrist_views if v < V]
            if len(valid_wrist_indices) > 0:
                # Stack wrist views along batch dim → [B * Nw, 3, H, W]
                wrist_stack = torch.cat(
                    [image_input[:, v] for v in valid_wrist_indices], dim=0,
                )
                wrist_out = self.da3_for_geostack(
                    wrist_stack, target_input_hw=self._side_wrist_input_hw,
                )
                wrist_feat_all = wrist_out["feats"][self._side_da3_tap_idx]   # [B*Nw, C, h_w, w_w]
                wrist_ray_all = wrist_out["ray"]                              # [B*Nw, 3, h_w, w_w]
                h_w, w_w = int(wrist_out["h_grid"]), int(wrist_out["w_grid"])
                wrist_coords = build_pos_orig_da3(
                    h_grid=h_w, w_grid=w_w,
                    device=wrist_feat_all.device, dtype=torch.float32,
                )
                # Split back into individual wrist views (and apply per-view Plücker)
                for i, v in enumerate(valid_wrist_indices):
                    feat_v = wrist_feat_all[i * B : (i + 1) * B]
                    ray_v = wrist_ray_all[i * B : (i + 1) * B]
                    # v3.2: per-wrist Plücker in world frame
                    if use_world_ray:
                        ray_v = compute_world_ray_6d(ray_v, extrinsics[:, v])
                    view_blocks.append({
                        "feat": feat_v, "coords": wrist_coords, "ray": ray_v,
                        "view_idx": v,
                    })

        # Build multi-view bank
        return self.side_stack_bank_builder.forward_multi_view(
            da3_views=view_blocks,
            t5_feat=t5_feats,
            t5_mask=t5_mask,
        )

    def _geostack_run_da3_and_build_per_layer(
        self,
        first_view_pixels: torch.Tensor,   # [B, 3, H, W]
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor], torch.Tensor]:
        """Run DA3 on the main camera and build per-injection-layer spatial tokens.

        Returns:
          geo_kv:           {layer_idx: [B, K_l, D]}   spatial tokens for cross-attn
          geo_da3_coords:   {layer_idx: [1, K_l, 2]}   normalized original-image coords
          vlm_coords:       [1, T_vision, 2]           Florence vision-token coords only
                                                       (callers assemble full merged coords)
        """
        from .geostack import build_pos_orig_vlm, build_pos_orig_da3

        out = self.da3_for_geostack(first_view_pixels)
        feats: list = out["feats"]          # list of [B, C, h_grid, w_grid]
        ray: torch.Tensor = out["ray"]      # [B, 3, h_grid, w_grid]
        device = first_view_pixels.device
        B = first_view_pixels.shape[0]
        h_grid = int(out["h_grid"])
        w_grid = int(out["w_grid"])

        # DA3 token coords — uniform grid in normalized original-image coords
        # (no padding; the DA3 frame is an aspect-preserved resize so coord
        # mapping is the identity in [0,1]² ↔ original).
        da3_coords = build_pos_orig_da3(h_grid=h_grid, w_grid=w_grid,
                                        device=device, dtype=torch.float32)  # [1, K, 2]

        # Per-level spatial tokens
        per_level_tokens: Dict[str, torch.Tensor] = {}
        ray_flat = ray.permute(0, 2, 3, 1).reshape(B, h_grid * w_grid, 3).contiguous()
        coords_b = da3_coords.expand(B, -1, -1).to(ray_flat.dtype)
        for level_name, feat in zip(self._geostack_feat_levels, feats):
            B_, C_, h_, w_ = feat.shape
            feat_flat = feat.permute(0, 2, 3, 1).reshape(B_, h_ * w_, C_).contiguous()
            tokens = self.geostack_token_builders[level_name](
                feat=feat_flat, coords=coords_b, ray=ray_flat,
            )  # [B, K, D]
            per_level_tokens[level_name] = tokens

        # Assemble per-layer K/V + coords
        geo_kv: Dict[int, torch.Tensor] = {}
        geo_da3_coords: Dict[int, torch.Tensor] = {}
        for li in self._geostack_inject_layers:
            level_name = self._geostack_inject_level_map[li]
            geo_kv[li] = per_level_tokens[level_name]
            geo_da3_coords[li] = da3_coords     # [1, K, 2] broadcastable

        # Florence vision-token coords (linear-stretch back-mapping). For 224×224
        # input + DaViT 32× downsample → 7×7 grid.
        vlm_coords = build_pos_orig_vlm(
            h_grid=self._geostack_florence_h, w_grid=self._geostack_florence_w,
            device=device, dtype=torch.float32,
        )  # [1, h_v*w_v, 2]
        return geo_kv, geo_da3_coords, vlm_coords

    # ============================= Florence2 encoder =============================
    def forward_vlm(
        self,
        input_ids: torch.LongTensor,        # [B, L]
        pixel_values: torch.FloatTensor,    # [B, V, C, H, W]
        image_mask: torch.Tensor,           # [B, V] (bool or 0/1)
        extrinsics: Optional[torch.Tensor] = None,   # [B, V, 4, 4] OpenCV w2c — for v2C wrist GeoStack
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text + multi-view images via Florence2 encoder.

        Returns:
          { "vlm_features": [B, T_enc, D], "aux_visual_inputs": [B, (V-1)*N, D] }
        """
        B, V = pixel_values.shape[:2]
        flat_mask = image_mask.view(-1).to(torch.bool)         # [B*V]
        flat_images = pixel_values.flatten(0, 1)                # [B*V, C, H, W]

        # A4 from 2026-06-11 audit: fast path when every view is valid
        # (which is the case for all active datasets — robotwin2, x2robot,
        # lerobotv21, lerobot_agibot — that set image_mask = ones). The slow
        # path is preserved bit-identically for future masked-view ablations.
        # Saves 3 GPU→CPU syncs per iter (.sum().item() + 2 bool-mask scatters).
        if bool(flat_mask.all()):
            valid_feats = self.vlm._encode_image(flat_images)   # [B*V, N, D]
            N, D = valid_feats.shape[1:]
            image_features = valid_feats.view(B, V, N, D)       # [B, V, N, D]
        else:
            num_valid = int(flat_mask.sum().item())
            if num_valid == 0:
                raise ValueError("At least one image view must be valid per batch.")

            valid_images = flat_images[flat_mask]               # [#valid, C, H, W]
            valid_feats = self.vlm._encode_image(valid_images)  # [#valid, N, D]
            N, D = valid_feats.shape[1:]

            image_features = valid_feats.new_zeros((B * V, N, D))
            image_features[flat_mask] = valid_feats
            image_features = image_features.view(B, V, N, D)    # [B, V, N, D]

        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)  # [B, L, D]

        merged_embeds, attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features[:, 0],  # first view: [B, N, D]
            inputs_embeds,         # [B, L, D]
        )

        # =========== GeoStack: set encoder context BEFORE calling encoder ===========
        # When alpha == 0 (e.g. pre-warmup or use_geostack disabled), the encoder
        # short-circuits the injection and runs bit-identically to baseline.
        if self.use_geostack and self.da3_for_geostack is not None:
            alpha = self.geostack_alpha_schedule.value()
            if alpha > 0.0:
                first_view = pixel_values[:, 0]  # [B, 3, H, W]
                geo_kv, geo_coords, vision_coords = self._geostack_run_da3_and_build_per_layer(first_view)
                # ---- Build full merged-sequence coords + spatial mask ----
                # Florence merge layout (modeling_florence2.py:2647-2664):
                #   merged = [image_features (N_vision) | task_prefix_embeds (L_text)]
                # Florence's DaViT typically emits N_vision = h_grid*w_grid + 1
                # (an extra global-pooled token). The SPATIAL bias only applies to
                # the first h_grid*w_grid tokens; the global token + text tokens
                # get sentinel coords and are masked OUT of the bias (zero-bias
                # rows → no spatial preference).
                N_spatial = int(self._geostack_florence_h * self._geostack_florence_w)  # 49
                N_vision = int(image_features.shape[2])                                  # 50 typically
                T_enc = int(merged_embeds.shape[1])
                if vision_coords.shape[1] != N_spatial:
                    raise RuntimeError(
                        f"[GeoStack] vision-coord grid h*w={vision_coords.shape[1]} "
                        f"mismatch — update geostack_florence_visual_{{h,w}}."
                    )
                if N_vision < N_spatial:
                    raise RuntimeError(
                        f"[GeoStack] Florence emitted N_vision={N_vision} < N_spatial={N_spatial}; "
                        f"can't place {N_spatial} spatial-grid coords into {N_vision} vision slots."
                    )
                full_vlm_coords = vision_coords.new_full((1, T_enc, 2), 0.5)
                full_vlm_coords[:, :N_spatial, :] = vision_coords    # spatial grid slots get real coords
                # spatial_mask: True ONLY for the spatial-grid prefix. The global
                # vision token (if any) AND text tokens are False → bias zeroed.
                spatial_mask = torch.zeros(1, T_enc, dtype=torch.bool, device=first_view.device)
                spatial_mask[:, :N_spatial] = True
                vision_mask = spatial_mask     # name reused; spatial-mask semantics

                enc = self.vlm.language_model.model.encoder
                enc.set_geo_context(
                    geo_kv=geo_kv,
                    geo_da3_coords=geo_coords,
                    geo_da3_valid=None,
                    vlm_coords=full_vlm_coords,
                    vision_mask=vision_mask,
                )
                try:
                    enc_out = enc(
                        attention_mask=attention_mask,
                        inputs_embeds=merged_embeds,
                    )[0]
                finally:
                    enc.clear_geo_context()
                aux_visual_inputs = image_features[:, 1:].reshape(B, -1, D)
                # v2C: wrist-aware aux cross-attention (no-op when use_wrist_geostack=False or alpha==0)
                if self.use_wrist_geostack and self.wrist_geostack is not None:
                    aux_visual_inputs = self._apply_wrist_geostack(
                        aux_visual_inputs=aux_visual_inputs,
                        pixel_values=pixel_values,
                        extrinsics=extrinsics,
                    )
                return {"vlm_features": enc_out, "aux_visual_inputs": aux_visual_inputs}

        # Baseline path (also used when alpha==0 to save the DA3 forward + token build)
        enc_out = self.vlm.language_model.model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=merged_embeds,
        )[0]  # [B, T_enc, D]

        aux_visual_inputs = image_features[:, 1:].reshape(B, -1, D)  # remaining views flattened
        # v2C: wrist-aware aux cross-attention (no-op when use_wrist_geostack=False or alpha==0)
        if self.use_wrist_geostack and self.wrist_geostack is not None:
            aux_visual_inputs = self._apply_wrist_geostack(
                aux_visual_inputs=aux_visual_inputs,
                pixel_values=pixel_values,
                extrinsics=extrinsics,
            )
        return {"vlm_features": enc_out, "aux_visual_inputs": aux_visual_inputs}

    # ===================== DA3 geometry conditioning =====================
    def _compute_geometry_tokens(
        self,
        batch_size: int,
        da3_features: Optional[torch.Tensor],
        object_masks: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        da3_feature_hw: Optional[tuple] = None,
        image_input_for_da3: Optional[torch.Tensor] = None,
        language_instruction: Optional[list] = None,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Build [B, K, H] geometry tokens, or return None when disabled.

        Raises a clear error if enabled but `da3_features` is missing, unless
        `allow_missing_geometry`/`allow_dummy_da3_features` is set (then zeros).
        """
        if not self.geometry_enabled or self.geometry_conditioner is None:
            return None

        gc = self.geometry_cfg

        # v2F path: self.geometry_conditioner is a GeoPerceiverV2F that contains
        # its OWN DA3 backbone (DA3LargeForGeoStack at aspect-preserved 252×336)
        # plus PE+ray enrichment plus K=160 perceiver. Skip the legacy da3_inline
        # path and call v2F directly with pixel_values.
        if getattr(self, "_use_geo_perceiver_v2f", False):
            if image_input_for_da3 is None:
                raise ValueError("v2F geometry path requires image_input_for_da3 (pixel_values).")
            return self.geometry_conditioner(image_input_for_da3, extrinsics=extrinsics)

        # Inline DA3 path: run DA3 on the raw image_input if no precomputed
        # features were provided. This is the spatialva-style "no precompute"
        # mode (da3_source == "da3_encoder_inline").
        if (
            da3_features is None
            and self.da3_inline is not None
            and image_input_for_da3 is not None
        ):
            # Posed multi-view DA3: pass extrinsics+intrinsics when (a) they
            # are present in the batch AND (b) config opt-in (use_posed_da3=True
            # or env XVLA_POSED_DA3=1). Else fall back to unposed (None,None).
            _use_posed = bool(gc.get("use_posed_da3", False)) or os.environ.get("XVLA_POSED_DA3", "0") == "1"
            if _use_posed and extrinsics is not None and intrinsics is not None:
                da3_features = self.da3_inline(image_input_for_da3, extrinsics, intrinsics)
            else:
                da3_features = self.da3_inline(image_input_for_da3)  # [B,V,C,h,w]

        # Inline G-SAM path (V2): when masks aren't provided in the batch but
        # gsam_inline is configured, compute them from the raw image_input +
        # per-sample language instruction. Output is [B, 1, H_mask, W_mask].
        # The downstream DA3LatentSegmenter resizes to the latent grid and
        # broadcasts across views.
        if (
            object_masks is None
            and self.gsam_inline is not None
            and image_input_for_da3 is not None
            and language_instruction is not None
        ):
            object_masks = self.gsam_inline(image_input_for_da3, language_instruction)
            # geometry_conditioner's segmenter accepts [B, M, H, W] for single-
            # view dense features OR [B, V, M, H, W] for multi-view. Our DA3
            # output is multi-view [B, V, C, h, w], so we need [B, V, 1, H, W].
            B, _, H_m, W_m = object_masks.shape
            V = da3_features.shape[1] if da3_features is not None and da3_features.dim() == 5 else 1
            object_masks = object_masks.unsqueeze(1).expand(B, V, 1, H_m, W_m).contiguous()

        allow_missing = bool(
            gc.get("allow_missing_geometry", False)
            or gc.get("allow_dummy_da3_features", False)
        )
        if da3_features is None:
            if allow_missing:
                K = int(gc["num_geometry_tokens"])
                H = self.config.hidden_size
                logger.warning(
                    "geometry_conditioning enabled but da3_features missing; "
                    "using zeros (allow_dummy_da3_features/allow_missing)."
                )
                return torch.zeros(batch_size, K, H, device=device, dtype=dtype)
            raise ValueError(
                "geometry_conditioning.enabled=True but no `da3_features` were "
                "provided in the batch/kwargs. Provide precomputed DA3 features "
                f"(key '{gc.get('da3_feature_key', 'da3_features')}'), or set "
                "geometry_conditioning.allow_dummy_da3_features=True for debug."
            )

        geom = self.geometry_conditioner(
            da3_features, object_masks, da3_feature_hw
        )  # [B,K,H]

        # T5 language tokens concatenated into the K/V bank for the gated
        # cross-attn adapters. The combined [spatial | T5_lang] bank gives
        # every policy token access to both geometry AND task language in a
        # single cross-attention step.
        if self.t5_inline is not None and language_instruction is not None:
            t5_feats, t5_mask = self.t5_inline(language_instruction)  # [B, L, d_t5], [B, L]
            t5_proj = self.t5_projector(t5_feats.to(geom.dtype))      # [B, L, hidden]
            # Concatenate spatial tokens with projected T5 language tokens.
            # Order: [spatial | t5_lang]. Padding positions in t5_proj are
            # zeroed via the mask (cross-attn doesn't take a padding mask
            # currently, so we explicitly null padded positions so they
            # contribute zero to softmax-weighted-sum after value projection).
            t5_proj = t5_proj * t5_mask.to(t5_proj.dtype).unsqueeze(-1)
            geom = torch.cat([geom, t5_proj], dim=1)                  # [B, K+L, hidden]

        if self.geometry_cfg.get("debug_shapes", False) and not self._geom_shape_logged:
            self._geom_shape_logged = True
            logger.info(
                "[geometry][shapes] da3_features=%s object_masks=%s "
                "geometry_tokens=%s",
                tuple(da3_features.shape),
                tuple(object_masks.shape) if object_masks is not None else None,
                tuple(geom.shape),
            )
        return geom.to(dtype)

    def apply_finetune_policy(self, finetune: Optional[dict] = None) -> Dict[str, int]:
        """
        Freeze / unfreeze parameter groups per `config.finetune`.

        Groups
        ------
          * backbone           -> Florence2 vlm
          * soft_prompts       -> transformer.soft_prompt_hub
          * action_head        -> transformer.action_encoder / action_decoder
          * geometry_modules   -> geometry_conditioner + transformer.geometry_fusion
          * last_n_policy_layers -> last N transformer blocks (the rest frozen)

        Returns a {group: trainable_param_count} summary. Safe to call when
        geometry is disabled (geometry groups are simply absent).
        """
        ft = {**self.finetune_cfg, **(finetune or {})}
        tr = self.transformer

        backbone = list(self.vlm.parameters())
        soft = list(tr.soft_prompt_hub.parameters()) if hasattr(tr, "soft_prompt_hub") else []
        head = list(tr.action_encoder.parameters()) + list(tr.action_decoder.parameters())
        geom = []
        if self.geometry_conditioner is not None:
            geom += list(self.geometry_conditioner.parameters())
        if getattr(tr, "geometry_fusion", None) is not None:
            geom += list(tr.geometry_fusion.parameters())
        # DA3-XVLA: gated action-to-spatial cross-attention adapters (one per
        # selected transformer block). New trainable params that must train
        # with the geometry modules, not get default-frozen with policy core.
        if getattr(tr, "spatial_cross_attn_layers", None) is not None:
            for adapter in tr.spatial_cross_attn_layers:
                # Skip nn.Identity placeholders (layers without adapter).
                if any(True for _ in adapter.parameters()):
                    geom += list(adapter.parameters())
        # DA3-XVLA: inline-encoder adapter params (multi-layer fusion LayerNorms
        # + Conv2d) live OUTSIDE da3_inline.model.* — they must train with the
        # other geometry modules, not get default-frozen with policy core.
        if getattr(self, "da3_inline", None) is not None:
            for name, p in self.da3_inline.named_parameters():
                if not name.startswith("model."):   # adapter, not backbone
                    geom.append(p)
        # T5 language encoder + projector: the projector is trainable (small
        # MLP that maps T5 d_model → hidden_size); the T5 encoder itself is
        # frozen via freeze_t5=True (no grads through .model.*).
        if getattr(self, "t5_projector", None) is not None:
            geom += list(self.t5_projector.parameters())
        # Spatial-language Method A/B modules. Keep frozen DA3/T5 backbones out
        # of this trainable group; they are explicitly re-frozen below.
        if getattr(self, "spatial_lang", None) is not None:
            for name, p in self.spatial_lang.named_parameters():
                if name.startswith("da3.") or name.startswith("t5."):
                    continue
                geom.append(p)
        for attr in ("spatial_refiner", "spatial_injection_layers"):
            mod = getattr(tr, attr, None)
            if mod is not None:
                geom += list(mod.parameters())

        special = set(map(id, backbone + soft + head + geom))
        # Policy "core" = transformer blocks + projections + pos_emb + norm.
        core = [p for p in self.parameters() if id(p) not in special]

        def _set(params, flag):
            for p in params:
                p.requires_grad = bool(flag)

        _set(backbone, ft.get("train_backbone", False))
        _set(soft, ft.get("train_soft_prompts", True))
        _set(head, ft.get("train_action_head", True))
        _set(geom, ft.get("train_geometry_modules", True))

        # Flow-matching action-expert transformer:
        #   train_last_n_policy_layers = 0   -> frozen
        #                              = -1  -> train the WHOLE policy
        #                                       transformer (all blocks +
        #                                       pos_emb + norm + projections)
        #                              = N>0 -> train only the last N blocks
        _set(core, False)
        n_last = int(ft.get("train_last_n_policy_layers", 0))
        n_blocks = len(tr.blocks)
        if n_last < 0 or n_last >= n_blocks:
            _set(core, True)                      # full action expert
        elif n_last > 0:
            for blk in list(tr.blocks)[-n_last:]:
                _set(list(blk.parameters()), True)

        if getattr(self, "spatial_lang", None) is not None:
            self.spatial_lang.freeze_backbones()

        def _count(params):
            return int(sum(p.numel() for p in params if p.requires_grad))

        summary = {
            "backbone": _count(backbone),
            "soft_prompts": _count(soft),
            "action_head": _count(head),
            "geometry_modules": _count(geom),
            "policy_last_n": _count(core),
        }

        total = int(sum(p.numel() for p in self.parameters()))
        trainable = int(sum(p.numel() for p in self.parameters() if p.requires_grad))
        trainable_modules = sorted({
            name.rsplit(".", 1)[0] if "." in name else name
            for name, p in self.named_parameters() if p.requires_grad
        })
        pct = 100.0 * trainable / max(1, total)
        logger.info("[geometry][finetune] per-group trainable: %s", summary)
        logger.info(
            "[geometry][finetune] total=%d trainable=%d (%.2f%%)",
            total, trainable, pct,
        )
        logger.info(
            "[geometry][finetune] trainable modules (%d): %s",
            len(trainable_modules),
            ", ".join(list(trainable_modules)[:40])
            + (" ..." if len(trainable_modules) > 40 else ""),
        )
        summary.update(_total=total, _trainable=trainable)
        return summary

    # ================================= training =================================
    def forward(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        action: torch.Tensor,  # [B, T=num_actions, D=dim_action]
        da3_features: Optional[torch.Tensor] = None,
        object_masks: Optional[torch.Tensor] = None,
        da3_feature_hw: Optional[tuple] = None,
        language_instruction: Optional[list] = None,   # V2: raw text per sample for inline G-SAM
        extrinsics: Optional[torch.Tensor] = None,     # [B,V,4,4] w2c — posed DA3
        intrinsics: Optional[torch.Tensor] = None,     # [B,V,3,3] K   — posed DA3
        image_input_da3: Optional[torch.FloatTensor] = None,  # dual-res: aspect-preserved DA3 input
    ) -> Dict[str, torch.Tensor]:
        """
        1) Encode multimodal inputs.
        2) Diffusion-style noisy mixture of actions: x_t = t*noise + (1-t)*gt.
        3) Space-specific preprocessing, prediction, and supervised loss.
        """
        enc = self.forward_vlm(input_ids, image_input, image_mask, extrinsics=extrinsics)

        B = input_ids.shape[0]
        t = (torch.rand(1, device=input_ids.device)
             + torch.arange(B, device=input_ids.device) / B) % (1 - 1e-5)

        action_noisy = torch.randn_like(action) * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
        proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)

        geometry_tokens = self._compute_geometry_tokens(
            B, da3_features, object_masks, action.device, action.dtype, da3_feature_hw,
            image_input_for_da3=(image_input_da3 if image_input_da3 is not None else image_input),
            language_instruction=language_instruction,
            extrinsics=extrinsics, intrinsics=intrinsics,
        )

        spatial_lang_banks = None
        if self.spatial_lang_enabled and self.spatial_lang is not None:
            spatial_lang_banks = self.spatial_lang(
                image_input=(image_input_da3 if image_input_da3 is not None else image_input),
                language_instruction=language_instruction,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
            )

        # GeoStack v3 side stack bank (None if disabled). Built ONCE per fwd.
        # Multi-view mode needs ALL views (image_input as [B, V, 3, H, W]) so
        # main + wrist can each get their own DA3 forward at their own resolution.
        # v3.2: extrinsics also passed for world-frame Plücker rays.
        spatial_lang_bank, spatial_lang_bank_pad = self._build_spatial_lang_bank(
            image_input, language_instruction, extrinsics=extrinsics,
        )

        # v3.5: ask transformer for the side stack's final state too if aux loss enabled.
        # Default (aux disabled): byte-identical to prior behavior — transformer returns
        # just pred_action.
        if self.side_stack_aux_enabled and spatial_lang_bank is not None:
            pred_action, side_state = self.transformer(
                domain_id=domain_id,
                action_with_noise=action_noisy_m,
                t=t,
                proprio=proprio_m,
                geometry_tokens=geometry_tokens,
                spatial_lang_banks=spatial_lang_banks,
                spatial_lang_bank=spatial_lang_bank,
                spatial_lang_bank_pad=spatial_lang_bank_pad,
                return_side_state=True,
                **enc,
            )
        else:
            pred_action = self.transformer(
                domain_id=domain_id,
                action_with_noise=action_noisy_m,
                t=t,
                proprio=proprio_m,
                geometry_tokens=geometry_tokens,
                spatial_lang_banks=spatial_lang_banks,
                spatial_lang_bank=spatial_lang_bank,
                spatial_lang_bank_pad=spatial_lang_bank_pad,
                **enc,
            )
            side_state = None

        loss_dict = self.action_space.compute_loss(pred_action, action)

        # v3.5: AUX action-prediction loss off side stack final state. Provides
        # a direct supervised gradient to side stack + bank builder so they
        # actually train (vs the v3.2g/3/4 failures where they stayed at fresh init).
        # Per-component logging (aux_pos, aux_rot, aux_grip) so the dashboard
        # shows which channel dominates — if aux loss is purely gripper-driven,
        # the side stack might be learning a degenerate representation.
        if (self.side_stack_aux_enabled
                and self.side_stack_aux_head is not None
                and side_state is not None):
            # v3.6: pass preprocessed proprio if aux head was built with proprio fusion
            aux_proprio = proprio_m if self.side_stack_aux_head.proprio_proj is not None else None
            aux_pred = self.side_stack_aux_head(
                side_state,
                bank_padding_mask=spatial_lang_bank_pad,
                proprio=aux_proprio,
            )
            aux_loss_dict = self.action_space.compute_loss(aux_pred, action)
            # Add per-component aux losses (each scaled by aux_weight). Names prefixed
            # with "aux_" so train.py / wandb / log parsing groups them naturally.
            for k, v in aux_loss_dict.items():
                loss_dict[f"aux_{k}"] = self.side_stack_aux_weight * v

        if (
            self.spatial_aux_heads is not None
            and spatial_lang_banks is not None
            and self.training
        ):
            loss_dict.update(
                self.spatial_aux_heads(
                    spatial_lang_banks,
                    action,
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                )
            )

        return loss_dict

    # ================================= inference =================================
    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        steps: int = 10,
        da3_features: Optional[torch.Tensor] = None,
        object_masks: Optional[torch.Tensor] = None,
        da3_feature_hw: Optional[tuple] = None,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        image_input_da3: Optional[torch.FloatTensor] = None,
        language_instruction: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Iterative denoising (linear schedule).
        Applies action_space.postprocess at the end (e.g., sigmoid on gripper).
        """
        self.eval()
        enc = self.forward_vlm(input_ids, image_input, image_mask, extrinsics=extrinsics)

        B = input_ids.shape[0]
        D = self.action_space.dim_action

        x1 = torch.randn(B, self.num_actions, D, device=proprio.device, dtype=proprio.dtype)
        action = torch.zeros_like(x1)

        # DA3 geometry tokens depend only on the (static) features, so compute
        # them once outside the denoising loop.
        geometry_tokens = self._compute_geometry_tokens(
            B, da3_features, object_masks, proprio.device, proprio.dtype, da3_feature_hw,
            image_input_for_da3=(image_input_da3 if image_input_da3 is not None else image_input),
            extrinsics=extrinsics, intrinsics=intrinsics,
        )

        spatial_lang_banks = None
        if self.spatial_lang_enabled and self.spatial_lang is not None:
            spatial_lang_banks = self.spatial_lang(
                image_input=(image_input_da3 if image_input_da3 is not None else image_input),
                language_instruction=language_instruction,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
            )

        # v3 side stack bank — also static across denoising steps. At inference
        # we FORCE α=1.0 (the alpha schedule's _step buffer is non-persistent so
        # it loads as 0 by default → would make side stack silent without this).
        if self.side_stack_enabled and language_instruction is not None:
            self.set_geostack_step(10**9)   # force α=1.0
            spatial_lang_bank, spatial_lang_bank_pad = self._build_spatial_lang_bank(
                image_input, language_instruction, extrinsics=extrinsics,
            )
        else:
            spatial_lang_bank, spatial_lang_bank_pad = (None, None)

        steps = max(1, int(steps))
        for i in range(steps, 0, -1):
            t = torch.full((B,), i / steps, device=proprio.device, dtype=proprio.dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
            action = self.transformer(
                domain_id=domain_id,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
                geometry_tokens=geometry_tokens,
                spatial_lang_banks=spatial_lang_banks,
                spatial_lang_bank=spatial_lang_bank,
                spatial_lang_bank_pad=spatial_lang_bank_pad,
                **enc,
            )
        return self.action_space.postprocess(action)

    # =============================== FastAPI service =============================
    def _build_app(self, processor):
        """
        Minimal FastAPI app for XVLA inference.

        Args:
            processor: callable(images, text) -> Dict[str, torch.Tensor]
                       expected keys: "input_ids", "image_input", "image_mask"
        """
        if self.app is not None:
            return

        app = FastAPI()

        @app.post("/act")
        def act(payload: Dict[str, Any]):
            try:
                self.eval()
                # Decode up to 3 image inputs
                images = []
                for key in ("image0", "image1", "image2"):
                    if key not in payload: continue
                    v = json_numpy.loads(payload[key])
                    if isinstance(v, np.ndarray):
                        if v.ndim == 1:  # encoded bytes
                            v = cv2.imdecode(v, cv2.IMREAD_COLOR)
                        images.append(Image.fromarray(v))
                    elif isinstance(v, (list, tuple)):
                        images.append(Image.fromarray(np.array(v)))
                    elif isinstance(v, str):
                        images.append(Image.open(v))
                if not images:
                    return JSONResponse({"error": "No valid images found."}, status_code=400)

                # Multimodal preprocessing by processor
                inputs = processor(images, payload["language_instruction"])
                if not {"input_ids", "image_input", "image_mask"}.issubset(inputs):
                    return JSONResponse({"error": "Processor returned incomplete inputs."}, status_code=400)

                # Build proprio/domain tensors
                proprio = torch.as_tensor(np.asarray(json_numpy.loads(payload["proprio"])))
                domain_id = torch.tensor([int(payload["domain_id"])], dtype=torch.long)

                # Align to model's device/dtype
                device = next(self.parameters()).device
                dtype = next(self.parameters()).dtype

                def to_model(t: torch.Tensor) -> torch.Tensor:
                    if not isinstance(t, torch.Tensor):
                        t = torch.as_tensor(t)
                    # cast floats to model dtype, keep integral/bool as-is
                    return t.to(device=device, dtype=dtype) if t.is_floating_point() else t.to(device=device)

                inputs = {k: to_model(v) for k, v in inputs.items()}
                inputs.update({
                    "proprio": to_model(proprio.unsqueeze(0)),
                    "domain_id": domain_id.to(device),
                })

                # Optional precomputed DA3 geometry inputs.
                if self.geometry_enabled and "da3_features" in payload:
                    da3 = torch.as_tensor(np.asarray(json_numpy.loads(payload["da3_features"])))
                    inputs["da3_features"] = to_model(da3.unsqueeze(0))
                    if "object_masks" in payload:
                        om = torch.as_tensor(np.asarray(json_numpy.loads(payload["object_masks"])))
                        inputs["object_masks"] = to_model(om.unsqueeze(0))

                # Inference
                steps = int(payload.get("steps", 10))
                action = self.generate_actions(**inputs, steps=steps).squeeze(0).float().cpu().numpy()
                return JSONResponse({"action": action.tolist()})

            except Exception:
                logging.error(traceback.format_exc())
                return JSONResponse({"error": "Request failed"}, status_code=400)

        self.app = app

    def run(self, processor, host: str = "0.0.0.0", port: int = 8000):
        """
        Launch the FastAPI service.
        """
        self._build_app(processor)
        assert self.app is not None
        uvicorn.run(self.app, host=host, port=port)
