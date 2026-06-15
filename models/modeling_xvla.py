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
from typing import Any, Dict, Optional

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
            geometry_conditioning=self.geometry_cfg if self.geometry_enabled else None,
        )

        # DA3 -> compressed geometry tokens (only built when enabled).
        self.geometry_conditioner: Optional[SegmentedDA3GeometryConditioner] = None
        if self.geometry_enabled:
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

        # Force-set the Perceiver resampler query at unit-scale init, AFTER
        # all submodule init. (HF's recursive `apply(_init_weights)` or some
        # earlier path was leaving this parameter at near-zero; explicit
        # re-init here is bulletproof and matches typical positional-embedding
        # init scale.) See `_reinit_geometry_modules` for the full re-init.
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

    # ============================= Florence2 encoder =============================
    def forward_vlm(
        self,
        input_ids: torch.LongTensor,        # [B, L]
        pixel_values: torch.FloatTensor,    # [B, V, C, H, W]
        image_mask: torch.Tensor,           # [B, V] (bool or 0/1)
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

        enc_out = self.vlm.language_model.model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=merged_embeds,
        )[0]  # [B, T_enc, D]

        aux_visual_inputs = image_features[:, 1:].reshape(B, -1, D)  # remaining views flattened
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
        # DA3-XVLA: inline-encoder adapter params (multi-layer fusion LayerNorms
        # + Conv2d) live OUTSIDE da3_inline.model.* — they must train with the
        # other geometry modules, not get default-frozen with policy core.
        if getattr(self, "da3_inline", None) is not None:
            for name, p in self.da3_inline.named_parameters():
                if not name.startswith("model."):   # adapter, not backbone
                    geom.append(p)

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
        enc = self.forward_vlm(input_ids, image_input, image_mask)

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

        pred_action = self.transformer(
            domain_id=domain_id,
            action_with_noise=action_noisy_m,
            t=t,
            proprio=proprio_m,
            geometry_tokens=geometry_tokens,
            **enc,
        )
        return self.action_space.compute_loss(pred_action, action)

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
    ) -> torch.Tensor:
        """
        Iterative denoising (linear schedule).
        Applies action_space.postprocess at the end (e.g., sigmoid on gripper).
        """
        self.eval()
        enc = self.forward_vlm(input_ids, image_input, image_mask)

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
