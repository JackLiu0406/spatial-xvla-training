# ------------------------------------------------------------------------------
# DA3-T5-XVLA: VLM-free variant of X-VLA.
#
# Replaces Florence-2 with:
#   - DA3 (depth-anything-3) frozen encoder for vision
#   - T5 encoder (frozen) for language
#
# Architecture:
#   images   ──► DA3 inline encoder    ──► da3_tokens   [B, N_da3, hidden]
#   language ──► T5 encoder (frozen)   ──► t5_tokens    [B, N_text, hidden]
#                                                              ▲
#   [action+noise, proprio, time] ──► SoftPromptedTransformer ─┘
#                                          (cross-attends to da3 + t5 tokens
#                                           via existing vlm_features /
#                                           aux_visual_inputs slots)
#
# Trains with the same dataset format as X-VLA (datasets/dataset.py) and the
# same flow-matching x0-prediction loss (action_hub.EE6DActionSpace).
# ------------------------------------------------------------------------------

from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers import T5EncoderModel, T5Tokenizer

from .action_hub import build_action_space
from .configuration_xvla import XVLAConfig
from .da3_inline import DA3InlineEncoder
from .transformer import SoftPromptedTransformer


DEFAULT_T5_NAME = "t5-base"


class DA3T5XVLA(PreTrainedModel):
    """
    VLM-free variant: DA3 + T5 + action expert.

    Drop-in replacement for `XVLA.forward(...)` signature with the SAME
    output keys (position_loss / rotate6D_loss / gripper_loss) so existing
    `train.py` works without modification, given a meta whose images are
    raw uint8 tensors and a `language_instruction` field of raw text.
    """

    config_class = XVLAConfig
    base_model_prefix = "da3t5xvla"
    supports_gradient_checkpointing = True

    def __init__(self, config: XVLAConfig):
        super().__init__(config)
        self.config = config

        # ----- action space -----
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

        # ----- vision: DA3 inline (always-on, since no Florence-2) -----
        self.da3 = DA3InlineEncoder(
            model_name=getattr(config, "da3_model_name", "depth-anything/DA3-BASE"),
            use_bf16=True,
            gradient_checkpointing=True,
            freeze=True,
            feature_size=64,
            feature_layer="last",
            process_res=504,
        )
        # DA3 outputs [B, V, 64*64, 768]; project to hidden_size.
        da3_feat_dim = 768
        self.da3_proj = nn.Linear(da3_feat_dim, config.hidden_size)
        nn.init.normal_(self.da3_proj.weight, std=0.02)
        nn.init.zeros_(self.da3_proj.bias)

        # ----- language: T5 encoder (frozen) -----
        t5_name = getattr(config, "t5_model_name", DEFAULT_T5_NAME)
        self.t5_tokenizer = T5Tokenizer.from_pretrained(t5_name)
        self.t5_encoder = T5EncoderModel.from_pretrained(t5_name)
        for p in self.t5_encoder.parameters():
            p.requires_grad_(False)
        # T5-base hidden = 768; project to action-expert hidden.
        t5_hidden = self.t5_encoder.config.d_model
        self.t5_proj = nn.Linear(t5_hidden, config.hidden_size)
        nn.init.normal_(self.t5_proj.weight, std=0.02)
        nn.init.zeros_(self.t5_proj.bias)

        # ----- action expert (reuse X-VLA's SoftPromptedTransformer) -----
        # We feed:
        #   vlm_features      ← DA3 tokens  (post-projection, [B, V*N_da3, H])
        #   aux_visual_inputs ← T5 tokens   (post-projection, [B, T_text, H])
        # Internal vlm_proj / aux_visual_proj are then near-identities (just
        # LayerNorm/Linear). Could also be made strict-identity by setting
        # them to identity init — left as small linear for now.
        self.transformer = SoftPromptedTransformer(
            num_domains=config.num_domains,
            multi_modal_input_size=config.hidden_size,  # we pre-project to hidden_size
            hidden_size=config.hidden_size,
            mlp_ratio=config.mlp_ratio,
            num_heads=config.num_heads,
            depth=config.depth,
            dim_action=dim_action,
            dim_propio=dim_proprio,
            dim_time=config.dim_time,
            len_soft_prompts=config.len_soft_prompts,
            max_len_seq=config.max_len_seq,
            use_hetero_proj=config.use_hetero_proj,
            geometry_conditioning=None,  # no separate geometry path; DA3 IS our vision
        )
        self.num_actions = config.num_actions
        self.action_mode = config.action_mode.lower()

    # =================================================================
    # Encoding helpers
    # =================================================================
    def _encode_da3(self, image_input: torch.FloatTensor) -> torch.Tensor:
        """
        image_input : [B, V, C, H, W]  (already normalized at deploy or
                      via the dataset's image_aug — DA3InlineEncoder applies its
                      own ImageNet normalize on raw uint8, so for compatibility
                      with the X-VLA dataset which passes normalized floats we
                      undo their normalize first if needed. For now we just
                      pass through and rely on DA3InlineEncoder.preprocess
                      handling float input.)
        Returns: [B, V*N, hidden_size]
        """
        # DA3InlineEncoder.forward returns [B, V, C_feat, h, w].
        feats = self.da3(image_input)  # [B, V, C_da3, h, w]
        B, V, C, h, w = feats.shape
        # Flatten spatial + view → tokens, project to hidden.
        feats = feats.permute(0, 1, 3, 4, 2).reshape(B, V * h * w, C)  # [B, V*N, C]
        return self.da3_proj(feats)  # [B, V*N, hidden]

    @torch.no_grad()
    def _tokenize_text(self, language_instruction: List[str], device):
        enc = self.t5_tokenizer(
            language_instruction,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    def _encode_t5(self, language_instruction: List[str], device) -> torch.Tensor:
        input_ids, attention_mask = self._tokenize_text(language_instruction, device)
        # T5 encoder forward (frozen) → last_hidden_state [B, T_text, t5_hidden]
        with torch.no_grad():
            out = self.t5_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.t5_proj(out.last_hidden_state)  # [B, T_text, hidden]

    # =================================================================
    # Training forward
    # =================================================================
    def forward(
        self,
        image_input: torch.FloatTensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        language_instruction: List[str],
        # accept and ignore X-VLA-style fields so train.py keeps working:
        input_ids: Optional[torch.LongTensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        da3_features: Optional[torch.Tensor] = None,
        object_masks: Optional[torch.Tensor] = None,
        da3_feature_hw: Optional[tuple] = None,
    ) -> Dict[str, torch.Tensor]:
        B = image_input.shape[0]
        device = image_input.device

        # ----- encode vision (DA3) -----
        da3_tokens = self._encode_da3(image_input)  # [B, V*N, H]

        # ----- encode language (T5) -----
        t5_tokens = self._encode_t5(language_instruction, device)  # [B, T_text, H]

        # ----- flow-matching noise (same as X-VLA) -----
        t = (torch.rand(1, device=device) + torch.arange(B, device=device) / B) % (1 - 1e-5)
        action_noisy = torch.randn_like(action) * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
        proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)

        pred_action = self.transformer(
            domain_id=domain_id,
            vlm_features=da3_tokens,           # DA3 in place of Florence VLM
            aux_visual_inputs=t5_tokens,       # T5 in place of Florence aux views
            action_with_noise=action_noisy_m,
            proprio=proprio_m,
            t=t,
            geometry_tokens=None,
        )
        return self.action_space.compute_loss(pred_action, action)

    # =================================================================
    # Inference (deploy)
    # =================================================================
    @torch.no_grad()
    def generate_actions(
        self,
        image_input: torch.FloatTensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        language_instruction: List[str],
        steps: int = 10,
        # accept X-VLA-style args for deploy script compatibility:
        input_ids: Optional[torch.LongTensor] = None,
        image_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Iterative denoising (same schedule as X-VLA.generate_actions).
        Returns: [B, num_actions, dim_action]
        """
        self.eval()
        device = image_input.device
        B = image_input.shape[0]
        D = self.action_space.dim_action

        da3_tokens = self._encode_da3(image_input)
        t5_tokens = self._encode_t5(language_instruction, device)

        x1 = torch.randn(B, self.num_actions, D, device=device, dtype=proprio.dtype)
        action = torch.zeros_like(x1)
        steps = max(1, int(steps))
        for i in range(steps, 0, -1):
            t = torch.full((B,), i / steps, device=device, dtype=proprio.dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
            action = self.transformer(
                domain_id=domain_id,
                vlm_features=da3_tokens,
                aux_visual_inputs=t5_tokens,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
                geometry_tokens=None,
            )
        return self.action_space.postprocess(action)
