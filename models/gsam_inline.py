# ------------------------------------------------------------------------------
# Inline language-conditioned segmentation (Grounded-SAM) for V2.
#
# Sits parallel to `DA3InlineEncoder` and runs *inside* the training forward,
# turning (multi-view RGB, language instruction) -> per-sample object masks.
# Those masks are then fed into the existing `DA3LatentSegmenter` (which the
# `SegmentedDA3GeometryConditioner` already owns) to soft-gate the DA3 latent
# before the Perceiver resampler.
#
# Spatialva-style efficiency: bf16 autocast, frozen weights (no backward).
# Grounding-DINO batches across samples in one call. SAM is per-sample because
# the box set differs per image, but each call is tiny (~50-100 ms on H100).
# ------------------------------------------------------------------------------
from __future__ import annotations
import logging
from contextlib import nullcontext
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class GroundedSAMInline(nn.Module):
    """
    Inline language-conditioned segmenter. Pairs GroundingDINO-tiny + SAM-vit-base
    (HF Transformers, no GroundingDINO CUDA build needed).

    forward(image_input, instructions) -> [B, 1, H_mask, W_mask] mask tensor
        image_input  : [B, V, 3, H, W] ImageNet-normalized (same as X-VLA's input)
        instructions : list[str] of length B (per-sample language text)

    Only runs on the *main view* (view 0). The same mask is used for all views
    downstream (broadcast in DA3LatentSegmenter). Trade-off: fewer SAM calls,
    accepts the assumption that the main-view mask is the most informative
    semantic anchor.
    """

    def __init__(
        self,
        dino_id: str = "IDEA-Research/grounding-dino-tiny",
        sam_id: str = "facebook/sam-vit-base",
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        freeze: bool = True,
        use_bf16: bool = True,
        main_view_only: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import (
                AutoProcessor, AutoModelForZeroShotObjectDetection,
                SamModel, SamProcessor,
            )
        except Exception as exc:
            raise ImportError("transformers is required for inline G-SAM") from exc
        logger.info("[gsam_inline] loading %s + %s", dino_id, sam_id)
        self.dino_proc = AutoProcessor.from_pretrained(dino_id)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).eval()
        self.sam = SamModel.from_pretrained(sam_id).eval()
        self.sam_proc = SamProcessor.from_pretrained(sam_id)
        self.box_thr = float(box_threshold)
        self.text_thr = float(text_threshold)
        self.use_bf16 = bool(use_bf16)
        self.main_view_only = bool(main_view_only)

        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)
            logger.info("[gsam_inline] frozen (no backward through G-SAM)")

        # ImageNet denorm buffers (X-VLA input was normalized with these stats)
        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std",  torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def reload_pretrained_weights(self) -> None:
        """
        Reload Grounding-DINO + SAM pretrained weights AFTER
        ``XVLA.from_pretrained`` runs. Same root cause as the DA3 fix in
        ``da3_inline.reload_pretrained_weights``: HF's ``from_pretrained``
        treats every key absent from the X-VLA checkpoint as "missing" and
        re-inits it, wiping the G-DINO + SAM weights that __init__ just loaded.
        Call once, right after ``XVLA.from_pretrained(...)``.
        """
        try:
            from transformers import AutoModelForZeroShotObjectDetection, SamModel
        except Exception as exc:
            raise ImportError("transformers is required for inline G-SAM") from exc
        dino_id = getattr(self.dino.config, "_name_or_path", None) or "IDEA-Research/grounding-dino-tiny"
        sam_id  = getattr(self.sam.config,  "_name_or_path", None) or "facebook/sam-vit-base"
        logger.info("[gsam_inline] reloading pretrained %s + %s (HF re-init wiped them)",
                    dino_id, sam_id)
        device = next(self.parameters()).device
        src_dino = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id)
        miss, unexp = self.dino.load_state_dict(src_dino.state_dict(), strict=False)
        del src_dino
        if miss:   logger.warning("[gsam_inline] DINO missing %d keys (first 3): %s", len(miss),  miss[:3])
        if unexp:  logger.warning("[gsam_inline] DINO unexpected %d keys (first 3): %s", len(unexp), unexp[:3])
        src_sam = SamModel.from_pretrained(sam_id)
        miss, unexp = self.sam.load_state_dict(src_sam.state_dict(), strict=False)
        del src_sam
        if miss:   logger.warning("[gsam_inline] SAM missing %d keys (first 3): %s", len(miss),  miss[:3])
        if unexp:  logger.warning("[gsam_inline] SAM unexpected %d keys (first 3): %s", len(unexp), unexp[:3])
        # Re-freeze (loading state_dict resets requires_grad on the reloaded params).
        for p in self.parameters():
            p.requires_grad_(False)
        # Push back to original device.
        self.to(device)
        logger.info("[gsam_inline] reload done, G-SAM frozen")

    @staticmethod
    def _normalize_prompt(text: str) -> str:
        t = (text or "").strip().lower()
        if not t:
            t = "object"
        return t if t.endswith(".") else t + "."

    @torch.no_grad()
    def _denorm_to_uint8_pil_list(self, normed: torch.Tensor) -> List["Image.Image"]:
        """[V_or_B, 3, H, W] ImageNet-normed -> list of PIL.Image (RGB)."""
        from PIL import Image
        x = normed.float() * self._std.to(normed.device) + self._mean.to(normed.device)
        x = (x.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        x = x.permute(0, 2, 3, 1).contiguous().cpu().numpy()       # [N, H, W, 3]
        return [Image.fromarray(x[i]) for i in range(x.shape[0])]

    @torch.no_grad()
    def forward(
        self,
        image_input: torch.Tensor,
        instructions: List[str],
    ) -> torch.Tensor:
        """
        Returns masks of shape [B, 1, H_mask, W_mask] (soft binary, in {0,1}).
        These are mask images at the input H×W resolution; the downstream
        DA3LatentSegmenter bilinearly resizes them to the DA3 latent grid.
        """
        if image_input.dim() != 5:
            raise ValueError(f"image_input must be [B,V,3,H,W], got {tuple(image_input.shape)}")
        B, V, C, H, W = image_input.shape
        device = image_input.device

        # Grab main view per sample (cheaper than V calls; broadcast to V later).
        main = image_input[:, 0]                                   # [B,3,H,W]
        pil_imgs = self._denorm_to_uint8_pil_list(main)             # list of B PIL
        prompts = [self._normalize_prompt(t) for t in instructions]

        ctx = (torch.autocast(device.type, dtype=torch.bfloat16)
               if self.use_bf16 and device.type == "cuda" else nullcontext())
        with ctx:
            # GroundingDINO: batched across samples in ONE call
            di = self.dino_proc(images=pil_imgs, text=prompts,
                                return_tensors="pt", padding=True).to(device)
            dout = self.dino(**di)
            target_sizes = [(img.size[1], img.size[0]) for img in pil_imgs]   # (H,W)
            res = self.dino_proc.post_process_grounded_object_detection(
                dout, di["input_ids"],
                box_threshold=self.box_thr, text_threshold=self.text_thr,
                target_sizes=target_sizes,
            )

            # SAM: per-sample (boxes vary per image). Each call is small.
            masks_per_sample: List[torch.Tensor] = []
            for b in range(B):
                boxes = res[b].get("boxes")
                if boxes is None or boxes.numel() == 0:
                    masks_per_sample.append(torch.zeros(1, H, W, device=device))
                    continue
                si = self.sam_proc(pil_imgs[b], input_boxes=[boxes.tolist()],
                                   return_tensors="pt").to(device)
                sout = self.sam(**si)
                m = self.sam_proc.image_processor.post_process_masks(
                    sout.pred_masks.cpu(),
                    si["original_sizes"].cpu(),
                    si["reshaped_input_sizes"].cpu(),
                )[0]                                                  # [M,1,H,W] or [M,k,H,W]
                m = m.float()
                if m.dim() == 4:
                    m = m[:, 0]                                       # keep first SAM proposal
                # Union over M objects -> 1 mask per sample
                m = m.amax(0, keepdim=True).to(device)                # [1,H,W]
                masks_per_sample.append(m)

        masks = torch.stack(masks_per_sample, dim=0)                 # [B,1,H,W]
        return masks.clamp_(0.0, 1.0)
