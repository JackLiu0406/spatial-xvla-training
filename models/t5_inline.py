"""
Frozen T5 encoder for language conditioning of geometry tokens.

Adds a pre-trained T5 text encoder that runs on the raw language instruction
strings. The per-token T5 embeddings get projected to the policy hidden_dim
and concatenated to the spatial tokens, producing an enlarged K/V bank that
all policy tokens cross-attend to in the gated cross-attn adapters.

Architecture intent (paired with full_sequence spatial cross-attn):
    spatial_tokens [B, K, H]  ─┐
                                ├─► combined_kv [B, K + L, H]  ──► cross-attn K/V
    T5_lang_tokens [B, L, H]  ─┘                                  (Q = full policy seq)

So action+vlm+aux+soft tokens can attend to BOTH "where things are" (spatial)
AND "what the task is" (pure-text language) in a single attention step.

Load contract (mirrors da3_inline.py):
  * T5 weights get wiped by HF's `from_pretrained` loader because they're absent
    from the X-VLA-PT checkpoint. Call ``reload_pretrained_weights()`` after
    ``XVLA.from_pretrained(...)`` to restore them (see train.py).

Defaults:
  * t5-base (220 M params, d_model=768)
  * Frozen (set ``freeze=True``); training-time forward is wrapped in no_grad
  * bf16 weights (saves memory; matches autocast dtype)
  * max_length=64 (RoboTwin instructions are short; cap for memory)
"""
from typing import List, Tuple, Optional
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _build_text_encoder(model_name: str):
    """Load a frozen text ENCODER, backbone-agnostic.

    * encoder-decoder models (T5) → ``T5EncoderModel`` (encoder half only)
    * encoder-only models (ModernBERT / BERT / etc.) → ``AutoModel``

    Both expose ``.config`` (with ``hidden_size`` or ``d_model``) and a forward
    returning ``.last_hidden_state`` given ``input_ids`` + ``attention_mask``,
    so the rest of this module is unchanged across encoders.
    """
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(model_name)
    if getattr(cfg, "is_encoder_decoder", False) or getattr(cfg, "model_type", "") == "t5":
        from transformers import T5EncoderModel
        return T5EncoderModel.from_pretrained(model_name)
    return AutoModel.from_pretrained(model_name)


class T5InlineEncoder(nn.Module):
    """
    Wraps a (frozen) T5 encoder for language-token generation.

    Forward: list[str] (batch of instructions) → ([B, L, d_t5], [B, L] mask)
    where L is the padded max instruction length in the batch.
    """

    def __init__(
        self,
        model_name: str = "t5-base",
        use_bf16: bool = True,
        freeze: bool = True,
        max_length: int = 64,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoTokenizer
        except Exception as exc:
            raise ImportError("transformers required for T5InlineEncoder") from exc

        self.model_name = str(model_name)
        self.max_length = int(max_length)
        self.use_bf16 = bool(use_bf16)
        self._freeze = bool(freeze)

        logger.info("[t5_inline] loading text encoder %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = _build_text_encoder(model_name)
        if use_bf16:
            self.model = self.model.to(torch.bfloat16)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
            logger.info("[t5_inline] text encoder frozen (no backward through encoder)")

    @property
    def hidden_size(self) -> int:
        # ModernBERT/BERT use `hidden_size`; T5 uses `d_model`.
        c = self.model.config
        return int(getattr(c, "hidden_size", None) or c.d_model)

    def forward(self, instructions: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        instructions : list[str], length B
            Raw language instructions for the batch.

        Returns
        -------
        features : Tensor, [B, L, d_t5], dtype matches self.use_bf16
            Per-token T5 encoder outputs.
        attention_mask : Tensor, [B, L], bool
            True for valid tokens, False for padding. To be passed to the
            downstream cross-attention as key_padding_mask (or inverted).
        """
        if not isinstance(instructions, (list, tuple)):
            raise TypeError(
                f"T5InlineEncoder.forward expected list[str], got {type(instructions)}"
            )
        if not all(isinstance(s, str) for s in instructions):
            raise TypeError(
                "T5InlineEncoder.forward expected list[str]; got non-str entry"
            )

        tok = self.tokenizer(
            list(instructions),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        input_ids = tok["input_ids"].to(device)
        attention_mask = tok["attention_mask"].to(device).bool()

        # T5 encoder forward. Wrap in no_grad when frozen — same pattern as
        # DA3 inline: avoids holding the computation graph that we can't use.
        if self._freeze:
            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask.long(),
                ).last_hidden_state
        else:
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask.long(),
            ).last_hidden_state

        return out, attention_mask

    # ------------------------------------------------------------------
    # Load contract (mirrors DA3InlineEncoder.reload_pretrained_weights)
    # ------------------------------------------------------------------
    def reload_pretrained_weights(self) -> None:
        """
        Re-load T5 weights AFTER ``XVLA.from_pretrained`` finishes.

        HF's loader treats every key absent from the X-VLA-PT checkpoint as
        "missing" — for T5InlineEncoder.model.* that means the T5 weights we
        loaded in __init__ get wiped to whatever PreTrainedModel._init_weights
        defaults to (zero in practice). This restores the actual T5 weights.
        """
        logger.info("[t5_inline] reloading pretrained %s (HF re-init wiped it)", self.model_name)
        src = _build_text_encoder(self.model_name)
        if self.use_bf16:
            src = src.to(torch.bfloat16)
        missing, unexpected = self.model.load_state_dict(src.state_dict(), strict=False)
        del src
        if unexpected:
            logger.warning("[t5_inline] %d unexpected keys (first 3): %s",
                           len(unexpected), unexpected[:3])
        if missing:
            logger.warning("[t5_inline] %d missing keys (first 3): %s",
                           len(missing), missing[:3])
        if self._freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
            logger.info("[t5_inline] reload done, T5 frozen")
