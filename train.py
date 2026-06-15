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

import os
import math
import time
import json
import random
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW

from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import create_dataloader
from models.modeling_xvla import XVLA
from models.processing_xvla import XVLAProcessor

import logging
import os
import sys
import psutil

# ============================================================
# logger
# ============================================================
def get_logger(name="train", output_dir=None, accelerator=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False 
    if logger.handlers:
        return logger
    is_main = accelerator is None or accelerator.is_main_process
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    if is_main:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(level)
        logger.addHandler(ch)
    if output_dir and is_main:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, "train.log"), mode="a")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)
    return logger


# ============================================================
# Argument Parser
# ============================================================
def get_args_parser():
    parser = argparse.ArgumentParser("XVLA Training", add_help=False)

    # I/O
    parser.add_argument("--models", type=str, required=True, help="Path or HF repo for pretrained XVLA")
    parser.add_argument("--output_dir", type=str, default="runnings", help="Directory to save checkpoints")

    # Data
    parser.add_argument("--train_metas_path", type=str, required=True, help="Path to training metadata")
    parser.add_argument("--batch_size", type=int, default=16)

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_coef", type=float, default=1.0, help="LR multiplier for soft prompts")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Schedule
    parser.add_argument("--iters", type=int, default=1000000)
    parser.add_argument("--freeze_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--use_cosine_decay", action="store_true", default=False)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)

    # Logging / saving
    parser.add_argument("--save_interval", type=int, default=50000)
    parser.add_argument("--log_interval", type=int, default=20)

    # System
    parser.add_argument("--seed", type=int, default=0)

    # DA3-XVLA: opt-in selective fine-tuning via config.finetune. When NOT set,
    # baseline behavior is preserved (all params trainable, LR schedule governs
    # freezing exactly as upstream).
    parser.add_argument("--apply_finetune_policy", action="store_true", default=False,
                        help="Apply requires_grad freezing per config.finetune")

    return parser


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


def build_optimizer(model: XVLA, lr: float, weight_decay: float, betas=(0.9, 0.95), lr_coef_soft=1.0):
    """Split param groups by module type with different learning rates."""
    vlm_params = list(model.vlm.parameters())
    soft_prompt_params = list(model.transformer.soft_prompt_hub.parameters())
    action_params = list(model.transformer.action_decoder.parameters()) + list(model.transformer.action_encoder.parameters())

    # DA3-XVLA: geometry conditioner + cross-attention fusion get their own
    # group so they can be trained at the action-head LR. Empty when disabled,
    # which leaves baseline param groups byte-for-byte identical.
    geometry_params = []
    if getattr(model, "geometry_conditioner", None) is not None:
        geometry_params += list(model.geometry_conditioner.parameters())
    if getattr(model.transformer, "geometry_fusion", None) is not None:
        geometry_params += list(model.transformer.geometry_fusion.parameters())

    # DA3-XVLA: split da3_inline params into two LR routes.
    #   da3_params         — backbone-only (model.*); VLM-style low LR
    #                        (lr * lr_coef_soft) to preserve depth pretraining.
    #   da3_adapter_params — multi-layer fusion adapter (LayerNorms + 1×1 Conv);
    #                        fresh init, wants the full action-head LR. Folded
    #                        into the geometry group so it cosines with the
    #                        other geometry modules.
    # When DA3 is frozen (default) both lists are empty for backbone params and
    # the adapter list only contains items if feature_layer=="multi", keeping
    # baseline param groups byte-for-byte identical in the single-layer case.
    da3_params = []
    da3_adapter_params = []
    if getattr(model, "da3_inline", None) is not None:
        for name, p in model.da3_inline.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("model."):
                da3_params.append(p)
            else:
                da3_adapter_params.append(p)
    geometry_params += da3_adapter_params

    exclude = set(map(id, vlm_params + soft_prompt_params + action_params + geometry_params + da3_params))
    transformer_core_params = [p for p in model.parameters() if id(p) not in exclude]
    param_groups = [
        {"name": "vlm", "params": vlm_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "transformer_core", "params": transformer_core_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "soft_prompts", "params": soft_prompt_params, "lr": lr * lr_coef_soft, "weight_decay": weight_decay},
        {"name": "action_heads", "params": action_params, "lr": lr, "weight_decay": weight_decay},
    ]
    if geometry_params:
        param_groups.append(
            {"name": "geometry", "params": geometry_params, "lr": lr, "weight_decay": weight_decay}
        )
    if da3_params:
        param_groups.append(
            {"name": "da3", "params": da3_params, "lr": lr * lr_coef_soft, "weight_decay": weight_decay}
        )
    # 8-bit Adam via bitsandbytes saves ~7-10 GB of optimizer state per rank
    # (quantizes the fp32 m and v moments to 8-bit). Required to fit bs16 with
    # unfrozen DA3 + 8L cross-attn on H100 80GB. No reported quality loss.
    if os.environ.get("XVLA_OPTIM_8BIT", "0") == "1":
        try:
            from bitsandbytes.optim import AdamW8bit
            print("[build_optimizer] AdamW8bit (bitsandbytes) — fp32 moments quantized to 8-bit", flush=True)
            return AdamW8bit(param_groups, betas=betas)
        except ImportError as e:
            print(f"[build_optimizer][WARN] XVLA_OPTIM_8BIT=1 but bitsandbytes import failed: {e}; falling back to fp32 AdamW", flush=True)
    return AdamW(param_groups, betas=betas)


def set_group_lr(optim: torch.optim.Optimizer, name: str, lr: float):
    for g in optim.param_groups: 
        if g["name"] == name: g["lr"] = lr


def get_group_lr(optim: torch.optim.Optimizer, name: str) -> float:
    for g in optim.param_groups:
        if g["name"] == name: return g["lr"]
    return 0.0


def linear_warmup_cosine(step, start, warmup, total, base_lr, min_ratio):
    """Linear warmup followed by cosine decay."""
    if step < start: return 0.0
    progress = step - start
    if progress < warmup:
        return base_lr * (progress / max(1, warmup))
    remain = max(1, total - (start + warmup))
    ratio = 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - warmup) / remain)))
    return base_lr * (min_ratio + (1 - min_ratio) * ratio)


def update_group_lrs(optim, step, args):
    """Elegant group-wise LR scheduler."""
    base = {
        "vlm": args.learning_rate * args.learning_coef,
        "transformer_core": args.learning_rate,
        "soft_prompts": args.learning_rate * args.learning_coef,
        "action_heads": args.learning_rate,
        # Only present when geometry conditioning is enabled.
        "geometry": args.learning_rate,
        # Only present when DA3 is unfrozen (config.geometry_conditioning.freeze_da3=False).
        # Uses VLM-style low LR to avoid destroying DA3's depth pretraining.
        "da3": args.learning_rate * args.learning_coef,
    }
    has_geometry = any(g["name"] == "geometry" for g in optim.param_groups)
    if not has_geometry:
        base.pop("geometry")
    has_da3 = any(g["name"] == "da3" for g in optim.param_groups)
    if not has_da3:
        base.pop("da3")
    def schedule(step, base_lr):
        return linear_warmup_cosine(step, args.freeze_steps, args.warmup_steps, args.iters, base_lr, args.min_lr_ratio)
    if step < args.freeze_steps:
        set_group_lr(optim, "vlm", 0.0)
        set_group_lr(optim, "transformer_core", 0.0)
        set_group_lr(optim, "soft_prompts", base["soft_prompts"])
        set_group_lr(optim, "action_heads", base["action_heads"])
        if has_geometry:
            set_group_lr(optim, "geometry", base["geometry"])
        if has_da3:
            set_group_lr(optim, "da3", 0.0)   # DA3 stays frozen during initial freeze
    else:
        for name, base_lr in base.items():
            new_lr = schedule(step, base_lr) if args.use_cosine_decay else base_lr
            set_group_lr(optim, name, new_lr)


# ============================================================
# Main Training
# ============================================================
def main(args):
    output_dir = Path(args.output_dir)
    # Perf: enable TF32 for any fp32 matmul paths (heads, loss, fp32 init ops).
    # bf16 autocast already covers the hot path; TF32 catches the rest.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # DA3-XVLA DDP perf knobs.
    #   XVLA_DDP_STATIC_GRAPH=1 → static_graph=True. DDP scans for unused params
    #     ONCE on iter 1, caches the bucket plan, reuses forever. Correct for
    #     frozen-DA3 scenarios because the unused set is identical every iter.
    #     Strict superset of find_unused for the perf-critical path.
    #   XVLA_DDP_FIND_UNUSED=1 (legacy) → per-iter autograd traversal. ~8x slower.
    _static_graph = os.environ.get("XVLA_DDP_STATIC_GRAPH", "0") == "1"
    _find_unused = os.environ.get("XVLA_DDP_FIND_UNUSED", "0") == "1"
    _accel_kwargs = dict(log_with="tensorboard", project_dir=output_dir)
    # A6 from 2026-06-11 audit: broadcast_buffers=False — the model has no
    # BatchNorm/SyncBN running buffers (LayerNorms only), so DDP's per-iter
    # named_buffers walk is pure overhead. ~1-2% of wall samples per py-spy.
    if _static_graph:
        _accel_kwargs["kwargs_handlers"] = [DistributedDataParallelKwargs(
            static_graph=True, broadcast_buffers=False)]
    elif _find_unused:
        _accel_kwargs["kwargs_handlers"] = [DistributedDataParallelKwargs(
            find_unused_parameters=True, broadcast_buffers=False)]
    else:
        _accel_kwargs["kwargs_handlers"] = [DistributedDataParallelKwargs(
            broadcast_buffers=False)]
    accelerator = Accelerator(**_accel_kwargs)
    accelerator.init_trackers("XVLA-Training")
    
    accelerator.wait_for_everyone()
    logger = get_logger(__name__, output_dir=output_dir, accelerator=accelerator)
    
    set_seed(args.seed + accelerator.process_index)
    logger.info(f"Args: {args}")

    # Load model & processor
    model = XVLA.from_pretrained(args.models)
    processor = XVLAProcessor.from_pretrained(args.models)

    # DA3-XVLA: HF re-initialized da3_inline.* (they're not in the X-VLA
    # checkpoint), wiping the DA3 pretrained weights. Restore them — but ONLY
    # if the checkpoint we just loaded doesn't already contain trained DA3
    # weights (which happens when resuming from a prior training run that had
    # DA3 unfrozen). Otherwise we'd blow away the fine-tuned weights.
    def _ckpt_has_prefix(prefix: str) -> bool:
        try:
            from safetensors import safe_open as _so
            import glob as _g
            st = os.path.join(args.models, "model.safetensors")
            if os.path.exists(st):
                with _so(st, framework="pt") as f:
                    return any(k.startswith(prefix) for k in f.keys())
            shards = _g.glob(os.path.join(args.models, "model-*.safetensors"))
            for s in shards:
                with _so(s, framework="pt") as f:
                    if any(k.startswith(prefix) for k in f.keys()):
                        return True
            bin_p = os.path.join(args.models, "pytorch_model.bin")
            if os.path.exists(bin_p):
                import torch as _t
                _sd = _t.load(bin_p, map_location="cpu", weights_only=True)
                hit = any(k.startswith(prefix) for k in _sd.keys())
                del _sd
                return hit
        except Exception as _e:
            logger.warning(f"_ckpt_has_prefix({prefix}) inspection failed: {_e}")
        return False

    if getattr(model, "da3_inline", None) is not None:
        if _ckpt_has_prefix("da3_inline."):
            logger.info("Checkpoint contains trained da3_inline weights — skipping reload_pretrained_weights (resume)")
        else:
            model.da3_inline.reload_pretrained_weights()
    # Same fix for V2: HF wiped G-SAM weights too (Grounding-DINO + SAM are
    # 1.4k+ params that get re-inited because they weren't in the X-VLA ckpt).
    # Restore them so segmentation actually produces meaningful masks.
    if getattr(model, "gsam_inline", None) is not None:
        if _ckpt_has_prefix("gsam_inline."):
            logger.info("Checkpoint contains trained gsam_inline weights — skipping reload_pretrained_weights (resume)")
        else:
            model.gsam_inline.reload_pretrained_weights()
    # Same problem on the Perceiver query — force-reinit to std=1.0 so the
    # geometry path produces unit-scale tokens from step 0 (otherwise it
    # collapses to denormals through the residual+LayerNorm pipeline).
    if getattr(model, "geometry_conditioner", None) is not None:
        # Auto-detect whether the loaded checkpoint already contains trained
        # geometry_fusion weights (a previous training run's ckpt) vs a fresh
        # base whose safetensors lacks them (new run). If trained weights are
        # present, skip the re-init so we don't wipe DA3 fusion learning. The
        # env var XVLA_SKIP_GEOMETRY_REINIT still works as a manual override.
        ckpt_has_geometry = False
        try:
            from safetensors import safe_open
            st_path = os.path.join(args.models, "model.safetensors")
            bin_path = os.path.join(args.models, "pytorch_model.bin")
            import glob as _glob
            shard_paths = _glob.glob(os.path.join(args.models, "model-*.safetensors"))
            if os.path.exists(st_path):
                with safe_open(st_path, framework="pt") as f:
                    ckpt_has_geometry = any(
                        ("geometry_fusion" in k) or ("geometry_conditioner" in k)
                        for k in f.keys()
                    )
            elif shard_paths:
                for sp in shard_paths:
                    with safe_open(sp, framework="pt") as f:
                        if any(("geometry_fusion" in k) or ("geometry_conditioner" in k)
                               for k in f.keys()):
                            ckpt_has_geometry = True; break
            elif os.path.exists(bin_path):
                # Resume from a .bin checkpoint (e.g. ckpt-50000 in our runs).
                # torch.load+weights_only=True only reads the key list, cheap.
                import torch as _t
                sd = _t.load(bin_path, map_location="cpu", weights_only=True)
                ckpt_has_geometry = any(
                    ("geometry_fusion" in k) or ("geometry_conditioner" in k)
                    for k in sd.keys()
                )
                del sd
        except Exception as e:
            logger.warning(f"Could not inspect checkpoint weights at {args.models}: {e}")

        if ckpt_has_geometry:
            logger.info("Checkpoint contains trained geometry weights — skipping _reinit_geometry_modules (resume)")
        elif os.environ.get("XVLA_SKIP_GEOMETRY_REINIT", "0") == "1":
            logger.info("XVLA_SKIP_GEOMETRY_REINIT=1 — keeping checkpoint geometry weights (manual override)")
        else:
            logger.info("Checkpoint has no geometry weights — calling _reinit_geometry_modules (fresh start)")
            model._reinit_geometry_modules()

    # Florence-2 (VLM) gradient checkpointing — env-gated opt-in. Frees ~10-15 GB
    # of VLM activation memory per rank at the cost of ~20-30% backward compute.
    # Required to fit bs>=12 with unfrozen DA3 on 80 GB H100. Uses HF's standard
    # API; safe with DDP when use_reentrant=False (HF default in recent versions).
    if os.environ.get("XVLA_VLM_GRADIENT_CHECKPOINTING", "0") == "1":
        if hasattr(model, "vlm") and hasattr(model.vlm, "gradient_checkpointing_enable"):
            try:
                model.vlm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                logger.info("VLM (Florence-2) gradient_checkpointing ENABLED via env XVLA_VLM_GRADIENT_CHECKPOINTING=1")
            except TypeError:
                # older HF signature without kwargs
                model.vlm.gradient_checkpointing_enable()
                logger.info("VLM (Florence-2) gradient_checkpointing ENABLED (legacy HF signature)")
            except Exception as e:
                logger.warning(f"Failed to enable VLM gradient_checkpointing: {e}")

    # DA3-XVLA: optional selective fine-tuning (opt-in; baseline unaffected).
    if args.apply_finetune_policy:
        summary = model.apply_finetune_policy()
        logger.info(f"Applied finetune policy: {summary}")
        # apply_finetune_policy sweeps every param NOT explicitly listed in
        # {backbone, soft, head, geom} into the "core" group and unfreezes it
        # when train_last_n_policy_layers=-1. That includes DA3 and G-SAM.
        # Re-freeze them ONLY when the config asks for it; otherwise honor the
        # unfreeze (e.g. freeze_da3=False means we want DA3 trainable).
        geom_cfg = dict(getattr(model.config, "geometry_conditioning", {}) or {})
        freeze_flags = {
            "da3_inline":  bool(geom_cfg.get("freeze_da3",  True)),
            "gsam_inline": bool(geom_cfg.get("freeze_gsam", True)),
        }
        for sub_name in ("da3_inline", "gsam_inline"):
            sub = getattr(model, sub_name, None)
            if sub is None:
                continue
            if not freeze_flags[sub_name]:
                logger.info(f"{sub_name}: leaving UNFROZEN per config (freeze_{sub_name.split('_')[0]}=False)")
                continue
            n = 0
            for p in sub.parameters():
                if p.requires_grad:
                    p.requires_grad_(False); n += 1
            if n > 0:
                logger.info(f"Re-froze {n} params in {sub_name} (apply_finetune_policy had unfrozen them)")
        # Recompute trainable count post-freeze
        trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        total = int(sum(p.numel() for p in model.parameters()))
        logger.info(f"Post-freeze: total={total} trainable={trainable} ({100*trainable/total:.2f}%)")

    # Iterable dataloader (don't wrap with prepare)
    train_dataloader = create_dataloader(
        batch_size=args.batch_size,
        metas_path=args.train_metas_path,
        num_actions=model.num_actions,
        action_mode=model.action_mode,
        training=True,
        # DA3-XVLA: only non-None when geometry conditioning is enabled, so
        # the baseline data path is completely unaffected otherwise.
        geometry_conditioning=(
            model.config.geometry_conditioning
            if getattr(model, "geometry_enabled", False) else None
        ),
    )

    # Optimizer
    optim = build_optimizer(
        model=model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        lr_coef_soft=args.learning_coef,
    )
    model, optim = accelerator.prepare(model, optim)

    # Training loop
    model.train()
    global_step, t0 = 0, time.time()
    logger.info(f"🚀 Start training for {args.iters} iterations | world_size={accelerator.num_processes}")
    
    for batch in train_dataloader:
        # Encode language
        lang = processor.encode_language(batch["language_instruction"])
        # V2 (inline G-SAM) needs the raw text strings; pass them through
        # unchanged so model.forward gets a list[str].
        raw_instructions = batch.pop("language_instruction", None)
        inputs = {**batch, **lang}
        inputs = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
                  for k, v in inputs.items()}
        # V2 only: re-attach raw instructions so inline G-SAM can use them.
        _underlying = model.module if hasattr(model, "module") else model
        if getattr(_underlying, "gsam_inline", None) is not None:
            inputs["language_instruction"] = raw_instructions
        # Update LR per group
        update_group_lrs(optim, global_step, args)

        # Forward & backward
        loss_dict: Dict[str, torch.Tensor] = model(**inputs)
        loss = sum(loss_dict.values())
        accelerator.backward(loss)
        if args.max_grad_norm:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # Per-group gradient norms (R2 diagnostic) — captured BEFORE optim.step
        # so we see actual grads. Computed only at log_interval to avoid overhead.
        # Per-group gradient norms: opt-in diagnostic via XVLA_GROUP_GNORM=1.
        # Default OFF: was syncing the GPU + iterating ~879M params every
        # log_interval, adding visible overhead vs colleague's training loop.
        group_grad_norms: Dict[str, float] = {}
        if os.environ.get("XVLA_GROUP_GNORM", "0") == "1" and global_step % args.log_interval == 0:
            for g in optim.param_groups:
                sq = 0.0
                for p in g["params"]:
                    if p.grad is not None:
                        sq += float(p.grad.detach().pow(2).sum().item())
                group_grad_norms[g["name"]] = sq ** 0.5

        optim.step()
        # A5 from 2026-06-11 audit: set_to_none=True avoids the in-place fill
        # on every gradient buffer (~879 M trainable params). ~2% wall-clock.
        optim.zero_grad(set_to_none=True)

        # Logging — A11 from audit: defer all .item() / GPU↔CPU syncs to
        # rank 0 only. Non-main ranks no longer block on log_interval.
        if global_step % args.log_interval == 0 and accelerator.is_main_process:
            logs = {k: v.detach().float().item() for k, v in loss_dict.items()}
            logs["loss_total"] = float(loss.detach().item())
            logs.update({f"lr_{g['name']}": g["lr"] for g in optim.param_groups})
            logs.update({f"gnorm_{name}": v for name, v in group_grad_norms.items()})
            accelerator.log(logs, step=global_step)

            dt = (time.time() - t0) / args.log_interval
            t0 = time.time()
            cpu_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**3
            gpu_mem = torch.cuda.memory_allocated() / 1024**3
            # R2: include per-group grad norms in console log so we can
            # tell at a glance which groups have meaningful gradients.
            gn_str = " ".join(
                f"gn_{n[:4]}={v:.2e}" for n, v in group_grad_norms.items()
            )
            logger.info(
                f"[{global_step}/{args.iters}] "
                f"loss={logs['loss_total']:.4f} "
                f"lr_core={logs['lr_transformer_core']:.2e} "
                f"lr_vlm={logs['lr_vlm']:.2e} ({dt:.2f}s/it) "
                f"USED_CPU={cpu_mem:.2e} GB "
                f"USED_GPU={gpu_mem:.2e} GB "
                f"{gn_str}"
            )
        
        # Checkpointing
        global_step += 1
        if accelerator.is_main_process:
            if global_step == args.iters or global_step % args.save_interval == 0:
                save_dir = os.path.join(output_dir, f"ckpt-{global_step}")
                accelerator.print(f"💾 Saving model to {save_dir}")
                # safe_serialization=True trips on DA3's shared tensors
                # (head.scratch.output_conv2_aux.*); use the pickle format.
                accelerator.unwrap_model(model).save_pretrained(save_dir, safe_serialization=False)
                processor.save_pretrained(save_dir)
                with open(os.path.join(save_dir, "state.json"), "w") as f:
                    json.dump({"global_step": global_step}, f)
        if global_step >= args.iters: break

    accelerator.end_training()

# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("XVLA training script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
