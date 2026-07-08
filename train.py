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
from models.configuration_xvla import XVLAConfig
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
    parser.add_argument("--method_b_conservative_geometry_lr", action="store_true", default=False,
                        help="Spatial Method B only: use 1e-4 geometry LR for steps 0-5k, then normal schedule")
    parser.add_argument("--spatial_lang_method", type=str, default=None,
                        choices=["post_refiner", "final6_injection"],
                        help="Enable DA3/T5 spatial-language XVLA with Method A or B")
    parser.add_argument("--da3_model", type=str, default=None,
                        help="Override the DA3 geometry backbone (default depth-anything/DA3-Large-1.1; "
                             "e.g. depth-anything/DA3NESTED-GIANT-LARGE-1.1)")
    parser.add_argument("--da3_out_layers", type=int, nargs="+", default=None,
                        help="Override DA3 feature tap layers to match backbone depth "
                             "(DA3-Large 24-blk=[11,15,19,23]; GIANT 40-blk e.g. 19 26 33 39)")
    parser.add_argument("--text_encoder_name", type=str, default=None,
                        help="Override the language text encoder (default t5-base; "
                             "e.g. answerdotai/ModernBERT-large). Stored in config key t5_model_name.")
    parser.add_argument("--spatial_posed_da3", action="store_true",
                        help="Feed all views jointly into DA3 with camera extrinsics+intrinsics "
                             "(cross-view-consistent per-view geometry). Requires the loader to "
                             "attach extrinsics/intrinsics. Sets spatial_lang.use_posed_da3=True.")
    parser.add_argument("--spatial_depth_ray", action="store_true",
                        help="Append DA3 metric depth (log) as a 7th ray channel so each grid "
                             "token carries ray direction + distance (scale-aware ray). Needs "
                             "--spatial_posed_da3 (metric depth = GIANT da3_metric). "
                             "Sets spatial_lang.use_depth_ray=True.")
    parser.add_argument("--spatial_tokens", type=int, nargs=3, default=None,
                        metavar=("MAIN", "LEFT", "RIGHT"),
                        help="Override per-view perceiver token counts (main left right). "
                             "Default 96 64 64. More tokens = finer spatial banks, slightly "
                             "more perceiver + injection cross-attn compute.")
    parser.add_argument("--spatial_no_perceiver", action="store_true",
                        help="Bypass the perceiver downsampler; feed the full DA3 grid tokens "
                             "per view to language fusion (1:1 spatial correspondence, larger "
                             "banks, no learned compression). Sets spatial_lang.use_perceiver=False. "
                             "--spatial_tokens is then moot (banks = grid size).")
    parser.add_argument("--spatial_scale_start", type=float, default=0.01,
                        help="Initial fixed residual scale for DA3/T5 spatial updates")
    parser.add_argument("--spatial_scale_end", type=float, default=1.0,
                        help="Final fixed residual scale for DA3/T5 spatial updates")
    parser.add_argument("--spatial_scale_ramp_until", type=int, default=15000,
                        help="Step by which DA3/T5 spatial residual scale reaches 1.0")
    parser.add_argument("--fixed_geometry_lr", type=float, default=None,
                        help="Override geometry/new-module LR with a constant value")
    parser.add_argument("--fixed_xvla_core_lr", type=float, default=None,
                        help="Override XVLA core LR with a constant value")
    parser.add_argument("--xvla_core_cosine_start_lr", type=float, default=None,
                        help="Use a custom XVLA core cosine schedule starting from this LR")
    parser.add_argument("--xvla_core_cosine_min_lr", type=float, default=None,
                        help="Minimum LR for the custom XVLA core cosine schedule")
    parser.add_argument("--xvla_core_cosine_start_step", type=int, default=0,
                        help="Local step where the custom XVLA core cosine decay starts")
    parser.add_argument("--xvla_core_cosine_end_step", type=int, default=None,
                        help="Local step where the custom XVLA core cosine decay reaches min LR")
    parser.add_argument("--geometry_cosine_start_lr", type=float, default=None,
                        help="Use a custom geometry-group cosine schedule starting from this LR "
                             "(overrides --fixed_geometry_lr and the default geometry schedule)")
    parser.add_argument("--geometry_cosine_min_lr", type=float, default=None,
                        help="Minimum LR for the custom geometry cosine schedule")
    parser.add_argument("--geometry_cosine_start_step", type=int, default=0,
                        help="Local step where the geometry cosine decay starts")
    parser.add_argument("--geometry_cosine_end_step", type=int, default=None,
                        help="Local step where the geometry cosine reaches min LR (defaults to --iters)")
    parser.add_argument("--fixed_vlm_lr", type=float, default=None,
                        help="Override VLM/FLORENCE LR with a constant value")
    parser.add_argument("--spatial_aux_heads", action="store_true", default=False,
                        help="Enable auxiliary spatial endpoint and projected heatmap heads")
    parser.add_argument("--spatial_aux_endpoint_weight", type=float, default=0.002,
                        help="Loss weight for final-chunk bimanual EE6D endpoint auxiliary head")
    parser.add_argument("--spatial_aux_heatmap_weight", type=float, default=0.02,
                        help="Loss weight for projected EE heatmap auxiliary head")
    parser.add_argument("--spatial_aux_heatmap_sigma", type=float, default=1.25,
                        help="Gaussian sigma in DA3 patch units for EE heatmap targets")

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
    vlm_params = [p for p in model.vlm.parameters() if p.requires_grad]
    soft_prompt_params = [p for p in model.transformer.soft_prompt_hub.parameters() if p.requires_grad]
    action_params = (
        [p for p in model.transformer.action_decoder.parameters() if p.requires_grad]
        + [p for p in model.transformer.action_encoder.parameters() if p.requires_grad]
    )

    def _unique(params):
        out, seen = [], set()
        for p in params:
            if p is None or not p.requires_grad or id(p) in seen:
                continue
            out.append(p)
            seen.add(id(p))
        return out

    # DA3/T5 Spatial-Language XVLA training plan: three explicit groups.
    # Frozen DA3/T5 backbones are excluded entirely; the VLM group uses LR=0
    # until step 5k instead of toggling requires_grad under DDP.
    if getattr(model, "spatial_lang", None) is not None:
        model.spatial_lang.freeze_backbones()
        geometry_params = []
        for name, p in model.spatial_lang.named_parameters():
            if name.startswith("da3.") or name.startswith("t5."):
                continue
            geometry_params.append(p)
        for attr in ("spatial_refiner", "spatial_injection_layers"):
            mod = getattr(model.transformer, attr, None)
            if mod is not None:
                geometry_params += list(mod.parameters())
        if getattr(model, "spatial_aux_heads", None) is not None:
            geometry_params += list(model.spatial_aux_heads.parameters())
        geometry_params = _unique(geometry_params)
        vlm_params = _unique(vlm_params)

        frozen_backbone_ids = {
            id(p)
            for name, p in model.spatial_lang.named_parameters()
            if name.startswith("da3.") or name.startswith("t5.")
        }
        exclude = set(map(id, geometry_params + vlm_params)) | frozen_backbone_ids
        xvla_core_params = _unique(
            [p for p in model.parameters() if id(p) not in exclude]
        )
        return AdamW([
            {"name": "geometry", "params": geometry_params, "lr": 0.0, "weight_decay": weight_decay},
            {"name": "xvla_core", "params": xvla_core_params, "lr": 0.0, "weight_decay": weight_decay},
            {"name": "vlm", "params": vlm_params, "lr": 0.0, "weight_decay": weight_decay},
        ], betas=betas)

    # DA3-XVLA: geometry conditioner + cross-attention fusion get their own
    # group so they can be trained at the action-head LR. Empty when disabled,
    # which leaves baseline param groups byte-for-byte identical.
    geometry_params = []
    if getattr(model, "geometry_conditioner", None) is not None:
        geometry_params += list(model.geometry_conditioner.parameters())
    if getattr(model.transformer, "geometry_fusion", None) is not None:
        geometry_params += list(model.transformer.geometry_fusion.parameters())
    # DA3-XVLA: gated action-to-spatial cross-attention adapters land in the
    # geometry group so they cosine with the other geometry modules at full LR.
    if getattr(model.transformer, "spatial_cross_attn_layers", None) is not None:
        for adapter in model.transformer.spatial_cross_attn_layers:
            if any(True for _ in adapter.parameters()):  # skip nn.Identity slots
                geometry_params += list(adapter.parameters())
    # T5 projector (small MLP that maps T5 d_model → hidden_size). Frozen T5
    # encoder weights live in t5_inline.model.* and stay frozen — they are NOT
    # in any optimizer group. Only the projector trains, alongside the other
    # geometry modules at the full action-head LR.
    if getattr(model, "t5_projector", None) is not None:
        geometry_params += list(model.t5_projector.parameters())

    # Spatial-language Method A/B:
    #   - tokenizer projectors/perceivers/lang-fusers train at full geometry LR
    #   - Method A refiner / Method B injection layers train at full geometry LR
    #   - frozen DA3/T5 backbones remain excluded and explicitly frozen
    if getattr(model, "spatial_lang", None) is not None:
        for name, p in model.spatial_lang.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("da3.") or name.startswith("t5."):
                continue
            geometry_params.append(p)
    for attr in ("spatial_refiner", "spatial_injection_layers"):
        mod = getattr(model.transformer, attr, None)
        if mod is not None:
            for p in mod.parameters():
                if p.requires_grad:
                    geometry_params.append(p)
    if getattr(model, "spatial_aux_heads", None) is not None:
        for p in model.spatial_aux_heads.parameters():
            if p.requires_grad:
                geometry_params.append(p)

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

    # GeoStack-XVLA v2B: the two GeoStack module groups must train at the
    # action-head LR after freeze_steps (same as other geometry modules).
    #   - geostack_token_builders: top-level attr; otherwise would land in
    #     transformer_core_params and stay at lr=0 for freeze_steps, then 1e-4
    #     (eventually OK but undertrained for 5k steps).
    #   - geostack_fusers: also registered as submodule of the Florence encoder
    #     via attach_geostack, so list(model.vlm.parameters()) sweeps them into
    #     vlm_params. We must explicitly collect them into geometry_params AND
    #     exclude their ids from vlm_params, otherwise they'd train at lr_vlm
    #     = lr * lr_coef_vlm = 1e-5 (10× too low for fresh-init adapters).
    geostack_adapter_params = []
    geostack_adapter_param_ids = set()
    # v2C: include wrist_geostack (per-wrist fusers + world ray MLP) AND
    # wrist_da3_token_builder (dedicated builder, not shared with main's "deep").
    for attr in ("geostack_token_builders", "geostack_fusers", "wrist_geostack", "wrist_da3_token_builder"):
        mod = getattr(model, attr, None)
        if mod is None:
            continue
        for p in mod.parameters():
            if not p.requires_grad:
                continue
            geostack_adapter_params.append(p)
            geostack_adapter_param_ids.add(id(p))
    geometry_params += geostack_adapter_params
    # Strip GeoStack adapter ids out of vlm_params so the fusers aren't
    # double-counted in the vlm group (they live in both ModuleDicts).
    if geostack_adapter_param_ids:
        vlm_params = [p for p in vlm_params if id(p) not in geostack_adapter_param_ids]

    # GeoStack-XVLA v3 side stack: route the new modules into the geometry
    # group so (a) gn_geom in the training log actually reflects side stack
    # gradient (was previously going into gn_tran via the transformer_core
    # leftovers bucket), and (b) they train at constant base["geometry"] = lr
    # from step 0 instead of warmup-ramping with transformer_core (the side
    # stack is fresh-init and benefits from full LR immediately once α > 0).
    # Modules collected:
    #   - transformer.side_stack             — 6 standard transformer blocks
    #   - transformer.s_to_x_xattn           — 6 gated cross-attention adapters
    #   - side_stack_bank_builder            — DA3/T5/ray projectors, view+modality emb, g_ray/g_pos
    #   - side_stack_aux_head    (v3.5)     — aux action-prediction head (pool query + LN + MLP)
    side_stack_param_ids = set()
    side_stack_mod = getattr(getattr(model, "transformer", None), "side_stack", None)
    s_to_x_xattn = getattr(getattr(model, "transformer", None), "s_to_x_xattn", None)
    x_to_s_xattn = getattr(getattr(model, "transformer", None), "x_to_s_xattn", None)   # v3.7
    bank_builder = getattr(model, "side_stack_bank_builder", None)
    aux_head = getattr(model, "side_stack_aux_head", None)
    for mod in (side_stack_mod, s_to_x_xattn, x_to_s_xattn, bank_builder, aux_head):
        if mod is None:
            continue
        for p in mod.parameters():
            if not p.requires_grad:
                continue
            if id(p) in side_stack_param_ids:
                continue
            geometry_params.append(p)
            side_stack_param_ids.add(id(p))

    exclude = set(map(id, vlm_params + soft_prompt_params + action_params + geometry_params + da3_params))
    transformer_core_params = [p for p in model.parameters() if p.requires_grad and id(p) not in exclude]
    spatial_lang_on = getattr(model, "spatial_lang", None) is not None
    old_xvla_scale = lr_coef_soft if spatial_lang_on else 1.0
    param_groups = [
        {"name": "vlm", "params": vlm_params, "lr": 0.0, "lr_scale": lr_coef_soft, "weight_decay": weight_decay},
        {"name": "transformer_core", "params": transformer_core_params, "lr": 0.0, "lr_scale": old_xvla_scale, "weight_decay": weight_decay},
        {"name": "soft_prompts", "params": soft_prompt_params, "lr": lr * lr_coef_soft, "lr_scale": lr_coef_soft, "weight_decay": weight_decay},
        {"name": "action_heads", "params": action_params, "lr": lr * old_xvla_scale, "lr_scale": old_xvla_scale, "weight_decay": weight_decay},
    ]
    if geometry_params:
        param_groups.append(
            {"name": "geometry", "params": geometry_params, "lr": lr, "lr_scale": 1.0, "weight_decay": weight_decay}
        )
    if da3_params:
        param_groups.append(
            {"name": "da3", "params": da3_params, "lr": lr * lr_coef_soft, "lr_scale": lr_coef_soft, "weight_decay": weight_decay}
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


def cosine_decay_lr(step, start_lr, min_lr, cosine_start=50000, cosine_end=125000):
    if step < cosine_start:
        return start_lr
    progress = min((step - cosine_start) / max(1, cosine_end - cosine_start), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (start_lr - min_lr) * cosine


def action_lr_schedule(step):
    if step < 2000:
        return 2e-5
    if step < 10000:
        return 5e-5
    if step < 15000:
        alpha = (step - 10000) / 5000.0
        return 5e-5 + alpha * (2e-4 - 5e-5)
    if step < 50000:
        return 2e-4
    return cosine_decay_lr(step, start_lr=2e-4, min_lr=5e-5)


def geometry_lr_schedule(step, method_b_conservative=False):
    if method_b_conservative and step < 5000:
        return 1e-4
    if step < 50000:
        return 2e-4
    return cosine_decay_lr(step, start_lr=2e-4, min_lr=5e-5)


def vlm_lr_schedule(step):
    if step < 5000:
        return 0.0
    if step < 15000:
        alpha = (step - 5000) / 10000.0
        return 1e-6 + alpha * (1e-5 - 1e-6)
    if step < 50000:
        return 1e-5
    return cosine_decay_lr(step, start_lr=1e-5, min_lr=2.5e-6)


def spatial_scale_schedule(step, start=0.01, end=1.0, ramp_until=15000):
    if ramp_until <= 0:
        return float(end)
    return float(start) + (float(end) - float(start)) * min(step / float(ramp_until), 1.0)


def update_group_lrs(optim, step, args):
    """Elegant group-wise LR scheduler."""
    group_names = {g["name"] for g in optim.param_groups}
    if {"geometry", "xvla_core", "vlm"}.issubset(group_names):
        method_b_conservative = bool(getattr(args, "method_b_conservative_geometry_lr", False))
        if getattr(args, "fixed_geometry_lr", None) is not None:
            geom_lr = float(args.fixed_geometry_lr)
        elif getattr(args, "geometry_cosine_start_lr", None) is not None:
            if getattr(args, "geometry_cosine_min_lr", None) is None:
                raise ValueError("--geometry_cosine_min_lr is required with --geometry_cosine_start_lr")
            geom_lr = cosine_decay_lr(
                step,
                start_lr=float(args.geometry_cosine_start_lr),
                min_lr=float(args.geometry_cosine_min_lr),
                cosine_start=int(args.geometry_cosine_start_step),
                cosine_end=int(args.geometry_cosine_end_step or args.iters),
            )
        else:
            geom_lr = geometry_lr_schedule(step, method_b_conservative)
        if getattr(args, "fixed_xvla_core_lr", None) is not None:
            core_lr = float(args.fixed_xvla_core_lr)
        elif getattr(args, "xvla_core_cosine_start_lr", None) is not None:
            if getattr(args, "xvla_core_cosine_min_lr", None) is None:
                raise ValueError("--xvla_core_cosine_min_lr is required with --xvla_core_cosine_start_lr")
            core_lr = cosine_decay_lr(
                step,
                start_lr=float(args.xvla_core_cosine_start_lr),
                min_lr=float(args.xvla_core_cosine_min_lr),
                cosine_start=int(args.xvla_core_cosine_start_step),
                cosine_end=int(args.xvla_core_cosine_end_step or args.iters),
            )
        else:
            core_lr = action_lr_schedule(step)
        vlm_lr = (
            float(args.fixed_vlm_lr)
            if getattr(args, "fixed_vlm_lr", None) is not None
            else vlm_lr_schedule(step)
        )
        set_group_lr(optim, "geometry", geom_lr)
        set_group_lr(optim, "xvla_core", core_lr)
        set_group_lr(optim, "vlm", vlm_lr)
        return

    default_scale = {
        "vlm": args.learning_coef,
        "transformer_core": 1.0,
        "soft_prompts": args.learning_coef,
        "action_heads": 1.0,
        "geometry": 1.0,
        "da3": args.learning_coef,
    }
    base = {
        g["name"]: args.learning_rate * float(g.get("lr_scale", default_scale.get(g["name"], 1.0)))
        for g in optim.param_groups
    }
    has_geometry = "geometry" in base
    has_da3 = "da3" in base
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


def clear_param_group_grads(optim, group_name: str) -> None:
    for g in optim.param_groups:
        if g.get("name") != group_name:
            continue
        for p in g["params"]:
            p.grad = None


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

    # Load model & processor. Optional Method A/B override enables the
    # spatial-language config before HF builds the module tree.
    if args.spatial_lang_method is not None:
        cfg = XVLAConfig.from_pretrained(args.models)
        gc = dict(getattr(cfg, "geometry_conditioning", {}) or {})
        # Spatial-language Method A/B starts from vanilla XVLA plus the new
        # DA3/T5 action-slice pathway. Checkpoints may carry older geometry
        # experiments in config.json; force those off before module init so we
        # do not silently train side-stack/geostack variants.
        gc["enabled"] = False
        gc["use_spatial_cross_attention"] = False
        gc["t5_language_enabled"] = False
        gc["use_geostack"] = False
        gc["use_wrist_geostack"] = False
        gc["use_geo_perceiver_v2f"] = False
        side_cfg = dict(gc.get("side_stack", {}) or {})
        side_cfg["enabled"] = False
        if isinstance(side_cfg.get("aux_loss"), dict):
            aux_cfg = dict(side_cfg["aux_loss"])
            aux_cfg["enabled"] = False
            side_cfg["aux_loss"] = aux_cfg
        if isinstance(side_cfg.get("multi_view_da3"), dict):
            mv_cfg = dict(side_cfg["multi_view_da3"])
            mv_cfg["enabled"] = False
            side_cfg["multi_view_da3"] = mv_cfg
        gc["side_stack"] = side_cfg
        sl = dict(gc.get("spatial_lang", {}) or {})
        sl.update({
            "enabled": True,
            "method": args.spatial_lang_method,
            "main_tokens": 96,
            "left_tokens": 64,
            "right_tokens": 64,
            "da3_freeze": True,
            "t5_freeze": True,
            "spatial_scale_start": float(args.spatial_scale_start),
            "spatial_scale_end": float(args.spatial_scale_end),
            "spatial_scale_ramp_until": int(args.spatial_scale_ramp_until),
        })
        # Optional per-run backbone / text-encoder overrides (fall through to
        # config defaults — DA3-Large + t5-base — when the flags are omitted).
        if args.da3_model is not None:
            sl["da3_model"] = str(args.da3_model)
        if args.da3_out_layers is not None:
            sl["da3_out_layers"] = [int(x) for x in args.da3_out_layers]
        if args.text_encoder_name is not None:
            sl["t5_model_name"] = str(args.text_encoder_name)
        if args.spatial_posed_da3:
            sl["use_posed_da3"] = True
        if args.spatial_depth_ray:
            sl["use_depth_ray"] = True
        if args.spatial_tokens is not None:
            sl["main_tokens"], sl["left_tokens"], sl["right_tokens"] = (int(x) for x in args.spatial_tokens)
        if args.spatial_no_perceiver:
            sl["use_perceiver"] = False
        sl["aux_heads"] = {
            "enabled": bool(args.spatial_aux_heads),
            "endpoint_weight": float(args.spatial_aux_endpoint_weight),
            "heatmap_weight": float(args.spatial_aux_heatmap_weight),
            "heatmap_sigma": float(args.spatial_aux_heatmap_sigma),
            "patch_size": int(sl.get("da3_patch_size", 14)),
            "target_step": -1,
        }
        gc["spatial_lang"] = sl
        cfg.geometry_conditioning = gc
        model = XVLA.from_pretrained(args.models, config=cfg)
    else:
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
            # 1) single safetensors
            st = os.path.join(args.models, "model.safetensors")
            if os.path.exists(st):
                with _so(st, framework="pt") as f:
                    return any(k.startswith(prefix) for k in f.keys())
            # 2) sharded safetensors
            shards = _g.glob(os.path.join(args.models, "model-*.safetensors"))
            for s in shards:
                with _so(s, framework="pt") as f:
                    if any(k.startswith(prefix) for k in f.keys()):
                        return True
            # 3) single pytorch bin
            bin_p = os.path.join(args.models, "pytorch_model.bin")
            if os.path.exists(bin_p):
                import torch as _t
                _sd = _t.load(bin_p, map_location="cpu", weights_only=True)
                hit = any(k.startswith(prefix) for k in _sd.keys())
                del _sd
                return hit
            # 4) sharded pytorch bin — index.json is fastest; falls back to scanning shards
            bin_idx = os.path.join(args.models, "pytorch_model.bin.index.json")
            if os.path.exists(bin_idx):
                import json as _json
                with open(bin_idx) as _f:
                    _idx = _json.load(_f)
                return any(k.startswith(prefix) for k in _idx.get("weight_map", {}).keys())
            bin_shards = _g.glob(os.path.join(args.models, "pytorch_model-*.bin"))
            if bin_shards:
                import torch as _t
                for s in bin_shards:
                    _sd = _t.load(s, map_location="cpu", weights_only=True)
                    hit = any(k.startswith(prefix) for k in _sd.keys())
                    del _sd
                    if hit:
                        return True
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
    # Same problem with T5 encoder — HF wipes its weights. Restore them so the
    # language K/V tokens come from the actual pretrained T5, not random init.
    if getattr(model, "t5_inline", None) is not None:
        if _ckpt_has_prefix("t5_inline."):
            logger.info("Checkpoint contains trained t5_inline weights — skipping reload_pretrained_weights (resume)")
        else:
            model.t5_inline.reload_pretrained_weights()
    # Spatial-language Method A/B has its own frozen DA3 + frozen T5 backbones.
    # Reload them after HF from_pretrained, then reset trainable fresh modules
    # when the checkpoint did not already contain trained spatial_lang weights.
    if getattr(model, "spatial_lang", None) is not None:
        has_spatial_lang = (
            _ckpt_has_prefix("spatial_lang.")
            or _ckpt_has_prefix("transformer.spatial_refiner.")
            or _ckpt_has_prefix("transformer.spatial_injection_layers.")
        )
        if has_spatial_lang:
            logger.info("Checkpoint contains trained spatial-language weights — skipping reload/reset (resume)")
            model.spatial_lang.freeze_backbones()
        else:
            logger.info("[spatial_lang] reloading frozen DA3/T5 and resetting trainable modules")
            model.spatial_lang.reload_pretrained_weights()
            if os.environ.get("XVLA_SKIP_GEOMETRY_REINIT", "0") != "1":
                model.spatial_lang.reset_trainable_parameters()
                for attr in ("spatial_refiner", "spatial_injection_layers"):
                    mod = getattr(model.transformer, attr, None)
                    if mod is None:
                        continue
                    for m in mod.modules():
                        if hasattr(m, "reset_parameters") and callable(m.reset_parameters):
                            try:
                                m.reset_parameters()
                            except Exception:
                                pass
            model.spatial_lang.freeze_backbones()
    # GeoStack-XVLA v2B: same problem with the frozen DA3-Large branch — HF
    # wipes its 411M weights because they're not in the X-VLA-Pt checkpoint.
    # Restore them so DA3 produces real geometry features when alpha > 0.
    if getattr(model, "da3_for_geostack", None) is not None:
        if _ckpt_has_prefix("da3_for_geostack."):
            logger.info("Checkpoint contains trained da3_for_geostack weights — skipping reload_pretrained_weights (resume)")
        else:
            logger.info("[GeoStack] reloading DA3-Large pretrained weights after HF init wiped them")
            model.da3_for_geostack.reload_pretrained_weights()
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

    # GeoStack-XVLA v2B re-init: the same `_reinit_geometry_modules` call also
    # resets geostack_token_builders / geostack_fusers (HF's no_init_weights
    # left them as raw uninitialized memory because they were built inside
    # XVLA.__init__). Fire it independently of the legacy geometry_conditioner
    # path so GeoStack-only runs ALSO get the re-init.
    if getattr(model, "geostack_token_builders", None) is not None:
        ckpt_has_geostack = (
            _ckpt_has_prefix("geostack_token_builders.")
            or _ckpt_has_prefix("geostack_fusers.")
            or _ckpt_has_prefix("vlm.language_model.model.encoder.geo_fusers.")
        )
        if ckpt_has_geostack:
            logger.info("Checkpoint contains trained GeoStack adapter weights — skipping re-init (resume)")
        elif os.environ.get("XVLA_SKIP_GEOMETRY_REINIT", "0") == "1":
            logger.info("XVLA_SKIP_GEOMETRY_REINIT=1 — keeping GeoStack adapter weights (manual override)")
        else:
            logger.info("[GeoStack] re-initializing token_builders + fusers (HF no_init_weights left them as uninit memory)")
            model._reinit_geometry_modules()

    # GeoStack-XVLA v3 re-init: same problem class as v2B fusers. The side stack
    # blocks + cross-attn adapters + bank builder are all built inside
    # XVLA.__init__ (via the transformer's __init__ and _init_side_stack_bank_builder)
    # under HF's no_init_weights context, so their .weight/.bias buffers are
    # uninitialized memory. _reinit_geometry_modules now also handles them.
    if getattr(getattr(model, "transformer", None), "side_stack", None) is not None:
        ckpt_has_v3 = (
            _ckpt_has_prefix("transformer.side_stack.")
            or _ckpt_has_prefix("transformer.s_to_x_xattn.")
            or _ckpt_has_prefix("transformer.x_to_s_xattn.")  # v3.7: bidirectional
            or _ckpt_has_prefix("side_stack_bank_builder.")
            or _ckpt_has_prefix("side_stack_aux_head.")
        )
        if ckpt_has_v3:
            logger.info("Checkpoint contains trained v3 side stack weights — skipping re-init (resume)")
        elif os.environ.get("XVLA_SKIP_GEOMETRY_REINIT", "0") == "1":
            logger.info("XVLA_SKIP_GEOMETRY_REINIT=1 — keeping v3 side stack weights (manual override)")
        else:
            logger.info("[v3] re-initializing side stack + cross-attn adapters + bank builder")
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
            "t5_inline":   bool(geom_cfg.get("freeze_t5",   True)),
        }
        for sub_name in ("da3_inline", "gsam_inline", "t5_inline"):
            sub = getattr(model, sub_name, None)
            if sub is None:
                continue
            if not freeze_flags[sub_name]:
                logger.info(f"{sub_name}: leaving UNFROZEN per config (freeze_{sub_name.split('_')[0]}=False)")
                continue
            n = 0
            # For t5_inline, only freeze the .model.* params (encoder weights);
            # the t5_projector lives on the top-level XVLA module, NOT on
            # t5_inline, so this loop won't touch it.
            for p in sub.parameters():
                if p.requires_grad:
                    p.requires_grad_(False); n += 1
            if n > 0:
                logger.info(f"Re-froze {n} params in {sub_name} (apply_finetune_policy had unfrozen them)")
        if getattr(model, "spatial_lang", None) is not None:
            model.spatial_lang.freeze_backbones()
            logger.info("Re-froze spatial_lang DA3/T5 backbones")
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
            if (getattr(model, "geometry_enabled", False) or getattr(model, "spatial_lang_enabled", False)) else None
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
        # Also needed when T5 language tokens are enabled (T5 takes raw strings).
        _underlying = model.module if hasattr(model, "module") else model
        if (getattr(_underlying, "gsam_inline", None) is not None
                or getattr(_underlying, "t5_inline", None) is not None
                or getattr(_underlying, "spatial_lang", None) is not None):
            inputs["language_instruction"] = raw_instructions
        # Update LR per group
        update_group_lrs(optim, global_step, args)
        spatial_scale = spatial_scale_schedule(
            global_step,
            start=args.spatial_scale_start,
            end=args.spatial_scale_end,
            ramp_until=args.spatial_scale_ramp_until,
        )

        # GeoStack-XVLA v2B: advance the residual-alpha schedule. Without this
        # the AlphaSchedule._step buffer stays at 0 forever and α never ramps —
        # meaning GeoStack injection is permanently silent regardless of step.
        if getattr(_underlying, "set_geostack_step", None) is not None:
            _underlying.set_geostack_step(global_step)
        if getattr(_underlying, "set_spatial_residual_scale", None) is not None:
            _underlying.set_spatial_residual_scale(spatial_scale)

        # Forward & backward
        loss_dict: Dict[str, torch.Tensor] = model(**inputs)
        loss = sum(loss_dict.values())
        if not torch.isfinite(loss.detach()).all():
            logger.error(f"Non-finite loss at step {global_step}: {loss.detach()}")
            raise FloatingPointError(f"Non-finite loss at step {global_step}")
        accelerator.backward(loss)
        # Spatial-language training freezes VLM/FLORENCE for the first 5k
        # steps. Keep the group in the optimizer for the later ramp, but clear
        # any computed grads while its scheduled LR is zero so logs and clipping
        # reflect an actually frozen group.
        if getattr(_underlying, "spatial_lang", None) is not None:
            if any(g.get("name") == "vlm" and float(g.get("lr", 0.0)) == 0.0 for g in optim.param_groups):
                clear_param_group_grads(optim, "vlm")
        if args.max_grad_norm:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # Per-group gradient norms (R2 diagnostic) — captured BEFORE optim.step
        # so we see actual grads. Computed only at log_interval to avoid overhead.
        # Per-group gradient norms: opt-in diagnostic via XVLA_GROUP_GNORM=1.
        # Default OFF: was syncing the GPU + iterating ~879M params every
        # log_interval, adding visible overhead vs colleague's training loop.
        group_grad_norms: Dict[str, float] = {}
        _spatial_lang_on = getattr(_underlying, "spatial_lang", None) is not None
        if (os.environ.get("XVLA_GROUP_GNORM", "0") == "1" or _spatial_lang_on) and global_step % args.log_interval == 0:
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
            logs["loss"] = logs["loss_total"]
            logs.update({f"lr_{g['name']}": g["lr"] for g in optim.param_groups})
            logs.update({f"gnorm_{name}": v for name, v in group_grad_norms.items()})
            logs.update({f"grad_norm_{name}": v for name, v in group_grad_norms.items()})
            logs["spatial_scale"] = float(spatial_scale)
            spatial_stats = getattr(getattr(_underlying, "transformer", None), "last_spatial_stats", {}) or {}
            if spatial_stats:
                def _stat(name, default=0.0):
                    v = spatial_stats.get(name, default)
                    if torch.is_tensor(v):
                        return float(v.detach().float().item())
                    return float(v)
                logs["h_action_norm"] = _stat("h_action_norm")
                logs["spatial_update_norm_main"] = _stat("main")
                logs["spatial_update_norm_left"] = _stat("left")
                logs["spatial_update_norm_right"] = _stat("right")
                logs["spatial_update_norm_merge"] = _stat("merge")
                logs["spatial_update_ratio_main"] = _stat("ratio_main")
                logs["spatial_update_ratio_left"] = _stat("ratio_left")
                logs["spatial_update_ratio_right"] = _stat("ratio_right")
                logs["spatial_update_ratio_merge"] = _stat("ratio_merge")
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
            lr_core_log = logs.get("lr_xvla_core", logs.get("lr_transformer_core", 0.0))
            lr_geom_log = logs.get("lr_geometry", 0.0)
            sp_ratio = logs.get("spatial_update_ratio_main", 0.0)
            if _spatial_lang_on:
                if global_step > 0 and group_grad_norms.get("geometry", 0.0) == 0.0:
                    logger.warning(f"[health] geometry grad norm is zero at step {global_step}")
                if global_step < 5000 and group_grad_norms.get("vlm", 0.0) != 0.0:
                    logger.warning(f"[health] VLM grad norm nonzero before unfreeze at step {global_step}: {group_grad_norms.get('vlm'):.3e}")
                if global_step > 2000 and sp_ratio < 1e-4:
                    logger.warning(f"[health] spatial branch may be ignored at step {global_step}: main ratio={sp_ratio:.3e}")
                if global_step < 15000 and sp_ratio > 0.5:
                    logger.warning(f"[health] spatial update may be too strong at step {global_step}: main ratio={sp_ratio:.3e}")
            logger.info(
                f"[{global_step}/{args.iters}] "
                f"loss={logs['loss_total']:.4f} "
                f"lr_core={lr_core_log:.2e} "
                f"lr_geom={lr_geom_log:.2e} "
                f"lr_vlm={logs['lr_vlm']:.2e} ({dt:.2f}s/it) "
                f"sp_scale={spatial_scale:.3f} "
                f"sp_main_ratio={sp_ratio:.2e} "
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
