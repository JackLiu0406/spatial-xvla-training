#!/usr/bin/env bash
# VGGT-Omega K=160, multi-layer fusion, aspect-preserved native input.
#
# VGGT input: 240×320 — data-native resolution. patch_size=16 → 15×20 = 300
# patches per view. Aspect 320/240 = 4/3 = exact match to native. Zero resize,
# zero pixel invention, zero distortion.
#
# Multi-layer fusion: aggregator's 4 cached layers (4, 11, 17, 23) are
# LayerNorm'd per-tap → channel-concat (4 × 2C = 8192) → 1×1 Conv → 2C = 2048.
# Init mode "last_only" — Conv identity on the deepest tap so the adapter
# starts close to last-layer behavior (LN modulo) and learns the multi-tap mix.
#
# Recipe matches the K=160 anchors exactly:
#   bs=32×4 H200, lr=1e-4, learning_coef=0.1, freeze=1000, warmup=2000,
#   cosine→0.1×, 100k iters. Effective batch 128 — clean A/B vs the VGGT
#   K=160 (224 squished) anchor and against the DA3-BASE K=160 aspect run.
# (First-launch tried bs=16; live VRAM at 66 GB confirmed bs=32 fits in H200's
#  141 GB envelope, so switched back to the anchor recipe.)
#
# VLM (Florence-2) still sees 224×224 from the dual-image dataloader (the env
# vars XVLA_DA3_INPUT_H/W set the geometry-encoder branch only).
set -eo pipefail

SRC=${REPO_ROOT}
RUN=${RUNS_ROOT}/vggt_omega_K160_multi_aspect
BASE_CKPT="$RUN/ckpt_init"
META=${RUNS_ROOT}/meta_all_ct_3cam_clean.json

[ -d "$BASE_CKPT" ] || { echo "[err] ckpt_init missing: $BASE_CKPT"; exit 1; }
[ -f "$META" ]      || { echo "[err] meta missing: $META"; exit 1; }
grep -q '"geometry_backbone": "vggt"'   "$BASE_CKPT/config.json" || { echo "[err] not VGGT"; exit 1; }
grep -q '"vggt_feature_layer": "multi"' "$BASE_CKPT/config.json" || { echo "[err] not multi-layer"; exit 1; }
grep -q '"num_geometry_tokens": 160'    "$BASE_CKPT/config.json" || { echo "[err] not K=160"; exit 1; }
grep -q '"vggt_input_height": 240'      "$BASE_CKPT/config.json" || { echo "[err] aspect record missing"; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
OUT=$RUN/runs/vggt_k160_multi_aspect_${TS}
LOG=$RUN/_logs/vggt_k160_multi_aspect_${TS}.log
mkdir -p "$OUT"
echo "$LOG" > /tmp/vggt_k160_multi_aspect_log_path
echo "[$(date '+%H:%M:%S')] launching VGGT-Omega K=160 multi+aspect (VLM 224×224 + VGGT 240×320) — log: $LOG"

cd "$SRC"
nohup env CUDA_VISIBLE_DEVICES=4,5,6,7 \
  XVLA_DDP_STATIC_GRAPH=1 XVLA_RGB_INPUT=1 \
  XVLA_DA3_INPUT_H=240 XVLA_DA3_INPUT_W=320 \
  XVLA_VLM_GRADIENT_CHECKPOINTING=0 \
  XVLA_NUM_WORKERS=16 XVLA_PREFETCH_FACTOR=4 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} \
  PYTHONPATH="$SRC" PYTHONUNBUFFERED=1 \
  envs/xvla/bin/accelerate launch \
    --num_processes 4 --num_machines 1 \
    --mixed_precision bf16 --dynamo_backend no \
    --main_process_port 47110 \
    train.py \
    --models "$BASE_CKPT" \
    --train_metas_path "$META" \
    --output_dir "$OUT" \
    --batch_size 32 \
    --learning_rate 1e-4 --learning_coef 0.1 \
    --weight_decay 0.0 --betas 0.9 0.95 --max_grad_norm 1.0 \
    --iters 100000 --freeze_steps 1000 --warmup_steps 2000 \
    --use_cosine_decay --min_lr_ratio 0.1 \
    --save_interval 10000 --log_interval 20 \
    --seed 0 --apply_finetune_policy \
    > "$LOG" 2>&1 &
disown $!
echo "[$(date '+%H:%M:%S')] VGGT-Omega K=160 multi+aspect launched on GPUs 4-7"
