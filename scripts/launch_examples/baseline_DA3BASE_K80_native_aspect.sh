#!/usr/bin/env bash
# DA3-BASE K=80, dual-resolution image pipeline:
#   • VLM (Florence-2): 224×224 (unchanged, required by X-VLA processor)
#   • DA3 encoder:      252×336 — aspect-preserved upscale of native 240×320
#                       Both dims are DA3 patch-clean (252=18×14, 336=24×14)
#                       Aspect 336/252 = 4/3 = exact match to native 320/240
# Intrinsics in the HDF5 are for native 240×320; the dataloader now rescales
# them to (252, 336) before they reach da3_inline so posed-DA3 math is correct
# (see datasets/domain_handler/base.py — dual-resolution branch).
#
# Recipe matches the anchor (same as baseline_DA3BASE_K80_native224):
#   bs=32×4 H200, lr=1e-4, learning_coef=0.1, freeze=1000, warmup=2000,
#   cosine→0.1×, iters=100k, save_interval=10k.
# Only knob changed vs baseline_DA3BASE_K80_native224:
#   XVLA_DA3_INPUT_H=252 XVLA_DA3_INPUT_W=336 (dual-image dataloader on)
set -eo pipefail

SRC=${REPO_ROOT}
RUN=${RUNS_ROOT}/baseline_DA3BASE_K80_native_aspect
BASE_CKPT="$RUN/ckpt_init"
META=${RUNS_ROOT}/meta_all_ct_3cam_clean.json

[ -d "$BASE_CKPT" ] || { echo "[err] ckpt_init missing: $BASE_CKPT"; exit 1; }
[ -f "$META" ]      || { echo "[err] meta missing: $META"; exit 1; }
grep -q '"da3_model_name": "depth-anything/DA3-BASE"' "$BASE_CKPT/config.json" || { echo "[err] not BASE"; exit 1; }
grep -q '"num_geometry_tokens": 80'                   "$BASE_CKPT/config.json" || { echo "[err] not K=80"; exit 1; }
grep -q '"da3_input_height": 252'                     "$BASE_CKPT/config.json" || { echo "[err] aspect-preserved record missing"; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
OUT=$RUN/runs/da3base_k80_aspect_${TS}
LOG=$RUN/_logs/da3base_k80_aspect_${TS}.log
mkdir -p "$OUT"
echo "$LOG" > /tmp/da3base_k80_aspect_log_path
echo "[$(date '+%H:%M:%S')] launching DA3-BASE K=80 aspect-preserved (VLM 224×224 + DA3 252×336) — log: $LOG"

cd "$SRC"
nohup env CUDA_VISIBLE_DEVICES=4,5,6,7 \
  XVLA_DDP_STATIC_GRAPH=1 XVLA_POSED_DA3=1 XVLA_RGB_INPUT=1 \
  XVLA_DA3_NATIVE_INPUT=1 \
  XVLA_DA3_INPUT_H=252 XVLA_DA3_INPUT_W=336 \
  XVLA_VLM_GRADIENT_CHECKPOINTING=0 \
  XVLA_NUM_WORKERS=16 XVLA_PREFETCH_FACTOR=4 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} \
  PYTHONPATH="$SRC" PYTHONUNBUFFERED=1 \
  envs/xvla/bin/accelerate launch \
    --num_processes 4 --num_machines 1 \
    --mixed_precision bf16 --dynamo_backend no \
    --main_process_port 47095 \
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
echo "[$(date '+%H:%M:%S')] DA3-BASE K=80 aspect-preserved launched on GPUs 4-7"
