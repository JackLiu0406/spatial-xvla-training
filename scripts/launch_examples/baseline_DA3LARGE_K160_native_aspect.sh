#!/usr/bin/env bash
# DA3-LARGE K=160, dual-resolution image pipeline:
#   • VLM (Florence-2): 224×224 (unchanged, required by X-VLA processor)
#   • DA3-LARGE encoder: 252×336 — aspect-preserved upscale of native 240×320
#                        252=18×14, 336=24×14 (DA3 patch-clean)
#                        Aspect 336/252 = 4/3 = exact match to native 320/240
# Intrinsics rescaled in the dataloader from native (240,320) → (252,336) so
# posed-DA3 math is correct (see datasets/domain_handler/base.py dual branch).
#
# Direct A/B with baseline_DA3LARGE_K160_native224 — only the DA3 input
# resolution + aspect preservation differ; recipe and architecture identical.
# Recipe: bs=32×4 H200, lr=1e-4, learning_coef=0.1, freeze=1000, warmup=2000,
#         cosine→0.1×, iters=100k, save_interval=10k.
set -eo pipefail

SRC=${REPO_ROOT}
RUN=${RUNS_ROOT}/baseline_DA3LARGE_K160_native_aspect
BASE_CKPT="$RUN/ckpt_init"
META=${RUNS_ROOT}/meta_all_ct_3cam_clean.json

[ -d "$BASE_CKPT" ] || { echo "[err] ckpt_init missing: $BASE_CKPT"; exit 1; }
[ -f "$META" ]      || { echo "[err] meta missing: $META"; exit 1; }
grep -q '"da3_model_name": "depth-anything/DA3-LARGE"' "$BASE_CKPT/config.json" || { echo "[err] not LARGE"; exit 1; }
grep -q '"num_geometry_tokens": 160'                   "$BASE_CKPT/config.json" || { echo "[err] not K=160"; exit 1; }
grep -q '"da3_input_height": 252'                      "$BASE_CKPT/config.json" || { echo "[err] aspect record missing"; exit 1; }
grep -q '"da3_process_res": 252'                       "$BASE_CKPT/config.json" || { echo "[err] process_res not updated"; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
OUT=$RUN/runs/da3large_k160_aspect_${TS}
LOG=$RUN/_logs/da3large_k160_aspect_${TS}.log
mkdir -p "$OUT"
echo "$LOG" > /tmp/da3large_k160_aspect_log_path
echo "[$(date '+%H:%M:%S')] launching DA3-LARGE K=160 aspect-preserved (VLM 224×224 + DA3-LARGE 252×336) — log: $LOG"

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
    --main_process_port 47100 \
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
echo "[$(date '+%H:%M:%S')] DA3-LARGE K=160 aspect-preserved launched on GPUs 4-7"
