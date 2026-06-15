#!/usr/bin/env bash
# DA3-BASE K=80 native 224. Fresh 100k-iter training on 4 GPUs.
# Same recipe as the current DA3-LARGE K=160 anchor. Only knobs changed:
#   - da3_model_name: DA3-LARGE -> DA3-BASE
#   - da3_input_dim:  1024 -> 768
#   - num_geometry_tokens: 160 -> 80
set -eo pipefail

SRC=${REPO_ROOT}
RUN=${RUNS_ROOT}/baseline_DA3BASE_K80_native224
BASE_CKPT="$RUN/ckpt_init"
META=${RUNS_ROOT}/meta_all_ct_3cam_clean.json

[ -d "$BASE_CKPT" ] || { echo "[err] ckpt_init missing: $BASE_CKPT"; exit 1; }
[ -f "$META" ]      || { echo "[err] meta missing: $META"; exit 1; }
grep -q '"da3_model_name": "depth-anything/DA3-BASE"' "$BASE_CKPT/config.json" || { echo "[err] not BASE"; exit 1; }
grep -q '"num_geometry_tokens": 80'                   "$BASE_CKPT/config.json" || { echo "[err] not K=80"; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
OUT=$RUN/runs/da3base_k80_${TS}
LOG=$RUN/_logs/da3base_k80_${TS}.log
mkdir -p "$OUT"
echo "$LOG" > /tmp/da3base_k80_log_path
echo "[$(date '+%H:%M:%S')] launching DA3-BASE K=80 native 224 — log: $LOG"

cd "$SRC"
nohup env CUDA_VISIBLE_DEVICES=4,5,6,7 \
  XVLA_DDP_STATIC_GRAPH=1 XVLA_POSED_DA3=1 XVLA_RGB_INPUT=1 \
  XVLA_DA3_NATIVE_INPUT=1 \
  XVLA_VLM_GRADIENT_CHECKPOINTING=0 \
  XVLA_NUM_WORKERS=16 XVLA_PREFETCH_FACTOR=4 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} \
  PYTHONPATH="$SRC" PYTHONUNBUFFERED=1 \
  envs/xvla/bin/accelerate launch \
    --num_processes 4 --num_machines 1 \
    --mixed_precision bf16 --dynamo_backend no \
    --main_process_port 47090 \
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
echo "[$(date '+%H:%M:%S')] DA3-BASE K=80 launched"
