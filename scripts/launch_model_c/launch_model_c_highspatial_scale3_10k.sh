#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SRC:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUN="${RUN:-$SRC/runs/spatial_lang_xvla}"
BASE="${BASE:-$RUN/runs/method_b_final6_spatialboost_scale1p5_geom1e4_core1e5_vlm0_30k_20260702_144730/ckpt-30000}"
META="${META:-/work/jack/_meta_all_ct_3cam_clean.json}"
ACCELERATE="${ACCELERATE:-envs/xvla/bin/accelerate}"

[ -d "$BASE" ] || { echo "[err] base checkpoint missing: $BASE"; exit 1; }
[ -f "$META" ] || { echo "[err] meta missing: $META"; exit 1; }

RUN_ID="method_c_highspatial_scale3_geom2e4_core5e6_vlm0_10k_$(date +%Y%m%d_%H%M%S)"
OUT="$RUN/runs/$RUN_ID"
mkdir -p "$OUT" "$RUN/_logs"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export XVLA_DDP_STATIC_GRAPH=0
export XVLA_DDP_FIND_UNUSED=1
export XVLA_RGB_INPUT=1
export XVLA_VLM_GRADIENT_CHECKPOINTING=0
export XVLA_GROUP_GNORM=1
export XVLA_NUM_WORKERS=8
export XVLA_PREFETCH_FACTOR=2
export XVLA_DA3_INPUT_H=252
export XVLA_DA3_INPUT_W=336
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-$RUN/hf_cache}"
export PYTHONPATH="$SRC:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

MAIN_PORT="${MAIN_PORT:-49231}"
cp -f "$0" "$OUT/launch.sh"

cat > "$OUT/RUN_PLAN.txt" <<EOF
Model C high-spatial contribution stress test.

Base checkpoint: $BASE
Base identity: Method B final-6 injection, C resume, fixed spatial scale 1.5.
Local training steps: 10000

Purpose:
  Test whether much larger spatial residual contribution improves over Model C.

Training settings:
  spatial_scale: fixed 3.0
  geometry LR: fixed 2e-4
  XVLA core LR: fixed 5e-6
  VLM LR: fixed 0
  DA3/T5 backbones: frozen by model/training code
  aux heads: disabled
  batch size: 32 per GPU
  GPUs: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

Evaluation notes:
  Evaluate checkpoints at 2.5k/5k/7.5k/10k if available.
  Also sweep eval-time spatial scale 1.5, 2.0, 2.5, 3.0 because training at 3.0 may not mean eval at 3.0 is optimal.
EOF

cd "$SRC"
"$ACCELERATE" launch \
  --num_processes 4 --num_machines 1 \
  --mixed_precision bf16 --dynamo_backend no \
  --main_process_port "$MAIN_PORT" \
  train.py \
  --models "$BASE" \
  --spatial_lang_method final6_injection \
  --spatial_scale_start 3.0 \
  --spatial_scale_end 3.0 \
  --spatial_scale_ramp_until 0 \
  --fixed_geometry_lr 2e-4 \
  --fixed_xvla_core_lr 5e-6 \
  --fixed_vlm_lr 0 \
  --train_metas_path "$META" \
  --output_dir "$OUT" \
  --batch_size 32 \
  --learning_rate 2e-4 --learning_coef 0.05 \
  --weight_decay 1e-8 \
  --betas 0.9 0.95 \
  --max_grad_norm 1.0 \
  --iters 10000 \
  --freeze_steps 0 --warmup_steps 0 \
  --save_interval 2500 --log_interval 20 \
  --seed 47 \
  2>&1 | tee "$OUT/train.log"
