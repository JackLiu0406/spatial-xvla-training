#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SRC:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUN="${RUN:-$SRC/runs/spatial_lang_xvla}"
BASE="${BASE:-$RUN/runs/method_b_final6_injection_4gpu_20260630_173540/ckpt-15000}"
META="${META:-/work/jack/_meta_all_ct_3cam_clean.json}"
ACCELERATE="${ACCELERATE:-envs/xvla/bin/accelerate}"

[ -d "$BASE" ] || { echo "[err] base checkpoint missing: $BASE"; exit 1; }
[ -f "$META" ] || { echo "[err] meta missing: $META"; exit 1; }

RUN_ID="method_b_from15k_highspatial_s1to3_core1e4to1e5_geom2e4_vlm0_to125k_$(date +%Y%m%d_%H%M%S)"
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

MAIN_PORT="${MAIN_PORT:-49243}"
cp -f "$0" "$OUT/launch.sh"

cat > "$OUT/RUN_PLAN.txt" <<EOF
Method B early high-spatial continuation from cumulative ckpt 15k.

Base checkpoint: $BASE
Base identity: original Method B final-6 injection at cumulative 15k, before XVLA has largely converged.
Local training steps: 110000
Final cumulative reference: 125000 Method B steps

Purpose:
  Push spatial contribution earlier while the XVLA/action path is still plastic.
  This replaces the previous queued early-20k / 20k-local high-spatial probe.

Training settings:
  spatial_scale: 1.0 -> 3.0 over first 15000 local steps
                 equivalent to cumulative 15k -> 30k
  geometry LR: fixed 2e-4
  XVLA core LR: 1e-4 until local step 15000, then cosine to 1e-5 by local step 110000
  VLM LR: fixed 0
  DA3/T5 backbones: frozen by model/training code
  aux heads: disabled
  batch size: 32 per GPU
  GPUs: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

Evaluation notes:
  Evaluate checkpoints every 5000 steps.
  Sweep eval-time spatial scale 1.5, 2.0, 2.5, 3.0, and 3.5.
EOF

cd "$SRC"
"$ACCELERATE" launch \
  --num_processes 4 --num_machines 1 \
  --mixed_precision bf16 --dynamo_backend no \
  --main_process_port "$MAIN_PORT" \
  train.py \
  --models "$BASE" \
  --spatial_lang_method final6_injection \
  --spatial_scale_start 1.0 \
  --spatial_scale_end 3.0 \
  --spatial_scale_ramp_until 15000 \
  --fixed_geometry_lr 2e-4 \
  --xvla_core_cosine_start_lr 1e-4 \
  --xvla_core_cosine_min_lr 1e-5 \
  --xvla_core_cosine_start_step 15000 \
  --xvla_core_cosine_end_step 110000 \
  --fixed_vlm_lr 0 \
  --train_metas_path "$META" \
  --output_dir "$OUT" \
  --batch_size 32 \
  --learning_rate 2e-4 --learning_coef 0.05 \
  --weight_decay 1e-8 \
  --betas 0.9 0.95 \
  --max_grad_norm 1.0 \
  --iters 110000 \
  --freeze_steps 0 --warmup_steps 0 \
  --save_interval 5000 --log_interval 20 \
  --seed 49 \
  2>&1 | tee "$OUT/train.log"
