#!/usr/bin/env bash
#SBATCH --job-name=blnLrg
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --nodelist=gpu-h200-103
# ↑ pinned: only gpu-h200-103 has /work/jack and /opt/da3xvla mounted; -102 / -105 don't.
#SBATCH --cpus-per-task=112
#SBATCH --mem=900G
#SBATCH --time=72:00:00
#SBATCH --output=${RUNS_ROOT}/baseline_DA3LARGE_K160/_logs/blnLrg-%j.out
#SBATCH --error=${RUNS_ROOT}/baseline_DA3LARGE_K160/_logs/blnLrg-%j.err
#
# Ablation cell: baseline recipe with ONLY DA3 model size swapped (BASE → LARGE).
# Everything else mirrors the user's baseline that was trained on 8×H100 bs=16:
#   - K=160, 504 upscaled, last-layer single-tap, posed multi-view, RGB input
#   - lr=1e-4, learning_coef=0.1, freeze_steps=1000, warmup_steps=2000
#   - iters=100000, cosine decay to 10%, apply_finetune_policy, seed=0
# Scaling: 4×H200 bs=32 → global batch 128, identical to 8×H100 bs=16.
#
# Base ckpt: X-VLA-Pt (no geometry weights) with config.json patched to enable
# geometry conditioning with DA3-LARGE + K=160 + 504 upscaled + last-layer.
# Geometry modules init fresh (HF re-init); DA3-LARGE backbone reloaded by
# da3_inline.reload_pretrained_weights() right after XVLA.from_pretrained.
set -eo pipefail

RUN=${RUNS_ROOT}/baseline_DA3LARGE_K160
SRC=${REPO_ROOT}
BASE_CKPT="$RUN/ckpt_init"
META=/opt/da3xvla/v1_smoke/meta_all_ct_3cam.json   # read-only; Roboreal data resolves under /shared_work/DATASETS/Roboreal/raw
[ -d "$BASE_CKPT" ]                                || { echo "[err] ckpt missing: $BASE_CKPT"; exit 1; }
[ -f "$META" ]                                     || { echo "[err] meta missing: $META"; exit 1; }
grep -q '"da3_model_name": "depth-anything/DA3-LARGE"' "$BASE_CKPT/config.json" || { echo "[err] config not DA3-LARGE"; exit 1; }
grep -q '"num_geometry_tokens": 160'               "$BASE_CKPT/config.json" || { echo "[err] config not K=160"; exit 1; }
grep -q '"da3_feature_layer": "last"'              "$BASE_CKPT/config.json" || { echo "[err] config not single-layer"; exit 1; }
grep -q '"da3_process_res": 504'                   "$BASE_CKPT/config.json" || { echo "[err] config not 504 upscaled"; exit 1; }

RUN_ID="blnLrg_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
OUT="$RUN/runs/$RUN_ID"; mkdir -p "$OUT"
echo "[info] SRC=$SRC  BASE_CKPT=$BASE_CKPT  META=$META  OUT=$OUT"
echo "[gpu] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader 2>/dev/null || true

MAIN_PORT=$((46000 + SLURM_JOB_ID % 5000))
# Recipe-canonical env flags. NB: XVLA_DA3_NATIVE_INPUT is intentionally OFF
# (baseline uses 504 upscaled, not 224 native — that's a separate ablation cell).
export XVLA_DDP_STATIC_GRAPH=1 XVLA_POSED_DA3=1 XVLA_RGB_INPUT=1
export XVLA_VLM_GRADIENT_CHECKPOINTING=0
export XVLA_NUM_WORKERS=16 XVLA_PREFETCH_FACTOR=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# DA3-LARGE weights are cached at ${HF_HOME:-$HOME/.cache/huggingface} (verified earlier).
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTHONPATH="$SRC:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

cd "$SRC"
envs/xvla/bin/accelerate launch \
  --num_processes 4 --num_machines 1 \
  --mixed_precision bf16 --dynamo_backend no \
  --main_process_port "$MAIN_PORT" \
  train.py \
  --models "$BASE_CKPT" \
  --train_metas_path "$META" \
  --output_dir "$OUT" \
  --batch_size 32 \
  --learning_rate 1e-4 --learning_coef 0.1 \
  --weight_decay 0.0 \
  --betas 0.9 0.95 \
  --max_grad_norm 1.0 \
  --iters 100000 \
  --freeze_steps 1000 --warmup_steps 2000 \
  --use_cosine_decay --min_lr_ratio 0.1 \
  --save_interval 10000 --log_interval 20 \
  --seed 0 --apply_finetune_policy \
  2>&1 | tee "$OUT/train.log"
echo "[done] $OUT (baseline recipe + DA3-LARGE swap, 4×H200 bs=32, 100k iters)"
