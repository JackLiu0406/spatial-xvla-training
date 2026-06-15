#!/usr/bin/env bash
# DA3-BASE K=160, dpt_fused tap, aspect-preserved 252×336.
#
# This is the architectural change we discussed: tap DA3's own pretrained DPT
# head right before the depth-output Conv (i.e. the input to head.scratch.output_conv2,
# captured via a forward hook in da3_inline.py). That latent is the OUTPUT of
# DA3's full DPT pyramid — multi-layer fused via the pretrained refinenets,
# spatially recovered via cascaded upsampling, refined with ResidualConvUnits
# (ReLU + 3×3 convs). Channel dim = features//2 = 64 for DA3-BASE.
#
# Why this should be strictly better than our hand-rolled multi-layer adapter:
#   - DA3's DPT pyramid is PRETRAINED (no cold-start fusion)
#   - Has non-linearity + residual structure (vs our linear LN+Conv adapter)
#   - Spatially aware (cascaded upsampling), not just channel-mixing
#   - ZERO new trainable params (head is frozen with the backbone)
#
# Knobs changed vs baseline_DA3BASE_K160_native_aspect:
#   da3_feature_layer: "last"  -> "dpt_fused"
#   da3_input_dim:      768    -> 64       (DA3-BASE features // 2)
#
# Recipe identical to the K=160 aspect anchors:
#   bs=32×4 H200, lr=1e-4, learning_coef=0.1, freeze=1000, warmup=2000,
#   cosine→0.1×, iters=100k, save_interval=10k.
#
# VLM (Florence-2) still sees 224×224 from the dual-image dataloader.
set -eo pipefail

SRC=${REPO_ROOT}
RUN=${RUNS_ROOT}/baseline_DA3BASE_K160_dptfused_aspect
BASE_CKPT="$RUN/ckpt_init"
META=${RUNS_ROOT}/meta_all_ct_3cam_clean.json

[ -d "$BASE_CKPT" ] || { echo "[err] ckpt_init missing: $BASE_CKPT"; exit 1; }
[ -f "$META" ]      || { echo "[err] meta missing: $META"; exit 1; }
grep -q '"da3_model_name": "depth-anything/DA3-BASE"' "$BASE_CKPT/config.json" || { echo "[err] not BASE"; exit 1; }
grep -q '"da3_feature_layer": "dpt_fused"'           "$BASE_CKPT/config.json" || { echo "[err] not dpt_fused"; exit 1; }
grep -q '"da3_input_dim": 64'                        "$BASE_CKPT/config.json" || { echo "[err] da3_input_dim must be 64 for BASE dpt_fused"; exit 1; }
grep -q '"num_geometry_tokens": 160'                 "$BASE_CKPT/config.json" || { echo "[err] not K=160"; exit 1; }
grep -q '"da3_input_height": 252'                    "$BASE_CKPT/config.json" || { echo "[err] aspect record missing"; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
OUT=$RUN/runs/da3base_k160_dptfused_aspect_${TS}
LOG=$RUN/_logs/da3base_k160_dptfused_aspect_${TS}.log
mkdir -p "$OUT"
echo "$LOG" > /tmp/da3base_k160_dptfused_aspect_log_path
echo "[$(date '+%H:%M:%S')] launching DA3-BASE K=160 dpt_fused aspect (252×336) — log: $LOG"

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
    --main_process_port 47115 \
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
echo "[$(date '+%H:%M:%S')] DA3-BASE K=160 dpt_fused aspect launched on GPUs 4-7"
