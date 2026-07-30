#!/usr/bin/env bash
# Plain pi0 (no DA3) baseline on RoboPRO/roboreal, 4 GPUs. Full finetune from pi0_base.
# RGB is corrected via the SwapImageChannels transform baked into the pi0_roboreal_full config.
set -uo pipefail
export CONFIG_NAME=pi0_roboreal_full
export GPUS="${GPUS:-0,1,2,3}"
export BATCH_SIZE="${BATCH_SIZE:-96}"          # 24/GPU x 4
export NUM_WORKERS="${NUM_WORKERS:-8}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"  # no torch DA3 -> JAX can take more
export EXP_NAME="${EXP_NAME:-pi0_roboreal_baseline_$(date +%Y%m%d_%H%M%S)}"
exec bash /work/jack/openpi_src/launch_pi05_da3_inline_mp_gpus.sh
