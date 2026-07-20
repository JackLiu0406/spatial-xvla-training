#!/usr/bin/env bash
# MULTI-PROCESS inline pi0.5-DA3 (DDP-style): NPROC processes, ONE GPU each, tied together by
# jax.distributed. Each process runs its OWN torch DA3-GIANT on its OWN shard/GPU -> real per-GPU
# parallelism (separate GILs), no central 4GB feature gather. Mirrors how X-VLA (all-torch DDP) works.
set -uo pipefail

OPENPI=/work/jack/openpi_src/openpi
VENV=$OPENPI/.venv
CONFIG_NAME="${CONFIG_NAME:-pi05_roboreal_full_da3_inline}"
NPROC="${NPROC:-8}"
COORD="${COORD:-localhost:12355}"

export HF_LEROBOT_HOME=/work/jack/DATASETS
export HF_HOME=/work/jack/da3xvla_workspace/hf_cache
export HF_HUB_OFFLINE=1
export PATH="/work/jack/openpi_src/.uv_bin:$PATH"
export WANDB_MODE=disabled WANDB_DISABLED=true
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}"  # per-GPU JAX pool; leaves room for torch DA3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH

RUN_ID="pi05_da3_inline_mp_$(date +%Y%m%d_%H%M%S)"
OUT=/work/jack/pi05_runs/$RUN_ID
mkdir -p "$OUT"
echo "[boot $(date -Iseconds)] MP inline: NPROC=$NPROC (1 GPU each), config=$CONFIG_NAME -> $OUT"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -"$NPROC"

cd "$OPENPI"
pids=()
for i in $(seq 0 $((NPROC-1))); do
    CUDA_VISIBLE_DEVICES="$i" OPENPI_NUM_PROCESSES="$NPROC" OPENPI_PROCESS_ID="$i" OPENPI_COORDINATOR="$COORD" \
        "$VENV/bin/python" scripts/train.py "$CONFIG_NAME" --exp-name "$RUN_ID" \
        > "$OUT/proc_$i.log" 2>&1 &
    pids+=($!)
    echo "  launched proc $i on GPU $i (pid ${pids[-1]}) -> $OUT/proc_$i.log"
done
echo "all pids: ${pids[*]}  (proc_0.log has the rate)"
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[done $(date -Iseconds)] MP inline exit rc=$rc: $OUT"
