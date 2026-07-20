#!/usr/bin/env bash
# MULTI-PROCESS inline pi0.5-DA3 on an ARBITRARY set of GPUs (not just 0..N-1).
# GPUS = comma list of physical GPU ids; each becomes one jax.distributed rank (1 GPU each).
set -uo pipefail

OPENPI=/work/jack/openpi_src/openpi
VENV=$OPENPI/.venv
CONFIG_NAME="${CONFIG_NAME:-pi05_roboreal_full_da3_inline_v2}"
GPUS="${GPUS:-4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-96}"          # GLOBAL batch (24/GPU x 4)
NUM_WORKERS="${NUM_WORKERS:-4}"         # per-rank dataloader workers; 4 halves our concurrent ffmpeg
                                        # swscale contexts vs 8 (less EAGAIN pressure on the shared node),
                                        # and we're compute-bound so fewer workers don't bottleneck.
COORD="${COORD:-localhost:12377}"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NPROC="${#GPU_ARR[@]}"

export HF_LEROBOT_HOME=/work/jack/DATASETS
export HF_HOME=/work/jack/da3xvla_workspace/hf_cache
export HF_HUB_OFFLINE=1
export PATH="/work/jack/openpi_src/.uv_bin:$PATH"
export WANDB_MODE=disabled WANDB_DISABLED=true
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Bound CPU intra-op threads per process. With N ranks x num_workers dataloader procs each,
# uncapped OMP/MKL/ffmpeg threads pile up (saw 82k threads) and starve the video swscaler into
# EAGAIN, which crashed a rank. Caps here + single-threaded ffmpeg decode (fast_video.py) fix it.
# 240-core box -> default thread pools are ~240/pool/proc; x8 procs booting together = thread
# explosion -> pthread_create EAGAIN. Cap hard (GPU-bound work, so CPU threads don't matter).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-4}"
export TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-4}"
unset PYTHONPATH

# EXP_NAME keys the checkpoint dir (checkpoints/<config>/<exp>). Override it to RESUME an existing
# run (same exp -> openpi restores its latest checkpoint). RESUME=1 passes --resume. LOG_SUFFIX keeps
# resumed-run logs from clobbering the original proc_*.log (which hold the loss history).
EXP_NAME="${EXP_NAME:-pi05_da3_inline_v2_mp_$(date +%Y%m%d_%H%M%S)}"
OUT=/work/jack/pi05_runs/$EXP_NAME
mkdir -p "$OUT"
LOGSFX="${LOG_SUFFIX:-}"
echo "[boot $(date -Iseconds)] MP inline v2: GPUS=$GPUS NPROC=$NPROC bs=$BATCH_SIZE config=$CONFIG_NAME exp=$EXP_NAME resume=${RESUME:-0} -> $OUT"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

cd "$OPENPI"
pids=()
for i in $(seq 0 $((NPROC-1))); do
    G="${GPU_ARR[$i]}"
    CUDA_VISIBLE_DEVICES="$G" OPENPI_NUM_PROCESSES="$NPROC" OPENPI_PROCESS_ID="$i" OPENPI_COORDINATOR="$COORD" \
        "$VENV/bin/python" scripts/train.py "$CONFIG_NAME" --exp-name "$EXP_NAME" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" ${NUM_TRAIN_STEPS:+--num-train-steps $NUM_TRAIN_STEPS} ${RESUME:+--resume} \
        > "$OUT/proc_${i}${LOGSFX}.log" 2>&1 &
    pids+=($!)
    echo "  launched rank $i on physical GPU $G (pid ${pids[-1]}) -> $OUT/proc_${i}${LOGSFX}.log"
    # Stagger starts so N processes don't hit orbax-restore + XLA-compile thread/fd/mem spikes
    # simultaneously at boot (that pthread_create EAGAIN'd rank 4 and cascade-killed an 8-proc resume).
    [ "$i" -lt "$((NPROC-1))" ] && sleep "${STAGGER:-0}"
done
echo "all pids: ${pids[*]}  (proc_0${LOGSFX}.log has the rate)  EXP_NAME=$EXP_NAME"
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[done $(date -Iseconds)] MP inline v2 exit rc=$rc: $OUT"
