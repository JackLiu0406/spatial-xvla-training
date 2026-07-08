#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SRC:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUN="${RUN:-$SRC/runs/spatial_lang_xvla}"
CURRENT_RUN="${CURRENT_RUN:-$RUN/runs/method_b_from20k_auxheads_spatialboost_s0p35to2p0_geom2e4_core1e4_vlm1e5_20260702_213922}"
LAUNCH="${LAUNCH:-$SCRIPT_DIR/launch_model_c_highspatial_scale3_10k.sh}"
LOG=$RUN/_logs/queue_method_c_highspatial_after_aux_$(date +%Y%m%d_%H%M%S).log

mkdir -p "$RUN/_logs"
echo "[queue] waiting for aux-head Method B run to finish" | tee -a "$LOG"
echo "[queue] current: $CURRENT_RUN" | tee -a "$LOG"
echo "[queue] launch:  $LAUNCH" | tee -a "$LOG"

while true; do
  if [ -d "$CURRENT_RUN/ckpt-105000" ]; then
    if ! pgrep -af "train.py .*method_b_from20k_auxheads_spatialboost_s0p35to2p0_geom2e4_core1e4_vlm1e5_20260702_213922" >/dev/null; then
      break
    fi
  fi
  sleep 60
done

echo "[queue] aux-head run complete; launching Model C high-spatial test" | tee -a "$LOG"
MAIN_PORT=49231 bash "$LAUNCH" >> "$LOG" 2>&1
