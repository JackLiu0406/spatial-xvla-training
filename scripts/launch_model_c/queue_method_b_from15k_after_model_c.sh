#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SRC:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUN="${RUN:-$SRC/runs/spatial_lang_xvla}"
LAUNCH="${LAUNCH:-$SCRIPT_DIR/launch_method_b_from15k_highspatial_to125k.sh}"
LOG=$RUN/_logs/queue_method_b_early20k_highspatial_after_method_c_$(date +%Y%m%d_%H%M%S).log

mkdir -p "$RUN/_logs"
echo "[queue] waiting for Model C high-spatial test to finish" | tee -a "$LOG"
echo "[queue] launch: $LAUNCH" | tee -a "$LOG"

while true; do
  HIGHSPATIAL_RUN="$(find "$RUN/runs" -maxdepth 1 -type d -name 'method_c_highspatial_scale3_geom2e4_core5e6_vlm0_10k_*' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)"
  if [ -n "$HIGHSPATIAL_RUN" ] && [ -d "$HIGHSPATIAL_RUN/ckpt-10000" ]; then
    if ! pgrep -af "train.py .*method_c_highspatial_scale3_geom2e4_core5e6_vlm0_10k_" >/dev/null; then
      break
    fi
  fi
  sleep 60
done

echo "[queue] Model C high-spatial complete; launching Method B ckpt-15k high-spatial-to-125k test" | tee -a "$LOG"
MAIN_PORT=49243 bash "$LAUNCH" >> "$LOG" 2>&1
