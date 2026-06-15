#!/bin/bash
# ==============================================================================
# DA3-XVLA + RoboPRO — turnkey rebuild on a NEW (shared) server.
#
# Run this AFTER you have transferred the code tree to $WORK (see MIGRATION.md;
# you transfer code only — the conda envs are NOT copied, they are rebuilt
# here because conda --prefix envs + compiled curobo are not relocatable).
#
# Everything stays under $WORK (default /shared_work/jack) so it is safe on a
# shared server. No sudo. If no conda is found, a private Miniforge is
# installed under $WORK.
#
# Usage:
#   bash migration/setup_server.sh                # rebuild envs + patches
#   bash migration/setup_server.sh --with-assets  # also pull ~15GB RoboPRO assets
#
# Env overrides: WORK=/path  CONDA=/path/to/conda  TORCH_ARCH=12.0
# ==============================================================================
set -u
WORK="${WORK:-/shared_work/jack}"
WITH_ASSETS=0
[ "${1:-}" = "--with-assets" ] && WITH_ASSETS=1

XVLA="$WORK/DA3-XVLA"
DAVLA="$WORK/DA3-VLA"
RP="$DAVLA/RoboPRO"
RT="$RP/customized_robotwin"
XENV="$XVLA/envs/xvla"
RENV="$DAVLA/envs/robotwin"
CU128="https://download.pytorch.org/whl/cu128"
log(){ echo -e "\n=== $* ==="; }
die(){ echo "FATAL: $*" >&2; exit 1; }

[ -d "$XVLA/models" ]  || die "$XVLA not found — transfer the code tree first (MIGRATION.md)."
[ -d "$RT" ]           || die "$RT not found — transfer DA3-VLA/RoboPRO (with submodule) first."

# ---- 0. conda ---------------------------------------------------------------
log "0. locate conda"
CONDA="${CONDA:-}"
if [ -z "$CONDA" ]; then
  CONDA="$(command -v conda || true)"
fi
if [ -z "$CONDA" ] || ! "$CONDA" --version >/dev/null 2>&1; then
  MF="$DAVLA/miniforge"
  if [ ! -x "$MF/bin/conda" ]; then
    echo "no conda -> installing private Miniforge to $MF"
    curl -fsSL -o /tmp/mf.sh \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
      || die "miniforge download failed"
    bash /tmp/mf.sh -b -p "$MF" || die "miniforge install failed"
  fi
  CONDA="$MF/bin/conda"
fi
echo "conda = $CONDA ($("$CONDA" --version))"
# accept ToS for the default anaconda channels (clone/create needs it)
"$CONDA" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null
"$CONDA" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    2>/dev/null

# ---- 1. GPU arch ------------------------------------------------------------
log "1. detect GPU arch (for the curobo CUDA build)"
ARCH="${TORCH_ARCH:-}"
if [ -z "$ARCH" ]; then
  ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
fi
[ -n "$ARCH" ] || die "could not detect GPU compute capability; set TORCH_ARCH=XX.X"
echo "GPU compute capability = $ARCH  (RTX 5080 was 12.0)"
CUDA_HOME_GUESS="$(ls -d /usr/local/cuda-12.8 /usr/local/cuda 2>/dev/null | head -1)"
echo "CUDA toolkit = ${CUDA_HOME_GUESS:-<none found — curobo build needs nvcc 12.x>}"

# ---- 2. xvla env (DA3-XVLA model + DA3 extractor) ---------------------------
log "2. build xvla env  -> $XENV  (python 3.11, torch 2.7 cu128)"
if [ ! -x "$XENV/bin/python" ]; then
  "$CONDA" create -y -p "$XENV" python=3.11 || die "create xvla env failed"
fi
XPIP="$XENV/bin/pip"
"$XPIP" install --upgrade pip wheel >/dev/null
"$XPIP" install "torch==2.7.0" "torchvision==0.22.0" --index-url "$CU128" \
  || die "torch cu128 install (xvla) failed"
# X-VLA core deps (authoritative list lives in the repo)
"$XPIP" install -r "$XVLA/requirements.txt" || die "X-VLA requirements failed"
"$XPIP" install tensorboard
# Depth Anything 3 (editable, from the transferred clone) — pulls xformers etc.
"$XPIP" install -e "$XVLA/third_party/Depth-Anything-3" \
  || die "DA3 editable install failed"
"$XENV/bin/python" - <<'PY' || die "xvla sanity import failed"
import torch, transformers
from depth_anything_3.api import DepthAnything3
print("xvla OK: torch", torch.__version__, "cuda_avail", torch.cuda.is_available(),
      "| transformers", transformers.__version__, "| DA3 import OK")
PY

# ---- 3. robotwin env (RoboPRO sim + curobo) ---------------------------------
log "3. build robotwin env -> $RENV  (python 3.10, RoboTwin stack)"
if [ ! -x "$RENV/bin/python" ]; then
  "$CONDA" create -y -p "$RENV" python=3.10 || die "create robotwin env failed"
fi
RPIP="$RENV/bin/pip"
"$RPIP" install --upgrade pip wheel >/dev/null
# RoboTwin requirements, but force torch 2.7 cu128 (its pinned 2.4.1 has no
# Blackwell support; required for RTX 5080-class GPUs).
grep -viE '^torch(==|>=|$)|^torchvision' "$RT/script/requirements.txt" > /tmp/rt_req.txt
"$RPIP" install -r /tmp/rt_req.txt          || die "RoboTwin requirements failed"
"$RPIP" install "torch==2.7.0" "torchvision==0.22.0" --index-url "$CU128" \
  || die "torch cu128 install (robotwin) failed"
"$RPIP" install "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47" \
  || echo "WARN: pytorch3d build failed — retry manually if a baseline needs it"
"$CONDA" install -y -p "$RENV" -c conda-forge ffmpeg \
  || echo "WARN: conda ffmpeg failed; pkl->hdf5 video step needs ffmpeg on PATH"
# RoboTwin's own sapien/mplib source tweaks (from customized_robotwin/script/_install.sh)
SAPIEN_LOC="$("$RPIP" show sapien 2>/dev/null | awk '/Location/{print $2}')/sapien"
[ -f "$SAPIEN_LOC/wrapper/urdf_loader.py" ] && \
  sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$SAPIEN_LOC/wrapper/urdf_loader.py" || true
MPLIB_LOC="$("$RPIP" show mplib 2>/dev/null | awk '/Location/{print $2}')/mplib"
[ -f "$MPLIB_LOC/planner.py" ] && \
  sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$MPLIB_LOC/planner.py" || true

# curobo: build editable for THIS GPU arch (from the vendored v0.7.7 clone)
log "3b. build curobo (TORCH_CUDA_ARCH_LIST=${ARCH}+PTX) — slow, ~10-20 min"
( cd "$RT/envs/curobo" && \
  CUDA_HOME="${CUDA_HOME_GUESS:-/usr/local/cuda}" \
  PATH="${CUDA_HOME_GUESS:-/usr/local/cuda}/bin:$PATH" \
  TORCH_CUDA_ARCH_LIST="${ARCH}+PTX" \
  "$RENV/bin/python" -m pip install -e . --no-build-isolation ) \
  || die "curobo build failed (need nvcc 12.x; check CUDA toolkit)"
"$RENV/bin/python" -c "import sapien,mplib,curobo,torch;print('robotwin OK: sapien',sapien.__version__,'curobo',curobo.__version__,'torch',torch.__version__,'cuda',torch.cuda.is_available())" \
  || die "robotwin sanity import failed"

# ---- 4. assets (optional, ~15GB) -------------------------------------------
if [ "$WITH_ASSETS" = 1 ]; then
  log "4. download RoboPRO assets (~15GB)"
  ( cd "$RP" && "$RENV/bin/python" scripts/install/download_assets.py ) \
    || die "asset download failed"
else
  log "4. SKIPPED asset download (re-run with --with-assets, or:"
  echo "   cd $RP && $RENV/bin/python scripts/install/download_assets.py )"
fi

# ---- 5. embodiment config paths + symlink + patches -------------------------
log "5. configure embodiment paths, symlink, apply patches"
ln -sfn ../benchmark/assets "$RT/assets"
if [ -d "$RP/benchmark/assets/embodiments" ]; then
  ( cd "$RP/benchmark" && \
    "$RENV/bin/python" "$RT/script/update_embodiment_config_path.py" < /dev/null ) || true
fi
WORK="$WORK" bash "$XVLA/migration/apply_patches.sh"

# ---- 6. final verification --------------------------------------------------
log "6. verify DA3-XVLA pipeline (CPU, no GPU needed)"
( cd "$XVLA" && CUDA_VISIBLE_DEVICES="" "$XENV/bin/python" scripts/debug_geometry_conditioning.py 2>&1 | tail -1 )
( cd "$XVLA" && CUDA_VISIBLE_DEVICES="" "$XENV/bin/python" tests/test_geometry_conditioning.py 2>&1 | tail -1 )
( cd "$XVLA" && CUDA_VISIBLE_DEVICES="" "$XENV/bin/python" tests/test_da3_pipeline.py 2>&1 | tail -1 )

cat <<EOF

============================================================
SETUP COMPLETE
  xvla env     : $XENV/bin/python
  robotwin env : $RENV/bin/python
  assets       : $([ "$WITH_ASSETS" = 1 ] && echo downloaded || echo "NOT downloaded (see step 4)")
Next: verify a RoboPRO curobo sphere check (needs assets):
  cd $RT && source set_env.sh && export ROBOTWIN_BENCH_TASK=bench
  (see MIGRATION.md "Post-setup verification")
============================================================
EOF
