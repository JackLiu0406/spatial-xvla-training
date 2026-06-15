#!/bin/bash
# ==============================================================================
# Pull the migration tarball from a HuggingFace repo and unpack it under your
# writable root on the NEW server. Self-bootstraps huggingface_hub (no env
# needed yet). Then you run setup_server.sh (or pass --setup to chain it).
#
# Usage:
#   REPO=<user>/da3xvla-migrate WORK=/your/root bash pull_from_hf.sh
#   REPO=... WORK=... bash pull_from_hf.sh --setup        # also rebuild envs+assets
# Env:
#   REPO      (required) HF repo id used by push_to_hf.sh
#   WORK      destination root            (default /shared_work/jack)
#   HF_TOKEN  HF read token               (else a prior huggingface-cli login)
#   TARNAME   artifact name               (default da3xvla_migrate.tar.gz)
#   REPO_TYPE dataset|model               (default dataset)
# ==============================================================================
set -u
WORK="${WORK:-/shared_work/jack}"
REPO="${REPO:-}"
TARNAME="${TARNAME:-da3xvla_migrate.tar.gz}"
REPO_TYPE="${REPO_TYPE:-dataset}"
DO_SETUP=0
[ "${1:-}" = "--setup" ] && DO_SETUP=1
[ -n "$REPO" ] || { echo "FATAL: set REPO=<user>/<repo-name>"; exit 1; }
mkdir -p "$WORK" || { echo "FATAL: cannot create $WORK"; exit 1; }

HFPY="$(command -v python3 || command -v python)"
[ -n "$HFPY" ] || { echo "FATAL: no python3 on PATH"; exit 1; }
if ! "$HFPY" -c "import huggingface_hub" 2>/dev/null; then
  echo "[pull] installing huggingface_hub for $HFPY"
  "$HFPY" -m pip install -q --user huggingface_hub \
    || "$HFPY" -m pip install -q huggingface_hub \
    || { echo "FATAL: cannot install huggingface_hub"; exit 1; }
fi
TOKEN_ARG=""
[ -n "${HF_TOKEN:-}" ] && TOKEN_ARG="$HF_TOKEN"

echo "[pull] downloading $TARNAME from $REPO ($REPO_TYPE)"
TARPATH="$("$HFPY" - "$REPO" "$REPO_TYPE" "$TARNAME" "$TOKEN_ARG" <<'PY'
import sys
from huggingface_hub import hf_hub_download
repo, rtype, tarname, token = sys.argv[1:5]
print(hf_hub_download(repo_id=repo, filename=tarname, repo_type=rtype,
                      token=token or None))
PY
)"
[ -n "$TARPATH" ] && [ -f "$TARPATH" ] || { echo "FATAL: download failed (auth? set HF_TOKEN / huggingface-cli login)"; exit 1; }
echo "[pull] got $TARPATH ($(du -h "$TARPATH" | cut -f1))"

echo "[pull] extracting into $WORK"
tar -xzf "$TARPATH" -C "$WORK" || { echo "FATAL: extract failed"; exit 1; }
[ -d "$WORK/DA3-XVLA/migration" ] || { echo "FATAL: tree not where expected under $WORK"; exit 1; }
echo "[pull] code tree restored under $WORK"

if [ "$DO_SETUP" = 1 ]; then
  echo "[pull] chaining setup_server.sh --with-assets"
  WORK="$WORK" bash "$WORK/DA3-XVLA/migration/setup_server.sh" --with-assets
else
  cat <<EOF

[pull] DONE (code only). Next, rebuild envs on this server:
  cd $WORK/DA3-XVLA
  WORK=$WORK bash migration/setup_server.sh --with-assets
(or re-run this with --setup to do it now)
EOF
fi
