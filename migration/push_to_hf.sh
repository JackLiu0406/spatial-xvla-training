#!/bin/bash
# ==============================================================================
# Pack the DA3-XVLA + RoboPRO code tree into one tarball and push it to a
# (private) HuggingFace dataset repo. Transport for server->server migration
# when rsync/SSH between hosts is not available.
#
# Excludes the non-portable / re-derivable bulk: conda envs, RoboPRO assets,
# throwaway data, __pycache__, nested .git. (~300-450 MB tarball.)
#
# Usage:
#   REPO=<user>/da3xvla-migrate  bash migration/push_to_hf.sh
# Env:
#   REPO      (required) HF repo id, e.g. jackq/da3xvla-migrate
#   WORK      project root to pack         (default /shared_work/jack)
#   HF_TOKEN  HF write token               (else uses a prior huggingface-cli login)
#   TARNAME   artifact name in the repo    (default da3xvla_migrate.tar.gz)
#   REPO_TYPE dataset|model                (default dataset)
# ==============================================================================
set -u
WORK="${WORK:-/shared_work/jack}"
REPO="${REPO:-}"
TARNAME="${TARNAME:-da3xvla_migrate.tar.gz}"
REPO_TYPE="${REPO_TYPE:-dataset}"
TARPATH="/tmp/$TARNAME"
[ -n "$REPO" ] || { echo "FATAL: set REPO=<user>/<repo-name>"; exit 1; }
[ -d "$WORK/DA3-XVLA" ] || { echo "FATAL: $WORK/DA3-XVLA not found"; exit 1; }

# Pick a python with huggingface_hub (the xvla env has it).
HFPY=""
for p in "$WORK/DA3-XVLA/envs/xvla/bin/python" "$(command -v python3)" "$(command -v python)"; do
  [ -n "$p" ] && "$p" -c "import huggingface_hub" 2>/dev/null && { HFPY="$p"; break; }
done
if [ -z "$HFPY" ]; then
  HFPY="$(command -v python3 || command -v python)"
  echo "[push] installing huggingface_hub for $HFPY"
  "$HFPY" -m pip install -q huggingface_hub || { echo "FATAL: cannot get huggingface_hub"; exit 1; }
fi
TOKEN_ARG=""
[ -n "${HF_TOKEN:-}" ] && TOKEN_ARG="$HF_TOKEN"

echo "[push] packing $WORK/{DA3-XVLA,DA3-VLA} -> $TARPATH"
( cd "$WORK" && tar \
    --exclude='*/envs' \
    --exclude='*/__pycache__' \
    --exclude='.git' \
    --exclude='DA3-VLA/RoboPRO/benchmark/assets' \
    --exclude='DA3-VLA/RoboPRO/customized_robotwin/data' \
    -czf "$TARPATH" DA3-XVLA DA3-VLA ) || { echo "FATAL: tar failed"; exit 1; }
echo "[push] tarball size: $(du -h "$TARPATH" | cut -f1)"

echo "[push] ensuring repo $REPO ($REPO_TYPE, private) exists + uploading"
"$HFPY" - "$REPO" "$REPO_TYPE" "$TARPATH" "$TARNAME" "$TOKEN_ARG" <<'PY'
import sys
from huggingface_hub import create_repo, upload_file
repo, rtype, tarpath, tarname, token = sys.argv[1:6]
token = token or None
create_repo(repo, repo_type=rtype, private=True, exist_ok=True, token=token)
url = upload_file(path_or_fileobj=tarpath, path_in_repo=tarname,
                  repo_id=repo, repo_type=rtype, token=token)
print("[push] uploaded:", url)
PY
rc=$?
rm -f "$TARPATH"
[ $rc -eq 0 ] || { echo "FATAL: upload failed (auth? run 'huggingface-cli login' or set HF_TOKEN)"; exit 1; }

cat <<EOF

[push] DONE.
On the NEW server:
  REPO=$REPO WORK=<your-root> bash pull_from_hf.sh
(get pull_from_hf.sh from this same migration/ dir, or it is inside the tarball
 at DA3-XVLA/migration/pull_from_hf.sh)
EOF
