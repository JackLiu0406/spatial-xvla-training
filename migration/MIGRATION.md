# Moving DA3-XVLA + RoboPRO to another server

Everything lives under one root (here `/shared_work/jack`). The new server is
shared and you can only write to your own root there — this plan keeps
**100% of it inside that root**: private conda envs (`--prefix`), a private
Miniforge if no conda exists, no `sudo`, no system writes.

**Golden rule:** copy the *code*, **rebuild the envs**. Conda `--prefix` envs
bake absolute paths into shebangs/`.pth`/compiled `.so`, and curobo is a CUDA
extension compiled for a specific GPU arch + path. Copying `envs/` to another
host or path **will not work**. `setup_server.sh` rebuilds them.

Replace `NEWROOT` below with your writable root on the new server (can be the
same `/shared_work/jack`).

---

## 1. What to transfer (and what NOT to)

| Path | Size | Transfer? |
|---|---|---|
| `DA3-XVLA/` **excluding** `envs/` | ~146 MB | ✅ yes (our model, scripts, patches, migration/) |
| `DA3-XVLA/third_party/Depth-Anything-3` | ~48 MB | ✅ yes (pinned DA3 clone) |
| `DA3-VLA/RoboPRO/` **excluding** `envs/`, `benchmark/assets/`, `customized_robotwin/data/` | ~150 MB of code (the 4.2 GB figure is mostly `data/` + assets — exclude them) | ✅ yes (repo + **our patches**) |
| `DA3-VLA/RoboPRO/customized_robotwin/envs/curobo` | ~133 MB | ✅ yes (vendored v0.7.7 src + patch; ext rebuilt) |
| `DA3-XVLA/envs/` (9.7 GB) | — | ❌ **rebuild** |
| `DA3-VLA/envs/` (16 GB) | — | ❌ **rebuild** |
| `DA3-VLA/RoboPRO/benchmark/assets/` (15 GB) | — | ⚠️ re-download on new server (`download_assets.py`); copy only if bandwidth-bound |
| `~/.cache/huggingface` | — | ❌ outside the root; models re-download on first use |
| `DA3-VLA/RoboPRO/customized_robotwin/data/` (smoke demos) | — | ❌ throwaway test data |

### Transfer commands (run on the OLD server)

```bash
# 1. DA3-XVLA (code + patches + migration kit + DA3 clone), no envs
rsync -aP --exclude 'envs/' --exclude '__pycache__/' \
  /shared_work/jack/DA3-XVLA/  USER@NEWHOST:NEWROOT/DA3-XVLA/

# 2. RoboPRO repo WITH our patches, but no envs / assets / throwaway data
rsync -aP \
  --exclude 'envs/' \
  --exclude 'RoboPRO/benchmark/assets/' \
  --exclude 'RoboPRO/customized_robotwin/data/' \
  --exclude '__pycache__/' \
  /shared_work/jack/DA3-VLA/  USER@NEWHOST:NEWROOT/DA3-VLA/
```

`migration/` (this kit + the env lockfiles `*.environment.yml` /
`*.pip-freeze.txt`) travels inside `DA3-XVLA/` automatically.

> The curobo `_tmp.yml` config patches live under `benchmark/assets/` which is
> re-downloaded fresh — so they'd be lost. `apply_patches.sh` (run by
> `setup_server.sh` step 5) re-applies them idempotently, so this is handled.
> The `planner.py` and `world_mesh.py` patches live in code dirs and travel
> with the rsync; `apply_patches.sh` is still idempotent over them.

### Alternative transport: via a HuggingFace repo (no rsync/SSH needed)

Use this when the two servers can't reach each other directly. Packs the same
code-only tree into one tarball through a **private** HF dataset repo.

On the OLD server:
```bash
cd /shared_work/jack/DA3-XVLA
# Pass the token via env at run time — never commit it to a file.
REPO=<user>/da3xvla-migrate HF_TOKEN=<write-token> \
  bash migration/push_to_hf.sh
```
On the NEW server (only `python3` + pip needed; it self-bootstraps
`huggingface_hub`):
```bash
# fetch just the pull script first (it's also inside the tarball):
huggingface-cli download <user>/da3xvla-migrate da3xvla_migrate.tar.gz \
  --repo-type dataset --local-dir /tmp && tar -xzf /tmp/da3xvla_migrate.tar.gz \
  -C NEWROOT DA3-XVLA/migration/pull_from_hf.sh   # or copy it over by hand once
# then:
REPO=<user>/da3xvla-migrate WORK=NEWROOT HF_TOKEN=<read-token> \
  bash NEWROOT/DA3-XVLA/migration/pull_from_hf.sh --setup
```
`push_to_hf.sh` excludes the same things as the rsync path (envs, assets,
data, `.git`, `__pycache__`). `pull_from_hf.sh --setup` extracts the tree and
chains `setup_server.sh --with-assets`, so the new server goes from empty to
fully built in one command. Drop `--setup` to extract only and run
`setup_server.sh` yourself.

---

## 2. Rebuild on the NEW server

```bash
cd NEWROOT/DA3-XVLA
# envs only (assets pulled separately so you can stage the big download):
WORK=NEWROOT bash migration/setup_server.sh
# …or do everything including the ~15 GB asset pull in one go:
WORK=NEWROOT bash migration/setup_server.sh --with-assets
```

What it does (all under `NEWROOT`, idempotent, resumable):
1. Finds conda; if none, installs a **private Miniforge** to `NEWROOT/DA3-VLA/miniforge`.
2. Detects the new GPU's compute capability (for the curobo build).
3. Builds `xvla` env: py3.11 + torch 2.7 cu128 + `DA3-XVLA/requirements.txt` + DA3 editable.
4. Builds `robotwin` env: py3.10 + RoboTwin reqs (torch forced to 2.7 cu128) + pytorch3d + ffmpeg + sapien/mplib source tweaks + **curobo rebuilt for the new GPU arch**.
5. (`--with-assets`) downloads RoboPRO assets (~15 GB).
6. Recreates the `customized_robotwin/assets` symlink, regenerates embodiment
   config paths, applies all 5 source patches.
7. Runs the DA3-XVLA sanity tests (CPU).

Asset download alone, if you skipped it:
```bash
cd NEWROOT/DA3-VLA/RoboPRO
NEWROOT/DA3-VLA/envs/robotwin/bin/python scripts/install/download_assets.py
# then re-run patches (they touch the freshly-downloaded curobo configs):
WORK=NEWROOT bash NEWROOT/DA3-XVLA/migration/apply_patches.sh
cd NEWROOT/DA3-VLA/RoboPRO/benchmark && \
  NEWROOT/DA3-VLA/envs/robotwin/bin/python \
  ../customized_robotwin/script/update_embodiment_config_path.py </dev/null
```

---

## 3. Post-setup verification

```bash
# DA3-XVLA pipeline (CPU): all three must print "... passed"
cd NEWROOT/DA3-XVLA
CUDA_VISIBLE_DEVICES="" envs/xvla/bin/python scripts/debug_geometry_conditioning.py | tail -1
CUDA_VISIBLE_DEVICES="" envs/xvla/bin/python tests/test_geometry_conditioning.py | tail -1
CUDA_VISIBLE_DEVICES="" envs/xvla/bin/python tests/test_da3_pipeline.py | tail -1

# DA3 real extractor (GPU): expect (1,C,64,64)/(3,C,64,64) finite
envs/xvla/bin/python - <<'PY'
import numpy as np, torch
from models.da3_wrapper import DA3FeatureExtractor
e=DA3FeatureExtractor(backend="da3", model_name="depth-anything/DA3-BASE", device="cuda")
print(tuple(e.extract_features([np.random.randint(0,255,(360,640,3),np.uint8)]).shape))
PY

# curobo attach link registered (needs assets): expect 100
cd NEWROOT/DA3-VLA/RoboPRO/benchmark/assets/embodiments/aloha-agilex
NEWROOT/DA3-VLA/envs/robotwin/bin/python - <<'PY'
from curobo.wrap.reacher.motion_gen import MotionGenConfig, MotionGen
c=MotionGenConfig.load_from_robot_config("curobo_left.yml",{"cuboid":{},"mesh":{}},interpolation_dt=1/250,num_trajopt_seeds=2)
print("attached_object spheres:", MotionGen(c).robot_cfg.kinematics.kinematics_config.get_number_of_spheres("attached_object"))
PY

# RoboPRO smoke collection (needs assets, GPU): expect "episode 0 success!"
cd NEWROOT/DA3-VLA/RoboPRO/customized_robotwin
export PATH=NEWROOT/DA3-VLA/envs/robotwin/bin:$PATH
source set_env.sh && export ROBOTWIN_BENCH_TASK=bench CUDA_VISIBLE_DEVICES=0
python -u script/collect_data.py put_mouse_on_pad bench_demo_office_smoke
```

---

## 4. Reference lockfiles & gotchas

- `migration/xvla.environment.yml` / `xvla.pip-freeze.txt`,
  `robotwin.environment.yml` / `robotwin.pip-freeze.txt` — exact versions from
  the working envs. **Use as a version reference, not `pip install -r`**: the
  xvla freeze contains two unrelated editable lines from the shared env it was
  cloned from (`-e /shared_work/behavior1k-mp`,
  `-e /shared_work/markhsp/.../mcp-video-server`) — **ignore them**, they are
  not part of this project. `setup_server.sh` rebuilds clean and avoids them.
- Key pins: torch `2.7.0+cu128`, torchvision `0.22.0+cu128` (cu128 index =
  `https://download.pytorch.org/whl/cu128`), transformers `<=4.51.3`,
  sapien `3.0.0b1`, mplib `0.2.1`, curobo `v0.7.7`, DA3 commit `4173623`,
  pytorch3d commit `75ebeeae`.
- **Blackwell (sm_120 / RTX 5080-class) needs cu128 torch** — RoboTwin's pinned
  `torch==2.4.1` is replaced automatically. On a non-Blackwell GPU cu128 still
  works; only the curobo `TORCH_CUDA_ARCH_LIST` changes (auto-detected).
- curobo build needs an `nvcc` 12.x toolkit (`/usr/local/cuda-12.8` or
  `/usr/local/cuda`). If the new server lacks it, install a CUDA toolkit into
  the conda env (`conda install -p ENV -c nvidia cuda-toolkit=12.8`) and set
  `CUDA_HOME` before re-running step 3b.
- See `migration/PATCHES.md` for exactly what was changed and why.
