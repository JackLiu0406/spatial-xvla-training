# pi0.5 + DA3 — spatial-language VLA training pipeline

This branch (`da3-pi05`) contains the **full training pipeline** for the pi0.5+DA3 model: a
[openpi](https://github.com/Physical-Intelligence/openpi) JAX **pi0.5** policy (PaliGemma VLM + Gemma-300M
action expert, flow matching) augmented with an **X-VLA-style DA3 spatial-language addon** — frozen
DA3-GIANT 3D-geometry features, language-conditioned via ModernBERT, cross-attended into the action
expert's last 6 layers. It is the recipe behind `pi05-da3-robopro-tuned-v2` (final loss ~0.0011).

> Architecture summary, per-field I/O contract, and eval instructions live in the model cards on HF:
> `JackLiu0406/DA3-XVLA-roboreal-ablations/pi05-da3-robopro-tuned-v2-ckpt39999`.

---

## 1. Layout of this directory (`pi05-da3/`)

```
openpi/                      modified openpi package — THE trainable code
  src/openpi/
    models/gemma.py          _LinenCrossAttn (learnable logit gain = the "v2" fix),
                             SpatialActionInjection, Block/Module threading
    models/spatial_da3.py    SpatialBankBuilder, PerceiverDownsampler, ray/pos MLPs
    models/pi0_config.py     Pi0DA3Config (all DA3/v2 flags)
    models/pi0.py            _compute_banks (uint16/fp8 feature decode) + wiring
    training/config.py       the TrainConfigs (pi05_*_da3_inline_v2)
    training/da3_extractor.py inline DA3-GIANT feature extractor (torch) + rescale_intrinsics
    training/data_loader.py  DA3 batch hook, fast-video prefetch, DistributedSampler
    training/fast_video.py   GOP=1 fast decode + EAGAIN retry (shared-node hardening)
    training/optimizer.py    spatial_group_labels -> 3 LR groups (vlm / core / geom)
  scripts/{train.py, compute_norm_stats.py}
  pyproject.toml, uv.lock    dependency lock (uv workspace; packages/openpi-client is a member)
da3/da3_for_geostack.py      frozen DA3-GIANT loader the inline extractor imports
launch/
  launch_pi05_da3_inline_mp_gpus.sh   multi-GPU launcher (arbitrary GPU set, resume, batch/steps)
  launch_pi05_da3_inline_mp.sh        original 0..N-1 GPU launcher
```

`openpi/` is a near-vanilla openpi checkout with the DA3 addon dropped in-tree. Everything DA3-specific is
named `spatial_*`, `da3_*`, or lives in the files listed above.

---

## 2. Prerequisites

- **Hardware:** NVIDIA GPUs, ≥80 GB each (developed on H200 141 GB). Multi-GPU via one JAX process per GPU.
- **Python 3.11**, CUDA 12.x, [`uv`](https://github.com/astral-sh/uv).
- **External model weights** (downloaded once, cached under `HF_HOME`):
  - `depth-anything/DA3NESTED-GIANT-LARGE-1.1` — the frozen geometry backbone.
  - `answerdotai/ModernBERT-large` — the frozen language encoder.
  - `gs://openpi-assets/checkpoints/pi05_base/params` — the pi0.5 base weights (openpi weight loader fetches this).
- **Depth-Anything-3 source** on disk (the geostack imports from it). Clone
  `https://github.com/ByteDance-Seed/Depth-Anything-3` (or your fork) and note its `src/` path.

---

## 3. Setup

### 3a. Install openpi
```bash
cd pi05-da3/openpi
uv venv --python 3.11
uv sync                         # installs from uv.lock (JAX-CUDA, torch, lerobot, orbax, ...)
# sanity: the DA3 config must import
uv run python -c "from openpi.models.pi0_config import Pi0DA3Config; print('DA3 ok', Pi0DA3Config().enabled)"
```

### 3b. Point the DA3 extractor at the geometry stack
The inline extractor (`training/da3_extractor.py`) loads DA3-GIANT via `da3/da3_for_geostack.py`, which
imports the Depth-Anything-3 package. Both paths are **env-overridable** — no code edits needed:
```bash
export DA3_SRC=/abs/path/to/Depth-Anything-3/src          # the DA3 python 'src' dir
export DA3_GEOSTACK=/abs/path/to/pi05-da3/da3/da3_for_geostack.py
```
The extractor pins: model `DA3NESTED-GIANT-LARGE-1.1`, out_layers `(19,26,33,39)`, input `252×336`
(aspect-correct for 4:3 sources), `forward_chunk=16`, bf16, frozen. It also ships features as **uint16
bf16-bits** to halve the GPU→CPU→GPU transfer.

### 3c. Language cache (ModernBERT)
Language features are precomputed once per dataset (task string → `feat[64,1024] fp16`, `mask[64] bool`) and
cached to a pickle. Point the config's `da3_cache.lang_cache` at it (see `training/config.py`). For unseen
prompts at inference, run `answerdotai/ModernBERT-large` last_hidden_state, pad/truncate to 64 tokens.

### 3d. Dataset (LeRobot format, WITH camera calibration)
The DA3 branch needs **per-view camera extrinsics + intrinsics**, so the dataset must carry them:
- 3 camera views, **view order everywhere: `0=countertop/main, 1=left wrist, 2=right wrist`**.
- Per view: `observation.images.<countertop|left|right>` (RGB), `observation.<view>.extrinsic_cv` (3×4
  world→cam), `observation.<view>.intrinsic_cv` (3×3 at native resolution).
- Point the loader at the dataset via `HF_LEROBOT_HOME`; the config `repo_id` names the LeRobot dir.
```bash
export HF_LEROBOT_HOME=/abs/path/to/DATASETS
export HF_HOME=/abs/path/to/hf_cache
export HF_HUB_OFFLINE=1          # once weights are cached
```
> **Intrinsics/aspect note:** `data_loader.py` rescales `intrinsic_cv` from native (e.g. 240×320) to the DA3
> input (252×336) via `rescale_intrinsics(K, native_hw, da3_hw)` **before** the DA3 forward, so K always
> matches the tensor DA3 receives. Keep DA3 input dims aspect-correct (252×336 = 4:3) — do **not** let DA3
> stretch to a square, or the pose math desyncs from the pixels.

### Channel order (RGB/BGR) — IMPORTANT
Both the frozen DA3-GIANT and SigLIP were pretrained on **RGB**, so feeding BGR silently degrades them
(DA3 is frozen — it can't adapt). The training decoder returns `rgb24`, and there is **no swap in openpi**.
The catch is upstream: **the pre-2026-07 RoboReal LeRobot videos were built BGR-swapped** (beige cabinets
render blue). To correct that at load time — applied once, so the DA3 and SigLIP paths stay consistent —
set `DA3CacheConfig(bgr_to_rgb=True)` (already on for the `*_roboreal_full_da3_inline*` configs). The
dataset converter in `data_prep/convert_roboreal_merged_poses.py` has since been fixed (it no longer
`cvtColor`s), so datasets rebuilt with it are correct RGB → use `bgr_to_rgb=False` for those.

> The released `pi05-da3-robopro-tuned-v2` checkpoints on HF were trained **before** this fix (i.e. on the
> BGR-swapped frames). They are internally consistent, so eval must feed them the **same** swapped
> convention — do NOT "correct" eval frames to RGB for those checkpoints. Models trained *after* the fix
> (with `bgr_to_rgb=True`) expect correct **RGB** at eval.

---

## 4. Compute normalization stats (once per dataset)
```bash
cd pi05-da3/openpi
uv run python scripts/compute_norm_stats.py --config-name=pi05_roboreal_full_da3_inline_v2
# writes assets/<config>/<repo_id>/norm_stats.json (quantile stats for state + actions)
```

---

## 5. Run training

Use the multi-GPU launcher. It starts **one JAX process per GPU** (DDP-style via `jax.distributed`), each
running its own torch DA3-GIANT extractor on its shard.

```bash
cd pi05-da3
GPUS=0,1,2,3,4,5,6,7 \            # arbitrary physical GPU set (1 process each)
BATCH_SIZE=128 \                 # GLOBAL batch; must be divisible by #GPUs
NUM_WORKERS=4 \                  # per-rank dataloader workers (keep modest on shared nodes; see §7)
CONFIG_NAME=pi05_roboreal_full_da3_inline_v2 \
  bash launch/launch_pi05_da3_inline_mp_gpus.sh
```

Other env knobs the launcher understands:
| var | default | meaning |
|---|---|---|
| `GPUS` | `4,5,6,7` | comma list of physical GPU ids → jax ranks |
| `BATCH_SIZE` | `96` | global batch (`--batch-size`); must be `% NGPUS == 0` |
| `NUM_WORKERS` | `4` | dataloader workers per rank (`--num-workers`) |
| `NUM_TRAIN_STEPS` | *(config)* | override `--num-train-steps` (e.g. stop early) |
| `EXP_NAME` | timestamped | checkpoint dir key; **reuse the same name + `RESUME=1` to resume** |
| `RESUME` | *(unset)* | set `RESUME=1` to pass `--resume` (restore latest checkpoint) |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.75` | JAX pool cap; raise toward 0.85 for big per-GPU batch |
| `STAGGER` | `0` | seconds between rank launches (spread boot spikes) |

Run detached so it survives your shell:
```bash
setsid bash -c 'GPUS=... CONFIG_NAME=... bash launch/launch_pi05_da3_inline_mp_gpus.sh' \
  > run.log 2>&1 < /dev/null &
```
Per-rank logs land in `/work/.../pi05_runs/<EXP_NAME>/proc_*.log`; `proc_0*.log` has the step rate + loss.

---

## 6. The config & recipe (`pi05_roboreal_full_da3_inline_v2`)

**Model** (`Pi0Config(pi05=True, da3=Pi0DA3Config(...))`):
- `enabled=True`, `spatial_scale=2.0`, `num_inject_layers=6` (inject into action-expert layers 12–17),
  `da3_channels=1536`, grid 18×24, banks 128/96/96.
- **v2 tuning flags** (the whole difference from v1): `attn_logit_gain=True`, `attn_logit_gain_init=32.0`,
  `bank_token_embed=True`, `perceiver_query_std=0.05`, `spatial_init_std=0.01`.

**LR groups** (`optimizer.spatial_group_labels`, all cosine, 1k warmup, `decay_steps=50_000`):
| group | params | peak → floor |
|---|---|---|
| `vlm` | pretrained SigLIP + Gemma-2B + Gemma-300M action expert | 2.5e-5 → 2.5e-6 |
| `core` | fresh spatial modules (`spatial_bank_builder`, `spatial_inject`) | **5e-4 → 5e-5** |
| `geom` | fresh ray/depth MLP (`ray_mlp`) | 5e-4 → 1e-4 |

**Init:** `pi05_base` params via weight loader with `missing_regex` for the fresh spatial modules.
Optim: AdamW (b1 .9, b2 .95, wd 1e-10, clip 1.0), EMA 0.99. Default `num_train_steps=50_000`, `batch_size=128`.

> **Why v2:** in v1 the injection's Q/K attention barely trained (near-uniform → geometry averaged over).
> The learnable per-head logit gain (init 32) sharpens attention from step 0 — the single change that makes
> the last-6-layer cross-attention actually *select* spatial tokens. `pi05_roboreal_full_da3_inline` (no
> suffix) is the untuned v1; use `_v2`.

Other ready configs in `training/config.py`: `pi05_robotwin2_full_da3_inline_v2` (RoboTwin2.0). To add a
dataset, copy a `_v2` TrainConfig, change `repo_id` + `da3_cache` paths.

---

## 7. Operational notes (shared-node hardening — learned the hard way)

- **Re-encode videos to GOP=1.** LeRobot random-frame decode on large-GOP video is ~675 ms/sample and
  dominates step time. Re-encode every episode video to GOP=1 (all keyframes, crf 18) → ~16 ms/frame. This
  is the single biggest speedup (≈4×). `fast_video.py` then decodes single-threaded via `av.open`.
- **Video decode is crash-hardened.** On thread-starved shared nodes the ffmpeg swscaler throws EAGAIN
  (`Resource temporarily unavailable`); `fast_video.py` retries up to ~35 s instead of crashing the rank.
  Keep `NUM_WORKERS` modest (4) to bound concurrent swscale contexts.
- **Cap CPU thread pools.** On many-core boxes (240 cores) torch/BLAS default their pools to core count;
  N processes booting together can exhaust the cgroup pid budget → `pthread_create` EAGAIN → C++ abort. The
  launcher sets `OMP/MKL/OPENBLAS/TORCH_NUM_THREADS=4` and `da3_extractor.py` calls
  `torch.set_num_threads`. If a rank dies with `std::system_error: Resource temporarily unavailable` at
  boot, check `cat /sys/fs/cgroup/user.slice/user-<uid>.slice/pids.{current,max}` and kill orphaned workers.
- **Resume caveat (topology):** resuming a checkpoint saved by N processes onto **M ≠ N** processes hits an
  orbax cross-topology restore bug (`no addressable shards`). Resume on the **same GPU count** it was saved
  with; to change GPU count, either start fresh or convert the checkpoint offline.
- **Memory:** JAX pre-allocates `XLA_PYTHON_CLIENT_MEM_FRACTION` of each GPU up front, so `nvidia-smi`
  "used" is a reservation, not the peak. torch DA3-GIANT (~17 GB, chunk-bounded) lives outside that pool —
  leave room for it (0.75 at ≤48/GPU; 0.85 fits ~64/GPU; 128/GPU OOMs).

---

## 8. Validate before trusting a checkpoint
1. `model.compute_loss` on training-like data → ~0.001–0.003.
2. **Ablation:** `object.__setattr__(model.PaliGemma.llm.module, "spatial_scale", 0.0)` should raise loss to
   ~0.9 (proves the spatial pathway is load-bearing), then restore to 2.0.
3. **DA3-correctness (on a REAL observation):** loss approaches ~0.003 **and** swapping the left/right wrist
   inputs measurably changes sampled actions. (A random-input ablation only proves the injection is wired,
   not that the DA3 features are correct — the bank builder's pos/view/ray embeddings are non-zero even with
   zero features.)

---

Built on [openpi](https://github.com/Physical-Intelligence/openpi) (see `openpi/LICENSE`,
`openpi/LICENSE_GEMMA.txt`). DA3 addon + training recipe by this project.
