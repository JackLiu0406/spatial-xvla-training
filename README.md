# Model C — Spatial-Language XVLA (spatial boost) · Training Pipeline

A complete, runnable pipeline to **train / recreate Model C**: a bimanual VLA policy = stock **X-VLA** + a **spatial-language conditioning branch** ("Method B / `final6_injection`") that injects frozen geometry (DA3) + language (T5) into the action expert. "Spatial boost" = trained/evaluated at an elevated spatial-injection scale.

> This branch contains the **full source — clone and run, no patch step.** Output actions are `[B, 30, 20]` bimanual EE6D.
> (General ablation-framework docs are in `README_base_general.md`.)

---

## TL;DR — recreate Model C

```bash
# 1. env
conda env create -f environment.yml && conda activate xvla     # exact pins in migration/xvla.pip-freeze.txt

# 2. data: RoboReal HDF5 with per-camera extrinsic_cv / intrinsic_cv (see §Data)
# 3. base ckpt: original Method-B final6-injection ckpt-125000 (the Model C resume base)

# 4. train (this IS the Model C recipe)
export XVLA_DA3_INPUT_H=252 XVLA_DA3_INPUT_W=336        # dual-res DA3 branch (aspect-correct 4:3)
bash scripts/launch_model_c/launch_model_c_spatialboost_scale1p5_30k.sh
```
The launch script wraps `train.py` with the exact flags (below). Frozen DA3-Large + T5 backbones auto-download from HF on first run.

---

## What Model C is (architecture in 60 seconds)

Base = stock X-VLA: Florence-2 VLM + a **24-block action expert** (hidden 1024, 16 heads) → `[B,30,20]` bimanual EE6D. The spatial-language branch adds, **per camera view**:

```
RGB view ─► frozen DA3-Large ─► 4 feature layers + per-pixel ray
         ─► project/fuse to 1024 + 2D-grid emb + view emb + ray emb
         ─► Perceiver downsample ─► T5 language-fusion cross-attn ─► per-view "bank"
```
Those banks are **cross-attended into action-expert blocks 18–23 only** (the "final 6"), updating just the action slice `x[:, :30]`, scaled by a runtime `spatial_scale`:
```
h_action ← h_action + spatial_scale · CrossAttn(h_action, [main | left | right] banks)
```
Spatial tokens are **not appended** to the sequence; DA3/T5 are **frozen**. Per-view token budget: main 96, left 64, right 64.

**`spatial_scale` is a runtime scalar (`model.set_spatial_residual_scale(x)`), not a weight** — Model C trains at **1.5** and you sweep it at eval. Higher = geometry conditions the action more strongly ("spatial boost").

---

## Most important files (read in this order)

| # | File | What it is / why it matters |
|---|---|---|
| 1 | **`models/spatial_language.py`** | ⭐ **The innovation.** `SpatialLanguageTokenizer` (builds the per-view DA3+T5 banks) + `SpatialActionInjectionLayer` (main/left/right cross-attn + merge that injects into the action expert). Start here. |
| 2 | **`models/modeling_xvla.py`** | The X-VLA model + where the branch is wired in: `_build_spatial_lang_bank` (calls the tokenizer), the blocks-18–23 injection, and **`set_spatial_residual_scale`** (the eval knob). Config gate: `geometry_conditioning.spatial_lang.enabled`. |
| 3 | **`models/configuration_xvla.py`** | Config schema. The `geometry_conditioning.spatial_lang` sub-dict (method, tokens, scale, da3_model, t5 name) defines the model. |
| 4 | **`models/da3_for_geostack.py`** | Frozen **DA3-Large** geometry backbone wrapper — multi-layer features + per-pixel ray at the 18×24 patch grid. |
| 5 | **`models/t5_inline.py`** | Frozen **T5** text encoder wrapper (per-token language embeddings for the fusion step). |
| 6 | **`train.py`** | Training loop, optimizer param-groups, LR schedules, and **all CLI flags** (`--spatial_lang_method`, `--spatial_scale_*`, `--fixed_geometry_lr`, ...). |
| 7 | **`datasets/dataset.py` + `datasets/domain_handler/`** | Data loader. `get_camera_poses` reads `observation/<cam>/extrinsic_cv`+`intrinsic_cv` from RoboReal HDF5; env `XVLA_DA3_INPUT_H/W` produces the 252×336 DA3 image branch. |
| 8 | **`scripts/launch_model_c/launch_model_c_spatialboost_scale1p5_30k.sh`** | ⭐ The exact Model C launch command (recipe below). |
| — | `models/modeling_florence2.py`, `models/transformer.py`, `models/action_hub.py` | Base X-VLA pieces (VLM, action-expert transformer, EE6D action space) — stock X-VLA. |
| — | `models/{geometry_conditioning,geostack,da3_inline,gsam_inline}.py` | Other geometry-conditioning variants X-VLA supports; **not used by Model C** but imported by `modeling_xvla.py`, so must be present. |

---

## Training recipe (exact)

Model C = resume from original **Method-B `ckpt-125000`**, train **30k** more local steps:
```
--spatial_lang_method final6_injection
--spatial_scale_start 1.5 --spatial_scale_end 1.5 --spatial_scale_ramp_until 0   # flat 1.5
--fixed_geometry_lr 1e-4          # geometry / spatial modules
--fixed_xvla_core_lr 1e-5         # action-expert core
--fixed_vlm_lr 0                  # VLM frozen
--batch_size 32 --iters 30000 --save_interval 5000
```
Defaults give DA3-Large + t5-base (Model C's backbones). Checkpoints are **model-only** (no optimizer state); `train.py` restarts `global_step` at 0 on warm-start, so `--iters` is a *local* budget. AdamW(0.9,0.95), wd 1e-8, grad-clip 1.0, global batch 128 (4×H200).

## Data
RoboReal HDF5 (`robotwin2_clean` domain → `domain_id = 6`). Each camera stores `observation/<cam>/rgb`, `extrinsic_cv` (T,3,4 OpenCV world-to-camera), `intrinsic_cv` (T,3,3). Native 240×320 (4:3) → DA3 branch resized to 252×336 (uniform, aspect-preserved). **View order everywhere: 0 = main/scene cam, 1 = left wrist, 2 = right wrist.**

## Evaluation
```python
model.set_spatial_residual_scale(1.5)          # Model C trained scale; SWEEP 1.0 / 1.25 / 1.5 / 1.75 / 2.0
actions = model.generate_actions(
    input_ids, image_input,           # [B,3,3,224,224] VLM branch
    image_mask, domain_id=6, proprio, steps=10,
    image_input_da3=...,              # [B,3,3,252,336] DA3 branch (required)
    language_instruction=texts,       # list[str] (required)
    extrinsics=..., intrinsics=...,   # camera poses -> world-frame rays
)   # -> [B,30,20]: left = [:, :, 0:10] xyz/rot6d/gripper, right = [:, :, 10:20]
```
Actions are already un-normalized (EE / 6D-rotation). Eval clients live in `evaluation/` (robotwin-2.0, libero, calvin, ...).

## Provenance
X-VLA-PT → Method-B final6-injection (→ ckpt-125000) → **Model C: +30k steps @ spatial scale flat 1.5, geom LR 1e-4, core LR 1e-5, VLM frozen.**

## Repo layout
`models/` (all model code) · `train.py` · `datasets/` (loader + domain handlers) · `scripts/launch_model_c/` (launch + queue scripts) · `config_templates/` (baseline geometry-conditioning configs) · `evaluation/` (eval clients) · `migration/` (env pins, HF helpers) · `docs/` (design notes) · `README_base_general.md` (general ablation-framework readme).
