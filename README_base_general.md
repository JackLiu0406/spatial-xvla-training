# DA3-XVLA: Geometry-Conditioned X-VLA with Pluggable Spatial Encoders

A modular training pipeline for **X-VLA** (Vision-Language-Action policies) extended with **inline geometry conditioning** from two interchangeable spatial encoders:

- **DA3** ([Depth-Anything-3](https://github.com/DepthAnything/Depth-Anything-3)) — DinoV2-backbone monocular/multi-view depth + ray model. SMALL/BASE/LARGE/GIANT variants.
- **VGGT-Omega** — multi-view aggregator with frame + inter-frame attention streams. 1B variant.

Both encoders feed dense geometry features into the same downstream perceiver-resampler → cross-attention into the policy transformer. Backbone is **swappable via a single config key**; everything else is shared.

This repo is intentionally structured for **ablations**. It supports several axes of variation, including:

| Axis | Values | Where to set |
|---|---|---|
| Spatial encoder | `da3` &#124; `vggt` | `config.json: geometry_conditioning.geometry_backbone` |
| DA3 backbone size | `SMALL`, `BASE`, `LARGE`, `GIANT` | `config.json: da3_model_name` |
| Geometry tokens K | 80, 160, ... | `config.json: num_geometry_tokens` |
| Feature tap | `last` &#124; `multi` &#124; `dpt_fused` | `config.json: da3_feature_layer` (or `vggt_feature_layer`) |
| Multi-layer fusion init | `last_only` &#124; `uniform` | `config.json: da3_multi_fusion_init` |
| Input resolution | 224×224 (squished) &#124; aspect-preserved (e.g. 252×336) | env: `XVLA_DA3_INPUT_H`, `XVLA_DA3_INPUT_W` |
| Posed multi-view DA3 | on/off | env: `XVLA_POSED_DA3=1` |
| Native input bypass | on/off | env: `XVLA_DA3_NATIVE_INPUT=1` |

All combinations work via the same `train.py`. Concrete ablation configurations are in [`config_templates/`](config_templates/); ready-to-run launch scripts are in [`scripts/launch_examples/`](scripts/launch_examples/).

---

## What's where

```
.
├── train.py                  # Single entry point for training (DDP-aware, accelerate-launchable)
├── deploy.py                 # Inference / deployment harness
├── peft_train.py             # PEFT (LoRA etc.) training variant
├── requirements.txt          # Core Python deps
├── requirements_xvla.txt     # X-VLA-specific extras
├── environment.yml           # Conda environment lockfile
│
├── models/
│   ├── modeling_xvla.py      # XVLA top-level model. Switches DA3↔VGGT here.
│   ├── configuration_xvla.py # Config schema (defaults + back-compat aliases)
│   ├── processing_xvla.py    # Florence-2 image+text processor wrapper
│   ├── modeling_florence2.py # Florence-2 VLM backbone
│   ├── transformer.py        # Policy (flow-matching) transformer
│   ├── action_hub.py         # Per-embodiment action heads
│   ├── da3_inline.py         # ⭐ Inline DA3 encoder: last / multi / dpt_fused modes
│   ├── vggt_inline.py        # ⭐ Inline VGGT-Omega encoder: last / multi modes
│   ├── geometry_conditioning.py # geometry projector + perceiver resampler + cross-attn fusion
│   └── gsam_inline.py        # Optional GroundingDINO + SAM mask conditioning
│
├── datasets/
│   ├── dataset.py            # InfiniteDataReader + dual-resolution dataloader
│   ├── utils.py              # JPEG decode (BGR/RGB), interpolation
│   ├── domain_config.py      # Per-dataset weights + domain IDs
│   └── domain_handler/       # Per-dataset handlers (robotwin, droid, lerobot, ...)
│
├── third_party/
│   ├── Depth-Anything-3/     # Vendored DA3 source (DINOv2 + DPT/DualDPT heads)
│   └── vggt_omega/           # Vendored VGGT-Omega aggregator + heads
│
├── evaluation/               # Per-benchmark eval clients (CALVIN, LIBERO, SimPLER, vlabench, robotwin)
├── scripts/
│   ├── benchmark_dataloader.py
│   └── launch_examples/      # ⭐ 7 ready-to-run ablation launch scripts (sanitized)
├── config_templates/         # ⭐ The matching config.json per ablation cell
├── docs/                     # Geometry-conditioning + segmentation docs
├── tests/                    # Unit tests
└── migration/                # Patch/sync scripts for upstream X-VLA
```

The two starred groups (`*_inline.py` and `launch_examples/` + `config_templates/`) are the surfaces you'll typically touch when adapting this to a new project.

---

## How geometry conditioning is wired

```
              ┌─────────────┐
              │ image_input │  [B, V, 3, 224, 224]   ──► VLM (Florence-2)
              │  (VLM-res)  │
              └─────────────┘
              ┌──────────────────┐
              │ image_input_da3  │  [B, V, 3, H, W]   ──► geometry encoder
              │ (geom-encoder res│       (DA3 or VGGT-Omega — switched by config)
              │  aspect-preserved│
              │  if env vars set)│
              └──────────────────┘
                      │
                      ▼
          ┌──────────────────────────────┐
          │ DA3InlineEncoder OR          │   feature_layer = "last" | "multi" | "dpt_fused"
          │ VGGTInlineEncoder            │   freeze backbone, only adapters/projector train
          └──────────────┬───────────────┘
                         │ [B, V, C, h, w]
                         ▼
          ┌──────────────────────────────┐
          │ geometry_projector (MLP)     │   C → hidden_dim
          └──────────────┬───────────────┘
                         ▼
          ┌──────────────────────────────┐
          │ perceiver resampler          │   spatial 4096+ → K tokens (K=160)
          └──────────────┬───────────────┘
                         ▼
          ┌──────────────────────────────┐
          │ cross-attention fusion       │   policy queries attend to K geometry tokens
          └──────────────┬───────────────┘
                         ▼
                  policy transformer (X-VLA)
```

The dataloader emits both image tensors when `XVLA_DA3_INPUT_H/W` env vars are set; otherwise only the VLM tensor (224×224), used for both branches. Intrinsics are rescaled to match the geometry-encoder input resolution.

---

## How to switch between ablations

Each ablation = one `config.json` + one `launch.sh`. The `config_templates/` directory has the 7 we ran for this paper:

| Ablation cell | Backbone | K | Feature tap | Input res | Notes |
|---|---|---|---|---|---|
| `baseline_DA3BASE_K80_native224` | DA3-BASE | 80 | last | 224 sq | smallest |
| `baseline_DA3BASE_K80_native_aspect` | DA3-BASE | 80 | last | 252×336 | aspect-preserved |
| `baseline_DA3BASE_K160_native_aspect` | DA3-BASE | 160 | last | 252×336 | aspect anchor |
| `baseline_DA3BASE_K160_dptfused_aspect` | DA3-BASE | 160 | **dpt_fused** | 252×336 | DA3 own head tap |
| `baseline_DA3LARGE_K160` | DA3-LARGE | 160 | last | 224 sq | original anchor |
| `baseline_DA3LARGE_K160_native_aspect` | DA3-LARGE | 160 | last | 252×336 | aspect+LARGE |
| `vggt_omega_K160_multi_aspect` | VGGT-Omega | 160 | **multi** | 240×320 | multi-layer fusion |

**To run a cell:** copy the matching launch script, set `REPO_ROOT` and `RUNS_ROOT`, then `bash launch_examples/<cell>.sh`. Each script is annotated with what's being varied vs the anchor.

**To create a new cell:** copy any `config_templates/*.config.json` to your runs root, edit the keys you want to change, and write a `launch.sh` setting the right env vars + `--models` pointer.

---

## Key abstractions

### `models/da3_inline.py`

`DA3InlineEncoder` — wraps DA3 from `depth_anything_3.api` and supports three feature-extraction modes:

- **`feature_layer="last"`** — single deepest DINOv2 block, raw token features. Channel dim = backbone embed_dim (BASE: 768, LARGE: 1024).
- **`feature_layer="multi"`** — 4 DPT-tap layers (e.g. [5,7,9,11] for BASE), each LayerNorm'd, channel-concatenated, fused with a learned 1×1 Conv(4C→C). Channel dim stays at C. Init mode `last_only` (identity on deepest tap) or `uniform`.
- **`feature_layer="dpt_fused"`** — taps DA3's *own* pretrained DPT head right before the depth-output Conv. Channel dim = `features // 2` (BASE: 64). Zero new trainable params — uses DA3's full pyramid fusion + spatial recovery + non-linear refinement for free. Implemented via a `register_forward_hook` on `head.scratch.output_conv2`.

Posed multi-view: when `extrinsics` and `intrinsics` are passed (env `XVLA_POSED_DA3=1`), DA3's camera encoder fuses pose tokens with backbone features. Intrinsics are rescaled in the dataloader to match the geometry-encoder input resolution.

### `models/vggt_inline.py`

`VGGTInlineEncoder` — wraps `vggt_omega.models.VGGTOmega` aggregator with the same `forward(image, extrinsics?, intrinsics?) → [B,V,C,h,w]` signature so it's a drop-in replacement.

- **`feature_layer="last"`** — uses the last cached aggregator layer (block 23). Channel dim = 2 × embed_dim = 2048 (the concat of frame + inter-frame streams).
- **`feature_layer="multi"`** — uses all 4 cached layers [4, 11, 17, 23], each LayerNorm'd, concatenated, fused with 1×1 Conv(4·2048→2048). Init modes match DA3.

VGGT-Omega uses RoPE positional encoding internally, so **rectangular inputs work natively** (no pos-embed interpolation needed). Patch size is **16** for VGGT (vs DA3's 14), so input H/W must be multiples of 16.

VGGT does *not* consume extrinsics/intrinsics (it predicts pose internally). Those kwargs are accepted for signature parity with DA3 and ignored.

### `models/modeling_xvla.py`

The single line that switches backbones:

```python
backbone = str(self.geometry_cfg.get("geometry_backbone", "da3")).lower()
if backbone == "vggt":
    self.da3_inline = VGGTInlineEncoder(...)
else:
    self.da3_inline = DA3InlineEncoder(...)
```

Note: the attribute name `da3_inline` is kept regardless of backbone choice so downstream code (`_compute_geometry_tokens`, `apply_finetune_policy`, parameter-group split) is backbone-agnostic.

### `datasets/dataset.py` + `datasets/domain_handler/base.py`

Dual-resolution dataloader. Two transforms when `XVLA_DA3_INPUT_H/W` env vars are set:

```python
self.image_aug      = Resize(224, 224) + Jitter + ToTensor + Normalize(ImageNet)   # for VLM
self.image_aug_da3  = Resize(H, W)     + Jitter + ToTensor + Normalize(ImageNet)   # for geometry encoder
```

Each sample carries both `image_input` (224²) and `image_input_da3` (HxW). The DataLoader's default collate stacks them across the batch automatically.

Intrinsics from the HDF5 dataset are rescaled from native (e.g. 240×320) to the geometry-encoder size (e.g. 252×336) so DA3's posed-multi-view math operates on correctly-calibrated rays.

When env vars are unset, the dataloader emits only the VLM tensor — backward-compatible with the single-image regime.

---

## Installation

```bash
# Conda env
conda env create -f environment.yml
conda activate xvla

# Or pip
pip install -r requirements.txt -r requirements_xvla.txt

# DA3 and VGGT-Omega are vendored under third_party/ — no extra install needed,
# but Python needs to be able to import them. The inline encoders insert
# third_party/ into sys.path at import time. If your launcher mucks with
# PYTHONPATH, set PYTHONPATH="$REPO_ROOT" explicitly.
```

The pretrained encoder weights are downloaded from HuggingFace on first use:
- DA3: `depth-anything/DA3-{SMALL,BASE,LARGE,GIANT}`
- VGGT-Omega: `JackLiu0406/vggt-omega-1b` (gated — request access on HF)

The X-VLA-Pt starting checkpoint is at [`2toINF/X-VLA-Pt`](https://huggingface.co/2toINF/X-VLA-Pt). Use it as the `--models` source for fresh training runs.

---

## How to run a training

Minimum:

```bash
# 1. Build a ckpt_init directory: symlink X-VLA-Pt artifacts + a config.json
#    that selects your ablation (see config_templates/).
export CKPT_INIT=/path/to/your/ckpt_init
mkdir -p $CKPT_INIT
ln -s /path/to/X-VLA-Pt/* $CKPT_INIT/
cp config_templates/baseline_DA3BASE_K160_native_aspect.config.json $CKPT_INIT/config.json

# 2. Launch (single node, 4 GPUs)
export REPO_ROOT=$(pwd)
export RUNS_ROOT=/path/to/your/runs
bash scripts/launch_examples/baseline_DA3BASE_K160_native_aspect.sh
```

The launch script sets the right env vars (e.g. `XVLA_DA3_INPUT_H=252 XVLA_DA3_INPUT_W=336`) and invokes `accelerate launch train.py` with the right hyperparameters. Copy + edit one to define a new cell.

### Training hyperparameters (defaults from the launch scripts)

```
batch_size      32       per GPU (effective 128 with 4 GPUs)
learning_rate   1e-4     core
learning_coef   0.1      VLM lr is core × 0.1
weight_decay    0.0
betas           0.9 0.95
max_grad_norm   1.0
iters           100000
freeze_steps    1000     LR = 0 for adapter warmup phase
warmup_steps    2000     linear LR ramp 0 → core
cosine_decay    yes, min_lr_ratio=0.1
save_interval   10000
log_interval    20
```

---

## Compute notes

- All runs were done on **4×H200 (141GB each)**, batch_size=32 per GPU.
- DA3-BASE: ~16 GB VRAM after warmup, ~0.82 s/iter.
- DA3-LARGE: ~17 GB VRAM, ~1.0-1.1 s/iter.
- VGGT-Omega 1B: ~100+ GB VRAM, ~1.1 s/iter.
- Full 100k-iter run takes ~22-30h depending on cell.

The aspect-preserved variants (252×336 for DA3, 240×320 for VGGT) cost ~5-15% more compute than 224² but train measurably better (no anisotropic stretch).

---

## Ablation cookbook (one-knob recipes)

To swap **DA3 → VGGT** keeping everything else fixed:
1. Set `geometry_backbone: "vggt"` in config.
2. Set `vggt_input_dim: 2048` (VGGT is 2C = 2048-dim, vs DA3-BASE's 768).
3. Change `XVLA_DA3_INPUT_H/W` to multiples of 16 (e.g. 240, 320 — VGGT patch_size=16).

To enable **multi-layer fusion** (4-tap):
1. Set `da3_feature_layer: "multi"` (or `vggt_feature_layer: "multi"`).
2. Default `multi_fusion_init: "last_only"` makes step 0 approximately identical to single-layer behavior (modulo LayerNorm distortion).

To enable **DPT-head tap** (DA3 only):
1. Set `da3_feature_layer: "dpt_fused"`.
2. Set `da3_input_dim: features // 2` (32 for SMALL, 64 for BASE, 128 for LARGE/GIANT).
3. Zero new trainable params — uses DA3's frozen pretrained DPT head.

To change **input aspect**:
- Pick `H, W` that are multiples of 14 (DA3) or 16 (VGGT), close to your data's native aspect ratio.
- Set `XVLA_DA3_INPUT_H` and `XVLA_DA3_INPUT_W` env vars before launching.
- Set `XVLA_DA3_NATIVE_INPUT=1` to skip the in-model resize.

To change **K (geometry tokens)**:
- Set `num_geometry_tokens` in config. Anything from 32 to 256 works.

---

## Citation / lineage

This work builds on:

- **X-VLA** (Tsinghua / 2toINF) — ICLR 2026. [Paper](https://arxiv.org/pdf/2510.10274). The policy transformer + Florence-2 backbone + soft-prompt design.
- **Depth-Anything-3** (ByteDance) — multi-view depth. [GitHub](https://github.com/DepthAnything/Depth-Anything-3).
- **VGGT** family (Meta AI) — visual geometry transformers. VGGT-Omega is the 1B variant.

If you use this codebase, please cite the underlying papers above.

---

## License

This repo inherits the **Apache 2.0** license from X-VLA. Vendored third_party code (DA3, VGGT-Omega) retains their original licenses (see respective `LICENSE` files under `third_party/`).
