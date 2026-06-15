# Segmented DA3 Geometry Conditioning for X-VLA

An **optional, config-gated** extension that conditions the X-VLA flow-matching
action policy on compressed 3D/geometry tokens derived from a DA3 encoder,
fused into the policy via cross-attention.

When `geometry_conditioning.enabled = false` (the default) the model is
**byte-for-byte the baseline X-VLA** — no new modules are built, parameter
groups are unchanged, and old checkpoints load normally.

## What it does

```
baseline X-VLA tokens : images+lang -> vlm/aux tokens
                        proprio      -> state tokens
                        noisy action -> action tokens
                        flow time    -> timestep token
                        domain id    -> soft-prompt tokens

DA3 path (new)        : precomputed DA3 features [+ object masks]
                        -> mask weighting (down-weight background, not zero)
                        -> Linear/MLP projector  (C -> hidden_size)
                        -> Perceiver resampler    (variable N -> K tokens)
                        -> K geometry tokens [B, K, hidden_size]

fusion (new)          : Q = X-VLA policy tokens, K/V = geometry tokens
                        cross-attention -> fused policy tokens
                        -> existing Transformer policy
                        -> existing flow-matching action head
```

The DA3/segmented feature is **never** dumped raw into the main Transformer;
it is always compressed to a small fixed `num_geometry_tokens` (default 32).

## Expected input shapes

`da3_features` (any one of):

| Shape            | Meaning                       |
|------------------|-------------------------------|
| `[B, N, C]`      | tokenized, single view        |
| `[B, C, H, W]`   | dense feature map, single view|
| `[B, V, N, C]`   | tokenized, multi-view         |
| `[B, V, C, H, W]`| dense feature map, multi-view |

`object_masks` (optional): `[B,1,H,W]`, `[B,H,W]`, `[B,V,1,H,W]`, `[B,N]`,
`[B,1,N]`, `[B,V,N]`. Masks are bilinearly resized to the feature grid and
clamped to `[0,1]`. Weighting: `mask*feat + alpha*(1-mask)*feat`, where
`alpha = mask_background_weight`.

4-D ambiguity (`[B,C,H,W]` vs `[B,V,N,C]`) is resolved with
`geometry_conditioning.da3_input_dim`; if it is `null`, a 4-D tensor is
assumed dense `[B,C,H,W]`.

## How to enable

In the model's `config.json` (or when constructing `XVLAConfig`):

```json
"geometry_conditioning": {
  "enabled": true,
  "source": "precomputed_da3",
  "fusion_type": "cross_attention",
  "num_geometry_tokens": 32,
  "da3_input_dim": null,
  "geometry_hidden_dim": null,
  "mask_background_weight": 0.2,
  "use_object_mask": true,
  "freeze_da3": true,
  "cross_attention_layers": 1,
  "cross_attention_heads": 8,
  "cross_attention_dropout": 0.0,
  "fusion_position": "before_policy",
  "train_geometry_projector": true,
  "train_cross_attention": true,
  "allow_missing_geometry": false,
  "resampler_type": "perceiver",
  "debug_shapes": false
}
```

`geometry_hidden_dim: null` → forced to the policy `hidden_size`.
`da3_input_dim: null` → inferred from the first batch (the projector is built
lazily and frozen to that channel count thereafter).
`fusion_position`: `before_policy` (default), `inside_policy` (mid-stack), or
`after_policy`.

## Using precomputed DA3 features

`source="precomputed_da3"` reads `da3_features` (and optional `object_masks`)
from the batch / forward kwargs:

```python
loss = model(input_ids=..., image_input=..., image_mask=...,
             domain_id=..., proprio=..., action=...,
             da3_features=da3, object_masks=masks)   # [B,C,H,W], [B,1,H,W]
```

To train with geometry, the dataset must yield `da3_features` (and optionally
`object_masks`) per sample; `train.py` passes any extra tensor batch keys
straight through `model(**inputs)`, so no training-loop change is required —
only the dataset/handler needs to emit the keys.

If `enabled=true` but `da3_features` is missing: a clear `ValueError` is
raised, unless `allow_missing_geometry=true` (then zeros are used — debug
only).

`source="da3_encoder_stub"` raises `NotImplementedError` — the clean seam to
later drop in a real in-model DA3 encoder (replace
`SegmentedDA3GeometryConditioner.forward`'s feature acquisition).

## Where the fusion is inserted

`GeometryCrossAttentionFusion` lives **inside `SoftPromptedTransformer`**:

- `before_policy`: after action/vlm/aux/pos/soft-prompt tokens are assembled,
  before the Transformer blocks (default).
- `inside_policy`: once, after the middle Transformer block.
- `after_policy`: after the last block, before the action decoder.

The conditioner (`SegmentedDA3GeometryConditioner`) lives on the top-level
`XVLA` model and produces the geometry tokens passed into the transformer.

## Fine-tuning groups

`config.finetune` + `--apply_finetune_policy` (opt-in) freeze/unfreeze:
`train_backbone`, `train_soft_prompts`, `train_action_head`,
`train_geometry_modules`, `train_last_n_policy_layers`.
`build_optimizer` additionally exposes a dedicated `geometry` param group
(present only when geometry is enabled) at the action-head LR.

## Sanity check

```bash
python tests/test_geometry_conditioning.py
```

Backbone-free; runs on CPU in seconds. Verifies disabled==baseline,
all input-shape variants, both resamplers, all three fusion positions, and
action-shape parity with the baseline.

## Offline DA3 preprocessing pipeline

DA3 is **not** run inside the model forward pass. Features are precomputed
offline and looked up at data-loading time.

### 1. Precompute features

```bash
# Dummy backend (no DA3 install) — validates the whole pipeline:
python scripts/precompute_da3_features.py \
    --metas_path /path/to/metas --out_root /path/to/da3_features \
    --backend dummy --stride 4

# Real DA3:
python scripts/precompute_da3_features.py \
    --metas_path /path/to/metas --out_root /path/to/da3_features \
    --backend da3 --model_name depth-anything-3-base --device cuda
```

Output layout (keyed by the exact triplet the dataloader has):

```
<out_root>/manifest.json
<out_root>/<dataset_slug>/<traj_idx>/<frame_idx>.pt
    -> {"da3_features": [V,C,H,W] or [V,N,C],
        "object_masks": [V,1,h,w]  (optional),
        "camera_ids":   [str, ...]}
```

`(dataset_name, traj_idx, frame_idx)` is the same triplet available inside
`BaseHDF5Handler.iter_episode` (`frame_idx` indexes the raw HDF5 image array),
so attach is order-independent and unambiguous. Optional object masks are read
from `--mask_obs_keys` (one HDF5 key per camera) and saved feature-aligned.

### 2. Point the config at it

```json
"geometry_conditioning": {
  "enabled": true,
  "source": "precomputed_da3",
  "da3_feature_root": "/path/to/da3_features",
  "object_mask_root": null
}
```

`create_dataloader` → `InfiniteDataReader` builds a `DA3FeatureAttacher`
**only** when `enabled` and `da3_feature_root` are set; it injects it into the
HDF5 handler, which attaches `da3_features` (and `object_masks` if present) to
each sample. Missing files raise a clear error unless
`allow_missing_geometry=true` (then a zero placeholder keeps batches
collatable). When geometry is disabled the attacher is never built and the data
path is byte-for-byte the baseline.

### DA3 wrapper interface

`models/da3_wrapper.py::DA3FeatureExtractor` exposes
`extract_features(images) -> [V,C,H,W]`. `backend="dummy"` yields deterministic
random features (no DA3 install); `backend="da3"` loads real Depth Anything 3
and raises a clear `ImportError` with install instructions if it is missing.
The single seam to adapt to a different DA3 feature choice is
`DA3FeatureExtractor._load_da3` / `_encode_da3`.

### Real DA3 install (verified)

Depth Anything 3 is cloned to `third_party/Depth-Anything-3` and installed
**editable** into the private env `envs/xvla/` (kept inside `/shared_work/jack`
per the workspace constraint; the shared `xvla` conda env is left untouched).

```bash
# run the precompute script with the private env:
/shared_work/jack/DA3-XVLA/envs/xvla/bin/python scripts/precompute_da3_features.py \
    --metas_path /path/to/metas --out_root /path/to/da3_features \
    --backend da3 --model_name depth-anything/DA3-BASE --device cuda
```

The `da3` backend runs DA3's multi-view `inference()` over the `V` camera
views and exposes **per-view depth + confidence** as a dense
`[V, 2, feature_size, feature_size]` feature (default 64×64) — exactly the
dense `[B,V,C,H,W]` layout `SegmentedDA3GeometryConditioner` consumes.
Verified end-to-end on the RTX 5080 (`V=1 → (1,2,64,64)`,
`V=3 → (3,2,64,64)`, ~0.14–0.5 s/forward). To use richer encoder tokens
instead of depth, request `export_feat_layers` in `_encode_da3`.

> Note: handlers that override `iter_episode` and are not HDF5-based
> (`x2robot`, `AGIBOT` lerobot, `agiworld`) are not covered by the attach path
> yet; the BaseHDF5Handler family (Calvin, Libero, RoboTwin2, VLABench,
> robomind, droid, AIR real-world) is.

## Files

| File | Change |
|------|--------|
| `models/geometry_conditioning.py` | **new** — conditioner, resampler, fusion |
| `models/configuration_xvla.py` | `geometry_conditioning` + `finetune` blocks |
| `models/modeling_xvla.py` | build conditioner, compute tokens, finetune policy, forward/inference/serve wiring |
| `models/transformer.py` | optional fusion at before/inside/after policy |
| `train.py`, `peft_train.py` | geometry param group + LR schedule; `--apply_finetune_policy` |
| `models/da3_wrapper.py` | **new** — `DA3FeatureExtractor` (dummy / real DA3 API) |
| `third_party/Depth-Anything-3/` | **new** — cloned DA3 repo (editable-installed into `envs/xvla/`) |
| `scripts/precompute_da3_features.py` | **new** — offline preprocessing CLI |
| `datasets/da3_loader.py` | **new** — `DA3FeatureAttacher` + key scheme |
| `datasets/domain_handler/base.py` | gated, no-op-by-default feature attach |
| `datasets/dataset.py`, `datasets/__init__.py` | build/thread attacher from config |
| `tests/test_geometry_conditioning.py` | **new** — model-side sanity checks |
| `tests/test_da3_pipeline.py` | **new** — offline-pipeline sanity checks |
| `docs/geometry_conditioning.md` | **new** — this doc |
