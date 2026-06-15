# Segmented DA3 Geometry Conditioning (Cross-Attention) for X-VLA

Optional, fully config-gated extension. **Default is disabled**
(`geometry_conditioning.enabled=false`) → baseline X-VLA is byte-for-byte
unchanged.

## 1. Architecture

```
RGB + language
    -> X-VLA visual-language tokens

RGB
    -> DA3 encoder (OFFLINE)
    -> DA3 latent features                (saved to disk)

RGB + language/object query
    -> language-conditioned segmenter (OFFLINE, LangSAM/Grounded-SAM)
    -> object masks                       (saved to disk)

DA3 latent + object masks
    -> DA3LatentSegmenter   (soft mask on the LATENT, not a point cloud)
    -> flatten/tokenize + MLP projector (C -> D)
    -> GeometryTokenResampler (Perceiver, variable N -> fixed K)
    -> geometry tokens [B, K, D]

X-VLA policy tokens (VL + state + noisy-action + timestep + soft-prompt)
    -> GeometryCrossAttentionFusion (Q=policy, K/V=geometry)
    -> fused policy tokens
    -> existing Transformer policy
    -> existing flow-matching action head -> training loss
```

## 2. Why segment on the DA3 *latent* (not a point cloud)

We multiplicatively down-weight background **on the DA3 latent feature map**
before token compression. This (a) keeps the operation differentiable and
cheap, (b) needs no camera intrinsics/depth unprojection, (c) avoids a
point-cloud pipeline entirely, and (d) lets the resampler attend to a clean,
task-focused latent. Background is *attenuated* (soft mask, weight 0.2), not
deleted, so global scene context survives.

## 3. Expected DA3 feature shapes

`[B,C,H,W]` · `[B,N,C]` · `[B,V,C,H,W]` · `[B,V,N,C]`. The 4-D
dense-vs-tokenized ambiguity is resolved via `da3_input_dim` (else assumed
dense `[B,C,H,W]`).

## 4. Expected object-mask shapes

`[B,M,Hi,Wi]` · `[B,1,Hi,Wi]` · `[B,V,M,Hi,Wi]` · `[B,V,1,Hi,Wi]`
(also tolerates the `M`/`1` axis omitted). Masks are clamped to `[0,1]`,
bilinearly resized to the DA3 feature grid, and `M` objects reduced by
`mask_reduce` (`max` | `sum_clamp` | `separate_objects`→NotImplementedError).

## 5. Precompute object masks

```bash
python scripts/precompute_object_masks.py \
    --dataset-root <metas> --output-root <mask_root> \
    --text-query-source instruction \
    --segmentation-backend custom_stub        # or langsam / grounded_sam_stub
# layout: <mask_root>/episode_{idx:06d}/<camera>/frame_{t:06d}.pt
```
`langsam`/`grounded_sam_stub` lazily import their deps and raise a clear
install message if absent — **training never imports them**. `custom_stub`
needs nothing (centered-box mask) for end-to-end plumbing tests.
`--debug-vis` writes RGB+mask overlays.

## 6. Precompute DA3 features

```bash
# real DA3 (installed & verified in envs/xvla/):
/shared_work/jack/DA3-XVLA/envs/xvla/bin/python scripts/precompute_da3_features.py \
    --dataset-root <metas> --output-root <da3_root> \
    --da3-backend da3 --model-name depth-anything/DA3-BASE --device cuda
# no-install plumbing test:
python scripts/precompute_da3_features.py --dataset-root <metas> \
    --output-root <da3_root> --da3-backend dummy
# layout: <da3_root>/episode_{idx:06d}/<camera>/frame_{t:06d}.pt
```
`--da3-backend da3_stub|custom` raises a clear `NotImplementedError`
(the seam if DA3 is not wrapped). DA3 is **never imported by training**.

## 7. Enable training

`config.json`:
```json
"geometry_conditioning": {
  "enabled": true,
  "da3_feature_root": "/path/da3_features",
  "object_mask_root": "/path/object_masks",
  "use_da3_latent_segmentation": true,
  "segmentation_mode": "soft_mask",
  "background_weight": 0.2,
  "mask_reduce": "max",
  "num_geometry_tokens": 32,
  "da3_input_dim": null,
  "fusion_position": "before_policy",
  "cross_attention_heads": 8
},
"finetune": {
  "train_backbone": false, "train_soft_prompts": true,
  "train_action_head": true, "train_geometry_modules": true,
  "train_last_n_policy_layers": 0
}
```
Train (geometry features/masks are attached by the dataloader; pass
`--apply_finetune_policy` to freeze per `finetune`):
```bash
python train.py --models <ckpt> --train_metas_path <metas> \
    --output_dir runs/geom --apply_finetune_policy
```
Legacy keys (`mask_background_weight`, `use_object_mask`,
`allow_missing_geometry`, `resampler_type`, `source`) are still accepted as
aliases.

## 8. Dummy debug script

```bash
python scripts/debug_geometry_conditioning.py
```
Backbone-free, CPU, seconds. Cases:
A `DA3LatentSegmenter [B,C,H,W]+[B,M,Hi,Wi]`; B `[B,V,C,H,W]+[B,V,M,Hi,Wi]`;
C full enabled forward (before/after policy) shape == baseline;
D disabled == baseline; E missing masks + `allow_missing_masks=false` → clear
error; F missing masks + `allow_missing_masks=true` → unsegmented + warn.

## 9. Where things happen

| Concern | Location |
|---|---|
| DA3 latent segmentation | `models/geometry_conditioning.py::DA3LatentSegmenter` (inside `SegmentedDA3GeometryConditioner.forward`, before flatten) |
| Geometry token compression | `GeometryTokenResampler` (Perceiver, K=32) |
| Cross-attention fusion | `GeometryCrossAttentionFusion`, inside `SoftPromptedTransformer.forward` at `fusion_position` (before/after/inside policy) |
| Feature/mask loading | `datasets/da3_loader.py::DA3FeatureAttacher` (episode/timestep/camera; legacy + per-camera layouts) |
| Param freezing + summary | `XVLA.apply_finetune_policy` (prints total/trainable/modules) |

## 10. Not implemented yet

- In-model DA3 / segmentation (`da3_source="extractor_stub"`,
  `object_mask_source="segmentation_model_stub"` raise clearly) — offline by design.
- `mask_reduce="separate_objects"` (raises `NotImplementedError`; would change
  the downstream token count).
- LangSAM/Grounded-SAM are interface stubs unless their packages are installed;
  `custom_stub` (centered box) is the no-dependency fallback.
- Attach path covers HDF5 handlers (Calvin/Libero/RoboTwin2/VLABench/robomind/
  droid/AIR); non-HDF5 handlers (x2robot, AGIBOT-lerobot, agiworld) not yet.
- Point-cloud conversion is intentionally **out of scope**.
