# Model C Spatial-Language XVLA Training Pipeline

This branch carries the Model C DA3/T5 spatial-language XVLA training pipeline.

The exact local code commit is available as a compressed patch artifact because this training node has GitHub connector access but no local `gh` CLI or HTTPS git credential for a direct source push. See `APPLY_MODEL_C_PATCH.md` for the one-command path to materialize the full code changes from the Hugging Face artifact.

Patch artifact:

```text
https://huggingface.co/JackLiu0406/DA3-XVLA-roboreal-ablations/resolve/main/model-c-spatialboost-training-pipeline/model_c_spatialboost_pipeline.patch.gz
```

## Model Summary

Model C is a Method B continuation of the DA3/T5 spatial-language XVLA model:

- Base XVLA action expert: 24 transformer blocks, hidden dim 1024, 16 heads.
- Original token sequence remains `[action | vlm | aux_visual | soft_prompts] + pos_emb`.
- Spatial tokens are not appended to the XVLA sequence.
- Method B injects spatial information only into the action slice `x[:, :30]` after blocks 18-23.
- DA3/T5 backbones are frozen.
- Per-view spatial tokens: main 96, left wrist 64, right wrist 64.
- DA3 input is aspect-preserved `252x336`; Florence/VLM preprocessing remains `224x224`.

## Action Space

The output action shape is `[B, 30, 20]`, not `[B, 30, 10]`.

Each of the 30 action tokens is a timestep/action-chunk token. Each timestep encodes both end-effectors:

- Left arm: `actions[:, :, 0:10]` = xyz position, 6D rotation, gripper logit.
- Right arm: `actions[:, :, 10:20]` = xyz position, 6D rotation, gripper logit.

Do not split the 30 tokens into left/right token groups. Left/right routing happens through hidden branches inside the spatial injection module.

## Model C Training Recipe

Model C resumes from original Method B `ckpt-125000` and trains 30k local steps:

```bash
--spatial_lang_method final6_injection
--spatial_scale_start 1.5
--spatial_scale_end 1.5
--spatial_scale_ramp_until 0
--fixed_geometry_lr 1e-4
--fixed_xvla_core_lr 1e-5
--fixed_vlm_lr 0
--batch_size 32
--iters 30000
--save_interval 5000
```

At evaluation time, call:

```python
model.set_spatial_residual_scale(1.5)
```

Also sweep eval-time spatial scale `1.0, 1.25, 1.5, 1.75, 2.0`.

## Files Added by the Patch

The patch updates:

- `train.py`
- `models/modeling_xvla.py`
- `models/configuration_xvla.py`
- `models/spatial_language.py`
- `scripts/launch_model_c/README.md`
- `scripts/launch_model_c/launch_model_c_spatialboost_scale1p5_30k.sh`
- `scripts/launch_model_c/launch_model_c_highspatial_scale3_10k.sh`
- `scripts/launch_model_c/launch_method_b_from15k_highspatial_to125k.sh`
- `scripts/launch_model_c/queue_model_c_highspatial_after_aux.sh`
- `scripts/launch_model_c/queue_method_b_from15k_after_model_c.sh`

See `APPLY_MODEL_C_PATCH.md` for the exact apply command.