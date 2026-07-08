# Model C: DA3/T5 Spatial-Language XVLA Training Pipeline

This branch adds the DA3/T5 spatial-language XVLA implementation used for the
Model C ablation. Model C is a Method B continuation: it starts from a trained
Method B final-6 spatial-injection checkpoint, keeps the pretrained XVLA path
mostly stable, and increases the fixed spatial residual contribution.

## Architecture

Base policy:

- XVLA action expert with 24 transformer blocks.
- Hidden dim 1024, 16 attention heads.
- Input sequence remains unchanged:
  `[action_tokens | vlm_tokens | aux_visual_tokens | soft_prompts] + pos_emb`.
- Spatial tokens are not appended to the XVLA sequence.
- Only the action slice `x[:, :30]` receives DA3/T5 spatial updates.

Action space:

- Output shape is `[B, 30, 20]`.
- This is bimanual EE6D, 30 chunk timesteps, both end-effectors per timestep.
- Left arm: `actions[:, :, 0:10]` = xyz, rot6d, gripper logit.
- Right arm: `actions[:, :, 10:20]` = xyz, rot6d, gripper logit.

Spatial-language pathway:

- Three RGB views: main, left wrist, right wrist.
- DA3 input uses aspect-preserved `252x336`, separate from Florence's `224x224`.
- Frozen DA3 backbone extracts four intermediate feature taps.
- Each tap is projected to 1024, gets a learned layer embedding, then all four
  are concatenated channel-wise and fused back to 1024.
- Spatial metadata is added per DA3 patch token:
  `f_spatial = F_fused + view_embed + pos2d_embed + ray_embed`.
- Per-view Perceiver token budgets:
  main = 96, left wrist = 64, right wrist = 64.
- Frozen T5 encodes the instruction; T5 tokens are projected to 1024.
- Two language cross-attention layers condition each view's geometry tokens.

Method B spatial injection:

- XVLA blocks 0-17 run normally.
- Blocks 18-23 run normally, then the action slice is updated by one spatial
  injection layer.
- Each injection layer applies:
  - main/global spatial cross-attention to all 30 action timestep tokens;
  - left hidden branch attending only to left wrist spatial-language tokens;
  - right hidden branch attending only to right wrist spatial-language tokens;
  - merge projection from 2048 back to 1024.
- No learned gates are used. Spatial contribution is controlled by a fixed
  residual scale.

## Important Files

- `models/spatial_language.py`
  - DA3/T5 spatial-language tokenizer.
  - Per-view Perceivers.
  - language fusion.
  - Method A refiner and Method B injection modules.
  - optional auxiliary endpoint and heatmap heads.
- `models/modeling_xvla.py`
  - wires the spatial-language module into XVLA.
  - Method B injects only into blocks 18-23 and only into `x[:, :30]`.
- `models/configuration_xvla.py`
  - config/defaults for spatial-language XVLA.
- `train.py`
  - optimizer groups, LR schedules, fixed spatial scale schedule, logging,
    auxiliary spatial losses, and custom XVLA-core cosine continuation options.
- `scripts/launch_model_c/*.sh`
  - launchers and queue scripts for Model C and follow-up ablations.

## Model C Recipe

Model C was the best-performing continuation at the time this branch was
created. It resumed from original Method B `ckpt-125000` and trained for 30k
local steps with:

- `--spatial_lang_method final6_injection`
- fixed spatial residual scale `1.5`
- geometry LR `1e-4`
- XVLA core LR `1e-5`
- VLM LR `0`
- DA3/T5 backbones frozen
- batch size 32 per GPU, 4 H200 GPUs
- save every 5000 steps

Launcher:

```bash
SRC=/path/to/spatial-xvla-training \
RUN=/path/to/runs/spatial_lang_xvla \
BASE=/path/to/original_method_b/ckpt-125000 \
META=/path/to/meta_all_ct_3cam_clean.json \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/launch_model_c/launch_model_c_spatialboost_scale1p5_30k.sh
```

At evaluation time, call:

```python
model.set_spatial_residual_scale(1.5)
```

Also sweep `1.0, 1.25, 1.5, 1.75, 2.0` because eval-time spatial scale may
not peak exactly at the training scale.

## Follow-Up Launchers

### High-Spatial Model C Stress Test

Starts from Model C `ckpt-30000` and trains 10k local steps:

- fixed spatial scale `3.0`
- geometry LR `2e-4`
- XVLA core LR `5e-6`
- VLM LR `0`

```bash
SRC=/path/to/spatial-xvla-training \
RUN=/path/to/runs/spatial_lang_xvla \
BASE=/path/to/model_c/ckpt-30000 \
META=/path/to/meta_all_ct_3cam_clean.json \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/launch_model_c/launch_model_c_highspatial_scale3_10k.sh
```

### Early Method B High-Spatial-to-125k

Starts earlier from original Method B `ckpt-15000`, before XVLA is as
converged, and trains 110k local steps to cumulative 125k:

- spatial scale ramps `1.0 -> 3.0` over first 15k local steps
  (cumulative 15k to 30k).
- geometry LR fixed `2e-4`.
- XVLA core LR `1e-4` until local step 15k, then cosine to `1e-5` by local
  step 110k.
- VLM LR fixed `0`.

```bash
SRC=/path/to/spatial-xvla-training \
RUN=/path/to/runs/spatial_lang_xvla \
BASE=/path/to/original_method_b/ckpt-15000 \
META=/path/to/meta_all_ct_3cam_clean.json \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/launch_model_c/launch_method_b_from15k_highspatial_to125k.sh
```

## Queue Scripts

The queue scripts are simple polling launchers. They wait for the expected
final checkpoint of the previous run and verify that the previous `train.py`
process has exited before launching the next run.

```bash
nohup bash scripts/launch_model_c/queue_model_c_highspatial_after_aux.sh \
  > runs/spatial_lang_xvla/_logs/queue_model_c_highspatial_after_aux_nohup.log 2>&1 &

nohup bash scripts/launch_model_c/queue_method_b_from15k_after_model_c.sh \
  > runs/spatial_lang_xvla/_logs/queue_method_b_from15k_after_model_c_nohup.log 2>&1 &
```

Set `RUN`, `CURRENT_RUN`, `LAUNCH`, and `MAIN_PORT` if your directory layout
differs from the defaults.

## Training Health Signals

Logs include:

- `loss`
- `lr_core`, `lr_geom`, `lr_vlm`
- `spatial_scale`
- `sp_main_ratio`
- `gn_geom`, `gn_xvla`, `gn_vlm`

Healthy signs:

- geometry grad norm is nonzero;
- spatial ratio is nonzero;
- no NaN/Inf loss;
- GPU memory remains stable;
- Model C fixed-scale continuation keeps VLM grad at zero when `--fixed_vlm_lr 0`.

## Notes for Evaluation Machines

- The action output is `[B, 30, 20]`, not `[B, 30, 10]`.
- Do not split action tokens by arm. The 30 action tokens are timestep/chunk
  tokens. Arm routing is represented inside each timestep's 20-D action vector.
- The DA3/T5 backbones are expected to stay frozen for these runs.
- Florence preprocessing remains 224x224; DA3 preprocessing is separately
  aspect-preserved at 252x336.
