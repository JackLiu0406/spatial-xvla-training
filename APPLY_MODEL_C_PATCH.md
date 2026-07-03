# Applying the Model C Pipeline Patch

This branch contains the exact Model C training pipeline as a compressed patch bundle because the training node did not have a GitHub CLI or HTTPS credential for a direct `git push`.

To materialize the full branch locally:

```bash
git clone https://github.com/JackLiu0406/spatial-xvla-training.git
cd spatial-xvla-training
git checkout model-c-spatialboost-training-pipeline

cat model_c_patch_chunks/part_* > /tmp/model_c_spatialboost_pipeline.patch.gz.b64
base64 -d /tmp/model_c_spatialboost_pipeline.patch.gz.b64 \
  | gunzip > /tmp/model_c_spatialboost_pipeline.patch
git apply /tmp/model_c_spatialboost_pipeline.patch
```

The patch applies one commit equivalent to:

```text
4905f24 Add Model C spatial-language XVLA training pipeline
```

It updates:

- `train.py`
- `models/modeling_xvla.py`
- `models/configuration_xvla.py`
- adds `models/spatial_language.py`
- adds `scripts/launch_model_c/README.md`
- adds Model C launch and queue scripts under `scripts/launch_model_c/`

After applying the patch, read:

```text
scripts/launch_model_c/README.md
```

for the architecture summary, action-space details, and launch commands.