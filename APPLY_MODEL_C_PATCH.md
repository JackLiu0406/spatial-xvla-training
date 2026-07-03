# Applying the Model C Pipeline Patch

This branch documents the Model C training pipeline and points to the exact code patch artifact.

The training node had GitHub connector access but no local `gh` CLI or HTTPS git credential, so the full local commit was exported as a compressed patch bundle and uploaded to the existing Hugging Face ablations repository.

Patch artifact:

```text
https://huggingface.co/JackLiu0406/DA3-XVLA-roboreal-ablations/resolve/main/model-c-spatialboost-training-pipeline/model_c_spatialboost_pipeline.patch.gz
```

Checksum file:

```text
https://huggingface.co/JackLiu0406/DA3-XVLA-roboreal-ablations/resolve/main/model-c-spatialboost-training-pipeline/model_c_spatialboost_pipeline.sha256
```

To materialize the full code locally:

```bash
git clone https://github.com/JackLiu0406/spatial-xvla-training.git
cd spatial-xvla-training
git checkout model-c-spatialboost-training-pipeline

curl -L -o /tmp/model_c_spatialboost_pipeline.patch.gz \
  https://huggingface.co/JackLiu0406/DA3-XVLA-roboreal-ablations/resolve/main/model-c-spatialboost-training-pipeline/model_c_spatialboost_pipeline.patch.gz

gunzip -c /tmp/model_c_spatialboost_pipeline.patch.gz > /tmp/model_c_spatialboost_pipeline.patch
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