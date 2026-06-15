# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

"""
Sanity checks for the offline DA3 preprocessing pipeline.

Backbone- and dataset-free: exercises the DA3 wrapper (dummy + missing-install
error) and the feature attacher round-trip with on-disk .pt files.

    python tests/test_da3_pipeline.py
"""

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.da3_wrapper import DA3FeatureExtractor          # noqa: E402
from datasets.da3_loader import (                            # noqa: E402
    DA3FeatureAttacher,
    feature_path,
    dataset_slug,
)


def test_dummy_extractor_shape():
    ext = DA3FeatureExtractor(backend="dummy", image_size=224, feature_dim=128)
    imgs = [np.random.randint(0, 255, (480, 640, 3), np.uint8) for _ in range(3)]
    feats = ext.extract_features(imgs)
    assert feats.shape == (3, 128, 14, 14), feats.shape  # 224/16 = 14
    assert feats.dtype == torch.float32
    print("[ok] dummy extractor ->", tuple(feats.shape))


def test_missing_da3_clear_error():
    try:
        DA3FeatureExtractor(backend="da3")
    except ImportError as e:
        assert "not installed" in str(e).lower()
        print("[ok] missing DA3 raises clear ImportError")
        return
    except RuntimeError:
        # DA3 importable but constructor differs — also acceptable here.
        print("[ok] DA3 importable; constructor seam raised RuntimeError")
        return
    print("[ok] DA3 importable in this env (no error)")


def test_attacher_roundtrip():
    with tempfile.TemporaryDirectory() as root:
        ds, traj, frame = "robotwin2-foo/bar", 3, 7
        path = feature_path(root, ds, traj, frame)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "da3_features": torch.randn(3, 64, 8, 8),
                "object_masks": torch.rand(3, 1, 8, 8),
                "camera_ids": ["cam0", "cam1", "cam2"],
            },
            path,
        )
        att = DA3FeatureAttacher(root, use_object_mask=True)
        sample = {"image_input": torch.zeros(3, 3, 224, 224)}
        att.attach(sample, ds, traj, frame, num_views=3)
        assert sample["da3_features"].shape == (3, 64, 8, 8)
        assert sample["object_masks"].shape == (3, 1, 8, 8)
        # Slug must be filesystem-safe.
        assert "/" not in dataset_slug(ds)
        print("[ok] attacher round-trip (features + masks)")


def test_attacher_missing_policy():
    with tempfile.TemporaryDirectory() as root:
        strict = DA3FeatureAttacher(root, allow_missing=False)
        raised = False
        try:
            strict.attach({}, "ds", 0, 0, num_views=2)
        except FileNotFoundError as e:
            raised = True
            assert "precompute_da3_features" in str(e)
        assert raised, "strict attacher must raise on missing feature"

        lax = DA3FeatureAttacher(root, allow_missing=True, placeholder_dim=5)
        s = {}
        lax.attach(s, "ds", 0, 0, num_views=2)
        assert s["da3_features"].shape[0] == 2
        print("[ok] missing-feature policy (strict raises / lax zeros)")


def main():
    test_dummy_extractor_shape()
    test_missing_da3_clear_error()
    test_attacher_roundtrip()
    test_attacher_missing_policy()
    print("\nAll DA3 pipeline sanity checks passed.")


if __name__ == "__main__":
    main()
