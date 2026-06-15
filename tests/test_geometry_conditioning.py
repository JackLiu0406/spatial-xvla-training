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
Sanity checks for Segmented DA3 Geometry Conditioning.

These tests are intentionally backbone-free: they exercise the policy head
(`SoftPromptedTransformer`) and the geometry modules directly with dummy
tensors, so they run in seconds on CPU without Florence2 weights or a dataset.

Run directly:
    python tests/test_geometry_conditioning.py
or with pytest:
    pytest tests/test_geometry_conditioning.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.configuration_xvla import (  # noqa: E402
    default_geometry_conditioning,
    default_finetune,
)
from models.transformer import SoftPromptedTransformer  # noqa: E402
from models.geometry_conditioning import (  # noqa: E402
    SegmentedDA3GeometryConditioner,
    PerceiverResampler,
    GeometryCrossAttentionFusion,
)

torch.manual_seed(0)

# Small dims so the test is fast.
B, H = 2, 64
T_VLM, T_AUX = 5, 4
NUM_ACTIONS, DIM_ACTION, DIM_PROPRIO, DIM_TIME = 3, 14, 14, 16
DA3_C, DA3_H, DA3_W = 96, 8, 8
K = 8


def _make_transformer(geometry_cfg=None):
    return SoftPromptedTransformer(
        hidden_size=H,
        multi_modal_input_size=H,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        num_domains=3,
        dim_action=DIM_ACTION,
        dim_propio=DIM_PROPRIO,
        dim_time=DIM_TIME,
        len_soft_prompts=4,
        max_len_seq=128,
        use_hetero_proj=False,
        geometry_conditioning=geometry_cfg,
    )


def _dummy_policy_inputs():
    return dict(
        domain_id=torch.zeros(B, dtype=torch.long),
        vlm_features=torch.randn(B, T_VLM, H),
        aux_visual_inputs=torch.randn(B, T_AUX, H),
        action_with_noise=torch.randn(B, NUM_ACTIONS, DIM_ACTION),
        proprio=torch.randn(B, DIM_PROPRIO),
        t=torch.rand(B),
    )


def test_baseline_unchanged_when_disabled():
    """geometry_conditioning disabled -> identical I/O contract as upstream."""
    tr = _make_transformer(geometry_cfg=None).eval()
    out = tr(**_dummy_policy_inputs())
    assert out.shape == (B, NUM_ACTIONS, DIM_ACTION), out.shape
    assert tr.geometry_fusion is None
    print("[ok] baseline disabled path -> action shape", tuple(out.shape))


def test_conditioner_dense_with_mask():
    """Dense [B,C,H,W] DA3 + [B,1,H,W] mask -> [B,K,H] geometry tokens."""
    cfg = default_geometry_conditioning()
    cfg.update(enabled=True, num_geometry_tokens=K, da3_input_dim=DA3_C)
    cond = SegmentedDA3GeometryConditioner(cfg, hidden_dim=H).eval()

    da3 = torch.randn(B, DA3_C, DA3_H, DA3_W)
    mask = torch.rand(B, 1, DA3_H, DA3_W)
    geom = cond(da3, mask)
    assert geom.shape == (B, K, H), geom.shape

    # Mask at a different resolution should be bilinearly resized.
    mask_big = torch.rand(B, 1, DA3_H * 4, DA3_W * 4)
    assert cond(da3, mask_big).shape == (B, K, H)
    print("[ok] dense conditioner + mask -> geometry tokens", tuple(geom.shape))


def test_conditioner_shape_variants():
    """[B,N,C], [B,V,N,C], [B,V,C,H,W] all reduce to [B,K,H]."""
    cfg = default_geometry_conditioning()
    cfg.update(enabled=True, num_geometry_tokens=K, da3_input_dim=DA3_C,
               allow_missing_masks=True)  # no masks here -> pass-through
    cond = SegmentedDA3GeometryConditioner(cfg, hidden_dim=H).eval()

    assert cond(torch.randn(B, 20, DA3_C)).shape == (B, K, H)
    assert cond(torch.randn(B, 3, 12, DA3_C)).shape == (B, K, H)
    assert cond(torch.randn(B, 2, DA3_C, DA3_H, DA3_W)).shape == (B, K, H)
    print("[ok] tokenized / multi-view shape variants")


def test_adaptive_pool_resampler():
    cfg = default_geometry_conditioning()
    cfg.update(enabled=True, num_geometry_tokens=K, da3_input_dim=DA3_C,
               resampler_type="adaptive_pool", allow_missing_masks=True)
    cond = SegmentedDA3GeometryConditioner(cfg, hidden_dim=H).eval()
    assert cond(torch.randn(B, DA3_C, DA3_H, DA3_W)).shape == (B, K, H)
    print("[ok] adaptive_pool resampler fallback")


def test_fusion_module_shapes():
    fusion = GeometryCrossAttentionFusion(dim=H, num_heads=4, num_layers=2).eval()
    pol = torch.randn(B, 7, H)
    geom = torch.randn(B, K, H)
    out = fusion(pol, geom)
    assert out.shape == pol.shape, out.shape
    res = PerceiverResampler(dim=H, num_queries=K, num_heads=4)(torch.randn(B, 30, H))
    assert res.shape == (B, K, H)
    print("[ok] fusion + resampler module shapes")


def test_enabled_forward_matches_baseline_shape():
    """Enabled geometry forward must keep the action output shape identical."""
    cfg = default_geometry_conditioning()
    cfg.update(enabled=True, num_geometry_tokens=K, da3_input_dim=DA3_C,
               debug_shapes=True)

    base_out = _make_transformer(None).eval()(**_dummy_policy_inputs())

    for pos in ("before_policy", "inside_policy", "after_policy"):
        c = dict(cfg, fusion_position=pos)
        tr = _make_transformer(c).eval()
        cond = SegmentedDA3GeometryConditioner(c, hidden_dim=H).eval()
        geom = cond(torch.randn(B, DA3_C, DA3_H, DA3_W),
                    torch.rand(B, 1, DA3_H, DA3_W))
        out = tr(**_dummy_policy_inputs(), geometry_tokens=geom)
        assert out.shape == base_out.shape, (pos, out.shape, base_out.shape)
        print(f"[ok] fusion_position={pos} -> action shape {tuple(out.shape)}")


def test_finetune_defaults():
    ft = default_finetune()
    assert ft["train_backbone"] is False
    assert ft["train_geometry_modules"] is True
    print("[ok] finetune defaults are baseline-compatible")


def main():
    test_baseline_unchanged_when_disabled()
    test_conditioner_dense_with_mask()
    test_conditioner_shape_variants()
    test_adaptive_pool_resampler()
    test_fusion_module_shapes()
    test_enabled_forward_matches_baseline_shape()
    test_finetune_defaults()
    print("\nAll geometry-conditioning sanity checks passed.")


if __name__ == "__main__":
    main()
