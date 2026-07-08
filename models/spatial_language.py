"""Spatial-language conditioning for XVLA action tokens.

This module implements two disabled-by-default variants:

* Method A: a post-XVLA action-only spatial refiner.
* Method B: action-slice spatial injection after the final XVLA blocks.

Both variants share the same frozen DA3 + frozen T5 tokenization pathway. The
spatial tokens are never appended to XVLA's input sequence; they are used only
as cross-attention memory for the action-token slice.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .da3_for_geostack import DA3LargeForGeoStack
from .side_stack import compute_world_ray_6d
from .t5_inline import T5InlineEncoder


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(approximate="tanh"),
        nn.Linear(hidden_dim, out_dim),
    )


class ResidualCrossAttention(nn.Module):
    """Plain residual cross-attention: q <- q + MHA(LN(q), LN(kv), LN(kv))."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.kv_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=float(dropout), batch_first=True
        )

    def forward(
        self,
        q_hidden: torch.Tensor,
        kv_hidden: torch.Tensor,
        kv_padding_mask: Optional[torch.Tensor] = None,
        residual_scale: float = 1.0,
        return_update: bool = False,
    ) -> torch.Tensor:
        q = self.q_norm(q_hidden)
        kv = self.kv_norm(kv_hidden)
        out, _ = self.attn(
            q, kv, kv,
            key_padding_mask=kv_padding_mask,
            need_weights=False,
        )
        scaled = out * float(residual_scale)
        updated = q_hidden + scaled
        if return_update:
            return updated, scaled
        return updated


class ResidualSelfAttention(nn.Module):
    """Plain residual self-attention for the 30 action chunk tokens."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=float(dropout), batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        out, _ = self.attn(h, h, h, need_weights=False)
        return x + out


class ResidualMlp(nn.Module):
    """Plain residual MLP: x <- x + MLP(LN(x))."""

    def __init__(self, hidden_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        mlp_h = int(hidden_dim * float(mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_h),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_h, hidden_dim),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


class PerceiverDownsampler(nn.Module):
    """Learned-query perceiver downsampler: tokens [B,N,H] -> [B,K,H]."""

    def __init__(self, num_queries: int, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, int(num_queries), hidden_dim))
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.kv_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, int(num_heads), dropout=float(dropout), batch_first=True
        )
        self.mlp = ResidualMlp(hidden_dim, mlp_ratio=2.0, dropout=dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(
            self.q_norm(q),
            self.kv_norm(tokens),
            self.kv_norm(tokens),
            need_weights=False,
        )
        return self.mlp(q + out)


class LanguageFusionLayer(nn.Module):
    """geo <- geo + CrossAttn(geo, lang), then geo <- geo + MLP(geo)."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.xattn = ResidualCrossAttention(hidden_dim, num_heads, dropout)
        self.mlp = ResidualMlp(hidden_dim, mlp_ratio=4.0, dropout=dropout)

    def forward(
        self,
        geo: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        geo = self.xattn(geo, lang_tokens, kv_padding_mask=lang_padding_mask)
        return self.mlp(geo)


class LanguageFusionStack(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            LanguageFusionLayer(hidden_dim, num_heads, dropout)
            for _ in range(int(depth))
        ])

    def forward(
        self,
        geo: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            geo = layer(geo, lang_tokens, lang_padding_mask)
        return geo


class SpatialLanguageTokenizer(nn.Module):
    """Build task-conditioned DA3 geometry banks for main/left/right views."""

    VIEW_NAMES = ("main", "left", "right")

    def __init__(self, cfg: dict, hidden_dim: int = 1024):
        super().__init__()
        self.cfg = dict(cfg)
        self.hidden_dim = int(hidden_dim)
        self.view_indices = {
            "main": int(cfg.get("main_view", 0)),
            "left": int(cfg.get("left_view", 1)),
            "right": int(cfg.get("right_view", 2)),
        }
        self.view_token_counts = {
            "main": int(cfg.get("main_tokens", 64)),
            "left": int(cfg.get("left_tokens", 48)),
            "right": int(cfg.get("right_tokens", 48)),
        }
        out_layers = tuple(int(i) for i in cfg.get("da3_out_layers", [11, 15, 19, 23]))
        if len(out_layers) != 4:
            raise ValueError(f"spatial_lang.da3_out_layers must have 4 entries, got {out_layers}")
        self.out_layers = out_layers
        # Posed joint multi-view DA3: feed all views together with camera
        # extrinsics+intrinsics so DA3 returns cross-view-consistent per-view
        # geometry. Opt-in; requires the loader to attach extrinsics/intrinsics.
        self.use_posed_da3 = bool(cfg.get("use_posed_da3", False))

        self.da3 = DA3LargeForGeoStack(
            model_name=str(cfg.get("da3_model", "depth-anything/DA3-Large-1.1")),
            out_layers=out_layers,
            da3_input_h=int(cfg.get("da3_input_h", 252)),
            da3_input_w=int(cfg.get("da3_input_w", 336)),
            patch_size=int(cfg.get("da3_patch_size", 14)),
            use_bf16=bool(cfg.get("da3_use_bf16", True)),
        )
        if bool(cfg.get("da3_freeze", True)):
            for p in self.da3.parameters():
                p.requires_grad_(False)

        self.t5 = T5InlineEncoder(
            model_name=str(cfg.get("t5_model_name", "t5-base")),
            use_bf16=bool(cfg.get("t5_use_bf16", True)),
            freeze=bool(cfg.get("t5_freeze", True)),
            max_length=int(cfg.get("t5_max_length", 64)),
        )
        self.t5_projector = nn.Sequential(
            nn.Linear(self.t5.hidden_size, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        c_in = int(self.da3.embed_dim)
        self.layer_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c_in, self.hidden_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
            )
            for _ in range(4)
        ])
        self.layer_embed = nn.Parameter(torch.zeros(4, self.hidden_dim))
        self.layer_fuse = nn.Linear(4 * self.hidden_dim, self.hidden_dim)

        self.view_embed = nn.Embedding(int(cfg.get("num_views", 3)), self.hidden_dim)
        self.pos2d_mlp = _mlp(2, int(cfg.get("pos_mlp_hidden", 256)), self.hidden_dim)
        # Scale-aware ray: append DA3's per-pixel METRIC depth (log) to the
        # Plücker-6 ray so each grid token carries direction AND distance-along-
        # ray = a 3D-grounded descriptor (no explicit point cloud). Requires the
        # posed multi-view path (metric depth comes from GIANT's da3_metric).
        self.use_depth_ray = bool(cfg.get("use_depth_ray", False))
        ray_in = 7 if self.use_depth_ray else 6
        self.ray_mlp = _mlp(ray_in, int(cfg.get("ray_mlp_hidden", 256)), self.hidden_dim)
        # Perceiver bypass (A/B knob): when False, feed the full DA3 grid tokens
        # per view straight to language fusion — perfect 1:1 spatial
        # correspondence, no learned compression. Banks become grid-sized
        # (h*w per view) instead of view_token_counts; downstream is all
        # cross-attention so cost stays linear. Built ONLY when enabled so the
        # bypass arm has zero unused params → DDP can run with
        # find_unused_parameters=False (the fast path).
        self.use_perceiver = bool(cfg.get("use_perceiver", True))

        heads = int(cfg.get("perceiver_heads", 8))
        dropout = float(cfg.get("dropout", 0.0))
        self.perceivers = nn.ModuleDict({
            name: PerceiverDownsampler(k, self.hidden_dim, heads, dropout)
            for name, k in self.view_token_counts.items()
        }) if self.use_perceiver else nn.ModuleDict()
        self.lang_fusers = nn.ModuleDict({
            name: LanguageFusionStack(
                self.hidden_dim,
                int(cfg.get("lang_fusion_heads", 8)),
                depth=int(cfg.get("lang_fusion_layers", 2)),
                dropout=dropout,
            )
            for name in self.VIEW_NAMES
        })

    def reload_pretrained_weights(self) -> None:
        self.da3.reload_pretrained_weights()
        self.t5.reload_pretrained_weights()
        if bool(self.cfg.get("da3_freeze", True)):
            for p in self.da3.parameters():
                p.requires_grad_(False)
        if bool(self.cfg.get("t5_freeze", True)):
            for p in self.t5.parameters():
                p.requires_grad_(False)

    def freeze_backbones(self) -> None:
        if bool(self.cfg.get("da3_freeze", True)):
            self.da3.eval()
            for p in self.da3.parameters():
                p.requires_grad_(False)
        if bool(self.cfg.get("t5_freeze", True)):
            self.t5.eval()
            for p in self.t5.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen DA3/T5 are feature extractors in this run. Keep them in eval
        # mode even when the parent XVLA model is put into train mode, avoiding
        # dropout/stochastic behavior in frozen language/geometry backbones.
        if bool(self.cfg.get("da3_freeze", True)):
            self.da3.eval()
        if bool(self.cfg.get("t5_freeze", True)):
            self.t5.eval()
        return self

    def reset_trainable_parameters(self) -> None:
        for name, module in self.named_modules():
            if name.startswith("da3.") or name.startswith("t5."):
                continue
            if hasattr(module, "reset_parameters") and callable(module.reset_parameters):
                try:
                    module.reset_parameters()
                except Exception:
                    pass
        nn.init.normal_(self.layer_embed, mean=0.0, std=0.02)

    @staticmethod
    def _grid_coords(h_grid: int, w_grid: int, device, dtype) -> torch.Tensor:
        if h_grid <= 1:
            v = torch.zeros(h_grid, device=device, dtype=dtype)
        else:
            v = 2.0 * torch.arange(h_grid, device=device, dtype=dtype) / float(h_grid - 1) - 1.0
        if w_grid <= 1:
            u = torch.zeros(w_grid, device=device, dtype=dtype)
        else:
            u = 2.0 * torch.arange(w_grid, device=device, dtype=dtype) / float(w_grid - 1) - 1.0
        yy, xx = torch.meshgrid(v, u, indexing="ij")
        return torch.stack([xx, yy], dim=-1).reshape(1, h_grid * w_grid, 2)

    def _fuse_da3_layers(self, feats_per_layer, view_slot: int) -> torch.Tensor:
        projected = []
        for li, feat in enumerate(feats_per_layer):
            feat_v = feat[:, view_slot]  # [B, C, h, w]
            B, C, h, w = feat_v.shape
            flat = feat_v.permute(0, 2, 3, 1).reshape(B, h * w, C).contiguous()
            p = self.layer_projectors[li](flat)
            p = p + self.layer_embed[li].to(p.dtype).view(1, 1, -1)
            projected.append(p)
        return self.layer_fuse(torch.cat(projected, dim=-1))

    def _ray6_for_view(
        self,
        ray_dir: torch.Tensor,             # [B, V, 3, h, w]
        view_index: int,
        extrinsics: Optional[torch.Tensor],
    ) -> torch.Tensor:
        ray_v = ray_dir[:, view_index]     # [B, 3, h, w]
        if extrinsics is not None and view_index < extrinsics.shape[1]:
            return compute_world_ray_6d(ray_v, extrinsics[:, view_index])
        origin = torch.zeros_like(ray_v)
        return torch.cat([origin, ray_v], dim=1)

    def forward(
        self,
        image_input: torch.Tensor,              # [B, V, 3, H, W]
        language_instruction,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if image_input.dim() != 5:
            raise ValueError(f"SpatialLanguageTokenizer expects image_input [B,V,3,H,W], got {tuple(image_input.shape)}")
        if language_instruction is None:
            raise ValueError("SpatialLanguageTokenizer requires raw language_instruction list[str].")
        B, V = image_input.shape[:2]

        needed = [self.view_indices[n] for n in self.VIEW_NAMES]
        if max(needed) >= V:
            raise ValueError(f"spatial_lang view index {max(needed)} >= available views {V}")

        # Posed joint multi-view: feed all views TOGETHER with camera pose so
        # DA3 returns cross-view-consistent per-view geometry. Requires both
        # extrinsics and intrinsics in the batch; otherwise falls back to the
        # legacy per-view-independent forward (unposed).
        if self.use_posed_da3 and extrinsics is not None and intrinsics is not None:
            da3_out = self.da3.forward_multi_view(
                image_input, extrinsics=extrinsics, intrinsics=intrinsics
            )
        else:
            da3_out = self.da3.forward_multi_view(image_input)
        feats = da3_out["feats"]        # list[4] of [B,V,C,h,w]
        ray = da3_out["ray"]            # [B,V,3,h,w]
        depth = da3_out.get("depth")    # [B,V,1,h,w] metric, or None
        h_grid, w_grid = int(da3_out["h_grid"]), int(da3_out["w_grid"])
        coords = self._grid_coords(h_grid, w_grid, image_input.device, torch.float32)
        coords_b = coords.expand(B, -1, -1)
        proj_dtype = next(self.layer_fuse.parameters()).dtype
        pos_emb = self.pos2d_mlp(coords_b.to(proj_dtype))

        t5_feat, t5_mask = self.t5(language_instruction)
        lang_tokens = self.t5_projector(t5_feat.to(next(self.t5_projector.parameters()).dtype))
        lang_padding_mask = ~t5_mask

        out: Dict[str, torch.Tensor] = {}
        out["_grid_hw"] = torch.tensor([h_grid, w_grid], device=image_input.device, dtype=torch.long)
        out["_input_hw"] = torch.tensor(
            [image_input.shape[-2], image_input.shape[-1]],
            device=image_input.device,
            dtype=torch.long,
        )
        for name in self.VIEW_NAMES:
            view_idx = self.view_indices[name]
            fused = self._fuse_da3_layers(feats, view_idx)
            ray6 = self._ray6_for_view(ray, view_idx, extrinsics)        # [B,6,h,w]
            if self.use_depth_ray:
                # Append log-metric-depth as a 7th ray channel (scale-aware ray).
                if depth is not None:
                    dv = depth[:, view_idx].float()                      # [B,1,h,w]
                    dv = torch.log(dv.clamp(min=1e-3))
                else:
                    dv = torch.zeros(B, 1, h_grid, w_grid, device=image_input.device)
                ray_planes = torch.cat([ray6, dv.to(ray6.dtype)], dim=1)  # [B,7,h,w]
            else:
                ray_planes = ray6
            c_ray = ray_planes.shape[1]
            ray_flat = ray_planes.permute(0, 2, 3, 1).reshape(B, h_grid * w_grid, c_ray).contiguous()
            ray_emb = self.ray_mlp(ray_flat.to(next(self.ray_mlp.parameters()).dtype)).to(fused.dtype)
            view_ids = torch.full((B,), view_idx, device=image_input.device, dtype=torch.long)
            view_emb = self.view_embed(view_ids).to(fused.dtype).unsqueeze(1)
            spatial = fused + view_emb + pos_emb.to(fused.dtype) + ray_emb
            out[f"{name}_grid"] = spatial
            geo = self.perceivers[name](spatial) if self.use_perceiver else spatial
            out[name] = self.lang_fusers[name](
                geo, lang_tokens.to(geo.dtype), lang_padding_mask=lang_padding_mask
            )
        return out


class SpatialAuxHeads(nn.Module):
    """Auxiliary spatial supervision for Method A/B spatial-language banks.

    Two heads are trained only during supervised training:
      1. endpoint: pooled spatial-language tokens -> final bimanual EE6D action.
      2. heatmap: pre-Perceiver DA3 grid tokens -> projected final EE locations.

    Inference ignores these heads; they exist to keep the spatial branch useful.
    """

    def __init__(self, cfg: dict, hidden_dim: int = 1024, num_actions: int = 30, dim_action: int = 20):
        super().__init__()
        self.cfg = dict(cfg)
        self.hidden_dim = int(hidden_dim)
        self.num_actions = int(num_actions)
        self.dim_action = int(dim_action)
        self.enabled = bool(self.cfg.get("enabled", False))
        self.endpoint_weight = float(self.cfg.get("endpoint_weight", 0.002))
        self.heatmap_weight = float(self.cfg.get("heatmap_weight", 0.02))
        self.target_step = int(self.cfg.get("target_step", -1))
        self.heatmap_sigma = float(self.cfg.get("heatmap_sigma", 1.25))
        self.patch_size = float(self.cfg.get("patch_size", 14))

        self.endpoint_query = nn.Parameter(torch.empty(1, 1, self.hidden_dim))
        self.endpoint_q_norm = nn.LayerNorm(self.hidden_dim)
        self.endpoint_kv_norm = nn.LayerNorm(self.hidden_dim)
        self.endpoint_pool = nn.MultiheadAttention(
            self.hidden_dim,
            int(self.cfg.get("endpoint_heads", 8)),
            batch_first=True,
        )
        mlp_h = int(self.hidden_dim * float(self.cfg.get("endpoint_mlp_ratio", 2.0)))
        self.endpoint_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, mlp_h),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_h, self.dim_action),
        )

        self.heatmap_heads = nn.ModuleDict({
            "main": nn.Linear(self.hidden_dim, 2),
            "left": nn.Linear(self.hidden_dim, 1),
            "right": nn.Linear(self.hidden_dim, 1),
        })
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.endpoint_query, mean=0.0, std=0.02)
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "reset_parameters") and callable(module.reset_parameters):
                try:
                    module.reset_parameters()
                except Exception:
                    pass

    @staticmethod
    def _step_index(step: int, T: int) -> int:
        return step if step >= 0 else max(0, T + step)

    def _endpoint_losses(self, banks: Dict[str, torch.Tensor], action: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens = torch.cat([banks["main"], banks["left"], banks["right"]], dim=1)
        B = tokens.shape[0]
        q = self.endpoint_query.to(tokens.dtype).expand(B, -1, -1)
        pooled, _ = self.endpoint_pool(
            self.endpoint_q_norm(q),
            self.endpoint_kv_norm(tokens),
            self.endpoint_kv_norm(tokens),
            need_weights=False,
        )
        pred = self.endpoint_head(pooled[:, 0]).to(action.dtype)
        idx = self._step_index(self.target_step, action.shape[1])
        target = action[:, idx, :self.dim_action]

        pos_loss = (
            F.mse_loss(pred[:, 0:3], target[:, 0:3]) +
            F.mse_loss(pred[:, 10:13], target[:, 10:13])
        ) * 500.0
        rot_loss = (
            F.mse_loss(pred[:, 3:9], target[:, 3:9]) +
            F.mse_loss(pred[:, 13:19], target[:, 13:19])
        ) * 10.0
        grip_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(pred[:, 9], (target[:, 9] > 0.5).float()) +
            F.binary_cross_entropy_with_logits(pred[:, 19], (target[:, 19] > 0.5).float())
        )
        w = self.endpoint_weight
        return {
            "spatial_aux_endpoint_position_loss": pos_loss * w,
            "spatial_aux_endpoint_rotate6D_loss": rot_loss * w,
            "spatial_aux_endpoint_gripper_loss": grip_loss * w,
        }

    def _project_points(
        self,
        points_world: torch.Tensor,
        view_idx: int,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        h_grid: int,
        w_grid: int,
        image_h: int,
        image_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ext = extrinsics[:, view_idx].to(points_world.dtype)
        K = intrinsics[:, view_idx].to(points_world.dtype)
        R = ext[:, :3, :3]
        t = ext[:, :3, 3]
        cam = torch.bmm(R, points_world.unsqueeze(-1)).squeeze(-1) + t
        z = cam[:, 2].clamp_min(1e-6)
        u = K[:, 0, 0] * cam[:, 0] / z + K[:, 0, 2]
        v = K[:, 1, 1] * cam[:, 1] / z + K[:, 1, 2]
        gx = u / max(float(image_w), 1.0) * float(w_grid) - 0.5
        gy = v / max(float(image_h), 1.0) * float(h_grid) - 0.5
        valid = (
            torch.isfinite(gx) & torch.isfinite(gy) &
            (cam[:, 2] > 1e-5) &
            (gx >= -0.5) & (gx <= float(w_grid) - 0.5) &
            (gy >= -0.5) & (gy <= float(h_grid) - 0.5)
        )
        return torch.stack([gx, gy], dim=-1), valid

    def _gaussian_targets(
        self,
        centers: torch.Tensor,
        valid: torch.Tensor,
        h_grid: int,
        w_grid: int,
    ) -> torch.Tensor:
        B = centers.shape[0]
        yy, xx = torch.meshgrid(
            torch.arange(h_grid, device=centers.device, dtype=centers.dtype),
            torch.arange(w_grid, device=centers.device, dtype=centers.dtype),
            indexing="ij",
        )
        xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1).unsqueeze(0)
        dist2 = (xy[..., 0] - centers[:, None, 0]).pow(2) + (xy[..., 1] - centers[:, None, 1]).pow(2)
        target = torch.exp(-dist2 / max(2.0 * self.heatmap_sigma * self.heatmap_sigma, 1e-6))
        target = target * valid.to(target.dtype).view(B, 1)
        denom = target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return target / denom

    @staticmethod
    def _soft_ce(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        per = -(target * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1)
        valid_f = valid.to(per.dtype)
        return (per * valid_f).sum() / valid_f.sum().clamp_min(1.0)

    def _heatmap_losses(
        self,
        banks: Dict[str, torch.Tensor],
        action: torch.Tensor,
        extrinsics: Optional[torch.Tensor],
        intrinsics: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        zero = action.sum() * 0.0
        if extrinsics is None or intrinsics is None:
            return {"spatial_aux_heatmap_loss": zero}
        if "_grid_hw" not in banks or "_input_hw" not in banks:
            return {"spatial_aux_heatmap_loss": zero}
        h_grid = int(banks["_grid_hw"][0].item())
        w_grid = int(banks["_grid_hw"][1].item())
        image_h = int(banks["_input_hw"][0].item())
        image_w = int(banks["_input_hw"][1].item())
        if h_grid <= 0 or w_grid <= 0:
            return {"spatial_aux_heatmap_loss": zero}

        idx = self._step_index(self.target_step, action.shape[1])
        left_xyz = action[:, idx, 0:3]
        right_xyz = action[:, idx, 10:13]

        losses = []
        specs = (
            ("main", 0, ((left_xyz, 0), (right_xyz, 1))),
            ("left", 1, ((left_xyz, 0),)),
            ("right", 2, ((right_xyz, 0),)),
        )
        for name, view_idx, arms in specs:
            grid_tokens = banks.get(f"{name}_grid", None)
            if grid_tokens is None:
                continue
            logits = self.heatmap_heads[name](grid_tokens).float()  # [B,N,C]
            for points, channel in arms:
                centers, valid = self._project_points(
                    points, view_idx, extrinsics, intrinsics,
                    h_grid, w_grid, image_h, image_w,
                )
                target = self._gaussian_targets(centers, valid, h_grid, w_grid)
                losses.append(self._soft_ce(logits[..., channel], target, valid))
        if not losses:
            return {"spatial_aux_heatmap_loss": zero}
        return {"spatial_aux_heatmap_loss": torch.stack(losses).mean() * self.heatmap_weight}

    def forward(
        self,
        banks: Dict[str, torch.Tensor],
        action: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.enabled:
            return {}
        out = {}
        if self.endpoint_weight > 0.0:
            out.update(self._endpoint_losses(banks, action))
        if self.heatmap_weight > 0.0:
            out.update(self._heatmap_losses(banks, action, extrinsics, intrinsics))
        return out


class SpatialActionInjectionLayer(nn.Module):
    """Method B layer: main x-attn, left/right branches, merge."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.main_xattn = ResidualCrossAttention(hidden_dim, num_heads, dropout)
        self.left_branch = nn.Linear(hidden_dim, hidden_dim)
        self.right_branch = nn.Linear(hidden_dim, hidden_dim)
        self.left_xattn = ResidualCrossAttention(hidden_dim, num_heads, dropout)
        self.right_xattn = ResidualCrossAttention(hidden_dim, num_heads, dropout)
        self.merge_proj = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(
        self,
        h_action: torch.Tensor,
        main_spatial_lang: torch.Tensor,
        left_spatial_lang: torch.Tensor,
        right_spatial_lang: torch.Tensor,
        spatial_scale: float = 1.0,
        return_stats: bool = False,
    ) -> torch.Tensor:
        h0 = h_action
        h_action, main_update = self.main_xattn(
            h_action, main_spatial_lang,
            residual_scale=spatial_scale,
            return_update=True,
        )
        h_left = self.left_branch(h_action)
        h_right = self.right_branch(h_action)
        h_left, left_update = self.left_xattn(
            h_left, left_spatial_lang,
            residual_scale=spatial_scale,
            return_update=True,
        )
        h_right, right_update = self.right_xattn(
            h_right, right_spatial_lang,
            residual_scale=spatial_scale,
            return_update=True,
        )
        merge_update = self.merge_proj(torch.cat([h_left, h_right], dim=-1)) * float(spatial_scale)
        out = h_action + merge_update
        if return_stats:
            return out, {
                "h_action_norm": h0.detach().float().norm(),
                "main": main_update.detach().float().norm(),
                "left": left_update.detach().float().norm(),
                "right": right_update.detach().float().norm(),
                "merge": merge_update.detach().float().norm(),
            }
        return out


class SpatialActionRefinerLayer(nn.Module):
    """Method A layer: action self-attn + Method B injection + MLP."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = ResidualSelfAttention(hidden_dim, num_heads, dropout)
        self.inject = SpatialActionInjectionLayer(hidden_dim, num_heads, dropout)
        self.mlp = ResidualMlp(hidden_dim, mlp_ratio=4.0, dropout=dropout)

    def forward(
        self,
        h_action: torch.Tensor,
        main_spatial_lang: torch.Tensor,
        left_spatial_lang: torch.Tensor,
        right_spatial_lang: torch.Tensor,
        spatial_scale: float = 1.0,
        return_stats: bool = False,
    ) -> torch.Tensor:
        h_action = self.self_attn(h_action)
        if return_stats:
            h_action, stats = self.inject(
                h_action, main_spatial_lang, left_spatial_lang, right_spatial_lang,
                spatial_scale=spatial_scale,
                return_stats=True,
            )
            return self.mlp(h_action), stats
        h_action = self.inject(
            h_action, main_spatial_lang, left_spatial_lang, right_spatial_lang,
            spatial_scale=spatial_scale,
        )
        return self.mlp(h_action)


class SpatialActionRefiner(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, depth: int = 6, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            SpatialActionRefinerLayer(hidden_dim, num_heads, dropout)
            for _ in range(int(depth))
        ])

    def forward(
        self,
        h_action: torch.Tensor,
        banks: Dict[str, torch.Tensor],
        spatial_scale: float = 1.0,
        return_stats: bool = False,
    ) -> torch.Tensor:
        stats_all = []
        for layer in self.layers:
            if return_stats:
                h_action, stats = layer(
                    h_action, banks["main"], banks["left"], banks["right"],
                    spatial_scale=spatial_scale,
                    return_stats=True,
                )
                stats_all.append(stats)
            else:
                h_action = layer(
                    h_action, banks["main"], banks["left"], banks["right"],
                    spatial_scale=spatial_scale,
                )
        if return_stats:
            return h_action, stats_all
        return h_action


__all__ = [
    "SpatialLanguageTokenizer",
    "SpatialAuxHeads",
    "SpatialActionInjectionLayer",
    "SpatialActionRefiner",
]
