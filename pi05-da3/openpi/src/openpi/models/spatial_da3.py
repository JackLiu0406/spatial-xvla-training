"""DA3 spatial-language branch for pi0.5 (JAX/Flax nnx port of the X-VLA addon).

Faithful reimplementation of the TRAINABLE modules from
`DA3-XVLA-cache/models/spatial_language.py` (H=1024, GIANT C=1536, grid 18x24=432,
perceiver tokens 128/96/96, 7-ch scale-aware ray, ModernBERT language fusion).

The FROZEN DA3 backbone + ModernBERT run offline (features precached); this module
consumes their outputs as arrays and produces per-view "banks" that are cross-attended
into the action-expert's late blocks (see gemma.py `SpatialActionInjection`).

Only the bank BUILDER lives here (nnx, a submodule of Pi0). The injection layer lives
in gemma.py (linen, inside the action-expert scan). Both use identical X-VLA math.

Reference math (verified by the understand-phase spec):
- ResidualCrossAttention: out = q_hidden + scale * MHA(LN_q(q_hidden), LN_kv(kv), LN_kv(kv))
- MHA matches torch nn.MultiheadAttention: separate q/k/v/out Linears w/ bias, 1/sqrt(head_dim).
- GELU is the tanh approximation everywhere; LayerNorm eps=1e-5.
- Perceiver residual adds the RAW learned query (not the normalized one).
- View order everywhere: 0=main/countertop, 1=left wrist, 2=right wrist.
"""

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at

# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def _gelu(x):
    return nnx.gelu(x, approximate=True)  # tanh approximation (matches torch GELU(approximate="tanh"))


class MHACrossAttn(nnx.Module):
    """Multi-head cross-attention matching torch nn.MultiheadAttention math (no residual, no norm)."""

    def __init__(self, dim: int, num_heads: int, *, rngs: nnx.Rngs):
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.k_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.v_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.out_proj = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, q, kv, key_pad_mask=None):
        # q:[b,Lq,d]  kv:[b,Lk,d]  key_pad_mask:[b,Lk] True=pad (ignored)
        h = self.num_heads
        Q = einops.rearrange(self.q_proj(q), "b l (h d) -> b h l d", h=h)
        K = einops.rearrange(self.k_proj(kv), "b l (h d) -> b h l d", h=h)
        V = einops.rearrange(self.v_proj(kv), "b l (h d) -> b h l d", h=h)
        logits = jnp.einsum("bhqd,bhkd->bhqk", Q, K) * (self.head_dim**-0.5)
        if key_pad_mask is not None:
            logits = jnp.where(key_pad_mask[:, None, None, :], jnp.asarray(-1e30, logits.dtype), logits)
        probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(logits.dtype)
        ctx = jnp.einsum("bhqk,bhkd->bhqd", probs, V)
        ctx = einops.rearrange(ctx, "b h q d -> b q (h d)")
        return self.out_proj(ctx)


class ResidualCrossAttn(nnx.Module):
    """Pre-LN residual cross-attention: out = q_hidden + scale * MHA(LN_q(q_hidden), LN_kv(kv))."""

    def __init__(self, dim: int, num_heads: int, *, rngs: nnx.Rngs):
        self.q_norm = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        self.kv_norm = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        self.attn = MHACrossAttn(dim, num_heads, rngs=rngs)

    def __call__(self, q_hidden, kv_hidden, key_pad_mask=None, residual_scale: float = 1.0):
        q = self.q_norm(q_hidden)
        kv = self.kv_norm(kv_hidden)
        out = self.attn(q, kv, key_pad_mask=key_pad_mask)
        return q_hidden + residual_scale * out


class ResidualMlp(nnx.Module):
    """Pre-LN residual MLP: x + Linear2(gelu(Linear1(LN(x))))."""

    def __init__(self, dim: int, mlp_ratio: float, *, rngs: nnx.Rngs):
        hidden = int(dim * mlp_ratio)
        self.norm = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        self.fc1 = nnx.Linear(dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, dim, rngs=rngs)

    def __call__(self, x):
        return x + self.fc2(_gelu(self.fc1(self.norm(x))))


class ProjLN(nnx.Module):
    """Linear(in->H) -> gelu -> Linear(H->H) -> LayerNorm(H). Used for layer projectors & t5_projector."""

    def __init__(self, in_dim: int, dim: int, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(in_dim, dim, rngs=rngs)
        self.fc2 = nnx.Linear(dim, dim, rngs=rngs)
        self.norm = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)

    def __call__(self, x):
        return self.norm(self.fc2(_gelu(self.fc1(x))))


class Mlp2(nnx.Module):
    """Linear(in->hidden) -> gelu -> Linear(hidden->out). Used for ray_mlp & pos2d_mlp (no LN)."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(in_dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, out_dim, rngs=rngs)

    def __call__(self, x):
        return self.fc2(_gelu(self.fc1(x)))


class PerceiverDownsampler(nnx.Module):
    """432 grid tokens -> K learned-query tokens (single cross-attn + residual MLP)."""

    def __init__(self, dim: int, num_queries: int, num_heads: int, *, query_std: float = 0.02, rngs: nnx.Rngs):
        key = rngs.params()
        self.query = nnx.Param(jax.random.normal(key, (1, num_queries, dim)) * query_std)
        self.xattn = ResidualCrossAttn(dim, num_heads, rngs=rngs)
        self.mlp = ResidualMlp(dim, mlp_ratio=2.0, rngs=rngs)

    def __call__(self, tokens):
        b = tokens.shape[0]
        q = jnp.broadcast_to(self.query.value, (b, *self.query.value.shape[1:]))
        z = self.xattn(q, tokens, residual_scale=1.0)  # residual adds RAW q (matches X-VLA)
        return self.mlp(z)


class LanguageFusionStack(nnx.Module):
    """N x [cross-attn(bank, lang) + residual-MLP], with language padding mask."""

    def __init__(self, dim: int, depth: int, num_heads: int, *, rngs: nnx.Rngs):
        self.layers = [
            (ResidualCrossAttn(dim, num_heads, rngs=rngs), ResidualMlp(dim, mlp_ratio=4.0, rngs=rngs))
            for _ in range(depth)
        ]

    def __call__(self, geo, lang_tokens, lang_pad_mask):
        for xattn, mlp in self.layers:
            geo = xattn(geo, lang_tokens, key_pad_mask=lang_pad_mask, residual_scale=1.0)
            geo = mlp(geo)
        return geo


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


def compute_world_ray_6d(ray_local, ext_w2c):
    """ray_local [b,3,h,w] cam-local unit dir; ext_w2c [b,4,4] OpenCV world->cam.

    Returns [b,6,h,w] = concat([origin_world(camera center), dir_world]).
    """
    R_w2c = ext_w2c[:, :3, :3]  # [b,3,3]
    t_w2c = ext_w2c[:, :3, 3]  # [b,3]
    R_c2w = jnp.swapaxes(R_w2c, -1, -2)
    pos_world = -jnp.einsum("bij,bj->bi", R_c2w, t_w2c)  # [b,3] camera center in world
    b, _, h, w = ray_local.shape
    dir_world = jnp.einsum("bij,bjk->bik", R_c2w, ray_local.reshape(b, 3, h * w)).reshape(b, 3, h, w)
    origin = jnp.broadcast_to(pos_world[:, :, None, None], (b, 3, h, w))
    return jnp.concatenate([origin, dir_world], axis=1)  # [b,6,h,w]


def _grid_coords(h: int, w: int):
    v = 2.0 * jnp.arange(h) / (h - 1) - 1.0
    u = 2.0 * jnp.arange(w) / (w - 1) - 1.0
    yy, xx = jnp.meshgrid(v, u, indexing="ij")
    return jnp.stack([xx, yy], axis=-1).reshape(1, h * w, 2)  # [1,432,2] (x=u, y=v), row-major


# ---------------------------------------------------------------------------
# bank builder
# ---------------------------------------------------------------------------

_VIEWS = (("main", 0, 128), ("left", 1, 96), ("right", 2, 96))
_VIEWS_2 = (("main", 0, 128), ("wrist", 1, 96))


def view_specs(num_views: int):
    """Per-view (name, index, num_perceiver_tokens) specs for a given camera count.

    3 views = roboreal/aloha (countertop/left/right). 2 views = LIBERO (main/wrist)."""
    if num_views == 3:
        return _VIEWS
    if num_views == 2:
        return _VIEWS_2
    raise ValueError(f"SpatialBankBuilder supports num_views in (2, 3); got {num_views}")


class SpatialBankBuilder(nnx.Module):
    """Cached DA3 (feats[/ray/depth/extrinsics]) + ModernBERT feats -> per-view banks.

    posed=True builds a world-frame scale-aware ray embedding (needs ray/depth/extrinsics).
    posed=False (unposed, e.g. LIBERO) drops the ray embedding entirely — no geometry inputs."""

    def __init__(
        self,
        *,
        hidden_dim: int = 1024,
        da3_channels: int = 1536,
        num_layers: int = 4,
        num_views: int = 3,
        posed: bool = True,
        grid_hw: tuple[int, int] = (18, 24),
        lang_dim: int = 1024,  # ModernBERT-large last_hidden width (768) -> set by config
        num_heads: int = 8,
        lang_fusion_depth: int = 2,
        perceiver_query_std: float = 0.02,
        bank_token_embed: bool = False,
        rngs: nnx.Rngs,
    ):
        H = hidden_dim
        self.hidden_dim = H
        self.num_layers = num_layers
        self.grid_hw = grid_hw
        self.posed = posed
        self._views = view_specs(num_views)
        # (a) per-tap projectors + layer embed + fuse
        self.layer_projectors = [ProjLN(da3_channels, H, rngs=rngs) for _ in range(num_layers)]
        self.layer_embed = nnx.Param(jax.random.normal(rngs.params(), (num_layers, H)) * 0.02)
        self.layer_fuse = nnx.Linear(num_layers * H, H, rngs=rngs)
        # (b) scale-aware ray (Plucker-6 + log-depth = 7) — posed path only
        self.ray_mlp = Mlp2(7, 256, H, rngs=rngs) if posed else None
        # (c) 2D grid pos + per-view embedding
        self.pos2d_mlp = Mlp2(2, 256, H, rngs=rngs)
        self.view_embed = nnx.Embed(num_views, H, rngs=rngs)
        # (d) language projector (ModernBERT feat -> H)
        self.t5_projector = ProjLN(lang_dim, H, rngs=rngs)
        # (e) per-view perceiver + language fusion
        self.perceivers = {
            name: PerceiverDownsampler(H, k, num_heads, query_std=perceiver_query_std, rngs=rngs)
            for name, _, k in self._views
        }
        self.lang_fusers = {name: LanguageFusionStack(H, lang_fusion_depth, num_heads, rngs=rngs) for name, _, _ in self._views}
        # (f) v2: learned per-token embedding added to each view's FINAL bank tokens. Guarantees
        # persistent cross-token diversity — the quantity that drives softmax gradients to the
        # injection's Q/K (shared content cancels in the softmax jacobian, so without this the
        # attention pattern barely trains; measured ~1000x slower than V/out in v1).
        self.bank_token_embeds = (
            {name: nnx.Param(jax.random.normal(rngs.params(), (1, k, H)) * 0.05) for name, _, k in self._views}
            if bank_token_embed
            else None
        )

    def _fuse_layers(self, feats_v):
        # feats_v: [b, num_layers, C, h, w]  -> [b, 432, H]
        b, L, C, h, w = feats_v.shape
        parts = []
        for li in range(self.num_layers):
            flat = einops.rearrange(feats_v[:, li], "b c h w -> b (h w) c")  # row-major
            p = self.layer_projectors[li](flat) + self.layer_embed.value[li][None, None, :]
            parts.append(p)
        return self.layer_fuse(jnp.concatenate(parts, axis=-1))

    def _ray7(self, ray_v, depth_v, ext_v):
        # ray_v [b,3,h,w], depth_v [b,1,h,w], ext_v [b,4,4] -> [b,432,7]
        ray6 = compute_world_ray_6d(ray_v, ext_v)  # [b,6,h,w]
        logd = jnp.log(jnp.clip(depth_v.astype(jnp.float32), a_min=1e-3)).astype(ray6.dtype)  # [b,1,h,w]
        ray7 = jnp.concatenate([ray6, logd], axis=1)  # [b,7,h,w]
        return einops.rearrange(ray7, "b c h w -> b (h w) c")

    def __call__(self, feats, ray, depth, extrinsics, lang_feat, lang_mask):
        # feats [b,L,V,C,h,w]; lang_feat [b,Lt,lang_dim]; lang_mask [b,Lt] True=real token.
        # posed path also uses ray [b,V,3,h,w], depth [b,V,1,h,w], extrinsics [b,V,4,4];
        # unposed path ignores them (may be None) and drops the ray embedding.
        h, w = self.grid_hw
        pos_emb = self.pos2d_mlp(_grid_coords(h, w).astype(feats.dtype))  # [1,432,H]
        lang_tokens = self.t5_projector(lang_feat)  # [b,Lt,H]
        lang_pad = jnp.logical_not(lang_mask)  # True=pad
        banks = {}
        for name, vidx, _k in self._views:
            fused = self._fuse_layers(feats[:, :, vidx])  # [b,432,H]
            view_emb = self.view_embed(jnp.asarray(vidx))[None, None, :]  # [1,1,H]
            spatial = fused + view_emb + pos_emb  # [b,432,H]
            if self.posed:
                ray_flat = self._ray7(ray[:, vidx], depth[:, vidx], extrinsics[:, vidx])  # [b,432,7]
                spatial = spatial + self.ray_mlp(ray_flat.astype(feats.dtype))  # + ray_emb
            geo = self.perceivers[name](spatial)  # [b,K,H]
            bank = self.lang_fusers[name](geo, lang_tokens, lang_pad)  # [b,K,H]
            if self.bank_token_embeds is not None:
                bank = bank + self.bank_token_embeds[name].value
            banks[name] = bank
        return banks
