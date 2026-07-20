# Copyright 2024 Big Vision Authors.
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

"""Gemma adaptation for Pi, taken from big_vision.

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")
"""

from collections.abc import Sequence
import dataclasses
from typing import Literal, TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

PALIGEMMA_VOCAB_SIZE = 257_152


@dataclasses.dataclass
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)


Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]


def get_config(variant: Variant) -> Config:
    """Returns config for specified gemma variant."""
    if variant == "dummy":
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        )
    if variant == "gemma_300m":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b_lora":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=16.0), "ffn": lora.LoRAConfig(rank=16, alpha=16.0)},
        )
    if variant == "gemma_300m_lora":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=32.0), "ffn": lora.LoRAConfig(rank=32, alpha=32.0)},
        )
    raise ValueError(f"Unknown variant: {variant}")


@at.typecheck
class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x, cond):
        dtype = x.dtype  # original dtype, could be half-precision
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)  # compute variance in float32
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))  # compute normalization in float32
        if cond is None:
            # regular RMSNorm
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            normed_inputs = normed_inputs * (
                1 + scale
            )  # scale by learned parameter in float32 (matches Flax implementation)
            return normed_inputs.astype(dtype), None  # return in original dtype

        # adaptive RMSNorm
        modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
        scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
        normed_inputs = normed_inputs * (1 + scale) + shift  # scale and shift in float32
        return normed_inputs.astype(dtype), gate


@at.typecheck
class Embedder(nn.Module):
    """Embedder module."""

    vocab_size: int
    embed_dim: int

    def setup(self):
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
        )

    def encode(self, x):
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x):
        return jnp.dot(x, self.input_embedding_table.T)


@at.typecheck
class Attention(nn.Module):
    """Attention module."""

    configs: Sequence[Config]

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache):
        # all experts must share the same head dim, num heads, and num kv heads for self-attention to work
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        dtype = next(x.dtype for x in xs if x is not None)  # original dtype, could be half-precision

        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            if config.num_kv_heads == config.num_heads:
                qkv_einsum = lora.Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:
                q_einsum = lora.Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=config.lora_configs.get("attn"),
                )
                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))

        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        q = _apply_rope(q, positions=positions)
        q *= self.configs[0].head_dim ** -0.5

        k = _apply_rope(k, positions=positions)

        # should still be half-precision here (if input was half-precision)
        assert q.dtype == k.dtype == v.dtype == dtype

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)

        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)

        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        # big_neg = jnp.finfo(logits.dtype).min
        big_neg = -2.3819763e38  # See gemma/modules.py
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = lora.Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                start = end
            else:
                out.append(None)

        return out, (k, v)


@at.typecheck
class FeedForward(nn.Module):
    """Feed forward module."""

    features: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # original dtype, could be half-precision
        w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        ).astype(dtype)
        ff_gate = jnp.dot(x, w_gating[0])
        gate_value = nn.gelu(ff_gate)

        ff1 = jnp.dot(x, w_gating[1])
        activations = gate_value * ff1

        w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        ).astype(dtype)
        outputs = jnp.dot(activations, w_linear)
        assert outputs.dtype == dtype
        return outputs


class _LinenCrossAttn(nn.Module):
    """Pre-LN cross-attention. out_proj is zero-init by default (identity at step 0); out_init_std>0
    gives it a small random init so q/k/v receive gradient from step 0 (with zero-init their grads
    are zero until out_proj grows off zero, LoRA-style bootstrap)."""

    d_model: int
    num_heads: int
    out_init_std: float = 0.0
    logit_gain: bool = False  # learnable per-head gain on the attention logits (v2 fix)
    logit_gain_init: float = 1.0  # initial gain value; >1 = sharper attention at init (bigger softmax
    # jacobian => stronger, more consistent Q/K gradients; uniform attention scales them by ~1/K)

    @nn.compact
    def __call__(self, q_hidden, kv):
        hd = self.d_model // self.num_heads
        q = nn.LayerNorm(epsilon=1e-5, name="q_norm")(q_hidden)
        kvn = nn.LayerNorm(epsilon=1e-5, name="kv_norm")(kv)
        Q = einops.rearrange(nn.Dense(self.d_model, name="q_proj")(q), "b l (h d) -> b h l d", h=self.num_heads)
        K = einops.rearrange(nn.Dense(self.d_model, name="k_proj")(kvn), "b l (h d) -> b h l d", h=self.num_heads)
        V = einops.rearrange(nn.Dense(self.d_model, name="v_proj")(kvn), "b l (h d) -> b h l d", h=self.num_heads)
        logits = jnp.einsum("bhqd,bhkd->bhqk", Q, K) * (hd**-0.5)
        if self.logit_gain:
            # exp-parameterized per-head gain, init 1.0; multiplies dL/dQ,K so the attention
            # pattern can actually train (v1 measured Q/K ~1000x slower than V/out).
            log_gain = self.param(
                "log_gain", nn.initializers.constant(jnp.log(self.logit_gain_init)), (self.num_heads,)
            )
            logits = logits * jnp.exp(log_gain)[None, :, None, None].astype(logits.dtype)
        probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(Q.dtype)
        ctx = einops.rearrange(jnp.einsum("bhqk,bhkd->bhqd", probs, V), "b h q d -> b q (h d)")
        out_init = nn.initializers.normal(stddev=self.out_init_std) if self.out_init_std > 0 else nn.initializers.zeros
        return nn.Dense(
            self.d_model, name="out_proj", kernel_init=out_init, bias_init=nn.initializers.zeros
        )(ctx)


def spatial_bank_names(num_views: int) -> tuple[str, ...]:
    """Bank dict keys the injection expects, per camera count. Must match spatial_da3.view_specs."""
    if num_views == 3:
        return ("main", "left", "right")
    if num_views == 2:
        return ("main", "wrist")
    raise ValueError(f"SpatialActionInjection supports num_views in (2, 3); got {num_views}")


class SpatialActionInjection(nn.Module):
    """X-VLA Method-B injection (main x-attn -> per-view branches -> merge), scaled residual.

    init_std=0 (default): output projections zero-init => returns 0 (exact identity at step 0, pi05_base
    reproduced bit-exactly). init_std>0: small normal init on the xattn out_projs and merge_proj => a
    small nonzero spatial delta from step 0 and immediate gradient flow to all injection weights.
    Returns the DELTA to add to the action-token stream (external per-layer gate selects it).

    num_views=3 (roboreal/aloha): main -> {left,right} branches -> concat-merge.
    num_views=2 (LIBERO): main -> {wrist} branch -> merge (single-branch, half-width merge input).
    """

    d_model: int
    num_heads: int = 8
    spatial_scale: float = 2.0
    init_std: float = 0.0
    logit_gain: bool = False
    logit_gain_init: float = 1.0
    num_views: int = 3

    @nn.compact
    def __call__(self, h0, banks):
        s = self.spatial_scale
        std = self.init_std
        g = self.logit_gain
        gi = self.logit_gain_init
        merge_init = nn.initializers.normal(stddev=std) if std > 0 else nn.initializers.zeros
        h1 = h0 + s * _LinenCrossAttn(self.d_model, self.num_heads, std, g, gi, name="main_xattn")(h0, banks["main"])
        if self.num_views == 2:
            hW = nn.Dense(self.d_model, name="wrist_branch")(h1)
            hW = hW + s * _LinenCrossAttn(self.d_model, self.num_heads, std, g, gi, name="wrist_xattn")(hW, banks["wrist"])
            merge = nn.Dense(
                self.d_model, name="merge_proj", kernel_init=merge_init, bias_init=nn.initializers.zeros
            )(hW)
        else:
            hL = nn.Dense(self.d_model, name="left_branch")(h1)
            hR = nn.Dense(self.d_model, name="right_branch")(h1)
            hL = hL + s * _LinenCrossAttn(self.d_model, self.num_heads, std, g, gi, name="left_xattn")(hL, banks["left"])
            hR = hR + s * _LinenCrossAttn(self.d_model, self.num_heads, std, g, gi, name="right_xattn")(hR, banks["right"])
            merge = nn.Dense(
                self.d_model, name="merge_proj", kernel_init=merge_init, bias_init=nn.initializers.zeros
            )(jnp.concatenate([hL, hR], axis=-1))
        out = h1 + s * merge
        return out - h0


@at.typecheck
class Block(nn.Module):
    """Transformer block."""

    configs: tuple[Config, ...]

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    spatial_scale: float = 2.0
    spatial_init_std: float = 0.0
    spatial_logit_gain: bool = False
    spatial_logit_gain_init: float = 1.0
    spatial_num_views: int = 3

    @nn.compact
    def __call__(  # noqa: FBT002
        self, xs, kv_cache, positions, attn_mask, adarms_cond, deterministic=True, banks=None, layer_gate=None
    ):
        xs = sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        attn = Attention(configs=self.configs, name="attn")

        pre_attn = []
        gates = []
        for i, x in enumerate(xs):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
            pre_attn.append(x)
            gates.append(gate if x is not None else None)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        out = []
        gates = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
                x = lora.FeedForward(  # noqa: PLW2901
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=_name("mlp", i),
                    lora_config=config.lora_configs.get("ffn"),
                )(x)
            out.append(x)
            gates.append(gate if x is not None else None)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        # ---- DA3 spatial injection on the SUFFIX (action-expert, expert 1) stream ONLY ----
        # layer_gate is a per-layer scalar (0 for early blocks, 1 for the last N); at init the
        # injection delta is 0 (zero-init output) so this is an exact no-op regardless of the gate.
        if banks is not None and len(xs) > 1 and xs[1] is not None:
            suf = xs[1]
            delta = SpatialActionInjection(
                d_model=self.configs[1].width,
                spatial_scale=self.spatial_scale,
                init_std=self.spatial_init_std,
                logit_gain=self.spatial_logit_gain,
                logit_gain_init=self.spatial_logit_gain_init,
                num_views=self.spatial_num_views,
                name="spatial_inject_1",
            )(suf, banks)
            # cast back to the suffix (carry) dtype: the injection runs in fp32 but the scan carry is bf16
            xs = [xs[0], (suf + layer_gate.astype(delta.dtype) * delta).astype(suf.dtype)]
            xs = sharding.activation_sharding_constraint(xs)

        return xs, kv_cache


KVCache: TypeAlias = tuple[at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]]


@at.typecheck
class Module(nn.Module):
    """Transformer model, supporting a mixture of different weights for different tokens."""

    configs: Sequence[Config]  # list of configs, one for each expert
    embed_dtype: str

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.
    adarms: bool = False
    # DA3 spatial injection: inject per-view banks into the SUFFIX (action-expert) stream at the
    # last `num_spatial_layers` blocks, via a gated cross-attention inside the same scan.
    spatial_inject: bool = False
    num_spatial_layers: int = 6
    spatial_scale: float = 2.0
    spatial_init_std: float = 0.0  # 0 = zero-init (exact identity at step 0); >0 = small normal init
    spatial_logit_gain: bool = False  # learnable per-head attention-logit gain in the injection (v2)
    spatial_logit_gain_init: float = 1.0
    spatial_num_views: int = 3  # 3 = roboreal/aloha (main/left/right); 2 = LIBERO (main/wrist)

    def setup(self):
        # all experts must have the same depth
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,  # embedder for first expert only
            name="embedder",
        )
        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            static_argnums=(5,),  # 0=self, 6=deterministic
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        # in_axes maps the Block.__call__ args AFTER the carry `xs`:
        #   kv_cache(scan axis0), positions, mask, adarms_cond, deterministic  [+ banks, layer_gate]
        # banks are broadcast (same for all blocks, computed once); layer_gate is scanned on axis 0
        # so each block receives its own scalar gate[l] (0 for early blocks, 1 for the last N).
        base_in_axes = (0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast)
        in_axes = (*base_in_axes, nn.broadcast, 0) if self.spatial_inject else base_in_axes
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=in_axes,
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
            spatial_scale=self.spatial_scale,
            spatial_init_std=self.spatial_init_std,
            spatial_logit_gain=self.spatial_logit_gain,
            spatial_logit_gain_init=self.spatial_logit_gain_init,
            spatial_num_views=self.spatial_num_views,
        )
        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def __call__(
        self,
        # list of token arrays, one for each expert, or None if that expert should not be run
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,
        *,
        kv_cache: KVCache | None = None,
        deterministic: bool = True,
        banks: dict | None = None,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        if self.spatial_inject and banks is not None:
            depth = self.configs[0].depth
            # gate[l] = 1 for the last num_spatial_layers blocks, else 0 (per-layer scalar via scan axis 0)
            layer_gate = (jnp.arange(depth) >= depth - self.num_spatial_layers).astype(jnp.float32)
            embedded, kv_cache = self.layers(
                embedded, kv_cache, positions, mask, adarms_cond, deterministic, banks, layer_gate
            )
        else:
            embedded, kv_cache = self.layers(embedded, kv_cache, positions, mask, adarms_cond, deterministic)

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        return [
            f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache

    def init(self, use_adarms: Sequence[bool]):
        """Convenience method for initializing all parameters, necessary due to the quirks of linen."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        # dummy banks so the spatial-injection params are materialized during lazy_init (bank dim
        # == action-expert width == configs[1].width); 1 dummy token per view is enough.
        banks = None
        if self.spatial_inject:
            w = self.configs[1].width
            banks = {k: jnp.zeros((1, 1, w)) for k in spatial_bank_names(self.spatial_num_views)}
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)],
            banks=banks,
        )


def _apply_rope(x, *, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    assert radians.dtype == jnp.float32
    # radians.shape = [...,L,1,d=D/2]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    assert res.dtype == jnp.float32
    # The original bigvision impl allows RoPE to upcast to float32. It is then immediately downcast again to the cache
    # dtype when in inference mode (but not in training mode). I don't think any of this was intentional. Based on the
    # original DeepMind impl, as well as the widely-used transformers impl, it is ok to always downcast back to bfloat16
    # here.
    return res.astype(x.dtype)


def _name(name, i):
    # we name layers like this because we want the first expert's weights to have no suffix (e.g., "attn"), so that they
    # can be loaded seamlessly from the existing PaliGemma checkpoint. subsequent experts will have a suffix (e.g.,
    # "attn_1") and their weights will be initialized from scratch. in practice, we only use two experts -- PaliGemma,
    # and the action expert.
    if i == 0:
        return name
    return f"{name}_{i}"


def _gated_residual(x, y, gate):
    assert (x is None) == (y is None)
    if x is None:
        return None
    if gate is None:
        return x + y
    return x + y * gate
