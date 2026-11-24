# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

from magnetron import nn, Tensor

from magnetron_models.common import ModelBase
from magnetron_models.kvcache import KVLayerCache
from magnetron_models.models.qwen3hyb.config import Qwen3HybConfig, Qwen3HybLayerType
from magnetron_models.models.qwen3hyb.attn import GatedDeltaNet, GatedAttn
from magnetron_models.models.qwen3hyb.kvcache import HybridCache, LinearLayerCache
from magnetron_models.models.rope import precompute_freq_cache


# Qwen 3.5/3.6 and 3.8 27B all basically share the same architecture and modeling code
# that's why we call this Qwen3Hybrid as this modeling code is used for all of those.
# From Qwen3.5/3.6 dense variants to 3.8 27B dense.


class _Block(nn.Module):
    class _MLP(nn.Module):
        def __init__(self, cfg: Qwen3HybConfig) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
            self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
            self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

        def forward(self, x: Tensor) -> Tensor:
            return self.down_proj(self.up_proj(x) * self.gate_proj(x).silu())

    def __init__(self, cfg: Qwen3HybConfig, layer_idx: int) -> None:
        super().__init__()
        self.is_linear: bool = cfg.layer_type(layer_idx) is Qwen3HybLayerType.ATTN_LINEAR
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(cfg)
        else:
            self.self_attn = GatedAttn(cfg)
        self.mlp = self._MLP(cfg)
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, gamma_offset=1.0)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, gamma_offset=1.0)

    def forward(self, x: Tensor, cfreq: Tensor, sfreq: Tensor, cache: KVLayerCache | LinearLayerCache) -> Tensor:
        normed = self.input_layernorm(x)
        if self.is_linear:
            assert isinstance(cache, LinearLayerCache)
            h = x + self.linear_attn(normed, cache)
        else:
            assert isinstance(cache, KVLayerCache)
            h = x + self.self_attn(normed, cfreq, sfreq, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class Qwen3HybModel(ModelBase):
    arch = 'qwen3hyb'
    config_cls = Qwen3HybConfig

    @nn.layer_init(weight=nn.init.EmptyInitStrategy(), bias=nn.init.EmptyInitStrategy())
    def __init__(self, cfg: Qwen3HybConfig) -> None:
        super().__init__(cfg, is_trainable=False)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([_Block(cfg, l) for l in range(cfg.num_hidden_layers)])
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, gamma_offset=1.0)
        self.lm_head: nn.Linear | None = None
        if not cfg.tie_word_embeddings:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.rope_cache: tuple[Tensor, Tensor] = precompute_freq_cache(cfg.rotary_dim, cfg.rope_theta, cfg.max_position_embeddings)
        self.kv_cache = HybridCache(cfg)

    @property
    def max_context_len(self) -> int:
        return self.cfg.max_position_embeddings

    def reset_state(self) -> None:
        self.kv_cache.clear()

    def forward(self, x: Tensor) -> Tensor:
        p, T = self.kv_cache.pos, x.shape[1]
        cfreq, sfreq = self.rope_cache[0][p : p + T], self.rope_cache[1][p : p + T]
        h = self.embed_tokens(x)
        for layer, cache in zip(self.layers, self.kv_cache.layers, strict=True):
            h = layer(h, cfreq, sfreq, cache)
        self.kv_cache.advance(T)
        h = self.norm(h[:, -1:])
        return self.lm_head(h) if self.lm_head is not None else Tensor.einsum('...H,VH->...V', h, self.embed_tokens.weight)
