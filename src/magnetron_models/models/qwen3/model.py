# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import math

from magnetron import Tensor, nn, dtype
from magnetron_models.common import ModelBase
from magnetron_models.kvcache import KVLayerCache, KVCache
from magnetron_models.models.qwen3.config import Qwen3Config
from magnetron_models.models.rope import apply_rope, precompute_freq_cache


class _Attn(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        H, self.nh, self.nkv, self.hd = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.q_proj = nn.Linear(H, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(H, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(H, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, H, bias=False)
        self.q_norm = nn.RMSNorm(self.hd, eps=cfg.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.hd, eps=cfg.rms_norm_eps)

    def forward(self, x: Tensor, cfreq: Tensor, sfreq: Tensor, cache: KVLayerCache | None = None) -> Tensor:
        B, T, _ = x.shape
        heads = lambda t, n: t.reshape(B, T, n, self.hd).transpose(1, 2)
        q = apply_rope(self.q_norm(heads(self.q_proj(x), self.nh)), cfreq, sfreq)
        k = apply_rope(self.k_norm(heads(self.k_proj(x), self.nkv)), cfreq, sfreq)
        k, v = cache.append(k, heads(self.v_proj(x), self.nkv))
        S = k.shape[2]
        s = (Tensor.einsum('BKGTD,BKSD->BKGTS', q.reshape(B, self.nkv, -1, T, self.hd), k) * self.hd**-0.5).cast(dtype.float32)
        if T > 1:
            mask = Tensor.arange(S).reshape(1, -1) <= Tensor.arange(S - T, S).reshape(-1, 1)
            s = s + Tensor.where(mask, 0.0, float('-inf')).cast(dtype.float32)
        o = Tensor.einsum('BKGTS,BKSD->BKGTD', s.softmax(dim=-1).cast(v.dtype), v)
        return self.o_proj(o.reshape(B, self.nh, T, self.hd).transpose(1, 2).reshape(B, T, -1))


class _Block(nn.Module):
    class _MLP(nn.Module):
        def __init__(self, cfg: Qwen3Config) -> None:
            super().__init__()
            H, I = cfg.hidden_size, cfg.intermediate_size
            self.gate_proj = nn.Linear(H, I, bias=False)
            self.up_proj = nn.Linear(H, I, bias=False)
            self.down_proj = nn.Linear(I, H, bias=False)

        def forward(self, x: Tensor) -> Tensor:
            return self.down_proj(self.gate_proj(x).silu() * self.up_proj(x))

    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.self_attn = _Attn(cfg)
        self.mlp = self._MLP(cfg)
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, x: Tensor, cfreq: Tensor, sfreq: Tensor, cache: KVLayerCache | None = None) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cfreq, sfreq, cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3Model(ModelBase):
    arch = 'qwen3'
    config_cls = Qwen3Config

    @nn.layer_init(weight=nn.init.EmptyInitStrategy(), bias=nn.init.EmptyInitStrategy())
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__(cfg, is_trainable=False)
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([_Block(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head: nn.Linear | None = None
        if not cfg.tie_word_embeddings:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.rope_cache: tuple[Tensor, Tensor] = precompute_freq_cache(cfg.head_dim, cfg.rope_theta, cfg.max_position_embeddings)
        self.kv_cache = KVCache(cfg.num_key_value_heads, cfg.num_hidden_layers, cfg.max_position_embeddings, cfg.head_dim)

    def forward(self, x: Tensor) -> Tensor:  # (B,T) ids -> (B,1,V) logits of the last position only
        p, T = self.kv_cache.pos, x.shape[1]
        cfreq, sfreq = self.rope_cache[0][p : p + T], self.rope_cache[1][p : p + T]
        h = self.embed_tokens(x)
        for layer, cache in zip(self.layers, self.kv_cache.layers):
            h = layer(h, cfreq, sfreq, cache)
        h = self.norm(h[:, -1:])
        return self.lm_head(h) if self.lm_head is not None else Tensor.einsum('...H,VH->...V', h, self.embed_tokens.weight)
