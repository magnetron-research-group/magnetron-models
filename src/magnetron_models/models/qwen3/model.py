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

from magnetron import Tensor, nn
from magnetron_models.common import ModelBase
from magnetron_models.kvcache import KVLayerCache, KVCache
from magnetron_models.models.qwen3.config import Qwen3Config
from magnetron_models.models.rope import apply_rope, precompute_freq_cache


class _SWA(nn.Module):
    @nn.layer_init(weight=nn.init.EmptyInitStrategy(), bias=nn.init.EmptyInitStrategy())
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.hdim: int = cfg.head_dim
        self.nheads: int = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.nrep: int = self.nheads // self.nkv
        self.sw = cfg.sliding_window
        self.q_proj = nn.Linear(cfg.hidden_size, self.nheads * self.hdim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hdim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hdim, bias=False)
        self.o_proj = nn.Linear(self.nheads * self.hdim, cfg.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.hdim, eps=cfg.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.hdim, eps=cfg.rms_norm_eps)

    def forward(self, x: Tensor, cfreq: Tensor, sfreq: Tensor, idx: Tensor, cache: KVLayerCache | None = None) -> Tensor:
        B, T, _ = x.shape
        ck = self.k_norm(self.k_proj(x).reshape(B, T, self.nkv, self.hdim).transpose(1, 2))
        cv = self.v_proj(x).reshape(B, T, self.nkv, self.hdim).transpose(1, 2)
        q = self.q_norm(self.q_proj(x).reshape(B, T, self.nheads, self.hdim).transpose(1, 2))
        q, ck = apply_rope(q, ck, cfreq, sfreq, idx)
        if cache is not None:
            k, v = cache.append(ck, cv, self.sw)
        else:
            k, v = ck, cv
        qg = q.reshape(B, self.nkv, self.nrep, T, self.hdim)
        scores = Tensor.einsum('BKGTD,BKSD->BKGTS', qg, k) * (1.0 / math.sqrt(self.hdim))
        ql, kl = q.shape[2], k.shape[2]
        if ql == 1:
            attn = scores
        else:
            kpi = Tensor.arange(kl).reshape(1, -1)
            qpi = Tensor.arange(start=(kl - ql), stop=kl).reshape(-1, 1)
            mask = Tensor.where(kpi <= qpi, 0.0, -1e4).cast(scores.dtype).reshape(1, 1, 1, ql, kl)
            attn = mask + scores
        out = Tensor.einsum('BKGTS,BKSD->BKGTD', attn.softmax(dim=-1), v)
        return self.o_proj(out.reshape(B, self.nheads, T, self.hdim).transpose(1, 2).reshape(B, T, -1))


class _Block(nn.Module):
    class _MLP(nn.Module):
        @nn.layer_init(weight=nn.init.EmptyInitStrategy(), bias=nn.init.EmptyInitStrategy())
        def __init__(self, cfg: Qwen3Config) -> None:
            super().__init__()
            self.hidden_size: int = cfg.hidden_size
            self.inter_size: int = cfg.intermediate_size
            self.gate_proj = nn.Linear(self.hidden_size, self.inter_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.inter_size, bias=False)
            self.down_proj = nn.Linear(self.inter_size, self.hidden_size, bias=False)

        def forward(self, x: Tensor) -> Tensor:
            return self.down_proj(self.gate_proj(x).silu() * self.up_proj(x))

    @nn.layer_init(weight=nn.init.EmptyInitStrategy(), bias=nn.init.EmptyInitStrategy())
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.self_attn = _SWA(cfg)
        self.mlp = self._MLP(cfg)
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, x: Tensor, cfreq: Tensor, sfreq: Tensor, idx: Tensor, cache: KVLayerCache | None = None) -> Tensor:
        h = x + self.self_attn(self.input_layernorm(x), cfreq, sfreq, idx, cache)
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h


class Qwen3Model(ModelBase):
    @nn.layer_init(weight=nn.init.EmptyInitStrategy(), bias=nn.init.EmptyInitStrategy())
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([_Block(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head: nn.Linear | None = None
        if not cfg.tie_word_embeddings:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.rope_cache: tuple[Tensor, Tensor] = precompute_freq_cache(cfg.head_dim, cfg.rope_theta, cfg.max_position_embeddings)
        self.kv_cache = KVCache(cfg.num_key_value_heads, cfg.num_hidden_layers, cfg.max_position_embeddings, cfg.head_dim)

    def forward(self, x: Tensor, idx: Tensor) -> Tensor:
        h = self.embed_tokens(x)
        for i, l in enumerate(self.layers):
            csin, ccos = self.rope_cache
            h = l(h, ccos, csin, idx, self.kv_cache[i])
        h = self.norm(h)
        return Tensor.einsum('...H,VH->...V', h, self.embed_tokens.weight) if self.cfg.tie_word_embeddings else self.lm_head(h)
