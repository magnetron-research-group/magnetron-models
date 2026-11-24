# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

from typing import Any

from magnetron import dtype, Tensor, nn

from magnetron_models.kvcache import KVLayerCache
from magnetron_models.models.qwen3hyb.kvcache import LinearLayerCache
from magnetron_models.models.qwen3hyb.config import Qwen3HybConfig
from magnetron_models.models.rope import apply_rope


def _unit_lower_inv(x: Tensor) -> Tensor:
    n = x.shape[-1]
    if n == 1:
        return Tensor.ones_like(x)
    h = n >> 1
    if n - h == h:
        batch = x.shape[0]
        merged = _unit_lower_inv(Tensor.cat([x[..., :h, :h], x[..., h:, h:]], dim=0))
        t11, t22 = merged[:batch], merged[batch:]
    else:
        t11, t22 = _unit_lower_inv(x[..., :h, :h]), _unit_lower_inv(x[..., h:, h:])
    t21 = t22 @ x[..., h:, :h] @ t11
    top = Tensor.cat([t11, Tensor.zeros(*x.shape[:-2], h, n - h, dtype=x.dtype)], dim=-1)
    return Tensor.cat([top, Tensor.cat([t21, t22], dim=-1)], dim=-2)


def _l2norm(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    return x * (x.sqr().sum(dim=dim, keepdim=True) + eps).rsqrt()


def _softplus(x: Tensor) -> Tensor:
    return x.relu() + x.abs().neg().exp().log1p()


class GatedDeltaNet(nn.Module):
    @staticmethod
    def _chunk_gated_delta_rule(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        g: Tensor,
        b: Tensor,
        state: Tensor | None = None,
        use_qk_l2norm: bool = False,
        out_final_state: bool = False,
        chunk: int = 64,
    ) -> tuple[Tensor, Tensor | None]:
        idt = q.dtype
        q, k, v, b, g = [x.transpose(1, 2).contiguous().cast(dtype.float32) for x in (q, k, v, b, g)]
        if use_qk_l2norm:
            q = _l2norm(q, dim=-1, eps=1e-6)
            k = _l2norm(k, dim=-1, eps=1e-6)
        batch, nhead, seq, khead = k.shape
        vhead: int = v.shape[-1]
        chunk: int = min(chunk, seq)
        pad: int = (chunk - seq % chunk) % chunk
        q = q.pad((0, 0, 0, pad))
        k = k.pad((0, 0, 0, pad))
        v = v.pad((0, 0, 0, pad))
        b = b.pad((0, pad))
        g = g.pad((0, pad))
        seq_tot: int = seq + pad
        q = q * (1 / (q.shape[-1] ** 0.5))
        vb = v * b[..., None]
        kb = k * b[..., None]
        q, k, v, kb, vb = [x.reshape(x.shape[0], x.shape[1], -1, chunk, x.shape[-1]) for x in (q, k, v, kb, vb)]
        g = g.reshape(g.shape[0], g.shape[1], -1, chunk).cusum(dim=-1)
        mask = Tensor.ones(chunk, chunk).cast(dtype.boolean).triu(diagonal=0)
        decay_mask = ((g[..., :, None] - g[..., None, :]).tril().exp().cast(dtype.float32)).tril()
        attn = _unit_lower_inv(-((kb @ k.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0))
        v = attn @ vb
        kcd = attn @ (kb * g.exp()[..., None])
        rec_state: Tensor | None = Tensor.zeros(batch, nhead, khead, vhead, dtype=v.dtype) if state is None else state.cast(v.dtype)
        out: list[Tensor] = []
        for i in range(0, seq_tot // chunk):
            qi, ki, vi = q[:, :, i], k[:, :, i], v[:, :, i]
            attn = qi @ ki.transpose(-1, -2) * decay_mask[:, :, i]
            vp = kcd[:, :, i] @ rec_state
            vn = vi - vp
            attn_inter = (qi * g[:, :, i, :, None].exp()) @ rec_state
            out.append(attn_inter + attn @ vn)
            kor = (ki * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ vn
            rec_state = rec_state * g[:, :, i, -1, None, None].exp() + kor
        if not out_final_state:
            rec_state = None
        attn = Tensor.cat(out, dim=2)[:, :, :seq].transpose(1, 2).contiguous().cast(idt)
        return attn, rec_state

    @staticmethod
    def _recurrent_gated_delta_rule(
        q: Tensor, k: Tensor, v: Tensor, g: Tensor, b: Tensor, state: Tensor | None = None, use_qk_l2norm: bool = False, out_final_state: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        idt = q.dtype
        q, k, v, b, g = [x.transpose(1, 2).contiguous().cast(dtype.float32) for x in (q, k, v, b, g)]
        if use_qk_l2norm:
            q = _l2norm(q, dim=-1, eps=1e-6)
            k = _l2norm(k, dim=-1, eps=1e-6)
        batch, nhead, seq, khead = k.shape
        vhead: int = v.shape[-1]
        q = q * (1 / (q.shape[-1] ** 0.5))
        if state is None:
            state = Tensor.zeros(batch, nhead, khead, vhead, dtype=dtype.float32)
        else:
            state = state if state.dtype == dtype.float32 else state.cast(dtype.float32)
        out: list[Tensor] = []
        for i in range(seq):
            qt = q[:, :, i : i + 1]
            kt = k[:, :, i : i + 1]
            vt = v[:, :, i : i + 1]
            gt = g[:, :, i, None, None].exp()
            bt = b[:, :, i, None, None]
            state = gt * state
            kvm = kt @ state
            delta = (vt - kvm) * bt
            state = state + kt.transpose(-1, -2) * delta
            out.append(qt @ state)
        attn = out[0] if seq == 1 else Tensor.cat(out, dim=2)
        return attn.transpose(1, 2).contiguous().cast(idt), state if out_final_state else None

    def __init__(self, cfg: Qwen3HybConfig) -> None:
        super().__init__()
        self.cfg: Qwen3HybConfig = cfg
        self.nkh: int = cfg.linear_num_key_heads
        self.nvh: int = cfg.linear_num_value_heads
        self.hkd: int = cfg.linear_key_head_dim
        self.hvd: int = cfg.linear_value_head_dim
        self.kd: int = cfg.linear_key_dim
        self.vd: int = cfg.linear_val_dim
        self.cv: int = cfg.linear_conv_dim
        self.cvks: int = cfg.linear_conv_kernel_dim
        self.nrep: int = self.nvh // self.nkh
        self.conv1d = nn.Conv1D(self.cv, self.cv, self.cvks, groups=self.cv, bias=False)
        self.dt_bias = nn.Parameter(Tensor.empty(self.nvh, dtype=dtype.float32))
        self.A_log = nn.Parameter(Tensor.empty(self.nvh, dtype=dtype.float32))
        self.norm = nn.RMSNormGated(self.hvd, eps=cfg.rms_norm_eps)
        h: int = cfg.hidden_size
        self.in_proj_qkv = nn.Linear(h, self.cv, bias=False)
        self.in_proj_z = nn.Linear(h, self.vd, bias=False)
        self.in_proj_b = nn.Linear(h, self.nvh, bias=False)
        self.in_proj_a = nn.Linear(h, self.nvh, bias=False)
        self.out_proj = nn.Linear(self.vd, h, bias=False)

    def _conv(self, mqkv: Tensor, cache: LinearLayerCache | None) -> Tensor:
        seq: int = mqkv.shape[-1]
        kern: int = self.cvks
        pad = Tensor.cat([cache.conv, mqkv], dim=-1) if cache is not None and cache.is_primed else mqkv.pad((kern - 1, 0))
        if cache is not None:
            cache.conv = pad[:, :, -kern:]
        return self.conv1d(pad)[:, :, -seq:].silu()

    def forward(self, x: Tensor, cache: LinearLayerCache | None = None) -> Tensor:
        B, T, _ = x.shape
        mqkv = self._conv(self.in_proj_qkv(x).transpose(1, 2), cache).transpose(1, 2)
        q = mqkv[:, :, : self.kd].reshape(B, T, -1, self.hkd)
        k = mqkv[:, :, self.kd : (self.kd << 1)].reshape(B, T, -1, self.hkd)
        v = mqkv[:, :, (self.kd << 1) :].reshape(B, T, -1, self.hvd)
        z = self.in_proj_z(x).reshape(B, T, -1, self.hvd)
        b = self.in_proj_b(x).sigmoid()
        g = -self.A_log.exp() * _softplus(self.in_proj_a(x).cast(dtype.float32) + self.dt_bias)
        if self.nrep > 1:
            q = q.repeat_interleave(self.nrep, dim=2)
            k = k.repeat_interleave(self.nrep, dim=2)
        state: Tensor | None = cache.state if cache is not None and cache.is_primed else None
        chunk_rule = GatedDeltaNet._chunk_gated_delta_rule
        rec_rule = GatedDeltaNet._recurrent_gated_delta_rule
        rule = rec_rule if state is not None and T == 1 else chunk_rule
        kw: dict[str, Any] = {} if rule is rec_rule else {'chunk': self.cfg.delta_chunk_size}
        attn, pstate = rule(q, k, v, g, b, state=state, use_qk_l2norm=True, out_final_state=cache is not None, **kw)
        if cache is not None:
            cache.state = pstate.cast(cache.dt)
            cache.is_primed = True
        attn = self.norm(attn.reshape(-1, self.hvd), z.reshape(-1, self.hvd))
        return self.out_proj(attn.reshape(B, T, -1))


class GatedAttn(nn.Module):
    def __init__(self, cfg: Qwen3HybConfig) -> None:
        super().__init__()
        self.hd: int = cfg.head_dim
        self.nh: int = cfg.num_attention_heads
        self.nkv: int = cfg.num_key_value_heads
        h: int = cfg.hidden_size
        self.q_proj = nn.Linear(h, self.nh * (self.hd << 1), bias=False)
        self.k_proj = nn.Linear(h, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(h, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, h, bias=False)
        self.q_norm = nn.RMSNorm(self.hd, eps=cfg.rms_norm_eps, gamma_offset=1.0)
        self.k_norm = nn.RMSNorm(self.hd, eps=cfg.rms_norm_eps, gamma_offset=1.0)

    def forward(self, x: Tensor, cfreq: Tensor, sfreq: Tensor, cache: KVLayerCache) -> Tensor:
        B, T, _ = x.shape
        qg = self.q_proj(x).reshape(B, T, self.nh, self.hd << 1)
        q = apply_rope(self.q_norm(qg[:, :, :, : self.hd]).transpose(1, 2), cfreq, sfreq)
        gate = qg[:, :, :, self.hd :].reshape(B, T, -1)
        k = apply_rope(self.k_norm(self.k_proj(x).reshape(B, T, self.nkv, self.hd)).transpose(1, 2), cfreq, sfreq)
        k, v = cache.append(k, self.v_proj(x).reshape(B, T, self.nkv, self.hd).transpose(1, 2))
        S: int = k.shape[2]
        s = (Tensor.einsum('BKGTD,BKSD->BKGTS', q.reshape(B, self.nkv, -1, T, self.hd), k) * self.hd**-0.5).cast(dtype.float32)
        if T > 1:
            mask = Tensor.arange(S).reshape(1, -1) <= Tensor.arange(S - T, S).reshape(-1, 1)
            s = s + Tensor.where(mask, 0.0, float('-inf')).cast(dtype.float32)
        o = Tensor.einsum('BKGTS,BKSD->BKGTD', s.softmax(dim=-1).cast(v.dtype), v).reshape(B, self.nh, T, self.hd)
        return self.o_proj(gate.sigmoid() * o.transpose(1, 2).reshape(B, T, -1))
