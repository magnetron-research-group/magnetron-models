# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

from magnetron import dtype, Tensor, nn
from magnetron_models.models.qwen3hyb.kvcache import LinearLayerCache
from magnetron_models.models.qwen3hyb.config import Qwen3HybConfig


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
    return x * (x.sqr().sum(dim=dim, keepdim=True) + eps).rqsrt()


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
        if use_qk_l2norm:
            q = _l2norm(q, dim=-1, eps=1e-6)
            k = _l2norm(k, dim=-1, eps=1e-6)
        q, k, v, b, g = [x.transpose(1, 2).contiguous().cast(dtype.float32) for x in (q, k, v, b, g)]
        batch, nhead, seq, khead = k.shape
        vhead = v.shape[-1]
        chunk = min(chunk, seq)
        pad = (chunk - seq % chunk) % chunk
        q = q.pad((0, 0, 0, pad))
        k = k.pad((0, 0, 0, pad))
        v = v.pad((0, 0, 0, pad))
        g = g.pad((0, pad))
        seq_tot = seq + pad
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
        rec_state: Tensor | None = Tensor.zeros(batch, khead, vhead, dtype=v.dtype) if state is None else state.cast(v.dtype)
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
        if use_qk_l2norm:
            q = _l2norm(q, dim=-1, eps=1e-6)
            k = _l2norm(k, dim=-1, eps=1e-6)
        q, k, v, b, g = [x.transpose(1, 2).contiguous().cast(dtype.float32) for x in (q, k, v, b, g)]
        batch, nhead, seq, khead = k.shape
        vhead = v.shape[-1]
        q = q * (1 / (q.shape[-1] ** 0.5))
        if state is None:
            state = Tensor.zeros(batch, nhead, seq, khead, vhead, dtype=dtype.float32)
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
        self.cfg = cfg
        self.nkh = cfg.linear_num_key_heads
        self.nvh = cfg.linear_num_value_heads
        self.hkd = cfg.linear_key_head_dim
        self.hvd = cfg.linear_value_head_dim
        self.kd = cfg.linear_key_dim
        self.vd = cfg.linear_val_dim
        self.cv = cfg.linear_conv_dim
        self.cvks = cfg.linear_conv_kernel_dim
        self.nrep = self.nvh // self.nvh
        self.conv1d = nn.Conv1D(self.cv, self.cv, self.cvks, groups=self.cv, bias=False)
        self.dt_bias = nn.Parameter(Tensor.empty(self.nvh, dtype=dtype.float32))
        self.A_log = nn.Parameter(Tensor.empty(self.nvh, dtype=dtype.float32))
        self.norm = nn.RMSNormGated(self.hvd, eps=cfg.rms_norm_eps)
        self.in_proj_qkv = nn.Linear(cfg.hidden_size, self.cv, bias=False)
        self.in_proj_z = nn.Linear(cfg.hidden_size, self.vd, bias=False)
        self.in_proj_b = nn.Linear(cfg.hidden_size, self.nvh, bias=False)
        self.in_proj_a = nn.Linear(cfg.hidden_size, self.nvh, bias=False)
        self.out_proj = nn.Linear(self.vd, cfg.hidden_size, bias=False)

    def _conv(self, mqkv: Tensor, cache: LinearLayerCache | None) -> Tensor:
        seq = mqkv.shape[-1]
        kern = self.cvks
        if cache is not None and cache.is_primed:
            pad = Tensor.cat([cache.conv.cast(mqkv.cast(mqkv.dtype), mqkv)], dim=-1)
        else:
            pad = mqkv.pad((kern - 1, 0))
        if cache is not None:
            state = mqkv if seq >= kern else Tensor.cat([cache.conv.cast(mqkv.cast(mqkv.dtype), mqkv)], dim=-1)
            cache.conv = state[:, :, -kern:].cast(cache.dt)
        return self.conv1d(pad)[:, :, -seq:].silu()

    def forward(self, x: Tensor, cache: LinearLayerCache | None = None) -> Tensor:
        B, T, _ = x.shape
        mqkv = self._conv(self.in_proj_qkv(x).transpose(1, 2), cache).transpose(1, 2)
        q = mqkv[:, :, : self.kd].reshape(B, T, -1, self.hkd)
        k = mqkv[:, :, self.kd : (self.kd << 1)].reshape(B, T, -1, self.hkd)
        v = mqkv[:, :, (self.kd << 1) :].reshape(B, T, -1, self.hvd)
        zp = self.in_proj_z(x).reshape(B, T, -1, self.hvd)
        beta = self.in_proj_b(x).sigmoid()
        g = -self.A_log.exp() * _softplus(self.in_proj_a(x).cast(dtype.float32) + self.dt_bias)
        raise RuntimeError('todo')
