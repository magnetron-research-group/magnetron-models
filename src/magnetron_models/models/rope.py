# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

from magnetron import Tensor, dtype, context


def precompute_freq_cache(dim: int, theta: float, max_seq_len: int) -> tuple[Tensor, Tensor]:
    inv = (theta ** -(Tensor.arange(0, dim, 2, dtype=dtype.float32) / dim)).reshape(1, -1)
    freqs = inv * Tensor.arange(stop=max_seq_len, dtype=dtype.float32).reshape(max_seq_len, 1)
    hsin = freqs.sin()
    hcos = freqs.cos()
    sin = Tensor.cat([hsin, hsin], dim=-1).cast(context.get_default_dtype())
    cos = Tensor.cat([hcos, hcos], dim=-1).cast(context.get_default_dtype())
    return cos, sin


def apply_rope(q: Tensor, k: Tensor, cfreq: Tensor, sfreq: Tensor, idx: Tensor) -> tuple[Tensor, Tensor]:
    def _rope_rows(table: Tensor, idx: Tensor) -> Tensor:
        flattened = idx.reshape(-1, 1)
        rows = table.gather(0, flattened.expand(flattened.shape[0], table.shape[-1]))
        return rows.reshape(*idx.shape, table.shape[-1])

    def _rot_half(x: Tensor) -> Tensor:
        return Tensor.cat([-x[:, :, :, x.shape[-1] >> 1 :], x[:, :, :, : x.shape[-1] >> 1]], dim=1)

    cos = _rope_rows(cfreq, idx)
    sin = _rope_rows(sfreq, idx)
    batch, seq, ror = cos.shape
    cos = cos.reshape(batch, 1, seq, ror)
    sin = sin.reshape(batch, 1, seq, ror)
    if ror == q.shape[-1]:  # Full rotary (Qwen3)
        q_emb = (cos * q) + (sin * _rot_half(q))
        k_emd = (cos * k) + (sin * _rot_half(k))
        return q_emb, k_emd
    q_rot, q_pass = q[:, :, :, :ror], q[:, :, :, ror:]
    k_rot, k_pass = k[:, :, :, :ror], k[:, :, :, ror:]
    q_emb = Tensor.cat([(cos * q_rot) + (sin * _rot_half(q_rot)), q_pass], dim=-1)
    k_emb = Tensor.cat([(cos * k_rot) + (sin * _rot_half(k_rot)), k_pass], dim=-1)
    return q_emb, k_emb
