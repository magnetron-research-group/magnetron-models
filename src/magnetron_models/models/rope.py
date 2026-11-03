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


def precompute_freq_cache(dim: int, theta: float, max_seq_len: int) -> tuple[Tensor, Tensor]:  # (cosfreq, sinfreq)
    inv = (theta ** -(Tensor.arange(0, dim, 2, dtype=dtype.float32) / dim)).reshape(1, -1)
    freqs = inv * Tensor.arange(stop=max_seq_len, dtype=dtype.float32).reshape(-1, 1)
    freqs = Tensor.cat([freqs, freqs], dim=-1)
    ddt = context.get_default_dtype()
    return freqs.cos().cast(ddt), freqs.sin().cast(ddt)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:  # x: (B,H,T,D), cos&sin: (T,R) if R<D = partial rotary
    R, H = cos.shape[-1], cos.shape[-1] >> 1
    r = x[..., :R]
    r = cos * r + sin * Tensor.cat([-r[..., H:], r[..., :H]], dim=-1)
    return r if R == x.shape[-1] else Tensor.cat([r, x[..., R:]], dim=-1)
