# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

from magnetron import Tensor, dtype
from magnetron_models.kvcache import KVLayerCache
from magnetron_models.models.qwen3hyb.config import Qwen3HybConfig, Qwen3HybLayerType


class LinearLayerCache:
    def __init__(self, cfg: Qwen3HybConfig, batch_size: int = 1, dt=dtype.float32) -> None:
        self.dt = dt
        self.conv = Tensor.zeros(batch_size, cfg.linear_conv_dim, cfg.linear_conv_kernel_dim, dtype=dt)
        self.state = Tensor.zeros(batch_size, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim, dtype=dt)
        self.is_primed = False

    def clear(self) -> None:  # TODO: use zero_()
        self.conv = Tensor.zeros(*self.conv.shape, dtype=self.dt)
        self.state = Tensor.zeros(*self.state.shape, dtype=self.dt)


class HybridCache:
    def __init__(self, cfg: Qwen3HybConfig, batch_size: int = 1) -> None:
        self.layers: list[KVLayerCache | LinearLayerCache] = [
            KVLayerCache(batch_size=batch_size, num_kv_heads=cfg.num_key_value_heads, max_seq_len=cfg.max_position_embeddings, head_dim=cfg.head_dim)
            if cfg.layer_type(l) is Qwen3HybLayerType.ATTN_FULL
            else LinearLayerCache(cfg, batch_size=batch_size)
            for l in range(cfg.num_hidden_layers)
        ]
        self.pos: int = 0

    def __getitem(self, i: int) -> KVLayerCache | LinearLayerCache:
        return self.layers[i]

    def advance(self, toks: int) -> None:
        self.pos += toks

    def checkpoint(self) -> tuple[int, list[int | tuple[Tensor, Tensor, bool]]]:
        xr = [layer.pos if isinstance(layer, KVLayerCache) else (layer.conv, layer.state, layer.is_primed) for layer in self.layers]
        return self.pos, xr

    def clear(self) -> None:
        for layer in self.layers:
            layer.clear()
        self.pos = 0

    def restore(self, chkp: tuple[int, list[int | tuple[Tensor, Tensor, bool]]]) -> None:
        self.pos, layers = chkp
        for layer, saved in zip(self.layers, layers, strict=True):
            if isinstance(layer, KVLayerCache):
                layer.pos = saved
            else:
                layer.conv, layer.state, layer.is_primed = saved
