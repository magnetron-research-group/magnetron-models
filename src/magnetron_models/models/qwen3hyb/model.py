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

from magnetron_models.models.qwen3hyb.config import Qwen3HybConfig


# Qwen 3.5/3.6 and 3.8 27B all basically share the same architecture and modeling code
# that's why we call this Qwen3Hybrid as this modeling code is used for all of those.
# From Qwen3.5/3.6 dense variants to 3.8 27B dense.


class _MLP(nn.Module):
    def __init__(self, cfg: Qwen3HybConfig, imm_sz: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, imm_sz, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, imm_sz, bias=False)
        self.down_proj = nn.Linear(imm_sz, cfg.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.up_proj(x) * self.gate_proj(x).silu())
