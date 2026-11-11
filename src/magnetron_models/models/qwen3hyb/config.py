# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

from dataclasses import dataclass
from enum import unique, Enum
from magnetron_models.common import ConfigBase


@unique
class Qwen38ReasoningEffort(Enum):
    LOW = 'low'
    MEDIUM = 'medium'
    XHIGH = 'xhigh'


@unique
class Qwen3HybLayerType(Enum):  # Layer attention type
    ATTN_LINEAR = 'attn_linear'
    ATTN_FULL = 'attn_full'


@dataclass
class Qwen3HybConfig(ConfigBase):
    repo_id: str = 'Qwen/Qwen3.5-4B'
    vocab_size: int = 248320
    hidden_size: int = 2560
    intermediate_size: int = 9216
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    max_position_embeddings: int = 8192  # 262144
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    full_attention_interval: int = 4  # Every n-th layer is full attention, the rest is gated delta net.
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    delta_chunk_size: int = 64
    bos_token_id: int = 248044  # <|endoftext|>
    eos_token_id: int = 248046  # <|im_end|>
    enable_thinking: bool = False
    reasoning_effort: Qwen38ReasoningEffort | None = None

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def linear_key_dim(self) -> int:
        return self.linear_key_head_dim * self.linear_num_key_heads

    @property
    def linear_val_dim(self) -> int:
        return self.linear_value_head_dim * self.linear_num_value_heads

    @property
    def linear_conv_dim(self) -> int:
        return self.linear_val_dim + (self.linear_key_dim << 1)

    def layer_type(self, i: int) -> Qwen3HybLayerType:
        return Qwen3HybLayerType.ATTN_LINEAR if (i + 1) % self.full_attention_interval else Qwen3HybLayerType.ATTN_FULL

    @property
    def layer_types(self) -> list[Qwen3HybLayerType]:
        return [self.layer_type(l) for l in range(self.num_hidden_layers)]
