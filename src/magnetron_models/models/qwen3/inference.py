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
from magnetron import Tensor
from magnetron_models.inference import register_inference_adapter
from magnetron_models.inference.adapters import TextGenAdapter
from magnetron_models.models.qwen3.model import Qwen3Model


@register_inference_adapter
class Qwen3InferenceAdapter(TextGenAdapter):
    arch = 'qwen3'

    def __init__(self, model: Qwen3Model, tensors: dict[str, Tensor], tokenizer: Any) -> None: ...
