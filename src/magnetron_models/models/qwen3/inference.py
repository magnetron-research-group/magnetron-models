# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

from pathlib import Path
from typing import override
from magnetron import no_grad
from magnetron_models.inference import register_inference_adapter
from magnetron_models.inference.text import TextGenAdapter
from magnetron_models.models.qwen3.model import Qwen3Model


@register_inference_adapter
class Qwen3InferenceAdapter(TextGenAdapter):
    arch = 'qwen3'

    @override
    @classmethod
    @no_grad()
    def load(cls, snapshot: Path) -> Qwen3InferenceAdapter:
        from tokenizers import Tokenizer

        _, model = Qwen3Model.from_snapshot(snapshot)
        tok = Tokenizer.from_str(model.snapshot_metadata['tokenizer_json'])
        return cls(model, tok, frozenset({model.cfg.eos_token_id, model.cfg.bos_token_id}))

    @override
    def apply_chat_template(self, msgs: list[dict[str, str]]) -> str:
        return ''.join(f'<|im_start|>{msg["role"]}\n{msg["content"]}<|im_end|>\n' for msg in msgs) + '<|im_start|>assisant\n'
