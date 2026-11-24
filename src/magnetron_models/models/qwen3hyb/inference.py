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
from magnetron_models.models.qwen3hyb.model import Qwen3HybModel


@register_inference_adapter
class Qwen3HybInferenceAdapter(TextGenAdapter):
    arch = 'qwen3hyb'

    @override
    @classmethod
    @no_grad()
    def load(cls, snapshot: Path) -> Qwen3HybInferenceAdapter:
        from tokenizers import Tokenizer

        _, model = Qwen3HybModel.from_snapshot(snapshot)
        tok = Tokenizer.from_str(model.snapshot_metadata['tokenizer_json'])
        return cls(model, tok, frozenset({model.cfg.eos_token_id, model.cfg.bos_token_id}))

    @property
    def enable_thinking(self) -> bool:
        return self.model.cfg.enable_thinking

    @enable_thinking.setter
    def enable_thinking(self, on: bool) -> None:
        self.model.cfg.enable_thinking = on

    @override
    def apply_chat_template(self, msgs: list[dict[str, str]]) -> str:
        out: list[str] = []
        for msg in msgs:
            content = msg['content'].strip()
            if msg['role'] == 'assistant' and '</think>' in content:
                content = content.split('</think>')[-1].lstrip('\n')
            out.append(f'<|im_start|>{msg["role"]}\n{content}<|im_end|>\n')
        out.append('<|im_start|>assistant\n<think>\n' if self.enable_thinking else '<|im_start|>assistant\n<think>\n\n</think>\n\n')
        return ''.join(out)
