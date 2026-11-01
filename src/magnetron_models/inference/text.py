# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from typing import Any


class IncrementalDecoder:
    def __init__(self, tokenizer: Any) -> None:
        self.tok = tokenizer
        self.ids: list[int] = []
        self.emitted = 0

    def push(self, tok: int) -> str:
        self.ids.append(tok)
        txt = self.tok.decode(self.ids, skip_special_tokens=True)
        if txt.endswith('\ufffd'):
            return ''
        delta, self.emitted = txt[self.emitted :], len(txt)
        return delta

    def flush(self) -> str:
        txt = self.tok.decode(self.ids, skip_special_tokens=True)
        delta, self.emitted = txt[self.emitted :], len(txt)
        return delta
