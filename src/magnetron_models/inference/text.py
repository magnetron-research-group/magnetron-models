# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

from abc import abstractmethod
from typing import Any, override, Iterator, Protocol
from magnetron import no_grad, Tensor
from magnetron_models.inference import InferenceAdapter
from magnetron_models.inference.adapters import TextRequest, TextChunk, TextResponse, StopReason
from magnetron_models.kvcache import KVCache


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


class CausalLM(Protocol):
    kv_cache: KVCache

    def __call__(self, x: Tensor, idx: Tensor) -> Tensor: ...  # (1,T) ids -> (1,T,V) logits


class TextGenAdapter(InferenceAdapter[TextRequest, TextResponse, TextChunk]):
    def __init__(self, model: CausalLM, tokenizer: Any, stop_ids: frozenset[int]) -> None:
        self.model = model
        self.tok = tokenizer
        self.stop_ids = stop_ids

    @abstractmethod
    def apply_chat_template(self, msgs: list[dict[str, str]]) -> str: ...

    def encode(self, prompt: str | list[dict[str, str]] | Tensor) -> Tensor:
        match prompt:
            case Tensor():
                return prompt.reshape(1, -1)
            case list():
                prompt = self.apply_chat_template(prompt)
        return Tensor([self.tok.encode(prompt, add_special_tokens=False).ids], dtype=dtype.int64)

    @staticmethod
    def _sample(logits: Tensor, tmp: float, topk: int) -> Tensor:
        if tmp <= 0.0:
            return logits.argmax(dim=0).reshape(1)
        vals, idx = (logits / tmp).topk(max(1, min(topk, logits.shape[0])), dim=0, largest=True, sorted=False)
        return idx.gather(vals.softmax(dim=-1)).reshape(1, -1).multinomial(num_samples=1).reshape(1)

    def _decode_tokens(self, ids: Tensor, req: TextRequest) -> Iterator[int]:
        cache = self.model.kv_cache
        cache.clear()
        cap = cache[0].k.shape[2]
        T = ids.shape[1]
        logits = self.model(ids, Tensor.arange(T).reshape(1, -1))
        sample_tok = lambda: self._sample(logits[0, -1], req.temp, req.topk)
        tok = sample_tok()
        pos = T
        for _ in range(req.max_toks):
            if pos >= cap:
                return
            logits = self.model(
                tok.reshape(1, 1), Tensor([[pos]], dtype=dtype.int64)
            )  # Async device: enqueue step i+1 before reading token i so the device works while we detokenize
            nxt = sample_tok()
            tid = int(tok.item())
            if tid in self.stop_ids:
                return
            yield tid
            tok, pos = nxt, 1 + pos

    @override
    @no_grad()
    def stream(self, req: TextRequest) -> Iterator[TextChunk]:
        dec = IncrementalDecoder(self.tok)
        hold = max((len(stop) for stop in req.stop), default=1) - 1
        buf = ''
        prompt: Tensor = self.encode(req.prompt)
        for tok in self._decode_tokens(prompt, req):
            buf += dec.push(tok)
            if (cut := min((i for stop in req.stop if (i := buf.find(stop)) != -1), default=-1)) != -1:
                yield TextChunk(buf[:cut], token=tok)
                return
            emit = len(buf) - hold
            yield TextChunk(buf[:emit], token=tok)
            buf = buf[emit:]
        if tail := buf + dec.flush():
            yield TextChunk(tail)

    @override
    def collect_stream(self, req: TextRequest, chunks: Iterator[TextChunk]) -> TextResponse:
        text: list[str] = []
        toks: list[int] = []
        for chunk in chunks:
            text.append(chunk.delta)
            if chunk.token is not None:
                toks.append(chunk.token)
        reason = StopReason.LENGTH if len(toks) >= req.max_toks else StopReason.STOP
        return TextResponse(text=''.join(text), tokens=toks, stop_reason=reason)
