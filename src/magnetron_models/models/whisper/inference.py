# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import ctypes
from typing import Iterable, Any

from magnetron_models.inference import register_inference_adapter
from magnetron_models.inference.adapters import TranscribeAdapter, TranscribeRequest, TextChunk, TranscribeChunk
from magnetron_models.inference.text import IncrementalDecoder
from magnetron_models.models.whisper.audio_processor import LogMelSpectogram, chunks, load_audio_and_resample
from collections.abc import Iterator
from pathlib import Path
from dataclasses import dataclass, field
from magnetron import Tensor, context, dtype, no_grad
from magnetron_models.models.whisper.audio_processor import SAMPLE_RATE
from magnetron_models.models.whisper.model import WhisperModel
from magnetron_models.utils import bytes_to_f32_tensor

_PREV_LIM: int = 223  # n_text_ctx // 2 - 1


@dataclass
class _Session:
    lang: str | None
    history: list[int] = field(default_factory=list)


@register_inference_adapter
class WhisperInferenceAdapter(TranscribeAdapter):
    arch = 'whisper'

    def __init__(self, model: WhisperModel, tensors: dict[str, Tensor], tokenizer: Any) -> None:
        self.model: WhisperModel = model
        self.tokenizer = tokenizer
        meta = model.snapshot_metadata
        gen: dict[str, Any] = meta.get('gen_cfg') or {}
        hf: dict[str, Any] = meta['hf_cfg']
        self.mel = LogMelSpectogram.from_snapshot(tensors['mel_filters'], meta['pp_cfg'])
        self.lang2id: dict[str, int] = {k.strip('<|>'): v for k, v in gen['lang_to_id'].items()}
        self.id2lang: dict[int, str] = {v: k for k, v in self.lang2id.items()}
        self.lang2idt = Tensor(list(self.lang2id.values()), dtype=dtype.int64)
        self.task_to_id: dict[str, int] = gen['task_to_id']
        self.sot: int = gen.get('decoder_start_token_id', hf['decoder_start_token_id'])
        self.eos: int = gen.get('eos_token_id', hf['eos_token_id'])
        self.no_timestamps: int = gen.get('no_timestamps_token_id', tokenizer.token_to_id('<|notimestamps|>'))
        self.sot_prev: int = tokenizer.token_to_id('<|startofprev|>')
        self.suppress = self._bias(gen.get('suppress_tokens', hf.get('suppress_tokens')) or [])
        self.begin_suppress = self._bias(gen.get('begin_suppress_tokens', hf.get('begin_suppress_tokens')) or [])

    @classmethod
    @no_grad()
    def load(cls, snapshot: Path) -> WhisperInferenceAdapter:
        from tokenizers import Tokenizer

        tensors, model = WhisperModel.from_snapshot(snapshot)
        return cls(model, tensors, Tokenizer.from_str(model.snapshot_metadata['tokenizer_json']))

    @no_grad()
    def stream(self, req: TranscribeRequest) -> Iterator[TextChunk]:
        sess = _Session(req.language)
        t = 0.0
        first = True
        for win in self._windows(req.audio):
            dur: float = win.numel / SAMPLE_RATE
            enc: Tensor = self._encode(win)
            if sess.lang is None:
                sess.lang = self.id2lang[self.model.detect_language(enc, self.sot, self.lang2idt).item()]
            dec = IncrementalDecoder(self.tokenizer)
            toks: list[int] = []
            prompt = self._prompt(sess, req)
            for tok in self._decode(enc, prompt, req.max_toks):
                toks.append(tok)
                delta = dec.push(tok)
                if first and delta:
                    delta, first = delta.lstrip(), False
                yield TranscribeChunk(delta, token=tok, segment_start=t, segment_end=dur + t)
            if tail := dec.flush():
                yield TranscribeChunk(delta=tail, segment_start=t, segment_end=dur + t)
            if req.condition_on_previous:
                sess.history.extend(toks)
            t += dur

    def _prompt(self, sess: _Session, req: TranscribeRequest) -> list[int]:
        assert sess.lang is not None
        prev = [self.sot_prev, *sess.history[-_PREV_LIM:]] if sess.history else []
        return [*prev, self.sot, self.lang2id[sess.lang], self.task_to_id[req.task.value], self.no_timestamps]

    def _encode(self, win: Tensor) -> Tensor:
        mel = self.mel(win).cast(context.get_default_dtype())
        return self.model.encoder(mel.reshape(1, *mel.shape))

    def _decode(self, enc: Tensor, prompt: list[int], max_toks: int) -> Iterator[int]:
        dec = self.model.decoder
        dec.reset(enc)
        logits = dec(Tensor([prompt], dtype=dtype.int64))
        for i in range(min(max_toks, self.model.cfg.max_target_positions - len(prompt))):
            logits = logits + (self.suppress + self.begin_suppress if i == 0 else self.suppress)
            tok: int = logits.argmax(dim=0).item()
            if tok >= self.eos:
                return
            yield tok
            logits = dec(Tensor([[tok]], dtype=dtype.int64))

    def _windows(self, audio: Path | bytes | Tensor | Iterable[Tensor]) -> Iterator[Tensor]:
        match audio:
            case Path():
                yield from (win for _, win in chunks(load_audio_and_resample(audio)))  # Load from file and chunk
            case bytes():
                yield from (win for _, win in chunks(bytes_to_f32_tensor(audio)))  # Load from bytes buffer and chunk
            case Tensor():
                yield from (win for _, win in chunks(audio))  # Chunk from tensor directly
            case _:
                yield from audio  # Caller chunked already into <= 30s windows

    def _bias(self, token_ids: list[int]) -> Tensor:
        vocab = self.model.cfg.vocab_size
        mask = Tensor(token_ids, dtype=dtype.int64).bincount(minlength=vocab) > 0
        return Tensor.zeros(vocab, dtype=context.get_default_dtype()).masked_fill(mask, float('-inf'))
