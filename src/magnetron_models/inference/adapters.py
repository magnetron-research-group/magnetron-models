# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from magnetron import Tensor
from enum import Enum

from magnetron_models.inference import InferenceAdapter


### Request Data ###


@dataclass
class TextRequest:
    prompt: str | list[dict[str, str]] | Tensor  # raw text or chat msgs or already tokenized ids
    max_toks: int = 512
    temp: float = 0.0
    stop: list[str] = field(default_factory=list)


class TranscribeTask(Enum):
    TRANSCRIBE = 'transcribe'
    TRANSLATE = 'translate'


@dataclass
class TranscribeRequest:
    audio: Path | bytes | Tensor | Iterable[Tensor]  # File path, raw f32 PCM as bytes or tensor, or pre-cut into <= 30s windows
    language: str | None = None  # None = auto detect
    task: TranscribeTask = TranscribeTask.TRANSCRIBE
    max_toks: int = 224  # max tokens per window
    condition_on_previous: bool = True  # feed earlier text as <|startofprev|> context


@dataclass
class ImageRequest:
    prompt: str
    width: int = 512
    height: int = 512
    steps: int = 30
    seed: int | None = None


class StopReason(Enum):
    STOP = 'stop'
    LENGTH = 'length'
    CANCELLED = 'cancelled'


### Response Data ###


@dataclass
class TextResponse:
    text: str
    tokens: list[int]
    stop_reason: StopReason = StopReason.STOP


@dataclass
class ImageResponse:
    pixels: Tensor  # WxHxC RGBA uint8
    width: int
    height: int
    channels: int


### Stream Chunks ###


@dataclass
class TextChunk:
    delta: str
    token: int | None = None


@dataclass
class TranscribeChunk(TextChunk):
    segment_start: float | None = None
    segment_end: float | None = None


@dataclass
class ImageChunk:
    step: int
    total_steps: int
    preview: ImageResponse | None = None


### Base Adapters ###


class TextGenAdapter(InferenceAdapter[TextRequest, TextResponse, TextChunk]):
    def collect_stream(self, req: TextRequest, chunks: Iterator[TextChunk]) -> TextResponse:
        text: list[str] = []
        toks: list[int] = []
        for chunk in chunks:
            text.append(chunk.delta)
            if chunk.token is not None:
                toks.append(chunk.token)
        reason = StopReason.LENGTH if len(toks) >= req.max_toks else StopReason.STOP
        return TextResponse(text=''.join(text), tokens=toks, stop_reason=reason)


class TranscribeAdapter(InferenceAdapter[TranscribeRequest, TextResponse, TranscribeChunk]):
    def collect_stream(self, req: TranscribeRequest, chunks: Iterator[TextChunk]) -> TextResponse:
        text: list[str] = []
        toks: list[int] = []
        for chunk in chunks:
            text.append(chunk.delta)
            if chunk.token is not None:
                toks.append(chunk.token)
        return TextResponse(text=''.join(text), tokens=toks)


class ImageGenAdapter(InferenceAdapter[ImageRequest, ImageResponse, ImageChunk]):
    def collect_stream(self, req: ImageRequest, chunks: Iterator[ImageChunk]) -> ImageResponse:
        last = None
        for chunk in chunks:
            last = chunk
        assert last is not None and last.preview is not None, 'Final chunk must contain the image result'
        return last.preview
