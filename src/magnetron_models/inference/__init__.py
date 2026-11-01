# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import ClassVar, Generic, Self, TypeVar, Iterator
from pathlib import Path
from magnetron_models.utils import read_metadata

Request = TypeVar('Request')
Response = TypeVar('Response')
Chunk = TypeVar('Chunk')


class InferenceAdapter(ABC, Generic[Request, Response, Chunk]):
    arch: ClassVar[str]  # matches meta['arch']

    @classmethod
    @abstractmethod
    def load(cls, snapshot: Path) -> Self: ...

    @abstractmethod
    def stream(self, req: Request) -> Iterator[Chunk]: ...

    @abstractmethod
    def collect_stream(self, req: Request, chunks: Iterator[Chunk]) -> Response: ...

    def generate_blocking(self, req: Request) -> Response:
        return self.collect_stream(req, self.stream(req))


_ADAPTER_REGISTRY: dict[str, type[InferenceAdapter]] = {}


def register_inference_adapter(cls: type[InferenceAdapter]) -> type[InferenceAdapter]:
    assert cls.arch is not None and type(cls.arch) is str
    _ADAPTER_REGISTRY[cls.arch] = cls
    return cls


def load(snapshot: Path) -> InferenceAdapter:
    arch = read_metadata(snapshot)['arch']
    if arch not in _ADAPTER_REGISTRY:
        try:
            importlib.import_module(f'magnetron_models.models.{arch}.inference')
        except ModuleNotFoundError as e:
            if e.name != f'magnetron_models.models.{arch}' and e.name != f'magnetron_models.models.{arch}.inference':
                raise
    if arch not in _ADAPTER_REGISTRY:
        raise KeyError(f'No inference adapter found for {arch!r}')
    return _ADAPTER_REGISTRY[arch].load(snapshot)
