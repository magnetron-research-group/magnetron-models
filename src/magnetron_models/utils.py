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
from typing import Any
from magnetron import SnapshotStreamReader
from magnetron.snapshot import _FileManifest


def read_metadata(file_path: str | Path) -> dict[str, Any]:
    with SnapshotStreamReader(str(file_path)) as reader:
        return dict(_FileManifest.deserialize(reader.metadata).usr_metadata)
    raise RuntimeError(f'Failed to read snapshot metadata from: {file_path!r}')


def bytes_to_f32_tensor(buf: bytes) -> Tensor:
    assert (len(buf) & 3) == 0
    tensor = Tensor.empty(len(buf) >> 2, dtype=dtype.float32, device='cpu')
    tensor.sync()
    ctypes.memmove(tensor.data_ptr_mut, buf, len(buf))
    tensor.sync()
    return tensor
