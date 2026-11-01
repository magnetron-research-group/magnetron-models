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


def download_model(repo_id: str, filename: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError('huggingface_hub is required to download models. Install it with: uv sync --group examples') from e
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))
