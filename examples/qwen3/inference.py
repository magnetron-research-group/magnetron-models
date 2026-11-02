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
from magnetron_models.inference import load, InferenceAdapter
from magnetron_models.inference.adapters import TextRequest, TextChunk

# TODO: Upload
SNAPSHOT_FILE = Path('/home/mario/db/qwen3-4b-instruct-2507-bf16.mag')

qwen: InferenceAdapter = load(SNAPSHOT_FILE)
request = TextRequest(
    prompt=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Explain in two sentences why the sky is blue.'},
    ],
    max_toks=256,
    temp=0.7,
    topk=20,
)

chunk: TextChunk
for chunk in qwen.stream(request):
    print(chunk.delta, end='', flush=True)
print()
