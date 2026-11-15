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

from magnetron_models.utils import download_model
from magnetron_models.inference import load, InferenceAdapter
from magnetron_models.inference.adapters import TranscribeRequest, TranscribeChunk

SPEECH_FILE = Path(__file__).parent / 'speech.mp3'

whisper: InferenceAdapter = load(download_model('mario-sieg/magnetron-whisper', 'whisper-base-bf16.mag'))
request = TranscribeRequest(SPEECH_FILE)

# Streaming token by token
chunk: TranscribeChunk
for chunk in whisper.stream(request):
    print(chunk.delta, end='', flush=True)

print()

# Blocking all at once
print(whisper.generate_blocking(request).text)

# For interactive version with microphone, run interactive.py
