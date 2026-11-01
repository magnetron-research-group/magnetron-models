# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import queue
from collections.abc import Iterator
import sounddevice as sd
from magnetron import Tensor
from magnetron_models.inference import InferenceAdapter, load
from magnetron_models.inference.adapters import TranscribeChunk, TranscribeRequest
from magnetron_models.models.whisper.audio_processor import SAMPLE_RATE
from magnetron_models.extra.vad import FRAME_LEN, SpeechBandEnergyVAD, segments
from magnetron_models.utils import bytes_to_f32_tensor, download_model

HF_REPO_ID = 'mario-sieg/magnetron-whisper'
HF_MODEL_FILE = 'whisper-base-bf16.mag'
LANGUAGE: str | None = None  # None = auto detect on first speech


def mic_frames(device: int | str | None = None) -> Iterator[Tensor]:
    q: queue.Queue[bytes] = queue.Queue()

    def callback(indata: object, _frames: int, _time: object, status: sd.CallbackFlags) -> None:
        if status:
            print(f'[audio] {status}')
        q.put(bytes(indata))

    with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=FRAME_LEN, device=device, callback=callback):
        while True:
            yield bytes_to_f32_tensor(q.get())


def main() -> None:
    whisper: InferenceAdapter = load(download_model(HF_REPO_ID, HF_MODEL_FILE))
    request = TranscribeRequest(segments(mic_frames(), SpeechBandEnergyVAD()), language=LANGUAGE, condition_on_previous=False)
    print('listening... (ctrl+c to stop)', flush=True)
    seg_start: float | None = None
    chunk: TranscribeChunk
    try:
        for chunk in whisper.stream(request):
            if chunk.segment_start != seg_start:
                if seg_start is not None:
                    print()
                seg_start = chunk.segment_start
                chunk.delta = chunk.delta.lstrip()
            print(chunk.delta, end='', flush=True)
    except KeyboardInterrupt:
        print()


if __name__ == '__main__':
    main()
