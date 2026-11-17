# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterator
from itertools import pairwise

from magnetron import Tensor, dtype

from magnetron_models.models.whisper.audio_processor import SAMPLE_RATE

FRAME_MS: int = 30
FRAME_LEN: int = SAMPLE_RATE * FRAME_MS // 1000


class SpeechBandEnergyVAD:
    def __init__(
        self,
        lo_hz: float = 200.0,
        hi_hz: float = 3500.0,
        on_db: float = 6.0,
        off_db: float = 3.0,
        min_db: float = -60.0,
        floor_min_db: float = -90.0,
        fall: float = 0.2,
        min_window_ms: int = 2500,
        stationary_ms: int = 1500,
        stationary_db: float = 1.0,
    ) -> None:
        self.on_db = on_db
        self.off_db = off_db
        self.min_db = min_db
        self.floor_min_db = floor_min_db
        self.fall = fall
        self.history: deque[float] = deque(maxlen=max(2, min_window_ms // FRAME_MS))
        self.stationary_db = stationary_db
        self.window: deque[float] = deque(maxlen=max(2, stationary_ms // FRAME_MS))
        self.floor: float | None = None
        self.voiced = False
        self.snapped = False
        self.db = floor_min_db
        n = FRAME_LEN
        k = Tensor.arange(round(lo_hz * n / SAMPLE_RATE), round(hi_hz * n / SAMPLE_RATE) + 1, dtype=dtype.float32).reshape(1, -1)
        t = Tensor.arange(0, n, dtype=dtype.float32).reshape(-1, 1)
        hann = 0.5 - 0.5 * (t * (2 * math.pi / n)).cos()
        phase = t * k * (2 * math.pi / n)
        self.cos_basis = (hann * phase.cos()).contiguous()
        self.sin_basis = (hann * phase.sin()).contiguous()
        self.norm = 2.0 / (n * hann.sqr().sum().item())

    def energy_db(self, frame: Tensor) -> float:
        x = frame.reshape(1, -1)
        power = (x @ self.cos_basis).sqr().sum() + (x @ self.sin_basis).sqr().sum()
        return 10.0 * math.log10(power.item() * self.norm + 1e-20)

    def __call__(self, frame: Tensor) -> bool:
        db = self.db = self.energy_db(frame)
        self.window.append(db)
        self.history.append(db)
        if self.floor is None:
            self.floor = db
        if db < self.floor:
            self.floor += self.fall * (db - self.floor)
        self.floor = max(self.floor, min(self.history))
        self.snapped = False
        if len(self.window) == self.window.maxlen:
            mean = sum(self.window) / len(self.window)
            std = math.sqrt(sum((v - mean) ** 2 for v in self.window) / len(self.window))
            if std < self.stationary_db and mean > self.floor + self.off_db:
                self.floor, self.snapped = mean, True
        self.floor = max(self.floor, self.floor_min_db)
        thresh = self.off_db if self.voiced else self.on_db
        self.voiced = db > self.floor + thresh and db > self.min_db
        return self.voiced


def segments(
    frames: Iterator[Tensor],
    vad: SpeechBandEnergyVAD,
    start_frames: int = 3,
    end_silence_ms: int = 600,
    min_end_silence_ms: int = 120,
    short_end_silence_ms: int = 1000,
    short_s: float = 1.5,
    target_seg_s: float = 5.0,
    preroll_ms: int = 300,
    tail_ms: int = 300,
    max_seg_s: float = 12.0,
    split_search_s: float = 3.0,
    min_speech_s: float = 0.4,
    min_flux_db: float = 0.5,
) -> Iterator[Tensor]:
    ms = lambda v: max(1, int(v) // FRAME_MS)
    preroll: deque[Tensor] = deque(maxlen=ms(preroll_ms))
    end_frames, min_end_frames = ms(end_silence_ms), ms(min_end_silence_ms)
    short_end_frames, short_frames, target_frames = ms(short_end_silence_ms), ms(short_s * 1000), ms(target_seg_s * 1000)

    def required_silence(speech_frames: int) -> int:
        if speech_frames < short_frames:
            return short_end_frames
        t = min(1.0, (speech_frames - short_frames) / max(1, target_frames - short_frames))
        return round(end_frames + t * (min_end_frames - end_frames))

    tail_frames, max_frames, search_frames = ms(tail_ms), ms(max_seg_s * 1000), ms(split_search_s * 1000)
    min_voiced = ms(min_speech_s * 1000)
    seg: list[Tensor] = []
    dbs: list[float] = []
    voiced_run = silence_run = n_voiced = 0

    def emit(frames_: list[Tensor], dbs_: list[float], n: int) -> Iterator[Tensor]:
        if n < min_voiced or not frames_:
            return
        if len(dbs_) > 1 and sum(abs(b - a) for a, b in pairwise(dbs_)) / (len(dbs_) - 1) < min_flux_db:
            return
        yield Tensor.cat(frames_, dim=0)

    for frame in frames:
        voiced = vad(frame)
        voiced_run = voiced_run + 1 if voiced else 0
        if not seg:
            preroll.append(frame)
            if voiced_run >= start_frames:
                seg, dbs = list(preroll), [vad.floor or vad.db] * len(preroll)
                preroll.clear()
                silence_run, n_voiced = 0, voiced_run
            continue
        if vad.snapped:
            keep = max(0, len(seg) - len(vad.window))
            if sum(d > vad.floor + vad.on_db for d in dbs[:keep]) >= min_voiced:
                yield from emit(seg[:keep], dbs[:keep], n_voiced)
            seg, dbs = [], []
            continue
        seg.append(frame)
        dbs.append(vad.db)
        n_voiced += voiced
        silence_run = 0 if voiced_run >= start_frames else silence_run + 1
        speech_frames = len(seg) - silence_run
        if silence_run >= required_silence(speech_frames):
            yield from emit(seg[: speech_frames + tail_frames], dbs[:speech_frames], n_voiced)
            seg, dbs = [], []
        elif len(seg) >= max_frames:
            cut = min(range(len(seg) - search_frames, len(seg)), key=dbs.__getitem__) + 1
            yield from emit(seg[:cut], dbs[:cut], n_voiced)
            seg, dbs = seg[cut:], dbs[cut:]
            n_voiced = len(seg)
    if seg:
        speech_frames = len(seg) - silence_run
        yield from emit(seg[: speech_frames + tail_frames], dbs[:speech_frames], n_voiced)
