# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Silero ``VadDetector`` behaviour against input frame size: when does it fire, and does it fire at all.

Runs on the CPU and uses audio time, not wall time: speech is fed as fast as possible and each detector event is
stamped with the number of audio seconds fed so far. ``cpu_ms_per_frame`` is the real per-frame cost.
"""

from __future__ import annotations

import asyncio
import io
import time
import wave
from dataclasses import dataclass

import numpy as np
from xr_ai_models import TTSService
from xr_ai_vad import VadDetector

from .._store import RunStore
from .stt import UTTERANCES, _pcm16k

_RATE = 16_000


@dataclass
class VadPlan:
    """Frame sizes (ms) and silence settings to compare."""

    samples: int = 3
    frame_ms: tuple[int, ...] = (10, 20, 32, 40, 100)
    silence_durations: tuple[float, ...] = (0.4, 0.8)
    lead_silence_s: float = 1.0
    trail_silence_s: float = 3.0


def _wav_len(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        return w.getnframes() / w.getframerate()


async def run(tts: TTSService, store: RunStore, plan: VadPlan) -> None:
    """Feed synthesized utterances in fixed frames; record start/finalize audio-time and CPU cost."""
    speech = {name: _pcm16k(await tts.synthesize(text, response_format="wav")) for name, text in UTTERANCES.items()}
    for name, audio in speech.items():
        lead = np.zeros(int(plan.lead_silence_s * _RATE), "<i2")
        trail = np.zeros(int(plan.trail_silence_s * _RATE), "<i2")
        pcm = np.concatenate([lead, audio, trail])
        speech_start_s, speech_end_s = plan.lead_silence_s, plan.lead_silence_s + audio.size / _RATE
        for silence in plan.silence_durations:
            for frame_ms in plan.frame_ms:
                for _ in range(plan.samples):
                    await _one(store, name, pcm, frame_ms, silence, speech_start_s, speech_end_s)


async def _one(store: RunStore, name: str, pcm: np.ndarray, frame_ms: int, silence: float,
               speech_start_s: float, speech_end_s: float) -> None:
    """Feed the whole clip, counting every finalization (a pause inside the speech can split it)."""
    fed = 0.0
    finals: list[tuple[float, float]] = []  # (audio time fed at finalization, utterance seconds)
    starts: list[float] = []

    async def on_utterance(audio: bytes, rate: int) -> None:
        finals.append((fed, len(audio) / 2 / rate))

    async def on_start() -> None:
        starts.append(fed)

    detector = VadDetector(on_utterance, on_speech_start=on_start, silence_duration=silence)
    step = _RATE * frame_ms // 1000
    spent = 0.0
    frames = 0
    for offset in range(0, pcm.size - step + 1, step):
        fed = (offset + step) / _RATE
        t = time.perf_counter()
        await detector.feed(pcm[offset:offset + step].tobytes(), _RATE)
        spent += time.perf_counter() - t
        frames += 1
        await asyncio.sleep(0)  # let the detector's speech-start task run, as the real event loop would
    last = finals[-1][0] if finals else None
    store.add(
        "vad", {"utterance": name, "frame_ms": frame_ms, "silence_duration": silence,
                "speech_s": round(speech_end_s - speech_start_s, 2)},
        fired=bool(finals), n_finals=len(finals),
        start_lag=(starts[0] - speech_start_s) if starts else None,
        first_final_vs_speech_end=(finals[0][0] - speech_end_s) if finals else None,
        end_to_final=(last - speech_end_s) if last is not None else None,
        utt_s=finals[0][1] if finals else None, cpu_ms_per_frame=spent / frames * 1000.0, frames=frames,
    )
