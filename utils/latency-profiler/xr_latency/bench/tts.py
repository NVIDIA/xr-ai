# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TTS latency: batch synthesis vs streamed first chunk, against text length."""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass

from xr_ai_models import TTSService

from .._gpu import load_tags
from .._store import RunStore
from .._timing import now_ms

_SENTENCE = "The second gauge shows a slow drift in temperature, so please recheck it before the afternoon shift. "
CHAR_LENGTHS = (20, 60, 120, 240, 480)


@dataclass
class TtsPlan:
    """Sample count and text lengths in characters."""

    samples: int = 15
    lengths: tuple[int, ...] = CHAR_LENGTHS


def _text(chars: int) -> str:
    return (_SENTENCE * (chars // len(_SENTENCE) + 1))[:chars].rstrip() + "."


def _wav_seconds(data: bytes) -> float:
    with wave.open(io.BytesIO(data)) as w:
        return w.getnframes() / w.getframerate()


async def run(tts: TTSService, store: RunStore, plan: TtsPlan) -> None:
    """Record batch total, streamed first-chunk and total time, and audio duration for each text length."""
    stream = getattr(tts, "stream", None)
    for chars in plan.lengths:
        text = _text(chars)
        for i in range(plan.samples + 1):
            warm = i > 0
            start = now_ms()
            audio = await tts.synthesize(text, response_format="wav")
            total = now_ms() - start
            seconds = _wav_seconds(audio)
            store.add("tts.batch", {"chars": chars, "warm": warm, **load_tags()},
                      total=total, audio_s=seconds, rtf=total / 1000.0 / seconds)
            if stream is None:
                continue
            start = now_ms()
            first, chunks, pcm_bytes, rate = None, 0, 0, 24_000
            async for chunk in stream(text):
                if first is None:
                    first = now_ms() - start
                chunks += 1
                pcm_bytes += len(chunk.data)
                rate = chunk.sample_rate
            total = now_ms() - start
            seconds = pcm_bytes / 2 / rate
            store.add("tts.stream", {"chars": chars, "warm": warm, **load_tags()},
                      first_chunk=first, total=total, chunks=chunks, audio_s=seconds, rtf=total / 1000.0 / seconds)
