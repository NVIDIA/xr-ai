# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the xr-ai-vad utterance detector.

These tests exercise the state machine end-to-end via the adaptive-energy
fallback path so they pass without the silero-vad ONNX model.  The energy
gate is forced active by sabotaging the loaded silero model so the public
API is exercised exactly as a worker would call it.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from xr_ai_vad import VadDetector


SR        = 16_000
CHUNK_S   = 0.02            # 20 ms chunks (matches XR hub default cadence)
CHUNK_N   = int(SR * CHUNK_S)
SILENT    = (np.zeros(CHUNK_N, np.float32)).tobytes()


def _tone_bytes(amp: float, n: int = CHUNK_N) -> bytes:
    """One 20 ms chunk of a 1 kHz sine at the given amplitude."""
    t = np.arange(n, dtype=np.float32) / SR
    return (amp * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32).tobytes()


def _force_energy_fallback(vad: VadDetector) -> None:
    """Disable silero so the energy gate is the sole classifier under test."""
    vad._silero = None  # type: ignore[attr-defined]


async def _feed_many(vad: VadDetector, n: int, chunk: bytes) -> None:
    for _ in range(n):
        await vad.feed(chunk, SR, CHUNK_N)


@pytest.mark.asyncio
async def test_finalize_after_silence_emits_utterance():
    """A burst of speech followed by enough silence triggers on_utterance."""
    received: list[tuple[bytes, int]] = []

    async def on_utt(audio: bytes, sr: int) -> None:
        received.append((audio, sr))

    vad = VadDetector(
        on_utterance      = on_utt,
        silence_threshold = 0.005,
        silence_duration  = 0.10,   # short to keep the test fast
        min_speech        = 0.06,
        vad_noise_mult    = 2.0,
    )
    _force_energy_fallback(vad)

    # 8 × 20 ms = 160 ms of speech (> min_speech).
    await _feed_many(vad, 8, _tone_bytes(0.5))
    # 7 × 20 ms = 140 ms of silence (> silence_duration).
    await _feed_many(vad, 7, SILENT)

    assert len(received) == 1
    audio, sr = received[0]
    assert sr == SR
    # Output is int16 PCM — total bytes should match the audio held.
    assert len(audio) % 2 == 0
    # Sanity: utterance contains the speech we fed in (>= 160 ms worth).
    assert len(audio) // 2 >= int(SR * 0.16)


@pytest.mark.asyncio
async def test_speech_start_fires_once_at_min_speech_crossing():
    """on_speech_start should fire exactly once per utterance, at the moment
    cumulative speech first exceeds min_speech."""
    starts:    list[int] = []
    finalized: list[int] = []
    finalize_evt = asyncio.Event()

    async def on_start() -> None:
        starts.append(1)

    async def on_utt(_audio: bytes, _sr: int) -> None:
        finalized.append(1)
        finalize_evt.set()

    vad = VadDetector(
        on_utterance      = on_utt,
        on_speech_start   = on_start,
        silence_threshold = 0.005,
        silence_duration  = 0.10,
        min_speech        = 0.06,
        vad_noise_mult    = 2.0,
    )
    _force_energy_fallback(vad)

    # 10 × 20 ms = 200 ms of speech — well past min_speech (60 ms).
    await _feed_many(vad, 10, _tone_bytes(0.5))
    # Let the on_speech_start task (scheduled via create_task) run.
    await asyncio.sleep(0)
    assert starts == [1], "on_speech_start should fire exactly once per utterance"

    # Then silence to finalize.
    await _feed_many(vad, 7, SILENT)
    await asyncio.wait_for(finalize_evt.wait(), timeout=1.0)
    assert finalized == [1]

    # Start a second utterance — on_speech_start should fire again.
    await _feed_many(vad, 10, _tone_bytes(0.5))
    await asyncio.sleep(0)
    assert starts == [1, 1], "on_speech_start should re-arm for the next utterance"


@pytest.mark.asyncio
async def test_below_min_speech_does_not_emit():
    """Speech that does not cross min_speech must not produce an utterance."""
    received: list[bytes] = []

    async def on_utt(audio: bytes, _sr: int) -> None:
        received.append(audio)

    vad = VadDetector(
        on_utterance      = on_utt,
        silence_threshold = 0.005,
        silence_duration  = 0.10,
        min_speech        = 0.5,    # 500 ms — well above what we'll feed
        vad_noise_mult    = 2.0,
    )
    _force_energy_fallback(vad)

    # 4 × 20 ms = 80 ms of speech, below min_speech.
    await _feed_many(vad, 4, _tone_bytes(0.5))
    await _feed_many(vad, 10, SILENT)

    assert received == []


@pytest.mark.asyncio
async def test_reset_drops_in_progress_utterance():
    """reset() must drop buffered speech without invoking on_utterance."""
    received: list[bytes] = []

    async def on_utt(audio: bytes, _sr: int) -> None:
        received.append(audio)

    vad = VadDetector(
        on_utterance      = on_utt,
        silence_threshold = 0.005,
        silence_duration  = 0.10,
        min_speech        = 0.06,
        vad_noise_mult    = 2.0,
    )
    _force_energy_fallback(vad)

    await _feed_many(vad, 8, _tone_bytes(0.5))
    vad.reset()
    await _feed_many(vad, 10, SILENT)

    assert received == [], "reset() should drop the in-progress utterance"
