# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the xr-ai-vad utterance detector.

These tests exercise the state machine end-to-end with the Silero classifier
swapped for a deterministic stub.  The stub looks at the int16 PCM bytes it
receives and returns a probability of 0.9 for non-zero audio (speech) and
0.0 for silence, so the VadDetector boundary contract is unchanged.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from xr_ai_vad import VadDetector


SR        = 16_000
CHUNK_S   = 0.02            # 20 ms chunks (matches XR hub default cadence)
CHUNK_N   = int(SR * CHUNK_S)
SILENT    = (np.zeros(CHUNK_N, np.int16)).tobytes()


def _tone_bytes(amp: float, n: int = CHUNK_N) -> bytes:
    """One 20 ms chunk of a 1 kHz sine at the given amplitude as int16 PCM."""
    t = np.arange(n, dtype=np.float32) / SR
    f32 = amp * np.sin(2 * np.pi * 1000.0 * t)
    return (f32 * 32767).astype(np.int16).tobytes()


class _StubSilero:
    """Stand-in for the silero model: speech prob is high when audio is loud."""

    def __init__(self, threshold_rms: float = 0.005) -> None:
        self._threshold = threshold_rms

    def __call__(self, tensor, sample_rate: int) -> float:
        arr = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
        rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2))) if arr.size else 0.0
        return 0.9 if rms >= self._threshold else 0.0

    def reset_states(self) -> None:
        pass


def _install_stub(vad: VadDetector) -> None:
    """Replace the loaded silero model with a deterministic stub."""
    vad._silero = _StubSilero()  # type: ignore[attr-defined]


async def _feed_many(vad: VadDetector, n: int, chunk: bytes) -> None:
    for _ in range(n):
        await vad.feed(chunk, SR)


@pytest.mark.asyncio
async def test_finalize_after_silence_emits_utterance():
    """A burst of speech followed by enough silence triggers on_utterance."""
    received: list[tuple[bytes, int]] = []

    async def on_utt(audio: bytes, sr: int) -> None:
        received.append((audio, sr))

    vad = VadDetector(
        on_utterance      = on_utt,
        silence_duration  = 0.10,   # short to keep the test fast
        min_speech        = 0.06,
        silero_threshold  = 0.5,
    )
    _install_stub(vad)

    # 8 × 20 ms = 160 ms of speech (> min_speech).
    await _feed_many(vad, 8, _tone_bytes(0.5))
    # 7 × 20 ms = 140 ms of silence (> silence_duration).
    await _feed_many(vad, 7, SILENT)

    assert len(received) == 1
    audio, sr = received[0]
    assert sr == SR
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
        silence_duration  = 0.10,
        min_speech        = 0.06,
        silero_threshold  = 0.5,
    )
    _install_stub(vad)

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
        silence_duration  = 0.10,
        min_speech        = 0.5,    # 500 ms — well above what we'll feed
        silero_threshold  = 0.5,
    )
    _install_stub(vad)

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
        silence_duration  = 0.10,
        min_speech        = 0.06,
        silero_threshold  = 0.5,
    )
    _install_stub(vad)

    await _feed_many(vad, 8, _tone_bytes(0.5))
    vad.reset()
    await _feed_many(vad, 10, SILENT)

    assert received == [], "reset() should drop the in-progress utterance"


@pytest.mark.asyncio
async def test_real_silero_model_loads_in_consumer_venv():
    """Construct a `VadDetector` with NO stub injection — exercises the real
    `load_silero_vad(onnx=True)` import path so a missing `onnxruntime` (or
    any other runtime dep that `silero-vad` lists as optional but our ONNX
    backend requires) fails at this test rather than at first-mic-input in a
    consumer worker.

    The state-machine tests above stub `_silero` after construction, which
    masks dependency gaps in the real load. This is the regression guard for
    the bug where `simple-vlm-example`'s venv didn't have `onnxruntime` and
    crashed at runtime instead of at `uv sync`.
    """
    received: list[bytes] = []

    async def on_utt(audio: bytes, _sr: int) -> None:
        received.append(audio)

    # No `_install_stub` — the real silero ONNX model must load + run.
    vad = VadDetector(
        on_utterance      = on_utt,
        silence_duration  = 0.10,
        min_speech        = 0.06,
        silero_threshold  = 0.5,
    )

    # One round-trip through the real classifier to prove the inference path
    # also works (not just the import). Silero's float32 expectations require
    # real audio shape, so we send a non-silent chunk.
    await vad.feed(_tone_bytes(0.5), SR)
    # Feed silence long enough to flush whatever decision silero made; we
    # don't assert on the outcome (the real model may classify a 1 kHz tone
    # as not-speech), only on the lack of exceptions.
    await _feed_many(vad, 10, SILENT)


@pytest.mark.asyncio
async def test_real_silero_accepts_48khz_via_internal_resample():
    """48 kHz audio from a WebRTC track must classify without raising.

    Silero v5 only accepts 16 kHz / 8 kHz, so the detector resamples to
    16 kHz internally. This test feeds chunks at 48 kHz (LiveKit's default
    publish rate) and asserts the real silero classify path runs cleanly.
    Regression guard for the runtime crash where the detector forwarded
    a 48 kHz `sample_rate` straight to silero and got
    `ValueError: Input audio chunk is too short` because the 512-sample
    16 kHz window is ~10.6 ms at 48 kHz — silero's `_validate_input` for
    the (resampled) 48 kHz request expects 1536+ samples.
    """
    SR_48K = 48_000
    chunk_n = int(SR_48K * 0.02)  # 20 ms at 48 kHz = 960 samples
    t = np.arange(chunk_n, dtype=np.float32) / SR_48K
    speech_48k = ((0.5 * np.sin(2 * np.pi * 1000.0 * t)) * 32767).astype(np.int16).tobytes()
    silent_48k = (np.zeros(chunk_n, np.int16)).tobytes()

    received: list[bytes] = []

    async def on_utt(audio: bytes, _sr: int) -> None:
        received.append(audio)

    # No stub — real silero must accept the resampled stream.
    vad = VadDetector(
        on_utterance      = on_utt,
        silence_duration  = 0.10,
        min_speech        = 0.06,
        silero_threshold  = 0.5,
    )

    # Feed enough 48 kHz chunks that the resample buffer crosses
    # silero's 512-sample window at 16 kHz several times.
    for _ in range(10):
        await vad.feed(speech_48k, SR_48K)
    for _ in range(10):
        await vad.feed(silent_48k, SR_48K)

    # No assertion on `received` — silero may or may not classify a 1 kHz
    # tone as speech. The contract under test is "no ValueError raised".
