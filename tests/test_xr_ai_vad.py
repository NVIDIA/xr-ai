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
        assert sample_rate == 16_000
        assert tensor.numel() == 512
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
    assert starts == [1], "on_speech_start should fire exactly once per utterance"

    # Allow the mixed speech/silence window plus four complete quiet windows.
    await _feed_many(vad, 8, SILENT)
    await asyncio.wait_for(finalize_evt.wait(), timeout=1.0)
    assert finalized == [1]

    # Start a second utterance — on_speech_start should fire again.
    await _feed_many(vad, 10, _tone_bytes(0.5))
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


async def _feed_parts(vad, audio, sample_rate, frame_ms):
    frame_bytes = sample_rate * frame_ms // 1000 * 2
    for offset in range(0, len(audio), frame_bytes):
        await vad.feed(audio[offset:offset + frame_bytes], sample_rate)
        await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("frame_ms", [10, 20, 32, 40, 100])
@pytest.mark.parametrize("sample_rate", [16_000, 48_000])
async def test_window_timing_and_original_pcm_are_packet_independent(frame_ms, sample_rate):
    received = []
    starts = []
    elapsed = 0

    async def on_utt(audio, rate):
        received.append((audio, rate))

    async def on_start():
        starts.append(elapsed / sample_rate)

    vad = VadDetector(on_utt, on_speech_start=on_start, min_speech=0.15, silence_duration=0.09)
    _install_stub(vad)
    # Ten speech windows then three silence windows: finalization covers
    # exactly this audio, even when a packet straddles their boundary.
    speech = np.full(sample_rate * 320 // 1000, 10_000, np.int16)
    quiet = np.zeros(sample_rate * 96 // 1000, np.int16)
    audio = np.concatenate([speech, quiet]).tobytes()
    frame_bytes = sample_rate * frame_ms // 1000 * 2
    for offset in range(0, len(audio), frame_bytes):
        part = audio[offset:offset + frame_bytes]
        elapsed += len(part) // 2
        await vad.feed(part, sample_rate)
        await asyncio.sleep(0)

    assert len(starts) == 1
    assert 0.16 <= starts[0] < 0.16 + frame_ms / 1000 + 1 / sample_rate
    assert received == [(audio, sample_rate)]


@pytest.mark.asyncio
@pytest.mark.parametrize("sample_rate", [8_000, 44_100])
async def test_source_rate_window_boundaries_do_not_drift(sample_rate):
    received = []

    async def on_utt(audio, rate):
        received.append((audio, rate))

    vad = VadDetector(on_utt, min_speech=0.15, silence_duration=0.09)
    _install_stub(vad)
    # Fifty windows is an integral source-sample count at both rates.
    speech = np.full(sample_rate * 1600 // 1000, 10_000, np.int16)
    quiet = np.zeros(sample_rate * 96 // 1000, np.int16)
    audio = np.concatenate([speech, quiet]).tobytes()
    await _feed_parts(vad, audio, sample_rate, 10)
    assert received == [(audio, sample_rate)]


@pytest.mark.asyncio
async def test_short_speech_in_ten_ms_packets_finalizes():
    received = []

    async def on_utt(audio, rate):
        received.append((audio, rate))

    vad = VadDetector(on_utt, min_speech=0.15, silence_duration=0.4)
    _install_stub(vad)
    speech = np.full(SR // 5, 10_000, np.int16).tobytes()
    audio = speech + np.zeros(SR, np.int16).tobytes()
    await _feed_parts(vad, audio, SR, 10)
    assert len(received) == 1
    assert received[0][1] == SR
    assert received[0][0].startswith(speech)


@pytest.mark.asyncio
@pytest.mark.parametrize("frame_ms", [10, 1000])
async def test_multiple_utterances_in_one_feed_keep_audio_order(frame_ms):
    received = []
    events = []

    async def on_start():
        events.append("start")
        await asyncio.sleep(0)
        events.append("start complete")

    async def on_utt(audio, _rate):
        events.append("utterance")
        received.append(audio)

    vad = VadDetector(on_utt, on_speech_start=on_start, min_speech=0.06, silence_duration=0.06)
    _install_stub(vad)
    first = np.concatenate([np.full(3 * 512, 8_000, np.int16), np.zeros(2 * 512, np.int16)]).tobytes()
    second = np.concatenate([np.full(3 * 512, 16_000, np.int16), np.zeros(2 * 512, np.int16)]).tobytes()
    await _feed_parts(vad, first + second, SR, frame_ms)
    assert received == [first, second]
    assert events == ["start", "start complete", "utterance"] * 2


@pytest.mark.asyncio
async def test_feed_cancellation_cancels_awaited_speech_start():
    entered = asyncio.Event()
    exited = asyncio.Event()
    finalized = []

    async def on_start():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            exited.set()

    async def on_utt(audio, _rate):
        finalized.append(audio)

    vad = VadDetector(on_utt, on_speech_start=on_start, min_speech=0.06, silence_duration=0.06)
    _install_stub(vad)
    audio = np.concatenate([np.full(3 * 512, 10_000, np.int16), np.zeros(2 * 512, np.int16)]).tobytes()
    feed = asyncio.create_task(vad.feed(audio, SR))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        feed.cancel()
        with pytest.raises(asyncio.CancelledError):
            await feed
        assert exited.is_set()
        assert finalized == []
    finally:
        feed.cancel()
        await asyncio.gather(feed, return_exceptions=True)


@pytest.mark.asyncio
async def test_speech_start_failure_is_isolated_before_finalization(caplog):
    events = []

    async def on_start():
        events.append("start")
        raise RuntimeError("callback failed")

    async def on_utt(_audio, _rate):
        events.append("utterance")

    vad = VadDetector(on_utt, on_speech_start=on_start, min_speech=0.06, silence_duration=0.06)
    _install_stub(vad)
    audio = np.concatenate([np.full(3 * 512, 10_000, np.int16), np.zeros(2 * 512, np.int16)]).tobytes()
    await vad.feed(audio, SR)
    assert events == ["start", "utterance"]
    assert "on_speech_start callback raised" in caplog.text


@pytest.mark.asyncio
async def test_pre_roll_is_bounded_by_audio_duration():
    received = []

    async def on_utt(audio, _rate):
        received.append(audio)

    vad = VadDetector(on_utt, min_speech=0.06, silence_duration=0.06)
    _install_stub(vad)
    # Distinct quiet samples make any dropped, duplicated or reordered bytes visible.
    quiet = np.repeat(np.arange(1, 21, dtype=np.int16), 512).tobytes()
    speech = np.full(2 * 512, 10_000, np.int16).tobytes()
    tail = np.zeros(2 * 512, np.int16).tobytes()
    await _feed_parts(vad, quiet + speech + tail, SR, 10)
    assert received == [quiet[-10 * 512 * 2:] + speech + tail]


@pytest.mark.asyncio
async def test_reset_discards_unclassified_pcm():
    received = []

    async def on_utt(audio, _rate):
        received.append(audio)

    vad = VadDetector(on_utt, min_speech=0.01, silence_duration=0.06)
    _install_stub(vad)
    await vad.feed(np.full(160, 10_000, np.int16).tobytes(), SR)
    vad.reset()
    await vad.feed(np.zeros(4 * 512, np.int16).tobytes(), SR)
    assert received == []


@pytest.mark.asyncio
async def test_max_duration_keeps_remaining_audio_from_same_feed(monkeypatch):
    from xr_ai_vad import detector

    monkeypatch.setattr(detector, "_MAX_UTT_S", 0.1)
    received = []

    async def on_utt(audio, _rate):
        received.append(audio)

    vad = VadDetector(on_utt, min_speech=0.06, silence_duration=0.06)
    _install_stub(vad)
    audio = np.concatenate([np.full(10 * 512, 10_000, np.int16), np.zeros(2 * 512, np.int16)]).tobytes()
    await vad.feed(audio, SR)
    assert len(received) == 3
    assert b"".join(received) == audio
