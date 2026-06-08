# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the xr-ai-pipecat unified voice pipeline.

Each library FrameProcessor (VadStt, VoiceGate, Brain, StreamingTts) is
exercised in isolation with mocked dependencies (VAD, STT, TTS, gate).
The factory is smoke-tested by composing a minimal end-to-end pipeline
and confirming an audio in / audio out round-trip.

Tests use pipecat's :class:`PipelineWorker` / :class:`WorkerRunner` for
the full lifecycle (setup → StartFrame → process → EndFrame) and a
``_CaptureSink`` processor at the tail to collect emitted frames. This
hits the same code paths the real worker does, so test results reflect
what a deployed pipeline will see.
"""
from __future__ import annotations

import asyncio
import gc
import io
import wave
import warnings
from typing import AsyncIterator, Sequence

import numpy as np
import pytest
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

from xr_ai_pipecat import (
    BrainProcessor,
    BrainResponseEndFrame,
    GatedQueryFrame,
    ParticipantJoinedFrame,
    ParticipantLeftFrame,
    StreamingTtsProcessor,
    VadConfig,
    VadSttProcessor,
    VoiceGateProcessor,
)
from xr_ai_voicegate import VoiceGate, VoiceGateConfig


# ── helpers ─────────────────────────────────────────────────────────────────


def _silence_wav(sample_rate: int = 22050, ms: int = 40) -> bytes:
    n = max(1, int(sample_rate * ms / 1000))
    pcm = np.zeros(n, dtype=np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


class _CaptureSink(FrameProcessor):
    """Tail processor — collects every downstream frame it sees.

    ``enable_direct_mode`` skips the internal queue/task so frames land
    in ``self.frames`` synchronously, making assertion order obvious.
    Frames are forwarded so EndFrame can reach the Pipeline sink and
    signal the worker to shut down.
    """

    def __init__(self) -> None:
        super().__init__(enable_direct_mode=True)
        self.frames: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        self.frames.append(frame)
        await self.push_frame(frame, direction)


async def _run_chain(
    *processors: FrameProcessor,
    sends: Sequence[Frame],
    settle_s: float = 0.1,
    per_send_delay_s: float = 0.0,
) -> _CaptureSink:
    """Build a Pipeline(processors), start a PipelineWorker, feed
    ``sends`` through the worker's downstream queue, then drain with an
    ``EndFrame``. Returns the capture sink holding every downstream
    frame seen at the tail. The worker drives StartFrame propagation
    itself, so no manual setup is needed.

    ``per_send_delay_s`` introduces a sleep between queued frames so
    earlier ones can start executing before the next arrives — useful
    for interruption tests that need a previous frame to actually start
    work before the interrupt lands.
    """
    sink = _CaptureSink()
    pipeline = Pipeline([*processors, sink])
    worker = PipelineWorker(
        pipeline,
        cancel_on_idle_timeout = False,
        enable_rtvi            = False,
    )
    runner = WorkerRunner()
    await runner.add_workers(worker)

    async def drive() -> None:
        # The runner's setup happens inside .run(); give it a tick to
        # push StartFrame through every processor before we feed data.
        await asyncio.sleep(0.05)
        for i, f in enumerate(sends):
            await worker.queue_frame(f)
            if i < len(sends) - 1 and per_send_delay_s:
                await asyncio.sleep(per_send_delay_s)
        await asyncio.sleep(settle_s)
        await worker.queue_frame(EndFrame())

    await asyncio.gather(runner.run(), drive())
    return sink


class _FakeStt:
    """STTService double — returns canned text or raises on demand."""

    def __init__(self, text: str = "hello world") -> None:
        self.text         = text
        self.calls:        list[tuple[bytes, int]] = []
        self.raise_on_call = False

    async def transcribe(self, audio: bytes, *, sample_rate: int | None = None, channels: int = 1, timeout: float | None = None) -> str:
        self.calls.append((audio, sample_rate or 16000))
        if self.raise_on_call:
            raise RuntimeError("stt down")
        return self.text

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        pass


class _FakeTts:
    """TTSService double returning a tiny valid WAV at a fixed rate."""

    def __init__(self, sample_rate: int = 22050) -> None:
        self.sample_rate    = sample_rate
        self.calls:         list[str] = []
        self.raise_on_call  = False
        self.delay_s:       float = 0.0

    async def synthesize(self, text: str, *, response_format: str = "wav", timeout: float | None = None) -> bytes:
        self.calls.append(text)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.raise_on_call:
            raise RuntimeError("tts down")
        return _silence_wav(self.sample_rate)

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        pass


class _NullSink:
    async def play_wav(self, pid: str, wav_bytes: bytes) -> None:
        return


class _CallbackStubEndpoint:
    """Endpoint stub that records the audio / participant callbacks the
    input transport registers in its ``__init__``. ``stop`` is a no-op so
    tests can flip ``transport._started`` directly without the ZMQ run
    loop. Shared by the InputTransport audio/participant routing tests."""

    def __init__(self) -> None:
        self.audio_cb = None
        self.participant_cb = None

    def on_audio(self, cb) -> None:
        self.audio_cb = cb

    def on_participant(self, cb) -> None:
        self.participant_cb = cb

    def stop(self) -> None:
        return


# ════════════════════════════════════════════════════════════════════════════
# VadSttProcessor
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_vad_stt_emits_transcription_on_utterance(monkeypatch):
    """When the underlying VadDetector calls back with an utterance,
    the processor pushes ``UserStoppedSpeakingFrame`` then a
    ``TranscriptionFrame`` carrying the STT result."""
    stt = _FakeStt(text="hello agent")

    class _StubVad:
        def __init__(self, on_utterance, on_speech_start, **_):
            self._on_utt   = on_utterance
            self._on_start = on_speech_start

        async def feed(self, pcm_int16: bytes, sample_rate: int) -> None:
            await self._on_start()
            await self._on_utt(pcm_int16, sample_rate)

    monkeypatch.setattr("xr_ai_pipecat.processors.vad_stt.VadDetector", _StubVad)
    proc = VadSttProcessor(stt=stt, vad_cfg=VadConfig())
    frame = InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=16000, num_channels=1)
    frame.transport_source = "web-client"

    sink = await _run_chain(proc, sends=[frame])

    kinds = [type(f).__name__ for f in sink.frames]
    assert "UserStartedSpeakingFrame" in kinds
    assert "UserStoppedSpeakingFrame" in kinds
    transcripts = [f for f in sink.frames if isinstance(f, TranscriptionFrame)]
    assert [t.text for t in transcripts] == ["hello agent"]
    # The pid from transport_source must propagate to TranscriptionFrame
    # so VoiceGate (which keys off user_id) and any future
    # transport_source consumer see the real participant.
    assert transcripts[0].user_id         == "web-client"
    assert transcripts[0].transport_source == "web-client"
    assert stt.calls and stt.calls[0][1] == 16000


@pytest.mark.asyncio
async def test_vad_stt_swallows_empty_transcript(monkeypatch):
    stt = _FakeStt(text="")

    class _StubVad:
        def __init__(self, on_utterance, on_speech_start, **_):
            self._on_utt = on_utterance

        async def feed(self, pcm_int16: bytes, sample_rate: int) -> None:
            await self._on_utt(pcm_int16, sample_rate)

    monkeypatch.setattr("xr_ai_pipecat.processors.vad_stt.VadDetector", _StubVad)
    proc = VadSttProcessor(stt=stt, vad_cfg=VadConfig())
    frame = InputAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)
    frame.transport_source = "web-client"
    sink = await _run_chain(proc, sends=[frame])
    assert not any(isinstance(f, TranscriptionFrame) for f in sink.frames)


@pytest.mark.asyncio
async def test_vad_stt_drops_frame_with_missing_transport_source(monkeypatch):
    """Regression guard: a transport adapter that fails to populate
    ``transport_source`` used to silently degrade to ``pid=''``, which
    the hub then dropped on the floor. The processor now drops the
    frame and logs loudly instead of dispatching with an empty pid."""
    stt = _FakeStt(text="hello agent")

    fed: list[tuple[bytes, int]] = []

    class _StubVad:
        def __init__(self, on_utterance, on_speech_start, **_):
            self._on_utt   = on_utterance
            self._on_start = on_speech_start

        async def feed(self, pcm_int16: bytes, sample_rate: int) -> None:
            fed.append((pcm_int16, sample_rate))
            await self._on_start()
            await self._on_utt(pcm_int16, sample_rate)

    monkeypatch.setattr("xr_ai_pipecat.processors.vad_stt.VadDetector", _StubVad)
    proc = VadSttProcessor(stt=stt, vad_cfg=VadConfig())

    # transport_source intentionally left at its default (None).
    frame = InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=16000, num_channels=1)
    sink = await _run_chain(proc, sends=[frame])

    assert fed == [], "VAD must not be fed when transport_source is missing"
    assert not any(isinstance(f, TranscriptionFrame) for f in sink.frames)
    assert stt.calls == []


# ── early STOP probe ────────────────────────────────────────────────────────


class _StagedStt:
    """STTService double that returns canned text from a queue.

    Each call pops from ``texts`` (FIFO); when empty, falls back to
    ``default``. Lets a single test feed different transcripts to the
    probe call and the eventual VAD-finalize call.
    """

    def __init__(self, texts: list[str], default: str = "") -> None:
        self.texts        = list(texts)
        self.default      = default
        self.calls:       list[tuple[bytes, int]] = []
        self.raise_on_call = False

    async def transcribe(self, audio: bytes, *, sample_rate: int | None = None, channels: int = 1, timeout: float | None = None) -> str:
        self.calls.append((audio, sample_rate or 16000))
        if self.raise_on_call:
            raise RuntimeError("stt down")
        if self.texts:
            return self.texts.pop(0)
        return self.default

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        pass


class _StagedVad:
    """VadDetector double whose ``feed`` only triggers ``on_speech_start``
    on the first call; the test fires ``on_utterance`` explicitly via
    ``trigger_utterance`` so the probe / finalize race can be exercised
    deterministically.
    """

    instances: list = []

    def __init__(self, on_utterance, on_speech_start, **_):
        self._on_utt    = on_utterance
        self._on_start  = on_speech_start
        self._started   = False
        self.last_audio: bytes = b""
        self.last_sr:    int   = 0
        _StagedVad.instances.append(self)

    async def feed(self, pcm_int16: bytes, sample_rate: int) -> None:
        # Accumulate the audio the way the real detector buffers an
        # utterance — keeps a single concatenated snapshot so the test
        # can assert against the bytes seen by the probe.
        self.last_audio = self.last_audio + pcm_int16
        self.last_sr    = sample_rate
        if not self._started:
            self._started = True
            await self._on_start()

    async def trigger_utterance(self) -> None:
        await self._on_utt(self.last_audio, self.last_sr)


@pytest.mark.asyncio
async def test_vad_stt_stop_probe_schedules_on_speech_start(monkeypatch):
    """On the first ``on_speech_start`` after silence, the processor
    schedules a one-shot probe task. Waiting longer than
    ``stop_probe_after_s`` lets the probe run; the stub STT records the
    call so the probe firing is observable."""
    _StagedVad.instances.clear()
    stt = _StagedStt(texts=["something"])  # not STOP — probe is silent
    monkeypatch.setattr("xr_ai_pipecat.processors.vad_stt.VadDetector", _StagedVad)
    proc = VadSttProcessor(stt=stt, vad_cfg=VadConfig(stop_probe_after_s=0.05))

    frame = InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=16000, num_channels=1)
    frame.transport_source = "web-client"
    await _run_chain(proc, sends=[frame], settle_s=0.2)

    # Exactly one STT call — the probe's — because on_utterance never fired.
    assert len(stt.calls) == 1
    assert stt.calls[0][1] == 16000


@pytest.mark.asyncio
async def test_vad_stt_stop_probe_emits_interruption_on_stop_match(monkeypatch):
    """When the probe's partial transcript matches ``STOP_RE`` the
    processor pushes ``InterruptionFrame`` + the matched
    ``TranscriptionFrame`` + ``UserStoppedSpeakingFrame`` downstream
    immediately — without waiting for VAD's silence-window finalize."""
    _StagedVad.instances.clear()
    stt = _StagedStt(texts=["stop"])
    monkeypatch.setattr("xr_ai_pipecat.processors.vad_stt.VadDetector", _StagedVad)
    proc = VadSttProcessor(stt=stt, vad_cfg=VadConfig(stop_probe_after_s=0.05))

    frame = InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=16000, num_channels=1)
    frame.transport_source = "web-client"
    sink = await _run_chain(proc, sends=[frame], settle_s=0.2)

    kinds = [type(f).__name__ for f in sink.frames]
    assert "InterruptionFrame"        in kinds
    assert "UserStoppedSpeakingFrame" in kinds
    transcripts = [f for f in sink.frames if isinstance(f, TranscriptionFrame)]
    assert [t.text for t in transcripts] == ["stop"]
    # InterruptionFrame must arrive before the TranscriptionFrame so any
    # in-flight reasoning is cancelled before the gate sees STOP.
    int_idx = next(i for i, f in enumerate(sink.frames) if isinstance(f, InterruptionFrame))
    tf_idx  = next(i for i, f in enumerate(sink.frames) if isinstance(f, TranscriptionFrame))
    assert int_idx < tf_idx


@pytest.mark.asyncio
async def test_vad_stt_stop_probe_silent_on_non_stop_match(monkeypatch):
    """A non-STOP partial transcript discards the probe result and lets
    VAD-finalize handle the utterance via the usual path. No
    ``InterruptionFrame`` is pushed."""
    _StagedVad.instances.clear()
    stt = _StagedStt(texts=["hello agent what time is it"])
    monkeypatch.setattr("xr_ai_pipecat.processors.vad_stt.VadDetector", _StagedVad)
    proc = VadSttProcessor(stt=stt, vad_cfg=VadConfig(stop_probe_after_s=0.05))

    frame = InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=16000, num_channels=1)
    frame.transport_source = "web-client"
    sink = await _run_chain(proc, sends=[frame], settle_s=0.2)

    assert not any(isinstance(f, InterruptionFrame) for f in sink.frames)
    assert not any(isinstance(f, TranscriptionFrame) for f in sink.frames)


@pytest.mark.asyncio
async def test_vad_stt_stop_probe_cancelled_when_vad_finalizes_first(monkeypatch):
    """If VAD finalizes the utterance before the probe timer fires, the
    pending probe task is cancelled — STT is called exactly once (by
    the on_utterance path) and no probe-side STT call lands."""
    _StagedVad.instances.clear()
    stt = _StagedStt(texts=["hello agent"])
    monkeypatch.setattr("xr_ai_pipecat.processors.vad_stt.VadDetector", _StagedVad)
    proc = VadSttProcessor(stt=stt, vad_cfg=VadConfig(stop_probe_after_s=1.0))

    frame = InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=16000, num_channels=1)
    frame.transport_source = "web-client"

    # Drive the first frame so on_speech_start fires and the probe task
    # is scheduled; then trigger on_utterance manually well before the
    # 1-second probe timer expires.
    sink = _CaptureSink()
    pipeline = Pipeline([proc, sink])
    worker = PipelineWorker(pipeline, cancel_on_idle_timeout=False, enable_rtvi=False)
    runner = WorkerRunner()
    await runner.add_workers(worker)

    async def drive() -> None:
        await asyncio.sleep(0.05)
        await worker.queue_frame(frame)
        await asyncio.sleep(0.05)
        assert _StagedVad.instances, "VAD stub was not instantiated"
        await _StagedVad.instances[-1].trigger_utterance()
        await asyncio.sleep(0.1)
        await worker.queue_frame(EndFrame())

    await asyncio.gather(runner.run(), drive())

    # Only the on_utterance STT call — no probe call.
    assert len(stt.calls) == 1


@pytest.mark.asyncio
async def test_vad_stt_stop_probe_suppresses_duplicate_vad_finalize(monkeypatch):
    """After the probe fires STOP, the eventual VAD-finalize for the
    same utterance must NOT re-emit ``UserStoppedSpeakingFrame`` + a
    second ``TranscriptionFrame`` — otherwise the gate would re-fire
    its canned "Okay, I will stop." ack TTS."""
    _StagedVad.instances.clear()
    # Probe call returns "stop"; the eventual on_utterance call (if
    # the suppression failed) would return "stop now" — we must not see
    # that downstream.
    stt = _StagedStt(texts=["stop", "stop now"])
    monkeypatch.setattr("xr_ai_pipecat.processors.vad_stt.VadDetector", _StagedVad)
    proc = VadSttProcessor(stt=stt, vad_cfg=VadConfig(stop_probe_after_s=0.05))

    frame = InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=16000, num_channels=1)
    frame.transport_source = "web-client"

    sink = _CaptureSink()
    pipeline = Pipeline([proc, sink])
    worker = PipelineWorker(pipeline, cancel_on_idle_timeout=False, enable_rtvi=False)
    runner = WorkerRunner()
    await runner.add_workers(worker)

    async def drive() -> None:
        await asyncio.sleep(0.05)
        await worker.queue_frame(frame)
        # Wait for the probe to fire (>= stop_probe_after_s).
        await asyncio.sleep(0.2)
        # VAD now finalizes after silence — would normally push a fresh
        # UserStoppedSpeakingFrame + TranscriptionFrame.
        assert _StagedVad.instances
        await _StagedVad.instances[-1].trigger_utterance()
        await asyncio.sleep(0.1)
        await worker.queue_frame(EndFrame())

    await asyncio.gather(runner.run(), drive())

    transcripts = [f for f in sink.frames if isinstance(f, TranscriptionFrame)]
    assert [t.text for t in transcripts] == ["stop"], (
        "duplicate transcription from VAD-finalize must be suppressed "
        "after the probe already fired STOP"
    )
    # The probe's stop-emit ends with UserStoppedSpeakingFrame; VAD's
    # finalize would re-push one. With suppression we should see exactly
    # one of each.
    stops = [f for f in sink.frames if isinstance(f, UserStoppedSpeakingFrame)]
    assert len(stops) == 1
    # Only the probe-side STT call should have happened — the on_utterance
    # path bailed before its own STT call.
    assert len(stt.calls) == 1


@pytest.mark.asyncio
async def test_vad_stt_stop_probe_disabled_when_setting_zero(monkeypatch):
    """``stop_probe_after_s = 0`` opts out of the probe entirely — no
    background task is scheduled and the only STT call comes from
    on_utterance, matching the pre-probe behavior."""
    _StagedVad.instances.clear()
    stt = _StagedStt(texts=["stop"])
    monkeypatch.setattr("xr_ai_pipecat.processors.vad_stt.VadDetector", _StagedVad)
    proc = VadSttProcessor(stt=stt, vad_cfg=VadConfig(stop_probe_after_s=0.0))

    frame = InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=16000, num_channels=1)
    frame.transport_source = "web-client"

    sink = _CaptureSink()
    pipeline = Pipeline([proc, sink])
    worker = PipelineWorker(pipeline, cancel_on_idle_timeout=False, enable_rtvi=False)
    runner = WorkerRunner()
    await runner.add_workers(worker)

    async def drive() -> None:
        await asyncio.sleep(0.05)
        await worker.queue_frame(frame)
        # Give the world long enough for any (incorrectly-scheduled) probe
        # to fire; with the probe disabled, nothing happens here.
        await asyncio.sleep(0.2)
        assert _StagedVad.instances
        await _StagedVad.instances[-1].trigger_utterance()
        await asyncio.sleep(0.05)
        await worker.queue_frame(EndFrame())

    await asyncio.gather(runner.run(), drive())

    # Exactly one STT call — the on_utterance one. No probe ran.
    assert len(stt.calls) == 1
    transcripts = [f for f in sink.frames if isinstance(f, TranscriptionFrame)]
    assert [t.text for t in transcripts] == ["stop"]
    # The discriminating signal: with the probe disabled, the
    # InterruptionFrame the probe normally pushes on STOP-match must NOT
    # appear. (Without this assertion, the call-count check above would
    # pass even if the probe ran — its STT call and the on_utterance one
    # would either way total exactly one because the suppression flag
    # would gate out the duplicate.)
    assert not any(isinstance(f, InterruptionFrame) for f in sink.frames), (
        "probe must not push InterruptionFrame when stop_probe_after_s=0"
    )


@pytest.mark.asyncio
async def test_vad_stt_stop_probe_no_unawaited_coroutine_under_finalize_race(monkeypatch):
    """Regression guard: the probe-STOP-then-VAD-finalize sequence must
    not produce any "coroutine was never awaited" RuntimeWarnings.

    Production saw an intermittent
    ``coroutine 'FrameProcessor.__process_frame_task_handler' was never
    awaited`` right after a probe-STOP match. The fix awaits the
    cancelled probe task to completion before scheduling the next one
    (and before ``on_utterance`` clears its bookkeeping) so a cancelled
    probe never overlaps a fresh one. Capture all RuntimeWarnings
    raised during the sequence, then force GC so the unawaited-coroutine
    finalizer fires before the assertion runs.
    """
    _StagedVad.instances.clear()
    # Probe sees STOP; finalize would see "stop now" if suppression failed.
    stt = _StagedStt(texts=["stop", "stop now"])
    monkeypatch.setattr("xr_ai_pipecat.processors.vad_stt.VadDetector", _StagedVad)
    proc = VadSttProcessor(stt=stt, vad_cfg=VadConfig(stop_probe_after_s=0.05))

    frame = InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=16000, num_channels=1)
    frame.transport_source = "web-client"

    sink = _CaptureSink()
    pipeline = Pipeline([proc, sink])
    worker = PipelineWorker(pipeline, cancel_on_idle_timeout=False, enable_rtvi=False)
    runner = WorkerRunner()
    await runner.add_workers(worker)

    async def drive() -> None:
        await asyncio.sleep(0.05)
        await worker.queue_frame(frame)
        # Let the probe fire its STOP.
        await asyncio.sleep(0.15)
        # VAD finalize races 0.1s after the probe — matches the
        # production timing reported in the original incident.
        assert _StagedVad.instances
        await _StagedVad.instances[-1].trigger_utterance()
        await asyncio.sleep(0.1)
        await worker.queue_frame(EndFrame())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await asyncio.gather(runner.run(), drive())
        # Coroutine-never-awaited surfaces from the GC finalizer, not from
        # a raise — force a collection cycle so the warning lands inside
        # the catch_warnings block.
        for _ in range(3):
            gc.collect()
            await asyncio.sleep(0)

    unawaited = [
        w for w in caught
        if issubclass(w.category, RuntimeWarning)
        and "never awaited" in str(w.message)
    ]
    assert not unawaited, (
        "probe → finalize sequence leaked unawaited coroutines: "
        + ", ".join(str(w.message) for w in unawaited)
    )


# ════════════════════════════════════════════════════════════════════════════
# VoiceGateProcessor
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_voice_gate_processor_dispatches_query_frame_on_fresh_match():
    cfg = VoiceGateConfig(magic_phrases=("agent",))
    proc = VoiceGateProcessor(cfg=cfg, tts=_FakeTts())
    sink = await _run_chain(
        proc,
        sends=[TranscriptionFrame(text="agent what time is it", user_id="pid-1", timestamp="t")],
    )

    queries = [f for f in sink.frames if isinstance(f, GatedQueryFrame)]
    assert len(queries) == 1
    assert queries[0].text          == "what time is it"
    assert queries[0].fresh_match   is True
    assert queries[0].participant_id == "pid-1"


@pytest.mark.asyncio
async def test_voice_gate_processor_stop_emits_interruption_and_ack_text():
    cfg = VoiceGateConfig(magic_phrases=("agent",))
    proc = VoiceGateProcessor(cfg=cfg, tts=_FakeTts())
    sink = await _run_chain(
        proc,
        sends=[TranscriptionFrame(text="stop", user_id="pid-1", timestamp="t")],
    )

    # The order matters: InterruptionFrame must reach downstream before
    # the ack text so any in-flight reasoning is cancelled BEFORE the
    # ack itself gets routed back through TTS.
    indices_interrupt = [i for i, f in enumerate(sink.frames) if isinstance(f, InterruptionFrame)]
    indices_text      = [i for i, f in enumerate(sink.frames) if isinstance(f, TextFrame)]
    assert indices_interrupt and indices_text
    assert indices_interrupt[0] < indices_text[0]
    ack = next(f for f in sink.frames if isinstance(f, TextFrame))
    assert ack.text == "Okay, I will stop."


@pytest.mark.asyncio
async def test_voice_gate_processor_greeting_emitted_when_phrases_configured():
    cfg = VoiceGateConfig(magic_phrases=("agent",))
    proc = VoiceGateProcessor(cfg=cfg, tts=_FakeTts())
    sink = await _run_chain(
        proc,
        sends=[ParticipantJoinedFrame(participant_id="pid-1")],
    )

    texts = [f for f in sink.frames if isinstance(f, TextFrame)]
    assert len(texts) == 1
    assert texts[0].text.startswith("To talk to me")
    assert any(isinstance(f, ParticipantJoinedFrame) for f in sink.frames)


@pytest.mark.asyncio
async def test_voice_gate_processor_no_greeting_when_phrases_empty():
    """Always-on mode: no wake word means no opt-in UX to advertise."""
    cfg = VoiceGateConfig(magic_phrases=())
    proc = VoiceGateProcessor(cfg=cfg, tts=_FakeTts())
    sink = await _run_chain(
        proc,
        sends=[ParticipantJoinedFrame(participant_id="pid-1")],
    )
    texts = [f for f in sink.frames if isinstance(f, TextFrame)]
    assert texts == []


@pytest.mark.asyncio
async def test_voice_gate_processor_phrase_only_emits_no_query_frame():
    cfg = VoiceGateConfig(magic_phrases=("agent",))
    proc = VoiceGateProcessor(cfg=cfg, tts=_FakeTts())
    sink = await _run_chain(
        proc,
        sends=[TranscriptionFrame(text="agent", user_id="pid-1", timestamp="t")],
    )
    assert not any(isinstance(f, GatedQueryFrame) for f in sink.frames)


@pytest.mark.asyncio
async def test_voice_gate_processor_chime_routes_through_pipeline_audio_path():
    """When a fresh-match query fires AND the chime is enabled AND TTS
    has been observed, the gate's chime arrives downstream as
    ``OutputAudioRawFrame``s — not via a sidechannel."""
    cfg  = VoiceGateConfig(magic_phrases=("agent",), listening_chime=True)
    proc = VoiceGateProcessor(cfg=cfg, tts=_FakeTts())
    # Prime the chime by observing a TTS WAV first.
    proc.gate.observe_tts_wav(_silence_wav(24000))

    sink = await _run_chain(
        proc,
        sends=[TranscriptionFrame(text="agent, what time is it", user_id="pid-1", timestamp="t")],
    )
    audio_out = [f for f in sink.frames if isinstance(f, OutputAudioRawFrame)]
    assert audio_out, "chime should have emitted at least one OutputAudioRawFrame"
    assert all(f.transport_destination == "pid-1" for f in audio_out)


# ════════════════════════════════════════════════════════════════════════════
# BrainProcessor
# ════════════════════════════════════════════════════════════════════════════


class _StringBrain(BrainProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.handle_calls: list[tuple[str, str, bool]] = []

    async def handle_query(self, pid, text, fresh_match):
        self.handle_calls.append((pid, text, fresh_match))
        return f"answer: {text}"


class _IterBrain(BrainProcessor):
    def __init__(self, chunks: list[str]) -> None:
        super().__init__()
        self._chunks = chunks
        self.cancelled = False

    async def handle_query(self, pid, text, fresh_match) -> AsyncIterator[str]:
        async def _gen():
            try:
                for c in self._chunks:
                    yield c
                    await asyncio.sleep(0.001)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return _gen()


class _LifecycleBrain(BrainProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.joined:           list[str] = []
        self.left:             list[str] = []
        self.started_speaking: list[str] = []

    async def handle_query(self, pid, text, fresh_match):
        return ""

    async def on_participant_joined(self, pid: str) -> None:
        self.joined.append(pid)

    async def on_participant_left(self, pid: str) -> None:
        self.left.append(pid)

    async def on_user_started_speaking(self, pid: str) -> None:
        self.started_speaking.append(pid)


@pytest.mark.asyncio
async def test_brain_string_return_pushes_single_text_frame():
    brain = _StringBrain()
    sink = await _run_chain(
        brain,
        sends=[GatedQueryFrame(participant_id="pid-1", text="hi", fresh_match=True, pts_us=0)],
    )

    texts = [f for f in sink.frames if isinstance(f, TextFrame)]
    assert [t.text for t in texts] == ["answer: hi"]
    assert brain.handle_calls == [("pid-1", "hi", True)]


@pytest.mark.asyncio
async def test_brain_async_iter_return_pushes_text_frame_per_chunk():
    brain = _IterBrain(chunks=["alpha ", "beta ", "gamma."])
    sink = await _run_chain(
        brain,
        sends=[GatedQueryFrame(participant_id="pid-1", text="hi", fresh_match=True, pts_us=0)],
        settle_s=0.15,
    )
    texts = [f.text for f in sink.frames if isinstance(f, TextFrame)]
    assert texts == ["alpha ", "beta ", "gamma."]


@pytest.mark.asyncio
async def test_brain_does_not_cancel_on_user_started_speaking():
    """Regression guard: ``UserStartedSpeakingFrame`` is a hook, not a
    cancel signal. Cancelling on speech onset breaks two things:

    * any AEC leak of the agent's own TTS becomes self-cancel,
    * a quick follow-up utterance aborts the prior response BEFORE the
      voice gate even decides whether the new utterance was a query.

    The brain must keep streaming TextFrames; cancellation happens on
    the next GatedQueryFrame or on an explicit InterruptionFrame."""
    brain = _IterBrain(chunks=[f"chunk{i} " for i in range(5)])
    sink = await _run_chain(
        brain,
        sends=[
            GatedQueryFrame(participant_id="pid-1", text="hi", fresh_match=True, pts_us=0),
            UserStartedSpeakingFrame(),
        ],
        settle_s=0.3,
        per_send_delay_s=0.05,
    )
    assert brain.cancelled is False
    texts = [f.text for f in sink.frames if isinstance(f, TextFrame)]
    assert texts == [f"chunk{i} " for i in range(5)]


@pytest.mark.asyncio
async def test_brain_cancels_inflight_on_new_query_for_same_pid():
    """A fresh GatedQueryFrame supersedes any in-flight reasoning for
    the same pid — this is the contract that makes rapid follow-ups
    work without the user having to wait for the previous answer."""
    brain = _IterBrain(chunks=[f"chunk{i} " for i in range(200)])
    await _run_chain(
        brain,
        sends=[
            GatedQueryFrame(participant_id="pid-1", text="hi",   fresh_match=True, pts_us=0),
            GatedQueryFrame(participant_id="pid-1", text="hi 2", fresh_match=True, pts_us=1),
        ],
        settle_s=0.2,
        per_send_delay_s=0.05,
    )
    assert brain.cancelled is True


@pytest.mark.asyncio
async def test_brain_cancels_inflight_on_interruption_frame():
    brain = _IterBrain(chunks=[f"chunk{i} " for i in range(200)])
    await _run_chain(
        brain,
        sends=[
            GatedQueryFrame(participant_id="pid-1", text="hi", fresh_match=True, pts_us=0),
            InterruptionFrame(),
        ],
        settle_s=0.2,
        per_send_delay_s=0.05,
    )
    assert brain.cancelled is True


@pytest.mark.asyncio
async def test_brain_on_query_superseded_fires_on_second_query_while_inflight():
    """When a fresh ``GatedQueryFrame`` arrives while a prior query's
    task is still in-flight, ``on_query_superseded`` must fire so an
    agent can decide what to do with the previous response's queued
    downstream state (e.g. push an InterruptionFrame to drain queued
    TTS audio). The library default is a no-op; this test only checks
    the hook is invoked."""
    class _SupersedeRecorder(_IterBrain):
        def __init__(self) -> None:
            super().__init__(chunks=[f"chunk{i} " for i in range(200)])
            self.supersede_calls: list[str] = []

        async def on_query_superseded(self, pid: str) -> None:
            self.supersede_calls.append(pid)

    brain = _SupersedeRecorder()
    await _run_chain(
        brain,
        sends=[
            GatedQueryFrame(participant_id="pid-1", text="first",  fresh_match=True, pts_us=0),
            GatedQueryFrame(participant_id="pid-1", text="second", fresh_match=True, pts_us=1),
        ],
        settle_s=0.2,
        per_send_delay_s=0.05,
    )
    assert brain.supersede_calls == ["pid-1"]
    assert brain.cancelled is True


@pytest.mark.asyncio
async def test_brain_on_query_superseded_fires_when_prior_task_already_done():
    """Regression: the supersede hook must fire on every non-first
    query for a pid, even when the prior brain task has already
    completed. Real-world case: a short VLM response streams in
    quickly so the brain task is done, but its TTS audio is still
    playing out when the user fires a follow-up query. Gating the
    hook on ``_inflight[pid].done()`` (the old bug) caused the new
    query to queue behind the prior audio instead of interrupting it.

    Uses ``_StringBrain`` (one-shot string response) plus a generous
    ``per_send_delay_s`` so the first task finishes before the
    second query arrives. Asserts that at supersede time the prior
    task was indeed already done, so the assertion exercises the
    bug case rather than the in-flight case."""
    class _SupersedeRecorder(_StringBrain):
        def __init__(self) -> None:
            super().__init__()
            self.supersede_calls: list[str] = []
            self.prior_task_done_at_supersede: list[bool] = []

        async def on_query_superseded(self, pid: str) -> None:
            self.supersede_calls.append(pid)
            existing = self._inflight.get(pid)
            self.prior_task_done_at_supersede.append(
                existing is None or existing.done(),
            )

    brain = _SupersedeRecorder()
    await _run_chain(
        brain,
        sends=[
            GatedQueryFrame(participant_id="pid-1", text="first",  fresh_match=True, pts_us=0),
            GatedQueryFrame(participant_id="pid-1", text="second", fresh_match=True, pts_us=1),
        ],
        # Long enough that the one-shot ``_StringBrain`` response for
        # the first query has fully run and emitted its end frame
        # before the second query is queued.
        settle_s=0.3,
        per_send_delay_s=0.2,
    )
    assert brain.supersede_calls == ["pid-1"]
    assert brain.prior_task_done_at_supersede == [True], (
        "test must exercise the bug case: prior brain task already done "
        "when supersede fires"
    )
    # Both queries ran end-to-end.
    handled = [t for _pid, t, _fm in brain.handle_calls]
    assert "first"  in handled
    assert "second" in handled


@pytest.mark.asyncio
async def test_brain_on_query_superseded_fires_on_every_subsequent_query():
    """The hook fires once per non-first query — three queries fire
    it twice, four fire it three times, etc. Uses ``_StringBrain``
    so each prior task completes before the next query arrives,
    confirming the gate is per-pid query count, not brain-task state."""
    class _SupersedeRecorder(_StringBrain):
        def __init__(self) -> None:
            super().__init__()
            self.supersede_calls: list[str] = []

        async def on_query_superseded(self, pid: str) -> None:
            self.supersede_calls.append(pid)

    brain = _SupersedeRecorder()
    await _run_chain(
        brain,
        sends=[
            GatedQueryFrame(participant_id="pid-1", text="first",  fresh_match=True, pts_us=0),
            GatedQueryFrame(participant_id="pid-1", text="second", fresh_match=True, pts_us=1),
            GatedQueryFrame(participant_id="pid-1", text="third",  fresh_match=True, pts_us=2),
            GatedQueryFrame(participant_id="pid-1", text="fourth", fresh_match=True, pts_us=3),
        ],
        settle_s=0.3,
        per_send_delay_s=0.1,
    )
    # First query: no supersede. Each subsequent query: one supersede.
    assert brain.supersede_calls == ["pid-1", "pid-1", "pid-1"]


@pytest.mark.asyncio
async def test_brain_on_query_superseded_seen_state_cleared_on_participant_left():
    """``_seen_query`` is per-pid and must be cleared on
    ``ParticipantLeftFrame`` so a rejoin's first query is treated
    as cold (no supersede) rather than as a follow-up. Without
    this, an override that pushes ``InterruptionFrame`` on
    supersede would flush unrelated audio on every fresh session."""
    class _SupersedeRecorder(_StringBrain):
        def __init__(self) -> None:
            super().__init__()
            self.supersede_calls: list[str] = []

        async def on_query_superseded(self, pid: str) -> None:
            self.supersede_calls.append(pid)

    brain = _SupersedeRecorder()
    await _run_chain(
        brain,
        sends=[
            GatedQueryFrame(participant_id="pid-1",  text="first",  fresh_match=True, pts_us=0),
            ParticipantLeftFrame(participant_id="pid-1"),
            # Same pid rejoins (or different session): the first
            # query after the left frame must NOT fire the hook.
            GatedQueryFrame(participant_id="pid-1",  text="second", fresh_match=True, pts_us=1),
        ],
        settle_s=0.3,
        per_send_delay_s=0.1,
    )
    assert brain.supersede_calls == []


@pytest.mark.asyncio
async def test_brain_on_query_superseded_not_called_on_cold_path_first_query():
    """The cold path — first query, no in-flight task — must NOT call
    ``on_query_superseded``. There is nothing to supersede, and agents
    that override to push an InterruptionFrame would otherwise flush
    unrelated in-flight audio (e.g. a voice-gate chime)."""
    class _SupersedeRecorder(_StringBrain):
        def __init__(self) -> None:
            super().__init__()
            self.supersede_calls: list[str] = []

        async def on_query_superseded(self, pid: str) -> None:
            self.supersede_calls.append(pid)

    brain = _SupersedeRecorder()
    await _run_chain(
        brain,
        sends=[GatedQueryFrame(participant_id="pid-1", text="hi", fresh_match=True, pts_us=0)],
    )
    assert brain.supersede_calls == []


@pytest.mark.asyncio
async def test_brain_on_query_superseded_can_push_frames_downstream():
    """The hook is allowed to push frames — this is the whole point of
    the design (sample-side override pushes ``InterruptionFrame`` to
    drain queued TTS audio for the previous response). Verify a frame
    pushed from the hook reaches the downstream sink."""
    class _InterruptingBrain(_IterBrain):
        def __init__(self) -> None:
            super().__init__(chunks=[f"chunk{i} " for i in range(200)])

        async def on_query_superseded(self, pid: str) -> None:
            await self.push_frame(InterruptionFrame())

    brain = _InterruptingBrain()
    sink = await _run_chain(
        brain,
        sends=[
            GatedQueryFrame(participant_id="pid-1", text="first",  fresh_match=True, pts_us=0),
            GatedQueryFrame(participant_id="pid-1", text="second", fresh_match=True, pts_us=1),
        ],
        settle_s=0.2,
        per_send_delay_s=0.05,
    )
    assert any(isinstance(f, InterruptionFrame) for f in sink.frames), (
        "hook must be able to push frames to downstream sink"
    )


@pytest.mark.asyncio
async def test_brain_on_query_superseded_exception_is_swallowed_and_spawn_proceeds():
    """A misbehaving override must not break the supersede contract:
    the previous task is still cancelled and the new query still
    spawns. The exception is logged at the library boundary."""
    class _RaisingBrain(_IterBrain):
        def __init__(self) -> None:
            super().__init__(chunks=[f"chunk{i} " for i in range(200)])
            self.handle_calls: list[str] = []
            self.supersede_calls: list[str] = []

        async def handle_query(self, pid, text, fresh_match):
            self.handle_calls.append(text)
            return await super().handle_query(pid, text, fresh_match)

        async def on_query_superseded(self, pid: str) -> None:
            self.supersede_calls.append(pid)
            raise RuntimeError("boom")

    brain = _RaisingBrain()
    await _run_chain(
        brain,
        sends=[
            GatedQueryFrame(participant_id="pid-1", text="first",  fresh_match=True, pts_us=0),
            GatedQueryFrame(participant_id="pid-1", text="second", fresh_match=True, pts_us=1),
        ],
        settle_s=0.2,
        per_send_delay_s=0.05,
    )
    assert brain.supersede_calls == ["pid-1"]
    # Previous task still cancelled; new query still ran.
    assert brain.cancelled is True
    assert "first"  in brain.handle_calls
    assert "second" in brain.handle_calls


@pytest.mark.asyncio
async def test_brain_steers_transport_target_on_participant_joined():
    """Single-participant routing default: when a brain is constructed
    with a transport, the first ``ParticipantJoinedFrame`` steers the
    output transport at that pid so return-audio/return-data go to the
    right participant without per-sample wiring. The output transport
    silently drops audio when ``_target_participant`` is empty — this
    is what previously made the bug "agent thinks but says nothing".
    """
    class _FakeTransport:
        def __init__(self) -> None:
            self.target_calls:  list[str] = []
            self.cleanup_calls: list[str] = []

        def set_target_participant(self, pid: str) -> None:
            self.target_calls.append(pid)

        def cleanup_participant(self, pid: str) -> None:
            self.cleanup_calls.append(pid)

    transport = _FakeTransport()
    brain = _StringBrain()
    brain._transport = transport  # type: ignore[assignment]

    await _run_chain(
        brain,
        sends=[
            ParticipantJoinedFrame(participant_id="web-client"),
            ParticipantLeftFrame(participant_id="web-client"),
        ],
    )

    assert transport.target_calls  == ["web-client"]
    assert transport.cleanup_calls == ["web-client"]


@pytest.mark.asyncio
async def test_brain_no_transport_steering_when_not_configured():
    """Multi-pid samples may construct the brain without a transport;
    join/leave must then be a no-op on the transport side."""
    brain = _StringBrain()  # default: transport=None
    await _run_chain(
        brain,
        sends=[ParticipantJoinedFrame(participant_id="pid-1")],
    )
    # If we got here without an AttributeError, the None-transport path
    # is exercised. No assertion needed beyond that.


@pytest.mark.asyncio
async def test_brain_participant_lifecycle_hooks_fire():
    brain = _LifecycleBrain()
    sink = await _run_chain(
        brain,
        sends=[
            ParticipantJoinedFrame(participant_id="p1"),
            ParticipantLeftFrame(participant_id="p1"),
        ],
    )

    assert brain.joined == ["p1"]
    assert brain.left   == ["p1"]
    kinds = [type(f).__name__ for f in sink.frames]
    assert "ParticipantJoinedFrame" in kinds
    assert "ParticipantLeftFrame"   in kinds


@pytest.mark.asyncio
async def test_brain_user_started_speaking_hook_fires_for_joined_pids():
    """on_user_started_speaking fires for every joined pid (NOT just the
    in-flight ones), so the cold path — first utterance, nothing in
    flight yet — still gets the speculative-warmup hook. Tracking
    in-flight tasks here would mean the very first turn never sees
    camera warmup, which is precisely the case it was designed for."""
    brain = _IterBrain(chunks=[])
    started_for: list[str] = []

    async def speech_hook(pid: str) -> None:
        started_for.append(pid)

    brain.on_user_started_speaking = speech_hook  # type: ignore[method-assign]

    await _run_chain(
        brain,
        sends=[
            ParticipantJoinedFrame(participant_id="pid-1"),
            UserStartedSpeakingFrame(),
        ],
        settle_s=0.1,
        per_send_delay_s=0.05,
    )
    assert started_for == ["pid-1"]


@pytest.mark.asyncio
async def test_brain_user_started_speaking_hook_skipped_after_leave():
    brain = _IterBrain(chunks=[])
    started_for: list[str] = []

    async def speech_hook(pid: str) -> None:
        started_for.append(pid)

    brain.on_user_started_speaking = speech_hook  # type: ignore[method-assign]

    await _run_chain(
        brain,
        sends=[
            ParticipantJoinedFrame(participant_id="pid-1"),
            ParticipantLeftFrame(participant_id="pid-1"),
            UserStartedSpeakingFrame(),
        ],
        settle_s=0.1,
        per_send_delay_s=0.05,
    )
    assert started_for == []


# ════════════════════════════════════════════════════════════════════════════
# StreamingTtsProcessor
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_streaming_tts_sentence_boundary_triggers_synth():
    tts  = _FakeTts(sample_rate=22050)
    gate = VoiceGate(VoiceGateConfig(), audio_sink=_NullSink(), tts=tts)
    proc = StreamingTtsProcessor(tts=tts, voice_gate=gate)

    sink = await _run_chain(
        proc,
        sends=[TextFrame(text="hello"), TextFrame(text=" world. ")],
    )
    assert tts.calls == ["hello world."]
    audio = [f for f in sink.frames if isinstance(f, OutputAudioRawFrame)]
    assert audio, "synth produced no audio frames downstream"


@pytest.mark.asyncio
async def test_streaming_tts_parallel_synth_keeps_order():
    """Out-of-order completion of synth tasks must NOT reorder the
    output audio: the ordered sender awaits in FIFO. ``call_starts``
    records start order; the sender's FIFO is asserted via
    ``observe_tts_wav`` observation order, which fires on each completed
    WAV in the sender loop."""
    tts = _FakeTts()
    delays = {"first sentence.": 0.05, "second sentence.": 0.0}
    call_starts: list[str] = []
    orig_synth = tts.synthesize

    async def variable_delay_synth(text, **kw):
        call_starts.append(text)
        await asyncio.sleep(delays.get(text, 0))
        return await orig_synth(text, **kw)

    tts.synthesize = variable_delay_synth  # type: ignore[method-assign]

    gate = VoiceGate(VoiceGateConfig(), audio_sink=_NullSink(), tts=tts)
    observation_order: list[bytes] = []
    orig_observe = gate.observe_tts_wav

    def spy(wav):
        observation_order.append(wav)
        return orig_observe(wav)

    gate.observe_tts_wav = spy  # type: ignore[method-assign]

    proc = StreamingTtsProcessor(tts=tts, voice_gate=gate)
    sink = await _run_chain(
        proc,
        sends=[TextFrame(text="first sentence. second sentence. ")],
        settle_s=0.2,
    )

    # Both sentences were dispatched in declared order (first synth
    # task starts first), even though "second" completes first.
    assert call_starts == ["first sentence.", "second sentence."]
    # The sender loop is FIFO: it observes the first WAV before the
    # second, regardless of completion order.
    assert len(observation_order) == 2
    audio = [f for f in sink.frames if isinstance(f, OutputAudioRawFrame)]
    assert len(audio) >= 2


@pytest.mark.asyncio
async def test_streaming_tts_interruption_cancels_and_clears_pending():
    tts = _FakeTts()
    tts.delay_s = 0.2  # so we can interrupt before completion
    gate = VoiceGate(VoiceGateConfig(), audio_sink=_NullSink(), tts=tts)
    proc = StreamingTtsProcessor(tts=tts, voice_gate=gate)

    sink = await _run_chain(
        proc,
        sends=[
            TextFrame(text="abandoned sentence one. "),
            InterruptionFrame(),
        ],
        settle_s=0.4,
        per_send_delay_s=0.05,
    )
    audio = [f for f in sink.frames if isinstance(f, OutputAudioRawFrame)]
    assert audio == []
    # Pending buffer is cleared so a subsequent partial sentence does
    # NOT get concatenated to the abandoned fragment.
    assert proc._pending == ""  # noqa: SLF001


@pytest.mark.asyncio
async def test_streaming_tts_flushes_hub_return_audio_on_interrupt():
    """STOP must drop audio that's already paced into the hub.

    Cancelling synth + sender tasks only stops *new* audio. The hub's
    pacing pipe (and LiveKit jitter buffer behind it) keep playing
    whatever is already queued, so without an explicit flush the user
    hears the agent finish its current sentence before silence — STOP
    feels broken. On ``InterruptionFrame`` the processor must call
    ``transport.endpoint.flush_return_audio(target_participant)`` so the
    hub drops its pending audio at the source.
    """
    class _StubEndpoint:
        def __init__(self) -> None:
            self.flush_calls: list[str] = []

        async def flush_return_audio(self, pid: str) -> None:
            self.flush_calls.append(pid)

    class _StubTransport:
        def __init__(self, pid: str) -> None:
            self.endpoint           = _StubEndpoint()
            self.target_participant = pid

    tts  = _FakeTts()
    tts.delay_s = 0.2
    gate = VoiceGate(VoiceGateConfig(), audio_sink=_NullSink(), tts=tts)
    transport = _StubTransport("web-client")
    proc = StreamingTtsProcessor(
        tts=tts, voice_gate=gate, transport=transport,
    )

    await _run_chain(
        proc,
        sends=[
            TextFrame(text="abandoned sentence one. "),
            InterruptionFrame(),
        ],
        settle_s=0.4,
        per_send_delay_s=0.05,
    )

    assert transport.endpoint.flush_calls == ["web-client"]


@pytest.mark.asyncio
async def test_streaming_tts_no_flush_when_transport_unset():
    """Tests / standalone usage that construct the processor without a
    transport must still survive ``InterruptionFrame`` — no transport
    means no flush, not an AttributeError."""
    tts  = _FakeTts()
    tts.delay_s = 0.2
    gate = VoiceGate(VoiceGateConfig(), audio_sink=_NullSink(), tts=tts)
    proc = StreamingTtsProcessor(tts=tts, voice_gate=gate)  # transport=None

    sink = await _run_chain(
        proc,
        sends=[
            TextFrame(text="abandoned sentence one. "),
            InterruptionFrame(),
        ],
        settle_s=0.3,
        per_send_delay_s=0.05,
    )
    # If we got here without an AttributeError, the None-transport path
    # is exercised. Pending buffer should still be cleared.
    assert proc._pending == ""  # noqa: SLF001
    audio = [f for f in sink.frames if isinstance(f, OutputAudioRawFrame)]
    assert audio == []


@pytest.mark.asyncio
async def test_streaming_tts_no_flush_when_no_target_participant():
    """A transport with no target participant bound yet has no pid to
    flush. The processor must skip the flush rather than calling
    ``flush_return_audio("")`` which the hub drops on the floor."""
    class _StubEndpoint:
        def __init__(self) -> None:
            self.flush_calls: list[str] = []

        async def flush_return_audio(self, pid: str) -> None:
            self.flush_calls.append(pid)

    class _StubTransport:
        def __init__(self) -> None:
            self.endpoint           = _StubEndpoint()
            self.target_participant = ""

    tts  = _FakeTts()
    gate = VoiceGate(VoiceGateConfig(), audio_sink=_NullSink(), tts=tts)
    transport = _StubTransport()
    proc = StreamingTtsProcessor(
        tts=tts, voice_gate=gate, transport=transport,
    )

    await _run_chain(
        proc,
        sends=[InterruptionFrame()],
        settle_s=0.1,
    )
    assert transport.endpoint.flush_calls == []


@pytest.mark.asyncio
async def test_streaming_tts_observes_each_wav_through_gate():
    """observe_tts_wav must be invoked once per synthesized WAV so the
    gate's lazy chime can build at the TTS sample rate."""
    tts  = _FakeTts(sample_rate=24000)
    observations: list[bytes] = []
    gate = VoiceGate(VoiceGateConfig(), audio_sink=_NullSink(), tts=tts)
    orig_observe = gate.observe_tts_wav

    def spy(wav):
        observations.append(wav)
        return orig_observe(wav)

    gate.observe_tts_wav = spy  # type: ignore[method-assign]
    proc = StreamingTtsProcessor(tts=tts, voice_gate=gate)
    await _run_chain(proc, sends=[TextFrame(text="hi there. ")])
    assert len(observations) == 1


# ════════════════════════════════════════════════════════════════════════════
# XRMediaHubInputTransport
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_input_transport_populates_transport_source_from_chunk_pid():
    """The hub-side ``AudioChunk.participant_id`` must flow onto
    ``InputAudioRawFrame.transport_source`` — without it every
    downstream return-data / return-audio send routes to ``pid=''`` and
    the hub drops the message on the floor (production bug fixed in
    this commit)."""
    from xr_ai_agent import AudioChunk
    from xr_ai_pipecat.transport import (
        SAMPLE_RATE,
        XRMediaHubInputTransport,
    )
    from pipecat.transports.base_transport import TransportParams

    ep = _CallbackStubEndpoint()
    params = TransportParams(
        audio_in_enabled=True,
        audio_in_sample_rate=SAMPLE_RATE,
        audio_in_channels=1,
    )
    transport = XRMediaHubInputTransport(ep, params)
    # Mark started without spinning up the ZMQ run loop; the audio
    # callback gates on this flag.
    transport._started = True

    pushed: list[Frame] = []

    async def capture(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append(frame)

    transport.push_frame = capture  # type: ignore[method-assign]

    pcm_f32 = np.zeros(320, dtype=np.float32).tobytes()
    chunk = AudioChunk(
        pts_us         = 0,
        sample_rate    = SAMPLE_RATE,
        channels       = 1,
        samples        = 320,
        data           = pcm_f32,
        participant_id = "web-client",
        track_id       = "mic",
    )
    await ep.audio_cb(chunk)

    assert len(pushed) == 1
    frame = pushed[0]
    assert isinstance(frame, InputAudioRawFrame)
    assert frame.transport_source == "web-client"


@pytest.mark.asyncio
async def test_input_transport_emits_participant_joined_frame():
    """``ParticipantEvent(joined=True)`` from the hub must surface as a
    ``ParticipantJoinedFrame`` on the pipecat pipeline — otherwise the
    voice gate never greets and the brain never steers the output
    transport at a participant, so every TTS chunk gets dropped by
    ``XRMediaHubOutputTransport.write_audio_frame``."""
    from xr_ai_agent import ParticipantEvent
    from xr_ai_pipecat.transport import (
        SAMPLE_RATE,
        XRMediaHubInputTransport,
    )
    from pipecat.transports.base_transport import TransportParams

    ep = _CallbackStubEndpoint()
    params = TransportParams(
        audio_in_enabled=True,
        audio_in_sample_rate=SAMPLE_RATE,
        audio_in_channels=1,
    )
    transport = XRMediaHubInputTransport(ep, params)
    transport._started = True

    pushed: list[Frame] = []

    async def capture(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append(frame)

    transport.push_frame = capture  # type: ignore[method-assign]

    assert ep.participant_cb is not None, (
        "input transport must bind on_participant in __init__"
    )

    await ep.participant_cb(
        ParticipantEvent(participant_id="web-client", joined=True, pts_us=0),
    )

    assert len(pushed) == 1
    frame = pushed[0]
    assert isinstance(frame, ParticipantJoinedFrame)
    assert frame.participant_id == "web-client"


@pytest.mark.asyncio
async def test_input_transport_emits_participant_left_frame():
    """``ParticipantEvent(joined=False)`` from the hub must surface as a
    ``ParticipantLeftFrame`` so ``BrainProcessor`` can run per-pid
    teardown (cancel in-flight, clear target participant)."""
    from xr_ai_agent import ParticipantEvent
    from xr_ai_pipecat.transport import (
        SAMPLE_RATE,
        XRMediaHubInputTransport,
    )
    from pipecat.transports.base_transport import TransportParams

    ep = _CallbackStubEndpoint()
    params = TransportParams(
        audio_in_enabled=True,
        audio_in_sample_rate=SAMPLE_RATE,
        audio_in_channels=1,
    )
    transport = XRMediaHubInputTransport(ep, params)
    transport._started = True

    pushed: list[Frame] = []

    async def capture(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append(frame)

    transport.push_frame = capture  # type: ignore[method-assign]

    await ep.participant_cb(
        ParticipantEvent(participant_id="web-client", joined=False, pts_us=0),
    )

    assert len(pushed) == 1
    frame = pushed[0]
    assert isinstance(frame, ParticipantLeftFrame)
    assert frame.participant_id == "web-client"


@pytest.mark.asyncio
async def test_input_transport_drops_participant_event_before_start():
    """Same ``_started`` guard as ``_on_hub_audio`` — a late event
    arriving after teardown (or before ``StartFrame``) must be a no-op
    so the bridge doesn't race the pipeline shutdown."""
    from xr_ai_agent import ParticipantEvent
    from xr_ai_pipecat.transport import (
        SAMPLE_RATE,
        XRMediaHubInputTransport,
    )
    from pipecat.transports.base_transport import TransportParams

    ep = _CallbackStubEndpoint()
    params = TransportParams(
        audio_in_enabled=True,
        audio_in_sample_rate=SAMPLE_RATE,
        audio_in_channels=1,
    )
    transport = XRMediaHubInputTransport(ep, params)
    # Intentionally leave transport._started == False.

    pushed: list[Frame] = []

    async def capture(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append(frame)

    transport.push_frame = capture  # type: ignore[method-assign]

    await ep.participant_cb(
        ParticipantEvent(participant_id="web-client", joined=True, pts_us=0),
    )

    assert pushed == []


# ════════════════════════════════════════════════════════════════════════════
# make_voice_pipeline end-to-end smoke
# ════════════════════════════════════════════════════════════════════════════


class _EchoBrain(BrainProcessor):
    async def handle_query(self, pid, text, fresh_match) -> str:
        return f"echo {text}."


@pytest.mark.asyncio
async def test_make_voice_pipeline_audio_in_to_audio_out(monkeypatch):
    """End-to-end smoke: feed an InputAudioRawFrame at the head, expect
    OutputAudioRawFrame at the tail.

    Always-on voicegate config means every transcription dispatches as
    a query; the brain echoes the text; the streaming TTS synthesizes
    a WAV; the WAV's audio frames are pushed downstream.
    """
    from xr_ai_pipecat import make_voice_pipeline
    from xr_ai_pipecat.transport import XRMediaHubTransport

    stt = _FakeStt(text="hi pipeline")
    tts = _FakeTts(sample_rate=22050)

    class _StubVad:
        def __init__(self, on_utterance, on_speech_start, **_):
            self._on_utt   = on_utterance
            self._on_start = on_speech_start

        async def feed(self, pcm_int16: bytes, sample_rate: int) -> None:
            await self._on_start()
            await self._on_utt(pcm_int16, sample_rate)

    monkeypatch.setattr("xr_ai_pipecat.processors.vad_stt.VadDetector", _StubVad)

    transport = XRMediaHubTransport()
    try:
        pipeline, _task = make_voice_pipeline(
            transport      = transport,
            stt            = stt,
            tts            = tts,
            brain          = _EchoBrain(),
            vad_cfg        = VadConfig(),
            voice_gate_cfg = VoiceGateConfig(),
        )
        # Confirm the factory composed the expected wiring: Pipeline
        # body is [transport.input(), vad_stt, voice_gate, brain,
        # streaming_tts, transport.output()]. Pipeline.processors wraps
        # that with Source/Sink at indices 0 and 7.
        kinds = [type(p).__name__ for p in pipeline.processors]
        assert kinds == [
            "PipelineSource",
            "XRMediaHubInputTransport",
            "VadSttProcessor",
            "VoiceGateProcessor",
            "_EchoBrain",
            "StreamingTtsProcessor",
            "XRMediaHubOutputTransport",
            "PipelineSink",
        ]
    finally:
        transport.shutdown()

    # Now spin up a fresh, transport-less pipeline with new processor
    # instances to exercise an audio → text → audio round-trip. Reusing
    # the original processors fails because they're already linked into
    # the factory's pipeline; a fresh chain is simpler than rewiring.
    voice_gate_cfg = VoiceGateConfig()
    voice_gate_proc = VoiceGateProcessor(cfg=voice_gate_cfg, tts=tts)
    streaming_tts   = StreamingTtsProcessor(tts=tts, voice_gate=voice_gate_proc.gate)
    vad_stt         = VadSttProcessor(stt=stt, vad_cfg=VadConfig())
    brain           = _EchoBrain()

    in_frame = InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=16000, num_channels=1)
    in_frame.transport_source = "web-client"
    sink = await _run_chain(
        vad_stt, voice_gate_proc, brain, streaming_tts,
        sends=[in_frame],
        settle_s=0.6,
    )

    audio_out = [f for f in sink.frames if isinstance(f, OutputAudioRawFrame)]
    assert audio_out, "expected at least one OutputAudioRawFrame at the tail"
    assert tts.calls == ["echo hi pipeline."]


# ════════════════════════════════════════════════════════════════════════════
# Regression: brain tags TextFrames with pid (Bug #2)
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_brain_tags_text_frame_with_pid_for_string_return():
    """The brain MUST set ``transport_destination`` on every TextFrame.

    Downstream ``StreamingTtsProcessor`` reads
    ``frame.transport_destination or ""`` and copies it onto the
    resulting ``OutputAudioRawFrame``. Without the pid tag the empty
    string flows through and the hub drops every audio chunk on the
    floor — the "agent thinks but says nothing" failure mode.
    """
    brain = _StringBrain()
    sink = await _run_chain(
        brain,
        sends=[GatedQueryFrame(participant_id="web-client", text="hi", fresh_match=True, pts_us=0)],
    )

    texts = [f for f in sink.frames if isinstance(f, TextFrame)]
    assert texts, "brain produced no TextFrame"
    assert all(t.transport_destination == "web-client" for t in texts)


@pytest.mark.asyncio
async def test_brain_tags_text_frame_with_pid_for_async_iter_return():
    brain = _IterBrain(chunks=["alpha ", "beta."])
    sink = await _run_chain(
        brain,
        sends=[GatedQueryFrame(participant_id="web-client", text="hi", fresh_match=True, pts_us=0)],
        settle_s=0.15,
    )
    texts = [f for f in sink.frames if isinstance(f, TextFrame)]
    assert len(texts) == 2
    assert all(t.transport_destination == "web-client" for t in texts)


# ════════════════════════════════════════════════════════════════════════════
# Regression: brain emits BrainResponseEndFrame at end of turn (Bug #4)
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_brain_emits_response_end_after_string_turn():
    """One ``BrainResponseEndFrame`` per completed turn carries the full
    assembled text and pid — the downstream data-channel echo keys off
    this marker."""
    brain = _StringBrain()
    sink = await _run_chain(
        brain,
        sends=[GatedQueryFrame(participant_id="pid-1", text="hi", fresh_match=True, pts_us=42)],
    )
    ends = [f for f in sink.frames if isinstance(f, BrainResponseEndFrame)]
    assert len(ends) == 1
    assert ends[0].pid    == "pid-1"
    assert ends[0].text   == "answer: hi"
    assert ends[0].pts_us == 42


@pytest.mark.asyncio
async def test_brain_emits_response_end_after_streamed_turn():
    brain = _IterBrain(chunks=["one ", "two ", "three."])
    sink = await _run_chain(
        brain,
        sends=[GatedQueryFrame(participant_id="pid-1", text="q", fresh_match=True, pts_us=7)],
        settle_s=0.2,
    )
    ends = [f for f in sink.frames if isinstance(f, BrainResponseEndFrame)]
    assert len(ends) == 1
    assert ends[0].text == "one two three."
    assert ends[0].pid  == "pid-1"


@pytest.mark.asyncio
async def test_brain_does_not_emit_response_end_on_cancel():
    """Cancellation (new query or InterruptionFrame) supersedes the
    in-flight turn — the data-channel echo would surface a partial
    answer that contradicts the new turn, so the brain skips the end
    marker. The second turn still emits its own end marker normally."""
    brain = _IterBrain(chunks=[f"chunk{i} " for i in range(200)])
    sink = await _run_chain(
        brain,
        sends=[
            GatedQueryFrame(participant_id="pid-1", text="first",  fresh_match=True, pts_us=0),
            GatedQueryFrame(participant_id="pid-1", text="second", fresh_match=True, pts_us=1),
        ],
        settle_s=0.4,
        per_send_delay_s=0.05,
    )
    ends = [f for f in sink.frames if isinstance(f, BrainResponseEndFrame)]
    # Only the second turn's end marker survives — the first was cancelled.
    assert len(ends) == 1
    assert ends[0].pts_us == 1


# ════════════════════════════════════════════════════════════════════════════
# Regression: handle_query async-def-returning-async-iterator shape works (Bug #3)
# ════════════════════════════════════════════════════════════════════════════


class _StreamMethodBrain(BrainProcessor):
    """Locks in the simple-vlm / xr-render-demo handle_query shape:

        async def handle_query(...) -> AsyncIterator[str]:
            return self._stream(...)

    where ``_stream`` is itself an async-generator function. ``await``-
    ing handle_query resolves to the async-generator object the base
    brain then iterates with ``async for`` — the shape is valid Python
    and must remain supported, since both samples use it.
    """

    def __init__(self, chunks: list[str]) -> None:
        super().__init__()
        self._chunks = chunks

    async def handle_query(self, pid, text, fresh_match) -> AsyncIterator[str]:
        return self._stream(pid, text)

    async def _stream(self, pid: str, text: str) -> AsyncIterator[str]:
        for c in self._chunks:
            yield c
            await asyncio.sleep(0.001)


@pytest.mark.asyncio
async def test_brain_supports_handle_query_returning_async_generator_method():
    brain = _StreamMethodBrain(chunks=["foo ", "bar."])
    sink = await _run_chain(
        brain,
        sends=[GatedQueryFrame(participant_id="pid-1", text="hi", fresh_match=True, pts_us=0)],
        settle_s=0.15,
    )
    texts = [f.text for f in sink.frames if isinstance(f, TextFrame)]
    assert texts == ["foo ", "bar."]
    ends = [f for f in sink.frames if isinstance(f, BrainResponseEndFrame)]
    assert len(ends) == 1
    assert ends[0].text == "foo bar."


# ════════════════════════════════════════════════════════════════════════════
# Regression: StreamingTts data-channel echo (Bug #4)
# ════════════════════════════════════════════════════════════════════════════


class _RecordingTransport:
    """Transport double — captures every ``send_return_data`` call."""

    def __init__(self) -> None:
        self.sends: list = []

    async def send_return_data(self, msg) -> None:
        self.sends.append(msg)


@pytest.mark.asyncio
async def test_streaming_tts_echoes_data_when_topic_set():
    tts  = _FakeTts()
    gate = VoiceGate(VoiceGateConfig(), audio_sink=_NullSink(), tts=tts)
    transport = _RecordingTransport()
    proc = StreamingTtsProcessor(
        tts=tts, voice_gate=gate,
        transport=transport, text_topic="vlm.response",
    )

    await _run_chain(
        proc,
        sends=[
            BrainResponseEndFrame(pid="web-client", text="hello there", pts_us=99),
        ],
    )

    assert len(transport.sends) == 1
    msg = transport.sends[0]
    assert msg.participant_id == "web-client"
    assert msg.topic          == "vlm.response"
    assert msg.data           == b"hello there"
    assert msg.pts_us         == 99


@pytest.mark.asyncio
async def test_streaming_tts_skips_echo_when_topic_empty():
    """Samples whose brain pushes its own per-turn data echo (e.g.
    xr-render-demo) pass ``text_topic=""`` to opt out of the
    pipeline-level send and avoid duplicates."""
    tts  = _FakeTts()
    gate = VoiceGate(VoiceGateConfig(), audio_sink=_NullSink(), tts=tts)
    transport = _RecordingTransport()
    proc = StreamingTtsProcessor(
        tts=tts, voice_gate=gate,
        transport=transport, text_topic="",
    )
    await _run_chain(
        proc,
        sends=[BrainResponseEndFrame(pid="pid-1", text="hi", pts_us=0)],
    )
    assert transport.sends == []


@pytest.mark.asyncio
async def test_streaming_tts_flushes_trailing_text_on_response_end():
    """The brain may finish a turn with text that has no sentence-final
    punctuation (e.g. partial answer). End-of-response is the last
    chance to flush the buffer; otherwise the tail of the reply is
    silently dropped."""
    tts  = _FakeTts()
    gate = VoiceGate(VoiceGateConfig(), audio_sink=_NullSink(), tts=tts)
    proc = StreamingTtsProcessor(tts=tts, voice_gate=gate)

    sink = await _run_chain(
        proc,
        sends=[
            TextFrame(text="trailing fragment with no period"),
            BrainResponseEndFrame(pid="pid-1", text="trailing fragment with no period", pts_us=0),
        ],
        settle_s=0.15,
    )
    assert tts.calls == ["trailing fragment with no period"]
    audio = [f for f in sink.frames if isinstance(f, OutputAudioRawFrame)]
    assert audio, "expected audio for the flushed trailing fragment"


@pytest.mark.asyncio
async def test_streaming_tts_flushes_sentence_ending_with_closing_quote():
    """Sentence-final punctuation followed by a closing quote/bracket
    must still flush the trailing fragment.

    Regression: the voice-gate greeting ends with ``... what am I
    looking at?"`` — the ``?`` is the sentence end but the buffer's
    last char is ``"``. A plain ``endswith((".", "!", "?"))`` check
    misses this; the tail stays in pending and concatenates onto the
    next turn's response, so the user hears half the greeting up front
    and the other half glued to their first query reply.
    """
    tts  = _FakeTts()
    gate = VoiceGate(VoiceGateConfig(), audio_sink=_NullSink(), tts=tts)
    proc = StreamingTtsProcessor(tts=tts, voice_gate=gate)

    await _run_chain(
        proc,
        sends=[TextFrame(text='How are you? "I am fine."')],
        settle_s=0.15,
    )

    # Both sentences must be dispatched in one turn — no residual
    # waiting to be glued onto the next reply.
    assert tts.calls == ['How are you?', '"I am fine."']
    assert proc._pending == ""  # noqa: SLF001


# ════════════════════════════════════════════════════════════════════════════
# Regression: output transport rewrites destination to default sender (Bug #1)
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_output_transport_handle_frame_routes_pid_to_default_sender(monkeypatch):
    """Pipecat's ``BaseOutputTransport._handle_frame`` drops frames whose
    ``transport_destination`` is not a registered key in
    ``_media_senders``. By default only ``None`` is registered, so a
    frame tagged with a pid (the way the brain / TTS / chime tag every
    outbound frame) would be silently dropped — the audio bug. The
    override rewrites the destination to ``None`` before delegating to
    the base class so the default sender picks it up; the hub layer
    routes by ``_target_participant``.
    """
    from xr_ai_pipecat.transport import XRMediaHubOutputTransport
    from pipecat.transports.base_output import BaseOutputTransport
    from pipecat.transports.base_transport import TransportParams

    class _StubEndpoint:
        async def send_return_audio(self, *_a, **_kw) -> None:
            return

    transport = XRMediaHubOutputTransport(_StubEndpoint(), TransportParams())

    # Capture what the super-class sees so the assertion focuses on the
    # destination rewrite without needing the full media-sender lifecycle.
    seen: list = []

    async def fake_super_handle(self, frame):
        seen.append((frame, frame.transport_destination))

    monkeypatch.setattr(BaseOutputTransport, "_handle_frame", fake_super_handle)

    frame = OutputAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)
    frame.transport_destination = "web-client"
    await transport._handle_frame(frame)

    assert seen, "super._handle_frame was not invoked"
    delegated_frame, dest_at_super_entry = seen[0]
    assert delegated_frame is frame
    # The super-class (default media sender router) must see destination=None
    # so it accepts the frame...
    assert dest_at_super_entry is None
    # ...but the pid is restored afterward so any downstream tap/sink still
    # sees which participant the frame was addressed to.
    assert frame.transport_destination == "web-client", (
        "destination must be restored after delegating to the default sender"
    )


@pytest.mark.asyncio
async def test_output_transport_writes_audio_to_target_participant():
    """End-to-end inside the output transport: ``write_audio_frame``
    (the pipecat hook the media sender invokes per chunked output frame)
    must produce one ``send_return_audio`` whose ``participant_id``
    matches the configured target.

    The previous implementation overrode the non-existent
    ``write_raw_audio_frames`` instead — pipecat never invoked it, so
    every TTS chunk was dropped before reaching the hub. This is the
    regression that locks the right hook in."""
    from xr_ai_agent import AudioChunk
    from xr_ai_pipecat.transport import (
        TTS_NATIVE_SAMPLE_RATE,
        XRMediaHubOutputTransport,
    )
    from pipecat.transports.base_transport import TransportParams

    captured: list[AudioChunk] = []

    class _StubEndpoint:
        async def send_return_audio(self, chunk: AudioChunk) -> None:
            captured.append(chunk)

    params = TransportParams(
        audio_out_enabled=True,
        audio_out_sample_rate=TTS_NATIVE_SAMPLE_RATE,
        audio_out_channels=1,
    )
    transport = XRMediaHubOutputTransport(_StubEndpoint(), params)
    transport.set_target_participant("web-client")

    pcm = b"\x00\x00" * 320  # 320 int16 samples = 20 ms @ 16 kHz
    frame = OutputAudioRawFrame(audio=pcm, sample_rate=TTS_NATIVE_SAMPLE_RATE, num_channels=1)
    ok = await transport.write_audio_frame(frame)

    assert ok is True
    assert len(captured) == 1
    assert captured[0].participant_id == "web-client"
    assert captured[0].track_id       == "tts"
    assert captured[0].sample_rate    == TTS_NATIVE_SAMPLE_RATE


@pytest.mark.asyncio
async def test_output_transport_routes_audio_by_frame_pid_not_single_target():
    """Multi-client isolation: ``write_audio_frame`` must address each chunk
    at the frame's own ``transport_destination`` (stamped by that
    participant's ``MediaSender``), NOT a single room-wide
    ``_target_participant``. Two participants speaking must each get their own
    answer; participant A's audio must never be delivered to B.

    Pre-fix the output transport nulled every frame's destination and used
    ``self._target_participant`` (set on each ``ParticipantJoinedFrame``, so
    last-join-wins) for every chunk — so A's TTS answer was published on B's
    return-audio track. This locks per-frame routing in."""
    from xr_ai_agent import AudioChunk
    from xr_ai_pipecat.transport import (
        TTS_NATIVE_SAMPLE_RATE,
        XRMediaHubOutputTransport,
    )
    from pipecat.transports.base_transport import TransportParams

    captured: list[AudioChunk] = []

    class _StubEndpoint:
        async def send_return_audio(self, chunk: AudioChunk) -> None:
            captured.append(chunk)

    params = TransportParams(
        audio_out_enabled=True,
        audio_out_sample_rate=TTS_NATIVE_SAMPLE_RATE,
        audio_out_channels=1,
    )
    transport = XRMediaHubOutputTransport(_StubEndpoint(), params)
    # Simulate the pre-fix steering: brain set the room-wide target to the
    # last participant that joined. Per-frame routing must override this.
    transport.set_target_participant("bob")

    pcm = b"\x00\x00" * 320
    frame_a = OutputAudioRawFrame(audio=pcm, sample_rate=TTS_NATIVE_SAMPLE_RATE, num_channels=1)
    frame_a.transport_destination = "alice"
    frame_b = OutputAudioRawFrame(audio=pcm, sample_rate=TTS_NATIVE_SAMPLE_RATE, num_channels=1)
    frame_b.transport_destination = "bob"

    assert await transport.write_audio_frame(frame_a) is True
    assert await transport.write_audio_frame(frame_b) is True

    # Each chunk addressed at its own participant — not both at "bob".
    assert [c.participant_id for c in captured] == ["alice", "bob"]


@pytest.mark.asyncio
async def test_output_transport_write_audio_frame_returns_false_without_target():
    """No target participant configured — drop the frame at the hub
    boundary instead of emitting an unaddressable AudioChunk. Returning
    False also tells pipecat to skip the downstream push so a tail tap
    doesn't see a half-routed frame."""
    from xr_ai_pipecat.transport import XRMediaHubOutputTransport
    from pipecat.transports.base_transport import TransportParams

    captured: list = []

    class _StubEndpoint:
        async def send_return_audio(self, chunk) -> None:
            captured.append(chunk)

    transport = XRMediaHubOutputTransport(_StubEndpoint(), TransportParams())
    frame = OutputAudioRawFrame(audio=b"\x00\x00", sample_rate=22050, num_channels=1)
    ok = await transport.write_audio_frame(frame)
    assert ok is False
    assert captured == []


# ════════════════════════════════════════════════════════════════════════════
# E2E smoke: GatedQueryFrame → full pipeline → OutputAudioRawFrame reaches transport
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_gated_query_drives_audio_through_full_pipeline_to_transport():
    """The end-to-end "agent says something" path:

    GatedQueryFrame → brain (TextFrame tagged with pid) → StreamingTts
    (OutputAudioRawFrame tagged with pid) → output transport
    (write_raw_audio_frames invoked → send_return_audio).

    This is the path that was previously silently dropping every audio
    frame at the output transport's media-sender router. The assertion
    is on ``send_return_audio`` reaching the hub, not just on audio
    frames appearing at the tail — that's the bug we shipped the fix
    for.
    """
    from xr_ai_agent import AudioChunk
    from xr_ai_pipecat.transport import (
        TTS_NATIVE_SAMPLE_RATE,
        XRMediaHubOutputTransport,
    )
    from pipecat.transports.base_transport import TransportParams

    captured: list[AudioChunk] = []

    class _StubEndpoint:
        async def send_return_audio(self, chunk: AudioChunk) -> None:
            captured.append(chunk)

    params = TransportParams(
        audio_out_enabled=True,
        audio_out_sample_rate=TTS_NATIVE_SAMPLE_RATE,
        audio_out_channels=1,
    )
    output = XRMediaHubOutputTransport(_StubEndpoint(), params)
    output.set_target_participant("web-client")

    tts  = _FakeTts(sample_rate=TTS_NATIVE_SAMPLE_RATE)
    gate = VoiceGate(VoiceGateConfig(), audio_sink=_NullSink(), tts=tts)
    brain = _StringBrain()
    streaming_tts = StreamingTtsProcessor(tts=tts, voice_gate=gate)

    await _run_chain(
        brain, streaming_tts, output,
        sends=[GatedQueryFrame(
            participant_id="web-client", text="echo this", fresh_match=True, pts_us=0,
        )],
        settle_s=0.4,
    )

    assert captured, "no audio reached the hub via send_return_audio"
    assert all(c.participant_id == "web-client" for c in captured)


# ════════════════════════════════════════════════════════════════════════════
# Regression: factory wires text_topic through to StreamingTts (Bug #4)
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_make_voice_pipeline_wires_text_topic_through_streaming_tts(monkeypatch):
    """The factory's ``text_topic`` argument must reach the streaming
    TTS processor's data-echo path. Previously the parameter was
    explicitly unused (``# noqa: ARG001``) and the echo silently never
    fired."""
    from xr_ai_pipecat import make_voice_pipeline
    from xr_ai_pipecat.transport import XRMediaHubTransport

    transport = XRMediaHubTransport()
    try:
        pipeline, _task = make_voice_pipeline(
            transport      = transport,
            stt            = _FakeStt(),
            tts            = _FakeTts(),
            brain          = _StringBrain(),
            vad_cfg        = VadConfig(),
            voice_gate_cfg = VoiceGateConfig(),
            text_topic     = "vlm.response",
        )
        # The streaming-tts processor lives at index 5 in the wrapped
        # pipeline (source, input, vad_stt, voice_gate, brain, tts, output, sink).
        streaming_tts = pipeline.processors[5]
        assert isinstance(streaming_tts, StreamingTtsProcessor)
        assert streaming_tts._text_topic == "vlm.response"
        assert streaming_tts._transport is transport
    finally:
        transport.shutdown()


# ════════════════════════════════════════════════════════════════════════════
# Regression: transport subscribes to VIDEO so brain receives FrameSignals
# ════════════════════════════════════════════════════════════════════════════


def test_xr_media_hub_transport_subscribes_to_video_frames():
    """The ProcessorEndpoint must subscribe to the video category so
    that camera FrameSignals reach brain consumers (e.g. VLM workers
    that bind ``ep.on_frame``). A previous version of the transport
    filtered out video at the ZMQ subscription layer, causing
    ``_wait_for_camera_frame`` to time out and the VLM call to block
    indefinitely after a query."""
    from xr_ai_agent._processor import Subscribe
    from xr_ai_pipecat.transport import XRMediaHubTransport

    transport = XRMediaHubTransport()
    try:
        assert transport._ep._default_filter & Subscribe.VIDEO, (
            "transport must subscribe to VIDEO frames so brains receive "
            "camera FrameSignals"
        )
        # Audio and data are also required for the voice pipeline.
        assert transport._ep._default_filter & Subscribe.AUDIO
        assert transport._ep._default_filter & Subscribe.DATA
    finally:
        transport.shutdown()
