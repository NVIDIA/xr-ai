# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the xr-ai-voice unified voice pipeline.

Each library FrameProcessor (VadStt, VoiceGate, Assistant, StreamingTts) is
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
from typing import Any, AsyncIterator, Sequence

import nemo_relay
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

from xr_ai_voice import VadConfig
from xr_ai_voice._types import VoiceQuery
from xr_ai_voice._pipeline import _build_voice_pipeline
from xr_ai_voice._frames import (
    AssistantResponseEndFrame,
    GatedQueryFrame,
    ParticipantJoinedFrame,
    ParticipantLeftFrame,
)
from xr_ai_voice._processors import (
    _VoiceIOProcessor,
    StreamingTtsProcessor,
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
        self.run_started = asyncio.Event()
        self.run_finished = asyncio.Event()
        self.ready_to_receive = asyncio.Event()

    def on_audio(self, cb) -> None:
        self.audio_cb = cb

    def on_participant(self, cb) -> None:
        self.participant_cb = cb

    async def run(self) -> None:
        self.run_started.set()
        await self.run_finished.wait()

    async def wait_until_running(self) -> None:
        await self.ready_to_receive.wait()

    def stop(self) -> None:
        self.run_finished.set()


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

    monkeypatch.setattr("xr_ai_voice._processors.vad_stt.VadDetector", _StubVad)
    proc = VadSttProcessor(stt=stt, vad_cfg=VadConfig())
    frame = InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=16000, num_channels=1)
    frame.transport_source = "web-client"

    events = []
    subscriber = "xr-ai-voice-stt-scopes"
    nemo_relay.subscribers.register(subscriber, events.append)
    try:
        sink = await _run_chain(proc, sends=[frame])
        await nemo_relay.subscribers.flush_async()
    finally:
        nemo_relay.subscribers.deregister(subscriber)

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
    stt_start = next(
        event.to_dict()
        for event in events
        if event.name == "voice.stt"
        and event.to_dict().get("scope_category") == "start"
    )
    stt_result = next(
        event.to_dict() for event in events if event.name == "voice.stt.result"
    )
    assert stt_start["category"] == "function"
    assert stt_start["data"] == {
        "audio_bytes": 640,
        "audio_duration_ms": 20.0,
        "sample_rate": 16000,
    }
    assert stt_start["metadata"] | {
        "participant_id": "web-client",
        "mode": "final",
    } == stt_start["metadata"]
    assert stt_result["parent_uuid"] == stt_start["uuid"]
    assert stt_result["data"] == {"text": "hello agent"}


@pytest.mark.asyncio
async def test_vad_stt_anchors_transcript_to_speech_onset(monkeypatch):
    """The transcript carries the capture time of the audio that opened the
    utterance, not the time STT happened to finish.

    The hub stamps each AudioChunk; the transport forwards it as the frame's
    presentation timestamp. Stamping wall-clock after STT instead would bake in
    VAD hangover plus transcription latency, and that error persists into stored
    transcripts and time-relative recorded-frame lookups."""
    stt = _FakeStt(text="hello agent")

    class _StubVad:
        def __init__(self, on_utterance, on_speech_start, **_):
            self._on_utt   = on_utterance
            self._on_start = on_speech_start

        async def feed(self, pcm_int16: bytes, sample_rate: int) -> None:
            await self._on_start()
            await self._on_utt(pcm_int16, sample_rate)

    monkeypatch.setattr("xr_ai_voice._processors.vad_stt.VadDetector", _StubVad)
    proc = VadSttProcessor(stt=stt, vad_cfg=VadConfig())
    frame = InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=16000, num_channels=1)
    frame.transport_source = "web-client"
    frame.pts = 1_700_000_000_000_000  # ns

    sink = await _run_chain(proc, sends=[frame])

    transcripts = [f for f in sink.frames if isinstance(f, TranscriptionFrame)]
    assert transcripts[0].pts == 1_700_000_000_000_000


@pytest.mark.asyncio
async def test_vad_stt_cancels_probe_tasks_on_pipeline_shutdown(monkeypatch):
    """A pending early-STT probe must not survive pipeline shutdown — otherwise
    it can push a transcript (and start a turn) after the session ended."""
    stt = _FakeStt(text="hello agent")

    class _StubVad:
        def __init__(self, on_utterance, on_speech_start, **_):
            self._on_start = on_speech_start

        async def feed(self, pcm_int16: bytes, sample_rate: int) -> None:
            await self._on_start()  # opens a probe task, never finalizes

    monkeypatch.setattr("xr_ai_voice._processors.vad_stt.VadDetector", _StubVad)
    # A long probe delay guarantees the task is still pending at EndFrame.
    proc = VadSttProcessor(stt=stt, vad_cfg=VadConfig(stop_probe_after_s=30.0))
    frame = InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=16000, num_channels=1)
    frame.transport_source = "web-client"

    await _run_chain(proc, sends=[frame], settle_s=0.15)

    # _run_chain drains with an EndFrame; no probe task may outlive it.
    assert all(t.done() for t in proc._probe_task.values())  # noqa: SLF001


@pytest.mark.asyncio
async def test_vad_stt_swallows_empty_transcript(monkeypatch):
    stt = _FakeStt(text="")

    class _StubVad:
        def __init__(self, on_utterance, on_speech_start, **_):
            self._on_utt = on_utterance

        async def feed(self, pcm_int16: bytes, sample_rate: int) -> None:
            await self._on_utt(pcm_int16, sample_rate)

    monkeypatch.setattr("xr_ai_voice._processors.vad_stt.VadDetector", _StubVad)
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

    monkeypatch.setattr("xr_ai_voice._processors.vad_stt.VadDetector", _StubVad)
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
    monkeypatch.setattr("xr_ai_voice._processors.vad_stt.VadDetector", _StagedVad)
    proc = VadSttProcessor(stt=stt, vad_cfg=VadConfig(stop_probe_after_s=0.05))

    frame = InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=16000, num_channels=1)
    frame.transport_source = "web-client"
    await _run_chain(proc, sends=[frame], settle_s=0.2)

    # Exactly one STT call — the probe's — because on_utterance never fired.
    assert len(stt.calls) == 1
    assert stt.calls[0][1] == 16000


@pytest.mark.asyncio
async def test_vad_stt_retries_partial_probe_until_wake_phrase_is_recognized(monkeypatch):
    _StagedVad.instances.clear()
    stt = _StagedStt(texts=["hey", "hey agent place a cube"])
    partials: list[tuple[str, str]] = []

    async def on_partial(pid: str, text: str) -> bool:
        partials.append((pid, text))
        return text.startswith("hey agent")

    monkeypatch.setattr("xr_ai_voice._processors.vad_stt.VadDetector", _StagedVad)
    proc = VadSttProcessor(
        stt=stt,
        vad_cfg=VadConfig(stop_probe_after_s=0.05),
        on_partial_transcript=on_partial,
    )

    frame = InputAudioRawFrame(
        audio=b"\x00\x00" * 320,
        sample_rate=16000,
        num_channels=1,
    )
    frame.transport_source = "web-client"
    await _run_chain(proc, sends=[frame], settle_s=0.25)

    assert partials == [
        ("web-client", "hey"),
        ("web-client", "hey agent place a cube"),
    ]
    assert len(stt.calls) == 2


@pytest.mark.asyncio
async def test_vad_stt_stops_partial_probes_after_rejected_prefix(monkeypatch):
    _StagedVad.instances.clear()
    stt = _StagedStt(texts=["ordinary room conversation"])

    async def reject_partial(_pid: str, _text: str) -> bool | None:
        return None

    monkeypatch.setattr("xr_ai_voice._processors.vad_stt.VadDetector", _StagedVad)
    proc = VadSttProcessor(
        stt=stt,
        vad_cfg=VadConfig(stop_probe_after_s=0.05),
        on_partial_transcript=reject_partial,
    )
    frame = InputAudioRawFrame(
        audio=b"\x00\x00" * 320,
        sample_rate=16000,
        num_channels=1,
    )
    frame.transport_source = "web-client"

    await _run_chain(proc, sends=[frame], settle_s=0.25)

    assert len(stt.calls) == 1


@pytest.mark.asyncio
async def test_vad_stt_stop_probe_emits_interruption_on_stop_match(monkeypatch):
    """When the probe's partial transcript matches ``STOP_RE`` the
    processor pushes ``InterruptionFrame`` + the matched
    ``TranscriptionFrame`` + ``UserStoppedSpeakingFrame`` downstream
    immediately — without waiting for VAD's silence-window finalize."""
    _StagedVad.instances.clear()
    stt = _StagedStt(texts=["stop"])
    monkeypatch.setattr("xr_ai_voice._processors.vad_stt.VadDetector", _StagedVad)
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
    monkeypatch.setattr("xr_ai_voice._processors.vad_stt.VadDetector", _StagedVad)
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
    monkeypatch.setattr("xr_ai_voice._processors.vad_stt.VadDetector", _StagedVad)
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
async def test_vad_stt_final_transcription_waits_for_probe_cancellation(monkeypatch):
    _StagedVad.instances.clear()

    class _SerialStt(_StagedStt):
        def __init__(self) -> None:
            super().__init__(texts=[])
            self.first_started = asyncio.Event()
            self.active = 0
            self.max_active = 0

        async def transcribe(
            self,
            audio: bytes,
            *,
            sample_rate: int | None = None,
            channels: int = 1,
            timeout: float | None = None,
        ) -> str:
            self.calls.append((audio, sample_rate or 16000))
            call_number = len(self.calls)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                if call_number == 1:
                    self.first_started.set()
                    await asyncio.Event().wait()
                return "hey agent place a cube"
            finally:
                self.active -= 1

    stt = _SerialStt()
    monkeypatch.setattr("xr_ai_voice._processors.vad_stt.VadDetector", _StagedVad)
    proc = VadSttProcessor(stt=stt, vad_cfg=VadConfig(stop_probe_after_s=0.05))
    frame = InputAudioRawFrame(
        audio=b"\x00\x00" * 320,
        sample_rate=16000,
        num_channels=1,
    )
    frame.transport_source = "web-client"

    sink = _CaptureSink()
    pipeline = Pipeline([proc, sink])
    worker = PipelineWorker(pipeline, cancel_on_idle_timeout=False, enable_rtvi=False)
    runner = WorkerRunner()
    await runner.add_workers(worker)

    async def drive() -> None:
        await asyncio.sleep(0.05)
        await worker.queue_frame(frame)
        await asyncio.wait_for(stt.first_started.wait(), timeout=1.0)
        await _StagedVad.instances[-1].trigger_utterance()
        await asyncio.sleep(0.05)
        await worker.queue_frame(EndFrame())

    await asyncio.gather(runner.run(), drive())

    assert len(stt.calls) == 2
    assert stt.max_active == 1
    transcripts = [item for item in sink.frames if isinstance(item, TranscriptionFrame)]
    assert [item.text for item in transcripts] == ["hey agent place a cube"]


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
    monkeypatch.setattr("xr_ai_voice._processors.vad_stt.VadDetector", _StagedVad)
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
    monkeypatch.setattr("xr_ai_voice._processors.vad_stt.VadDetector", _StagedVad)
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
    monkeypatch.setattr("xr_ai_voice._processors.vad_stt.VadDetector", _StagedVad)
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
async def test_voice_gate_processor_stamps_query_with_speech_onset():
    """The dispatched query carries the transcript's speech-onset timestamp, so
    a turn is anchored to when the user spoke rather than to when STT
    finished."""
    cfg = VoiceGateConfig(magic_phrases=("agent",))
    proc = VoiceGateProcessor(cfg=cfg, tts=_FakeTts())
    tf = TranscriptionFrame(text="agent what time is it", user_id="pid-1", timestamp="t")
    tf.pts = 1_700_000_000_000_000  # ns

    sink = await _run_chain(proc, sends=[tf])

    queries = [f for f in sink.frames if isinstance(f, GatedQueryFrame)]
    assert queries[0].pts_us == 1_700_000_000_000  # ns -> µs


@pytest.mark.asyncio
async def test_voice_gate_processor_falls_back_to_wall_clock_without_pts():
    """A transcript with no capture time (e.g. injected without an originating
    audio frame) still gets a usable timestamp."""
    cfg = VoiceGateConfig(magic_phrases=("agent",))
    proc = VoiceGateProcessor(cfg=cfg, tts=_FakeTts())

    sink = await _run_chain(
        proc,
        sends=[TranscriptionFrame(text="agent hello", user_id="pid-1", timestamp="t")],
    )

    queries = [f for f in sink.frames if isinstance(f, GatedQueryFrame)]
    assert queries[0].pts_us > 0


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
async def test_voice_gate_processor_marks_followup_at_speech_start(monkeypatch):
    cfg = VoiceGateConfig(magic_phrases=("agent",))
    proc = VoiceGateProcessor(cfg=cfg, tts=_FakeTts())
    started_pids: list[str] = []
    monkeypatch.setattr(proc.gate, "begin_utterance", started_pids.append)

    started = UserStartedSpeakingFrame()
    started.transport_source = "pid-1"
    await _run_chain(proc, sends=[started])

    assert started_pids == ["pid-1"]


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


@pytest.mark.asyncio
async def test_voice_gate_processor_chimes_on_partial_wake_without_dispatching_early():
    cfg = VoiceGateConfig(
        magic_phrases=("agent", "hey agent"),
        listening_chime=True,
    )
    proc = VoiceGateProcessor(cfg=cfg, tts=_FakeTts())
    proc.gate.observe_tts_wav(_silence_wav(24000))

    sink = _CaptureSink()
    pipeline = Pipeline([proc, sink])
    worker = PipelineWorker(
        pipeline,
        cancel_on_idle_timeout=False,
        enable_rtvi=False,
    )
    runner = WorkerRunner()
    await runner.add_workers(worker)
    early_audio_count = 0

    async def drive() -> None:
        nonlocal early_audio_count
        await asyncio.sleep(0.05)
        acknowledged = await proc.handle_partial_transcript(
            "pid-1", "hey agent place a blue sphere",
        )
        assert acknowledged is True
        early_audio_count = sum(
            isinstance(frame, OutputAudioRawFrame) for frame in sink.frames
        )
        assert not any(isinstance(frame, GatedQueryFrame) for frame in sink.frames)
        await worker.queue_frame(TranscriptionFrame(
            text="hey agent place a blue sphere",
            user_id="pid-1",
            timestamp="t",
        ))
        await asyncio.sleep(0.05)
        await worker.queue_frame(EndFrame())

    await asyncio.gather(runner.run(), drive())

    audio_count = sum(
        isinstance(frame, OutputAudioRawFrame) for frame in sink.frames
    )
    queries = [frame for frame in sink.frames if isinstance(frame, GatedQueryFrame)]
    assert early_audio_count > 0
    assert audio_count == early_audio_count
    assert [query.text for query in queries] == ["place a blue sphere"]


@pytest.mark.asyncio
async def test_voice_gate_processor_classifies_partial_wake_prefixes():
    cfg = VoiceGateConfig(
        magic_phrases=("agent", "hey agent"),
        listening_chime=True,
    )
    proc = VoiceGateProcessor(cfg=cfg, tts=_FakeTts())

    assert await proc.handle_partial_transcript("pid-1", "hey") is False
    assert await proc.handle_partial_transcript("pid-1", "room conversation") is None


@pytest.mark.asyncio
async def test_voice_gate_processor_phrase_only_falls_back_to_final_chime():
    cfg = VoiceGateConfig(magic_phrases=("hey agent",), listening_chime=True)
    proc = VoiceGateProcessor(cfg=cfg, tts=_FakeTts())
    proc.gate.observe_tts_wav(_silence_wav(24000))

    sink = await _run_chain(
        proc,
        sends=[TranscriptionFrame(text="hey agent", user_id="pid-1", timestamp="t")],
    )

    assert any(isinstance(frame, OutputAudioRawFrame) for frame in sink.frames)
    assert not any(isinstance(frame, GatedQueryFrame) for frame in sink.frames)


# ════════════════════════════════════════════════════════════════════════════
# Private assistant processor
# ════════════════════════════════════════════════════════════════════════════


class _StringAssistant(_VoiceIOProcessor):
    def __init__(self, **callbacks: Any) -> None:
        super().__init__(self.handle, **callbacks)
        self.handle_calls: list[tuple[str, str]] = []

    async def handle(self, query: VoiceQuery) -> None:
        self.handle_calls.append((query.participant_id, query.text))
        await self.enqueue_response(
            query.participant_id,
            f"answer: {query.text}",
            pts_us=query.timestamp_us,
        )


class _IterAssistant(_VoiceIOProcessor):
    def __init__(self, chunks: list[str], **callbacks: Any) -> None:
        super().__init__(self.handle, **callbacks)
        self._chunks = chunks
        self.cancelled = False

    async def handle(self, query: VoiceQuery) -> None:
        async def _gen():
            try:
                for c in self._chunks:
                    yield c
                    await asyncio.sleep(0.001)
            except asyncio.CancelledError:
                self.cancelled = True
                raise

        await self.enqueue_response(
            query.participant_id,
            _gen(),
            pts_us=query.timestamp_us,
        )


class _LifecycleAssistant(_VoiceIOProcessor):
    def __init__(self) -> None:
        self.left:             list[str] = []
        super().__init__(
            self.handle,
            on_participant_left=self.on_participant_left,
        )

    async def handle(self, _query: VoiceQuery) -> None:
        pass

    async def on_participant_left(self, pid: str) -> None:
        self.left.append(pid)

class _InputOnlyAssistant(_VoiceIOProcessor):
    def __init__(self) -> None:
        self.queries: list[VoiceQuery] = []
        super().__init__(self.handle)

    async def handle(self, query: VoiceQuery) -> None:
        self.queries.append(query)


@pytest.mark.asyncio
async def test_runtime_input_can_enqueue_finite_voice_output():
    assistant = _StringAssistant()
    sink = await _run_chain(
        assistant,
        sends=[GatedQueryFrame(participant_id="pid-1", text="hi", fresh_match=True, pts_us=0)],
    )

    texts = [f for f in sink.frames if isinstance(f, TextFrame)]
    assert [t.text for t in texts] == ["answer: hi"]
    assert assistant.handle_calls == [("pid-1", "hi")]


@pytest.mark.asyncio
async def test_runtime_input_can_enqueue_incremental_voice_output():
    assistant = _IterAssistant(chunks=["alpha ", "beta ", "gamma."])
    sink = await _run_chain(
        assistant,
        sends=[GatedQueryFrame(participant_id="pid-1", text="hi", fresh_match=True, pts_us=0)],
        settle_s=0.15,
    )
    texts = [f.text for f in sink.frames if isinstance(f, TextFrame)]
    assert texts == ["alpha ", "beta ", "gamma."]


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_interrupting_response_does_not_cancel_query_delivery() -> None:
    query_started = asyncio.Event()
    release_query = asyncio.Event()

    async def input_sink(_query: VoiceQuery) -> None:
        query_started.set()
        await release_query.wait()

    assistant = _VoiceIOProcessor(input_sink)

    async def capture(
        _frame: Frame,
        _direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ) -> None:
        return None

    assistant.push_frame = capture  # type: ignore[method-assign]
    await assistant.enqueue_query("pid-1", "look")
    await asyncio.wait_for(query_started.wait(), 1.0)
    input_task = next(iter(assistant._input_tasks))  # noqa: SLF001

    await assistant.enqueue_response("pid-1", "answer", interrupt=True)
    await asyncio.wait_for(assistant._inflight["pid-1"], 1.0)  # noqa: SLF001

    assert not input_task.cancelled()
    release_query.set()
    await asyncio.wait_for(input_task, 1.0)


@pytest.mark.asyncio
async def test_failed_response_iterator_is_closed() -> None:
    class FailingStream:
        def __init__(self) -> None:
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self) -> str:
            raise RuntimeError("producer failed")

        async def aclose(self) -> None:
            self.closed = True

    async def input_sink(_query: VoiceQuery) -> None:
        return None

    stream = FailingStream()
    assistant = _VoiceIOProcessor(input_sink)

    async def capture(
        _frame: Frame,
        _direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ) -> None:
        return None

    assistant.push_frame = capture  # type: ignore[method-assign]
    await assistant.enqueue_response("pid-1", stream)
    await asyncio.wait_for(assistant._inflight["pid-1"], 1.0)  # noqa: SLF001

    assert stream.closed is True


async def test_external_response_stream_uses_normal_assistant_output_frames() -> None:
    frames: list[Frame] = []

    async def input_sink(_query: VoiceQuery) -> None:
        pass

    async def chunks() -> AsyncIterator[str]:
        yield "The kettle "
        yield "is boiling."

    assistant = _VoiceIOProcessor(input_sink)

    async def capture(frame: Frame, _direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        frames.append(frame)

    assistant.push_frame = capture  # type: ignore[method-assign]
    await assistant.enqueue_response("pid-1", chunks(), pts_us=42)
    task = assistant._inflight["pid-1"]  # noqa: SLF001
    _ = await task

    assert [frame.text for frame in frames if isinstance(frame, TextFrame)] == [
        "The kettle ",
        "is boiling.",
    ]
    end = next(frame for frame in frames if isinstance(frame, AssistantResponseEndFrame))
    assert (end.pid, end.text, end.pts_us) == ("pid-1", "The kettle is boiling.", 42)


@pytest.mark.asyncio
async def test_external_response_preserves_chunks_before_iterator_failure() -> None:
    frames: list[Frame] = []

    async def input_sink(_query: VoiceQuery) -> None:
        pass

    async def chunks() -> AsyncIterator[str]:
        yield "Partial response."
        raise RuntimeError("producer failed")

    assistant = _VoiceIOProcessor(input_sink)

    async def capture(
        frame: Frame,
        _direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ) -> None:
        frames.append(frame)

    assistant.push_frame = capture  # type: ignore[method-assign]
    await assistant.enqueue_response("pid-1", chunks(), pts_us=42)
    await assistant._inflight["pid-1"]  # noqa: SLF001

    end = next(frame for frame in frames if isinstance(frame, AssistantResponseEndFrame))
    assert end.text == "Partial response."


@pytest.mark.asyncio
async def test_external_responses_preserve_participant_fifo() -> None:
    frames: list[Frame] = []
    release_first = asyncio.Event()

    async def input_sink(_query: VoiceQuery) -> None:
        pass

    async def first_response() -> AsyncIterator[str]:
        yield "first "
        await release_first.wait()
        yield "done"

    assistant = _VoiceIOProcessor(input_sink)

    async def capture(
        frame: Frame,
        _direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ) -> None:
        frames.append(frame)

    assistant.push_frame = capture  # type: ignore[method-assign]
    await assistant.enqueue_response("pid-1", first_response())
    await asyncio.sleep(0)
    await assistant.enqueue_response("pid-1", "second")

    assert len(assistant._queued["pid-1"]) == 1  # noqa: SLF001
    release_first.set()
    async def wait_until_idle() -> None:
        while assistant._inflight:  # noqa: SLF001
            await asyncio.gather(*tuple(assistant._inflight.values()))  # noqa: SLF001

    await asyncio.wait_for(wait_until_idle(), 1.0)

    assert [frame.text for frame in frames if isinstance(frame, TextFrame)] == [
        "first ",
        "done",
        "second",
    ]


@pytest.mark.asyncio
async def test_assistant_does_not_cancel_on_user_started_speaking():
    """Regression guard: ``UserStartedSpeakingFrame`` is a hook, not a
    cancel signal. Cancelling on speech onset breaks two things:

    * any AEC leak of the agent's own TTS becomes self-cancel,
    * a quick follow-up utterance aborts the prior response BEFORE the
      voice gate even decides whether the new utterance was a query.

    The assistant must keep streaming TextFrames; cancellation happens on
    the next GatedQueryFrame or on an explicit InterruptionFrame."""
    assistant = _IterAssistant(chunks=[f"chunk{i} " for i in range(5)])
    sink = await _run_chain(
        assistant,
        sends=[
            GatedQueryFrame(participant_id="pid-1", text="hi", fresh_match=True, pts_us=0),
            UserStartedSpeakingFrame(),
        ],
        settle_s=0.3,
        per_send_delay_s=0.05,
    )
    assert assistant.cancelled is False
    texts = [f.text for f in sink.frames if isinstance(f, TextFrame)]
    assert texts == [f"chunk{i} " for i in range(5)]


@pytest.mark.asyncio
async def test_assistant_cancels_inflight_on_new_query_for_same_pid():
    """A fresh GatedQueryFrame supersedes any in-flight reasoning for
    the same pid — this is the contract that makes rapid follow-ups
    work without the user having to wait for the previous answer."""
    assistant = _IterAssistant(chunks=[f"chunk{i} " for i in range(200)])
    await _run_chain(
        assistant,
        sends=[
            GatedQueryFrame(participant_id="pid-1", text="hi",   fresh_match=True, pts_us=0),
            GatedQueryFrame(participant_id="pid-1", text="hi 2", fresh_match=True, pts_us=1),
        ],
        settle_s=0.2,
        per_send_delay_s=0.05,
    )
    assert assistant.cancelled is True


@pytest.mark.asyncio
async def test_assistant_cancels_inflight_on_interruption_frame():
    assistant = _IterAssistant(chunks=[f"chunk{i} " for i in range(200)])
    await _run_chain(
        assistant,
        sends=[
            GatedQueryFrame(participant_id="pid-1", text="hi", fresh_match=True, pts_us=0),
            InterruptionFrame(),
        ],
        settle_s=0.2,
        per_send_delay_s=0.05,
    )
    assert assistant.cancelled is True


@pytest.mark.asyncio
async def test_voice_io_closes_cancelled_response_stream() -> None:
    started = asyncio.Event()
    blocked = asyncio.Event()

    class HeldStream:
        def __init__(self) -> None:
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self) -> str:
            started.set()
            await blocked.wait()
            raise StopAsyncIteration

        async def aclose(self) -> None:
            self.closed = True

    stream = HeldStream()

    async def handle(_query: VoiceQuery) -> None:
        return None

    assistant = _VoiceIOProcessor(handle)

    async def capture(
        _frame: Frame,
        _direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ) -> None:
        return None

    assistant.push_frame = capture  # type: ignore[method-assign]
    await assistant.enqueue_response("pid-1", stream)
    task = assistant._inflight["pid-1"]  # noqa: SLF001
    await asyncio.wait_for(started.wait(), 1.0)
    await assistant._cancel_pid("pid-1")  # noqa: SLF001
    await asyncio.gather(task, return_exceptions=True)

    assert stream.closed is True


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_participant_left_clears_seen_output_state() -> None:
    async def input_sink(_query: VoiceQuery) -> None:
        return None

    assistant = _VoiceIOProcessor(input_sink)
    assistant._seen_output.add("pid-1")  # noqa: SLF001

    async def capture(
        _frame: Frame,
        _direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ) -> None:
        return None

    assistant.push_frame = capture  # type: ignore[method-assign]
    await assistant.process_frame(
        ParticipantLeftFrame(participant_id="pid-1"),
        FrameDirection.DOWNSTREAM,
    )

    assert "pid-1" not in assistant._seen_output  # noqa: SLF001


async def test_assistant_notifies_external_runtime_on_interruption_frame():
    interrupted: list[str | None] = []

    async def handle(_query: VoiceQuery) -> str:
        return "unused"

    async def on_interrupted(pid: str | None) -> None:
        interrupted.append(pid)

    assistant = _VoiceIOProcessor(handle, on_interrupted=on_interrupted)
    participant_frame = InterruptionFrame()
    participant_frame.transport_source = "pid-1"
    global_frame = InterruptionFrame()
    await _run_chain(assistant, sends=[participant_frame, global_frame])

    assert interrupted == ["pid-1", None]


@pytest.mark.asyncio
async def test_assistant_cancels_inflight_turn_on_pipeline_shutdown():
    """A handler task must not outlive the pipeline. Otherwise a turn keeps
    emitting text — and a turn observer keeps writing transcripts — after the
    session has ended."""
    class _SlowAssistant(_IterAssistant):
        def __init__(self) -> None:
            super().__init__(chunks=[f"chunk{i} " for i in range(10_000)])

    assistant = _SlowAssistant()
    await _run_chain(
        assistant,
        sends=[GatedQueryFrame(participant_id="pid-1", text="hi", fresh_match=True, pts_us=0)],
        settle_s=0.1,
    )

    # _run_chain drains with an EndFrame; the turn must be torn down by then.
    assert assistant._inflight == {}          # noqa: SLF001
    assert assistant.cancelled, "the in-flight handler task must be cancelled"


@pytest.mark.asyncio
async def test_handler_can_request_audio_interruption_when_superseded():
    """The explicit supersede option drains queued TTS audio for the prior response."""
    class _InterruptingAssistant(_IterAssistant):
        def __init__(self) -> None:
            super().__init__(
                chunks=[f"chunk{i} " for i in range(200)],
                interrupt_on_supersede=True,
            )

    assistant = _InterruptingAssistant()
    sink = await _run_chain(
        assistant,
        sends=[
            GatedQueryFrame(participant_id="pid-1", text="first",  fresh_match=True, pts_us=0),
            GatedQueryFrame(participant_id="pid-1", text="second", fresh_match=True, pts_us=1),
        ],
        settle_s=0.2,
        per_send_delay_s=0.05,
    )
    assert any(isinstance(f, InterruptionFrame) for f in sink.frames), (
        "handler request must push an interruption downstream"
    )


@pytest.mark.asyncio
async def test_assistant_steers_transport_target_on_participant_joined():
    """Single-participant routing default: when a assistant is constructed
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
    assistant = _StringAssistant()
    assistant._transport = transport  # type: ignore[assignment]

    await _run_chain(
        assistant,
        sends=[
            ParticipantJoinedFrame(participant_id="web-client"),
            ParticipantLeftFrame(participant_id="web-client"),
        ],
    )

    assert transport.target_calls  == ["web-client"]
    assert transport.cleanup_calls == ["web-client"]


@pytest.mark.asyncio
async def test_assistant_no_transport_steering_when_not_configured():
    """Multi-pid samples may construct the assistant without a transport;
    join/leave must then be a no-op on the transport side."""
    assistant = _StringAssistant()  # default: transport=None
    await _run_chain(
        assistant,
        sends=[ParticipantJoinedFrame(participant_id="pid-1")],
    )
    # If we got here without an AttributeError, the None-transport path
    # is exercised. No assertion needed beyond that.


@pytest.mark.asyncio
async def test_assistant_participant_left_callback_fires():
    assistant = _LifecycleAssistant()
    sink = await _run_chain(
        assistant,
        sends=[
            ParticipantJoinedFrame(participant_id="p1"),
            ParticipantLeftFrame(participant_id="p1"),
        ],
    )

    assert assistant.left   == ["p1"]
    kinds = [type(f).__name__ for f in sink.frames]
    assert "ParticipantJoinedFrame" in kinds
    assert "ParticipantLeftFrame"   in kinds


# ════════════════════════════════════════════════════════════════════════════
# StreamingTtsProcessor
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_streaming_tts_sentence_boundary_triggers_synth():
    tts  = _FakeTts(sample_rate=22050)
    gate = VoiceGate(VoiceGateConfig(), audio_sink=_NullSink(), tts=tts)
    proc = StreamingTtsProcessor(tts=tts, voice_gate=gate)

    events = []
    subscriber = "xr-ai-voice-tts-scopes"
    nemo_relay.subscribers.register(subscriber, events.append)
    try:
        sink = await _run_chain(
            proc,
            sends=[TextFrame(text="hello"), TextFrame(text=" world. ")],
        )
        await nemo_relay.subscribers.flush_async()
    finally:
        nemo_relay.subscribers.deregister(subscriber)
    assert tts.calls == ["hello world."]
    audio = [f for f in sink.frames if isinstance(f, OutputAudioRawFrame)]
    assert audio, "synth produced no audio frames downstream"
    tts_start = next(
        event.to_dict()
        for event in events
        if event.name == "voice.tts"
        and event.to_dict().get("scope_category") == "start"
    )
    assert tts_start["category"] == "function"
    assert tts_start["data"] == {"text": "hello world."}
    assert tts_start["metadata"]["participant_id"] is None


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
    # A pid-less interrupt drains every participant's state, so no pending
    # fragment survives to concatenate onto a subsequent partial sentence.
    assert proc._by_pid == {}  # noqa: SLF001


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
    # is exercised. A pid-less interrupt drains every participant's state.
    assert proc._by_pid == {}  # noqa: SLF001
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
async def test_input_transport_releases_startup_barrier_after_endpoint_starts():
    """The startup barrier releases independently of roster convergence."""
    from pipecat.frames.frames import StartFrame
    from pipecat.transports.base_transport import TransportParams
    from xr_ai_voice._transport import SAMPLE_RATE, XRMediaHubInputTransport

    endpoint = _CallbackStubEndpoint()
    started = asyncio.Event()
    transport = XRMediaHubInputTransport(
        endpoint,
        TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=SAMPLE_RATE,
            audio_in_channels=1,
        ),
        started_event=started,
    )

    try:
        start_task = asyncio.create_task(transport.start(StartFrame()))
        await endpoint.run_started.wait()
        assert not started.is_set()

        endpoint.ready_to_receive.set()
        assert await start_task is None
        assert started.is_set()
    finally:
        await transport.stop(EndFrame())

    assert not started.is_set()


@pytest.mark.asyncio
async def test_input_transport_populates_transport_source_from_chunk_pid():
    """The hub-side ``AudioChunk.participant_id`` must flow onto
    ``InputAudioRawFrame.transport_source`` — without it every
    downstream return-data / return-audio send routes to ``pid=''`` and
    the hub drops the message on the floor (production bug fixed in
    this commit)."""
    from xr_ai_hub import AudioChunk
    from xr_ai_voice._transport import (
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
    voice gate never greets and the assistant never steers the output
    transport at a participant, so every TTS chunk gets dropped by
    ``XRMediaHubOutputTransport.write_audio_frame``."""
    from xr_ai_hub import ParticipantEvent
    from xr_ai_voice._transport import (
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
    ``ParticipantLeftFrame`` so the private processor can run per-pid
    teardown (cancel in-flight, clear target participant)."""
    from xr_ai_hub import ParticipantEvent
    from xr_ai_voice._transport import (
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
    from xr_ai_hub import ParticipantEvent
    from xr_ai_voice._transport import (
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
# private pipeline assembly end-to-end smoke
# ════════════════════════════════════════════════════════════════════════════


class _EchoAssistant(_VoiceIOProcessor):
    def __init__(self) -> None:
        super().__init__(self.handle)

    async def handle(self, query: VoiceQuery) -> None:
        await self.enqueue_response(
            query.participant_id,
            f"echo {query.text}.",
            pts_us=query.timestamp_us,
        )


@pytest.mark.asyncio
async def test_private_pipeline_assembly_connects_audio_in_to_audio_out(monkeypatch):
    """End-to-end smoke: feed an InputAudioRawFrame at the head, expect
    OutputAudioRawFrame at the tail.

    Always-on voicegate config means every transcription dispatches as
    a query; the assistant echoes the text; the streaming TTS synthesizes
    a WAV; the WAV's audio frames are pushed downstream.
    """
    from xr_ai_voice._transport import HubVoiceTransport

    stt = _FakeStt(text="hi pipeline")
    tts = _FakeTts(sample_rate=22050)

    class _StubVad:
        def __init__(self, on_utterance, on_speech_start, **_):
            self._on_utt   = on_utterance
            self._on_start = on_speech_start

        async def feed(self, pcm_int16: bytes, sample_rate: int) -> None:
            await self._on_start()
            await self._on_utt(pcm_int16, sample_rate)

    monkeypatch.setattr("xr_ai_voice._processors.vad_stt.VadDetector", _StubVad)

    transport = HubVoiceTransport()
    try:
        pipeline, _task = _build_voice_pipeline(
            transport      = transport,
            stt            = stt,
            tts            = tts,
            io_processor       = _EchoAssistant(),
            vad_cfg        = VadConfig(),
            voice_gate_cfg = VoiceGateConfig(),
        )
        # Confirm the factory composed the expected wiring: Pipeline
        # body is [transport.input(), vad_stt, voice_gate, assistant,
        # streaming_tts, transport.output()]. Pipeline.processors wraps
        # that with Source/Sink at indices 0 and 7.
        kinds = [type(p).__name__ for p in pipeline.processors]
        assert kinds == [
            "PipelineSource",
            "XRMediaHubInputTransport",
            "VadSttProcessor",
            "VoiceGateProcessor",
            "_EchoAssistant",
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
    assistant           = _EchoAssistant()

    in_frame = InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=16000, num_channels=1)
    in_frame.transport_source = "web-client"
    sink = await _run_chain(
        vad_stt, voice_gate_proc, assistant, streaming_tts,
        sends=[in_frame],
        settle_s=0.6,
    )

    audio_out = [f for f in sink.frames if isinstance(f, OutputAudioRawFrame)]
    assert audio_out, "expected at least one OutputAudioRawFrame at the tail"
    assert tts.calls == ["echo hi pipeline."]


def test_private_pipeline_assembly_wires_early_wake_ack_for_chime_config():
    from xr_ai_voice._transport import HubVoiceTransport

    transport = HubVoiceTransport()
    try:
        pipeline, _worker = _build_voice_pipeline(
            transport=transport,
            stt=_FakeStt(),
            tts=_FakeTts(),
            io_processor=_EchoAssistant(),
            vad_cfg=VadConfig(),
            voice_gate_cfg=VoiceGateConfig(
                magic_phrases=("hey agent",),
                listening_chime=True,
            ),
        )
        vad_stt = next(
            processor
            for processor in pipeline.processors
            if isinstance(processor, VadSttProcessor)
        )
        assert vad_stt._on_partial_transcript is not None
    finally:
        transport.shutdown()


# ════════════════════════════════════════════════════════════════════════════
# private pipeline assembly idle-timeout knob
# ════════════════════════════════════════════════════════════════════════════


def test_private_pipeline_assembly_disables_idle_timeout_by_default():
    """Default: idle-timeout auto-cancel is OFF (pipecat's on-by-default is
    overridden), so a quiet session is never dropped for inactivity."""
    from xr_ai_voice._transport import HubVoiceTransport

    transport = HubVoiceTransport()
    try:
        _, worker = _build_voice_pipeline(
            transport      = transport,
            stt            = _FakeStt(),
            tts            = _FakeTts(),
            io_processor       = _EchoAssistant(),
            vad_cfg        = VadConfig(),
            voice_gate_cfg = VoiceGateConfig(),
        )
        assert worker._cancel_on_idle_timeout is False
        assert worker._cancel_runner_on_idle_timeout is False
        assert worker._idle_timeout_secs is None
    finally:
        transport.shutdown()


def test_private_pipeline_assembly_accepts_idle_timeout():
    """A positive idle_timeout_secs opts into pipecat's idle auto-cancel."""
    from xr_ai_voice._transport import HubVoiceTransport

    transport = HubVoiceTransport()
    try:
        _, worker = _build_voice_pipeline(
            transport      = transport,
            stt            = _FakeStt(),
            tts            = _FakeTts(),
            io_processor       = _EchoAssistant(),
            vad_cfg        = VadConfig(),
            voice_gate_cfg = VoiceGateConfig(),
            idle_timeout_secs = 300.0,
        )
        assert worker._cancel_on_idle_timeout is True
        assert worker._idle_timeout_secs == 300.0
    finally:
        transport.shutdown()


# ════════════════════════════════════════════════════════════════════════════
# Regression: assistant tags TextFrames with pid (Bug #2)
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_assistant_tags_finite_output_with_pid():
    """The assistant MUST set ``transport_destination`` on every TextFrame.

    Downstream ``StreamingTtsProcessor`` reads
    ``frame.transport_destination or ""`` and copies it onto the
    resulting ``OutputAudioRawFrame``. Without the pid tag the empty
    string flows through and the hub drops every audio chunk on the
    floor — the "agent thinks but says nothing" failure mode.
    """
    assistant = _StringAssistant()
    sink = await _run_chain(
        assistant,
        sends=[GatedQueryFrame(participant_id="web-client", text="hi", fresh_match=True, pts_us=0)],
    )

    texts = [f for f in sink.frames if isinstance(f, TextFrame)]
    assert texts, "assistant produced no TextFrame"
    assert all(t.transport_destination == "web-client" for t in texts)


@pytest.mark.asyncio
async def test_assistant_tags_incremental_output_with_pid():
    assistant = _IterAssistant(chunks=["alpha ", "beta."])
    sink = await _run_chain(
        assistant,
        sends=[GatedQueryFrame(participant_id="web-client", text="hi", fresh_match=True, pts_us=0)],
        settle_s=0.15,
    )
    texts = [f for f in sink.frames if isinstance(f, TextFrame)]
    assert len(texts) == 2
    assert all(t.transport_destination == "web-client" for t in texts)


# ════════════════════════════════════════════════════════════════════════════
# Regression: assistant emits AssistantResponseEndFrame at end of turn (Bug #4)
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_assistant_emits_response_end_after_finite_output():
    """One ``AssistantResponseEndFrame`` per completed turn carries the full
    assembled text and pid — the downstream data-channel echo keys off
    this marker."""
    assistant = _StringAssistant()
    sink = await _run_chain(
        assistant,
        sends=[GatedQueryFrame(participant_id="pid-1", text="hi", fresh_match=True, pts_us=42)],
    )
    ends = [f for f in sink.frames if isinstance(f, AssistantResponseEndFrame)]
    assert len(ends) == 1
    assert ends[0].pid    == "pid-1"
    assert ends[0].text   == "answer: hi"
    assert ends[0].pts_us == 42


@pytest.mark.asyncio
async def test_assistant_emits_response_end_after_streamed_output():
    assistant = _IterAssistant(chunks=["one ", "two ", "three."])
    sink = await _run_chain(
        assistant,
        sends=[GatedQueryFrame(participant_id="pid-1", text="q", fresh_match=True, pts_us=7)],
        settle_s=0.2,
    )
    ends = [f for f in sink.frames if isinstance(f, AssistantResponseEndFrame)]
    assert len(ends) == 1
    assert ends[0].text == "one two three."
    assert ends[0].pid  == "pid-1"


@pytest.mark.asyncio
async def test_input_only_handler_does_not_emit_an_empty_agent_response():
    assistant = _InputOnlyAssistant()
    sink = await _run_chain(
        assistant,
        sends=[
            GatedQueryFrame(
                participant_id="pid-1",
                text="publish this",
                fresh_match=True,
                pts_us=7,
            )
        ],
    )

    assert [query.text for query in assistant.queries] == ["publish this"]
    assert not any(isinstance(frame, TextFrame) for frame in sink.frames)
    assert not any(isinstance(frame, AssistantResponseEndFrame) for frame in sink.frames)


@pytest.mark.asyncio
async def test_assistant_does_not_emit_response_end_on_cancel():
    """Cancellation (new query or InterruptionFrame) supersedes the
    in-flight turn — the data-channel echo would surface a partial
    answer that contradicts the new turn, so the assistant skips the end
    marker. The second turn still emits its own end marker normally."""
    assistant = _IterAssistant(chunks=[f"chunk{i} " for i in range(200)])
    sink = await _run_chain(
        assistant,
        sends=[
            GatedQueryFrame(participant_id="pid-1", text="first",  fresh_match=True, pts_us=0),
            GatedQueryFrame(participant_id="pid-1", text="second", fresh_match=True, pts_us=1),
        ],
        settle_s=0.4,
        per_send_delay_s=0.05,
    )
    ends = [f for f in sink.frames if isinstance(f, AssistantResponseEndFrame)]
    # Only the second turn's end marker survives — the first was cancelled.
    assert len(ends) == 1
    assert ends[0].pts_us == 1


# ════════════════════════════════════════════════════════════════════════════
# Regression: handler async method returning an async iterator works
# ════════════════════════════════════════════════════════════════════════════


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
            AssistantResponseEndFrame(pid="web-client", text="hello there", pts_us=99),
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
    """Samples whose assistant pushes its own per-turn data echo (e.g.
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
        sends=[AssistantResponseEndFrame(pid="pid-1", text="hi", pts_us=0)],
    )
    assert transport.sends == []


@pytest.mark.asyncio
async def test_streaming_tts_flushes_trailing_text_on_response_end():
    """The assistant may finish a turn with text that has no sentence-final
    punctuation (e.g. partial answer). End-of-response is the last
    chance to flush the buffer; otherwise the tail of the reply is
    silently dropped."""
    tts  = _FakeTts()
    gate = VoiceGate(VoiceGateConfig(), audio_sink=_NullSink(), tts=tts)
    proc = StreamingTtsProcessor(tts=tts, voice_gate=gate)

    text_frame = TextFrame(text="trailing fragment with no period")
    # The handler tags every TextFrame with the addressed pid; the end frame
    # carries the same pid, so per-participant flush finds the pending buffer.
    text_frame.transport_destination = "pid-1"
    sink = await _run_chain(
        proc,
        sends=[
            text_frame,
            AssistantResponseEndFrame(pid="pid-1", text="trailing fragment with no period", pts_us=0),
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
    # Both sentences dispatched (nothing left in pending), and the pipeline
    # EndFrame tore down all per-participant state.
    assert proc._by_pid == {}  # noqa: SLF001


def _text_for(text: str, pid: str) -> TextFrame:
    f = TextFrame(text=text)
    f.transport_destination = pid
    return f


def _interrupt_for(pid: str) -> InterruptionFrame:
    f = InterruptionFrame()
    f.transport_source = pid
    return f


@pytest.mark.asyncio
async def test_streaming_tts_keeps_participants_separate():
    """Interleaved token streams for two participants must not share a pending
    buffer (which would glue their words into one sentence) or a sender (which
    would misroute audio). Each participant's sentence is synthesized on its own
    and its audio is addressed only to that participant."""
    tts  = _FakeTts()
    gate = VoiceGate(VoiceGateConfig(), audio_sink=_NullSink(), tts=tts)
    proc = StreamingTtsProcessor(tts=tts, voice_gate=gate)

    sink = await _run_chain(
        proc,
        sends=[
            _text_for("Alice", "alice"),   # partial — no sentence end yet
            _text_for("Bob", "bob"),       # interleaves before Alice completes
            _text_for(". ", "alice"),      # completes Alice's sentence
            _text_for(". ", "bob"),        # completes Bob's sentence
        ],
        settle_s=0.2,
    )
    # Per-participant pending: each sentence is synthesized on its own, never
    # concatenated across pids into "AliceBob.".
    assert sorted(tts.calls) == ["Alice.", "Bob."]
    # Each participant's audio is addressed only to that participant.
    dests = {
        f.transport_destination
        for f in sink.frames
        if isinstance(f, OutputAudioRawFrame)
    }
    assert dests == {"alice", "bob"}


@pytest.mark.asyncio
async def test_streaming_tts_interrupt_is_participant_scoped():
    """An interrupt for one participant cancels only their in-flight TTS and
    flushes only their hub audio — the other participant's stream is untouched."""
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
    tts.delay_s = 0.2  # so both syntheses are in flight when the interrupt lands
    gate = VoiceGate(VoiceGateConfig(), audio_sink=_NullSink(), tts=tts)
    transport = _StubTransport()
    proc = StreamingTtsProcessor(tts=tts, voice_gate=gate, transport=transport)

    sink = await _run_chain(
        proc,
        sends=[
            _text_for("Alice speaking. ", "alice"),
            _text_for("Bob speaking. ", "bob"),
            _interrupt_for("alice"),
        ],
        settle_s=0.4,
        per_send_delay_s=0.03,
    )
    # Only Alice's hub audio is flushed; Bob is not touched.
    assert transport.endpoint.flush_calls == ["alice"]
    audio = [f for f in sink.frames if isinstance(f, OutputAudioRawFrame)]
    # Alice's in-flight audio was cancelled; Bob's still reaches Bob.
    assert [f.transport_destination for f in audio if f.transport_destination == "alice"] == []
    assert any(f.transport_destination == "bob" for f in audio), (
        "bob's audio must survive an alice-scoped interrupt"
    )


@pytest.mark.asyncio
async def test_streaming_tts_releases_state_on_participant_left():
    """A departing participant's synthesis state is released, and no audio for
    that pid is emitted afterwards.

    This matters for the output transport: it drops the pid's media sender when
    the same frame reaches it, so a lingering synth task emitting audio later
    would make its lazy routing recreate the sender just released.
    """
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
    tts.delay_s = 0.2  # synthesis still in flight when the participant leaves
    gate = VoiceGate(VoiceGateConfig(), audio_sink=_NullSink(), tts=tts)
    transport = _StubTransport()
    proc = StreamingTtsProcessor(tts=tts, voice_gate=gate, transport=transport)

    sink = await _run_chain(
        proc,
        sends=[
            _text_for("Alice speaking. ", "alice"),
            _text_for("Bob speaking. ", "bob"),
            ParticipantLeftFrame(participant_id="alice"),
        ],
        settle_s=0.4,
        per_send_delay_s=0.03,
    )

    # Alice's state is gone; Bob's stream is untouched.
    assert "alice" not in proc._by_pid  # noqa: SLF001
    audio = [f for f in sink.frames if isinstance(f, OutputAudioRawFrame)]
    assert [f for f in audio if f.transport_destination == "alice"] == [], (
        "no audio may be emitted for a departed participant"
    )
    assert any(f.transport_destination == "bob" for f in audio)
    # Leaving is not an interruption — there is no live client left to flush.
    assert transport.endpoint.flush_calls == []
    # The frame still reaches the rest of the pipeline.
    assert any(isinstance(f, ParticipantLeftFrame) for f in sink.frames)


@pytest.mark.asyncio
async def test_streaming_tts_pidless_interrupt_flushes_every_active_participant():
    """A pid-less interruption drains every participant, so it must flush every
    participant's hub audio too — flushing only the fallback target would leave
    the others' already-paced audio playing out."""
    class _StubEndpoint:
        def __init__(self) -> None:
            self.flush_calls: list[str] = []

        async def flush_return_audio(self, pid: str) -> None:
            self.flush_calls.append(pid)

    class _StubTransport:
        def __init__(self) -> None:
            self.endpoint           = _StubEndpoint()
            self.target_participant = "alice"

    tts  = _FakeTts()
    tts.delay_s = 0.2
    gate = VoiceGate(VoiceGateConfig(), audio_sink=_NullSink(), tts=tts)
    transport = _StubTransport()
    proc = StreamingTtsProcessor(tts=tts, voice_gate=gate, transport=transport)

    await _run_chain(
        proc,
        sends=[
            _text_for("Alice speaking. ", "alice"),
            _text_for("Bob speaking. ", "bob"),
            InterruptionFrame(),  # no transport_source
        ],
        settle_s=0.4,
        per_send_delay_s=0.03,
    )

    assert set(transport.endpoint.flush_calls) == {"alice", "bob"}


# ════════════════════════════════════════════════════════════════════════════
# Regression: output transport rewrites destination to default sender (Bug #1)
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_output_transport_handle_frame_routes_pid_to_default_sender(monkeypatch):
    """Pipecat's ``BaseOutputTransport._handle_frame`` drops frames whose
    ``transport_destination`` is not a registered key in
    ``_media_senders``. By default only ``None`` is registered, so a
    frame tagged with a pid (the way the assistant / TTS / chime tag every
    outbound frame) would be silently dropped — the audio bug. The
    override rewrites the destination to ``None`` before delegating to
    the base class so the default sender picks it up; the hub layer
    routes by ``_target_participant``.
    """
    from xr_ai_voice._transport import XRMediaHubOutputTransport
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
    from xr_ai_hub import AudioChunk
    from xr_ai_voice._transport import (
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
async def test_output_transport_releases_media_sender_on_participant_left():
    """A departing participant's per-pid ``MediaSender`` is torn down so a
    long-lived hub with join/leave churn does not retain idle senders until
    pipeline shutdown."""
    from xr_ai_voice._transport import XRMediaHubOutputTransport
    from pipecat.transports.base_transport import TransportParams

    class _StubEndpoint:
        async def send_return_audio(self, *_a, **_kw) -> None:
            return

    class _StubSender:
        def __init__(self) -> None:
            self.cancelled = False

        async def cancel(self, _frame) -> None:
            self.cancelled = True

    transport = XRMediaHubOutputTransport(_StubEndpoint(), TransportParams())
    sender = _StubSender()
    transport._media_senders["alice"] = sender      # noqa: SLF001
    transport.set_target_participant("alice")

    await transport._release_destination("alice")    # noqa: SLF001

    assert sender.cancelled, "the departing participant's sender must be cancelled"
    assert "alice" not in transport._media_senders   # noqa: SLF001
    # The fallback target is cleared when the bound target participant leaves.
    assert transport.target_participant == ""


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
    from xr_ai_hub import AudioChunk
    from xr_ai_voice._transport import (
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
    # Simulate the pre-fix steering: assistant set the room-wide target to the
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
    from xr_ai_voice._transport import XRMediaHubOutputTransport
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

    GatedQueryFrame → assistant (TextFrame tagged with pid) → StreamingTts
    (OutputAudioRawFrame tagged with pid) → output transport
    (write_raw_audio_frames invoked → send_return_audio).

    This is the path that was previously silently dropping every audio
    frame at the output transport's media-sender router. The assertion
    is on ``send_return_audio`` reaching the hub, not just on audio
    frames appearing at the tail — that's the bug we shipped the fix
    for.
    """
    from xr_ai_hub import AudioChunk
    from xr_ai_voice._transport import (
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
    assistant = _StringAssistant()
    streaming_tts = StreamingTtsProcessor(tts=tts, voice_gate=gate)

    await _run_chain(
        assistant, streaming_tts, output,
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
async def test_private_pipeline_assembly_routes_text_through_streaming_tts(monkeypatch):
    """The factory's ``text_topic`` argument must reach the streaming
    TTS processor's data-echo path. Previously the parameter was
    explicitly unused (``# noqa: ARG001``) and the echo silently never
    fired."""
    from xr_ai_voice._transport import HubVoiceTransport

    transport = HubVoiceTransport()
    try:
        pipeline, _task = _build_voice_pipeline(
            transport      = transport,
            stt            = _FakeStt(),
            tts            = _FakeTts(),
            io_processor       = _StringAssistant(),
            vad_cfg        = VadConfig(),
            voice_gate_cfg = VoiceGateConfig(),
            text_topic     = "vlm.response",
        )
        # The streaming-tts processor lives at index 5 in the wrapped
        # pipeline (source, input, vad_stt, voice_gate, assistant, tts, output, sink).
        streaming_tts = pipeline.processors[5]
        assert isinstance(streaming_tts, StreamingTtsProcessor)
        assert streaming_tts._text_topic == "vlm.response"
        assert streaming_tts._transport is transport
    finally:
        transport.shutdown()


# ════════════════════════════════════════════════════════════════════════════
# Regression: transport subscribes to VIDEO so assistant receives FrameSignals
# ════════════════════════════════════════════════════════════════════════════


def test_xr_media_hub_transport_subscribes_to_video_frames():
    """The ProcessorEndpoint must subscribe to the video category so
    that camera FrameSignals reach assistant consumers (e.g. VLM workers
    that bind ``ep.on_frame``). A previous version of the transport
    filtered out video at the ZMQ subscription layer, causing
    ``_wait_for_camera_frame`` to time out and the VLM call to block
    indefinitely after a query."""
    from xr_ai_hub._processor import Subscribe
    from xr_ai_voice._transport import HubVoiceTransport

    transport = HubVoiceTransport()
    try:
        assert transport._ep._default_filter & Subscribe.VIDEO, (
            "transport must subscribe to VIDEO frames so assistants receive "
            "camera FrameSignals"
        )
        # Audio and data are also required for the voice pipeline.
        assert transport._ep._default_filter & Subscribe.AUDIO
        assert transport._ep._default_filter & Subscribe.DATA
    finally:
        transport.shutdown()
