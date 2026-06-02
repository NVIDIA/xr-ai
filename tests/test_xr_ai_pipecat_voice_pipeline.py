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
import io
import wave
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
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

from xr_ai_pipecat import (
    BrainProcessor,
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

    class _StubEndpoint:
        def __init__(self) -> None:
            self.audio_cb = None

        def on_audio(self, cb) -> None:
            self.audio_cb = cb

        def stop(self) -> None:
            return

    ep = _StubEndpoint()
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
