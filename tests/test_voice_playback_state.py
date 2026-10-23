# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speaking state reaches ``_VoiceIOProcessor`` from real TTS output.

Pipecat's output transport raises bot started/stopped speaking only for TTS
audio frames, so these tests run the real ``StreamingTtsProcessor`` into the
real ``DeviceIOHubOutputTransport`` (fake TTS service, fake hub endpoint)
behind a real ``_VoiceIOProcessor``. Injecting the speaking frames directly
would pass even when the production pipeline never emits them.
"""
from __future__ import annotations

import asyncio
import io
import wave
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import pytest
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InterruptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports import base_output
from pipecat.transports.base_transport import TransportParams
from pipecat.workers.runner import WorkerRunner
from xr_ai_voice._frames import AssistantResponseEndFrame
from xr_ai_voice._processors import StreamingTtsProcessor, VoiceGateProcessor, _VoiceIOProcessor
from xr_ai_voice._transport import TTS_NATIVE_SAMPLE_RATE, DeviceIOHubOutputTransport
from xr_ai_voice._types import VoiceQuery
from xr_ai_voicegate import VoiceGateConfig

_PID = "alice"
_TIMEOUT_S = 5.0


def _silence_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TTS_NATIVE_SAMPLE_RATE)
        wf.writeframes(bytes(2 * TTS_NATIVE_SAMPLE_RATE // 5))
    return buf.getvalue()


class _FakeTts:
    """Return 200 ms of silence for every sentence."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def synthesize(self, text: str, **_kwargs: object) -> bytes:
        self.calls.append(text)
        return _silence_wav()


class _GatedEndpoint:
    """Hub endpoint whose return-audio send blocks until ``release`` is set.

    Holding the send keeps the reply's audio "playing" for as long as a test
    needs, without depending on wall-clock pacing.
    """

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.first_chunk = asyncio.Event()

    async def send_return_audio(self, chunk: object) -> None:
        self.first_chunk.set()
        await self.release.wait()


async def _ignore_query(_query: VoiceQuery) -> None:
    return None


async def _until(predicate: Callable[[], bool]) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), _TIMEOUT_S)


async def _one_reply(text: str) -> AsyncIterator[str]:
    yield text


@asynccontextmanager
async def _pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fallback_s: float = 0.1,
) -> AsyncIterator[tuple[_VoiceIOProcessor, PipelineWorker, _FakeTts, _GatedEndpoint, VoiceGateProcessor]]:
    # Pipecat reads both timeouts when the audio task starts.
    monkeypatch.setattr(base_output, "BOT_VAD_STOP_SECS", 0.05)
    monkeypatch.setattr(base_output, "BOT_VAD_STOP_FALLBACK_SECS", fallback_s)
    endpoint = _GatedEndpoint()
    tts = _FakeTts()
    gate_processor = VoiceGateProcessor(cfg=VoiceGateConfig(), tts=tts)
    io_processor = _VoiceIOProcessor(_ignore_query)
    output = DeviceIOHubOutputTransport(
        endpoint,
        TransportParams(
            audio_out_enabled=True,
            audio_out_sample_rate=TTS_NATIVE_SAMPLE_RATE,
            audio_out_channels=1,
        ),
    )
    worker = PipelineWorker(
        Pipeline([
            io_processor,
            gate_processor,
            StreamingTtsProcessor(tts=tts, voice_gate=gate_processor.gate),
            output,
        ]),
        cancel_on_idle_timeout=False,
        enable_rtvi=False,
    )
    started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def on_pipeline_started(_worker: PipelineWorker, _frame: object) -> None:
        started.set()

    runner = WorkerRunner()
    await runner.add_workers(worker)
    run = asyncio.create_task(runner.run())
    try:
        await asyncio.wait_for(started.wait(), _TIMEOUT_S)
        yield io_processor, worker, tts, endpoint, gate_processor
    finally:
        endpoint.release.set()
        await worker.queue_frame(EndFrame())
        await asyncio.wait_for(run, _TIMEOUT_S)


@pytest.mark.asyncio
async def test_reply_audio_raises_speaking_then_silence(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _pipeline(monkeypatch) as (io_processor, _worker, tts, endpoint, _gate):
        await io_processor.enqueue_reply(_PID, _one_reply("Hello there."), request_id="r1")

        await _until(lambda: _PID in io_processor._speaking)  # noqa: SLF001
        assert tts.calls == ["Hello there."]

        endpoint.release.set()
        await _until(lambda: _PID not in io_processor._speaking)  # noqa: SLF001


@pytest.mark.asyncio
async def test_update_waits_for_reply_audio_to_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _pipeline(monkeypatch) as (io_processor, _worker, tts, endpoint, _gate):
        await io_processor.enqueue_reply(_PID, _one_reply("Hello there."), request_id="r1")
        await asyncio.wait_for(endpoint.first_chunk.wait(), _TIMEOUT_S)
        await _until(lambda: _PID in io_processor._speaking)  # noqa: SLF001
        await _until(lambda: not io_processor._inflight)  # noqa: SLF001

        await io_processor.enqueue_update(_PID, "Timer done.", source="timer")
        await asyncio.sleep(0.2)
        assert tts.calls == ["Hello there."]
        assert not io_processor._inflight  # noqa: SLF001

        endpoint.release.set()
        await _until(lambda: "Timer done." in tts.calls)


@pytest.mark.asyncio
async def test_interruption_clears_speaking(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _pipeline(monkeypatch) as (io_processor, worker, _tts, endpoint, _gate):
        await io_processor.enqueue_reply(_PID, _one_reply("Hello there."), request_id="r1")
        await _until(lambda: _PID in io_processor._speaking)  # noqa: SLF001

        interruption = InterruptionFrame()
        interruption.transport_source = _PID
        await worker.queue_frame(interruption)
        endpoint.release.set()

        await _until(lambda: _PID not in io_processor._speaking)  # noqa: SLF001


@pytest.mark.asyncio
async def test_idle_chime_clears_speaking_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _pipeline(monkeypatch, fallback_s=30.0) as (io_processor, _worker, tts, endpoint, gate):
        await gate.play_wav(_PID, _silence_wav())
        await _until(lambda: _PID in io_processor._speaking)  # noqa: SLF001

        await io_processor.enqueue_update(_PID, "Timer done.", source="timer")
        await asyncio.sleep(0.1)
        assert tts.calls == []

        endpoint.release.set()
        await _until(lambda: _PID not in io_processor._speaking)  # noqa: SLF001
        await _until(lambda: tts.calls == ["Timer done."])


@pytest.mark.asyncio
async def test_new_request_cancels_reply_audio_still_being_synthesized(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _pipeline(monkeypatch) as (io_processor, _worker, tts, endpoint, _gate):
        synthesis = asyncio.Event()
        cancelled: list[str] = []
        synthesize = tts.synthesize

        async def slow_first(text: str, **kwargs: object) -> bytes:
            if text == "Old answer.":
                try:
                    await synthesis.wait()
                except asyncio.CancelledError:
                    cancelled.append(text)
                    raise
            return await synthesize(text, **kwargs)

        tts.synthesize = slow_first  # type: ignore[method-assign]
        await io_processor.enqueue_reply(_PID, _one_reply("Old answer."), request_id="r1")
        await _until(lambda: not io_processor._inflight and tts.calls == [])  # noqa: SLF001
        await asyncio.sleep(0.05)
        assert _PID not in io_processor._speaking  # noqa: SLF001

        await io_processor.enqueue_reply(_PID, _one_reply("New answer."), request_id="r2")
        endpoint.release.set()

        await _until(lambda: cancelled == ["Old answer."])
        await _until(lambda: tts.calls == ["New answer."])


@pytest.mark.asyncio
async def test_new_request_cancels_reply_synthesis_after_a_playback_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _pipeline(monkeypatch) as (io_processor, _worker, tts, endpoint, _gate):
        endpoint.release.set()
        second = asyncio.Event()
        cancelled: list[str] = []
        synthesize = tts.synthesize

        async def slow_second(text: str, **kwargs: object) -> bytes:
            if text == "Old two.":
                try:
                    await second.wait()
                except asyncio.CancelledError:
                    cancelled.append(text)
                    raise
            return await synthesize(text, **kwargs)

        tts.synthesize = slow_second  # type: ignore[method-assign]
        await io_processor.enqueue_reply(_PID, _one_reply("Old one. Old two."), request_id="r1")
        await _until(lambda: _PID in io_processor._speaking)  # noqa: SLF001
        # The silence fallback stops speaking while the second sentence synthesizes.
        await _until(lambda: _PID not in io_processor._speaking)  # noqa: SLF001

        await io_processor.enqueue_reply(_PID, _one_reply("New answer."), request_id="r2")

        await _until(lambda: cancelled == ["Old two."])
        await _until(lambda: tts.calls == ["Old one.", "New answer."])


@pytest.mark.asyncio
async def test_queued_reply_does_not_let_an_urgent_update_cut_a_playing_one(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _pipeline(monkeypatch) as (io_processor, _worker, tts, endpoint, _gate):
        interruptions: list[str | None] = []
        push_frame = io_processor.push_frame

        async def record(frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
            if isinstance(frame, InterruptionFrame):
                interruptions.append(frame.transport_source)
            await push_frame(frame, direction)

        io_processor.push_frame = record  # type: ignore[method-assign]
        await io_processor.enqueue_update(_PID, "Fire alarm.", source="alarm", urgent=True)
        await _until(lambda: _PID in io_processor._speaking)  # noqa: SLF001
        interruptions.clear()

        await io_processor.enqueue_reply(_PID, _one_reply("Answer."), request_id="r1")
        await io_processor.enqueue_update(_PID, "Leave now.", source="alarm", urgent=True)
        await asyncio.sleep(0.1)
        assert tts.calls == ["Fire alarm."]

        endpoint.release.set()
        await _until(lambda: tts.calls == ["Fire alarm.", "Leave now.", "Answer."])
        assert interruptions == []


@pytest.mark.asyncio
async def test_another_participants_interruption_keeps_the_done_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _pipeline(monkeypatch) as (io_processor, worker, tts, endpoint, gate):
        endpoint.release.set()
        held = asyncio.Event()
        interrupted = asyncio.Event()
        process_frame = gate.process_frame

        # Hold Bob's response end in the gate until Alice's interruption has
        # passed it, so Bob's done marker waits in the gate's queue meanwhile.
        async def hold_bob(frame: Frame, direction: FrameDirection) -> None:
            if isinstance(frame, AssistantResponseEndFrame) and frame.pid == "bob":
                held.set()
                await interrupted.wait()
            await process_frame(frame, direction)
            if isinstance(frame, InterruptionFrame):
                interrupted.set()

        gate.process_frame = hold_bob  # type: ignore[method-assign]
        await io_processor.enqueue_reply("bob", _one_reply("One."), request_id="r1")
        await asyncio.wait_for(held.wait(), _TIMEOUT_S)
        await io_processor.enqueue_reply("bob", _one_reply("Two."), request_id="r1")

        interruption = InterruptionFrame()
        interruption.transport_source = _PID
        await worker.queue_frame(interruption)

        await _until(lambda: tts.calls == ["One.", "Two."])


@pytest.mark.asyncio
async def test_updates_wait_for_the_participant_after_a_barge_in(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _pipeline(monkeypatch) as (io_processor, worker, tts, endpoint, _gate):
        await io_processor.enqueue_reply(_PID, _one_reply("Hello there."), request_id="r1")
        await _until(lambda: _PID in io_processor._speaking)  # noqa: SLF001
        await io_processor.enqueue_update(_PID, "Timer done.", source="timer")

        started = UserStartedSpeakingFrame()
        started.transport_source = _PID
        await worker.queue_frame(started)
        interruption = InterruptionFrame()
        interruption.transport_source = _PID
        await worker.queue_frame(interruption)
        endpoint.release.set()
        # The interruption stops the bot, but the participant is still talking.
        await _until(lambda: _PID not in io_processor._speaking)  # noqa: SLF001
        await asyncio.sleep(0.2)
        assert tts.calls == ["Hello there."]

        stopped = UserStoppedSpeakingFrame()
        stopped.transport_source = _PID
        await worker.queue_frame(stopped)
        await _until(lambda: tts.calls == ["Hello there.", "Timer done."])
