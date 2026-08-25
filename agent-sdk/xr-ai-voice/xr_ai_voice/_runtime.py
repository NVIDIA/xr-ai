# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bidirectional agent-runtime boundary for participant-aware voice."""

from __future__ import annotations

import asyncio
from builtins import BaseExceptionGroup
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nemo_relay
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator
from xr_ai_hub import DataMessage
from xr_ai_models import STTService, TTSService
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, Topic, subscribe
from xr_ai_voicegate import VoiceGateConfig

from ._processors import VadConfig
from ._session import _VoiceSession
from ._transport import HubVoiceTransport
from ._types import VoiceQuery

_OPEN_STREAM_CAPACITY = 1024
_CLOSED_STREAM_CAPACITY = 1024
_TRANSCRIPT_CAPACITY = 32


class VoiceStreamClosedError(ValueError):
    """Raised when output targets a voice stream the consumer already closed."""


class UserQuery(BaseModel):
    """One accepted user query emitted by the voice input boundary."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    """Gate-accepted speech or direct typed text."""

    timestamp_us: int = Field(ge=0)
    """Input presentation time as Unix microseconds."""


class VoiceTranscript(BaseModel):
    """One final STT result emitted before voice-gate filtering."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    """Final text returned by STT before voice-gate filtering."""

    timestamp_us: int = Field(ge=0)
    """Speech presentation time as Unix microseconds."""


class VoiceParticipantJoined(BaseModel):
    """Notification that one participant joined the voice transport."""

    model_config = ConfigDict(extra="forbid")


class VoiceParticipantLeft(BaseModel):
    """Notification that one participant left the voice transport."""

    model_config = ConfigDict(extra="forbid")


class VoiceInterrupted(BaseModel):
    """Notification that participant-scoped or global voice work was interrupted."""

    model_config = ConfigDict(extra="forbid")


class VoiceOutput(BaseModel):
    """One complete response or one chunk of an incremental voice response."""

    model_config = ConfigDict(extra="forbid")

    text: str = ""
    """Complete response text or one incremental fragment."""

    response_id: str | None = Field(default=None, min_length=1)
    """Stable identifier shared by chunks of an incremental response."""

    final: bool = True
    """Whether this message completes the response."""

    interrupt: bool = False
    """Whether the first response message supersedes active voice output."""

    timestamp_us: int | None = Field(default=None, ge=0)
    """Optional originating input timestamp propagated to TTS."""

    @model_validator(mode="after")
    def validate_boundary(self) -> VoiceOutput:
        """Require identity for incremental output and text for finite output."""

        if not self.final and self.response_id is None:
            raise ValueError("non-final voice output needs a response_id")
        if self.response_id is None and not self.text.strip():
            raise ValueError("complete voice output must contain text")
        if self.interrupt and self.response_id is not None and not self.text.strip():
            raise ValueError("an empty stream terminator cannot interrupt output")
        return self


VOICE_TRANSCRIPT_TOPIC = Topic("voice.transcript", VoiceTranscript)
"""Non-empty final STT results queued for publication before gate filtering.

Delivery is best effort: the bounded queue drops its oldest pending item on
overflow, and shutdown discards queued items that have not been published.
"""

VOICE_OUTPUT_TOPIC = Topic("voice.output", VoiceOutput, telemetry="none")
"""Finite or incremental responses consumed by :class:`VoiceAgent`."""


@dataclass(slots=True)
class _ResponseTrace:
    participant_id: str
    source: str
    response_id: str
    correlation_id: str
    timestamp_us: int
    interrupt: bool
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    fragments: list[str] = field(default_factory=list)


class _ResponseStream(AsyncIterator[str]):
    def __init__(
        self,
        capacity: int,
        on_close: Callable[[_ResponseStream], None],
    ) -> None:
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=capacity)
        self.closed = asyncio.Event()
        self._on_close = on_close

    async def send(self, text: str) -> None:
        if not text or self.closed.is_set():
            return
        queued = asyncio.create_task(self.queue.put(text))
        closed = asyncio.create_task(self.closed.wait())
        try:
            done, _ = await asyncio.wait(
                (queued, closed),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if closed in done and not queued.done():
                queued.cancel()
        finally:
            if not queued.done():
                queued.cancel()
            if not closed.done():
                closed.cancel()
            await asyncio.gather(queued, closed, return_exceptions=True)

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        if not self.queue.empty():
            return self.queue.get_nowait()
        if self.closed.is_set():
            raise StopAsyncIteration
        queued = asyncio.create_task(self.queue.get())
        closed = asyncio.create_task(self.closed.wait())
        try:
            done, _ = await asyncio.wait(
                (queued, closed),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if queued in done:
                return queued.result()
            raise StopAsyncIteration
        except asyncio.CancelledError:
            await self.aclose()
            raise
        finally:
            if not queued.done():
                queued.cancel()
            if not closed.done():
                closed.cancel()
            await asyncio.gather(queued, closed, return_exceptions=True)

    async def aclose(self) -> None:
        """Release blocked producers and evict the stream from its owner."""

        if self.closed.is_set():
            return
        self.closed.set()
        self._on_close(self)


class VoiceAgent(Agent):
    """Own voice media lifecycle and bridge runtime input and output topics."""

    def __init__(
        self,
        *,
        query_topic: Topic[UserQuery],
        stt: STTService,
        tts: TTSService,
        vad: VadConfig,
        voice_gate: VoiceGateConfig,
        probes: Mapping[str, Callable[[], Awaitable[bool]]] | None = None,
        ready_file: Path | None = None,
        closeables: Iterable[Any] = (),
        text_topic: str = "agent.response",
        idle_timeout_secs: float | None = None,
        transport: HubVoiceTransport | None = None,
        response_capacity: int = 32,
        text_input: bool = True,
        participant_joined_topic: Topic[VoiceParticipantJoined] | None = None,
        participant_left_topic: Topic[VoiceParticipantLeft] | None = None,
        interrupted_topic: Topic[VoiceInterrupted] | None = None,
        interrupt_on_supersede: bool = False,
    ) -> None:
        if response_capacity <= 0:
            raise ValueError("voice response capacity must be positive")
        super().__init__()
        self._session = _VoiceSession(
            stt=stt,
            tts=tts,
            vad=vad,
            voice_gate=voice_gate,
            probes=probes,
            ready_file=ready_file,
            closeables=closeables,
            text_topic=text_topic,
            idle_timeout_secs=idle_timeout_secs,
            transport=transport,
        )
        self.query_topic = query_topic
        self.response_capacity = response_capacity
        self.text_input = text_input
        self.participant_joined_topic = participant_joined_topic
        self.participant_left_topic = participant_left_topic
        self.interrupted_topic = interrupted_topic
        self.interrupt_on_supersede = interrupt_on_supersede
        self._runtime: AgentRuntime | None = None
        self._source = "voice"
        self._output_lock = asyncio.Lock()
        self._streams: dict[tuple[str, str, str], _ResponseStream] = {}
        self._response_traces: dict[tuple[str, str, str], _ResponseTrace] = {}
        self._closed_streams: dict[tuple[str, str, str], None] = {}
        self._lifecycle_tasks: set[asyncio.Task[None]] = set()
        self._participant_lifecycle_tails: dict[str, asyncio.Task[None]] = {}
        self._transcript_queue: asyncio.Queue[
            tuple[str, VoiceTranscript]
        ] | None = None
        self._transcript_task: asyncio.Task[None] | None = None

    async def run(self, runtime: AgentRuntime, *, source: str = "voice") -> None:
        """Run the owned voice session and bridge it to a running runtime."""

        if self._runtime is not None:
            raise RuntimeError("voice agent is already running")
        if not runtime.running:
            raise RuntimeError("agent runtime must be running")
        if not source.strip():
            raise ValueError("voice agent source must not be empty")
        self._runtime = runtime
        self._source = source
        unsubscribe_data: Callable[[], None] | None = None
        try:
            await self._session.__aenter__()
            self._transcript_queue = asyncio.Queue(maxsize=_TRANSCRIPT_CAPACITY)
            self._transcript_task = asyncio.create_task(
                self._deliver_transcripts(runtime),
                name="voice-transcript-delivery",
            )
            try:
                if self.text_input:
                    unsubscribe_data = self._session.endpoint.on_data(self._on_data)
                await self._session.run(
                    self._publish_input,
                    on_transcript=self._publish_transcript,
                    on_participant_joined=self._participant_joined,
                    on_participant_left=self._participant_left,
                    on_interrupted=(
                        self._publish_interrupted
                        if self.interrupted_topic is not None
                        else None
                    ),
                    interrupt_on_supersede=self.interrupt_on_supersede,
                )
            finally:
                if unsubscribe_data is not None:
                    unsubscribe_data()
                await self._stop_transcript_delivery()
                tasks = tuple(self._lifecycle_tasks)
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                self._participant_lifecycle_tails.clear()
                await asyncio.gather(
                    *(stream.aclose() for stream in tuple(self._streams.values()))
                )
                self._closed_streams.clear()
                await self._session.close()
        finally:
            self._runtime = None
            self._source = "voice"

    @subscribe(VOICE_OUTPUT_TOPIC)
    async def output(self, output: VoiceOutput, ctx: RuntimeContext) -> None:
        """Send one voice message using participant and producer metadata."""

        metadata = ctx.metadata
        participant_id = metadata.participant_id
        if participant_id is None:
            raise ValueError("voice output requires a participant")
        timestamp_us = (
            output.timestamp_us
            if output.timestamp_us is not None
            else metadata.timestamp_us
        )
        if output.response_id is None:
            async with self._output_lock:
                with self._response_scope(
                    participant_id=participant_id,
                    source=metadata.source,
                    response_id=None,
                    correlation_id=metadata.correlation_id,
                    text=output.text,
                    fragment_count=1,
                    interrupt=output.interrupt,
                    timestamp_us=timestamp_us,
                    streaming=False,
                    status="completed",
                ):
                    await self._session.enqueue_response(
                        participant_id,
                        output.text,
                        interrupt=output.interrupt,
                        pts_us=timestamp_us,
                    )
            return

        key = (participant_id, metadata.source, output.response_id)
        stream: _ResponseStream
        async with self._output_lock:
            if key in self._closed_streams:
                return
            existing = self._streams.get(key)
            if existing is None:
                if output.final:
                    if not output.text.strip():
                        raise VoiceStreamClosedError("voice stream terminator has no open response")
                    with self._response_scope(
                        participant_id=participant_id,
                        source=metadata.source,
                        response_id=output.response_id,
                        correlation_id=metadata.correlation_id,
                        text=output.text,
                        fragment_count=1,
                        interrupt=output.interrupt,
                        timestamp_us=timestamp_us,
                        streaming=False,
                        status="completed",
                    ):
                        await self._session.enqueue_response(
                            participant_id,
                            output.text,
                            interrupt=output.interrupt,
                            pts_us=timestamp_us,
                        )
                    return
                if len(self._streams) >= _OPEN_STREAM_CAPACITY:
                    oldest_key = next(iter(self._streams))
                    await self._streams[oldest_key].aclose()
                stream = _ResponseStream(
                    self.response_capacity,
                    lambda closed: self._discard_stream(key, closed),
                )
                trace = _ResponseTrace(
                    participant_id=participant_id,
                    source=metadata.source,
                    response_id=output.response_id,
                    correlation_id=metadata.correlation_id,
                    timestamp_us=timestamp_us,
                    interrupt=output.interrupt,
                )
                await self._session.enqueue_response(
                    participant_id,
                    stream,
                    interrupt=output.interrupt,
                    pts_us=timestamp_us,
                )
                if stream.closed.is_set():
                    self._remember_closed_stream(key)
                    return
                self._streams[key] = stream
                self._response_traces[key] = trace
            else:
                stream = existing
                if output.interrupt:
                    raise ValueError("only the first chunk of a voice stream may interrupt")

            trace = self._response_traces[key]
            trace.fragments.append(output.text)
            trace.interrupt = trace.interrupt or output.interrupt

        await stream.send(output.text)
        if output.final:
            async with self._output_lock:
                if self._streams.get(key) is stream:
                    self._finish_response_trace(key, status="completed")
                    await stream.aclose()

    async def _publish_input(self, query: VoiceQuery) -> None:
        runtime = self._running_runtime()
        if query.interrupted_output and self.interrupted_topic is not None:
            await runtime.publish(
                self.interrupted_topic,
                VoiceInterrupted(),
                participant_id=query.participant_id,
                source=self._source,
            )
        await runtime.publish(
            self.query_topic,
            UserQuery(text=query.text, timestamp_us=query.timestamp_us),
            participant_id=query.participant_id,
            source=self._source,
        )

    async def _publish_transcript(
        self,
        participant_id: str,
        text: str,
        timestamp_us: int,
    ) -> None:
        """Queue one final STT result without waiting for runtime subscribers."""

        queue = self._transcript_queue
        if queue is None:
            return
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            else:
                queue.task_done()
                logger.warning(
                    "voice transcript queue full; dropped oldest transcript"
                )
        queue.put_nowait(
            (
                participant_id,
                VoiceTranscript(text=text, timestamp_us=timestamp_us),
            )
        )

    async def _deliver_transcripts(self, runtime: AgentRuntime) -> None:
        queue = self._transcript_queue
        assert queue is not None
        while True:
            participant_id, transcript = await queue.get()
            try:
                await runtime.publish(
                    VOICE_TRANSCRIPT_TOPIC,
                    transcript,
                    participant_id=participant_id,
                    source=self._source,
                )
            except asyncio.CancelledError:
                raise
            except BaseExceptionGroup as error:
                if runtime.running:
                    logger.error(
                        "voice transcript publication failed pid={!r}: {}",
                        participant_id,
                        error,
                    )
            except Exception as error:
                if runtime.running:
                    logger.error(
                        "voice transcript publication failed pid={!r}: {}",
                        participant_id,
                        error,
                    )
            finally:
                queue.task_done()

    async def _stop_transcript_delivery(self) -> None:
        task = self._transcript_task
        self._transcript_task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        queue = self._transcript_queue
        self._transcript_queue = None
        if queue is None:
            return
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            queue.task_done()

    async def _participant_joined(self, participant_id: str) -> None:
        topic = self.participant_joined_topic
        if topic is None:
            return
        runtime = self._running_runtime()
        self._start_participant_lifecycle_task(
            participant_id,
            lambda: runtime.publish(
                topic,
                VoiceParticipantJoined(),
                participant_id=participant_id,
                source=self._source,
            ),
            name=f"voice-participant-joined:{participant_id}",
        )

    async def _participant_left(self, participant_id: str) -> None:
        topic = self.participant_left_topic
        if topic is None:
            return
        runtime = self._running_runtime()
        self._start_participant_lifecycle_task(
            participant_id,
            lambda: runtime.publish(
                topic,
                VoiceParticipantLeft(),
                participant_id=participant_id,
                source=self._source,
            ),
            name=f"voice-participant-left:{participant_id}",
        )

    def _publish_interrupted(self, participant_id: str | None) -> None:
        runtime = self._running_runtime()
        topic = self.interrupted_topic
        assert topic is not None
        self._start_lifecycle_task(
            runtime.publish(
                topic,
                VoiceInterrupted(),
                participant_id=participant_id,
                source=self._source,
            ),
            name=f"voice-interrupted:{participant_id or 'all'}",
        )

    def _start_lifecycle_task(
        self,
        operation: Awaitable[None],
        *,
        name: str,
    ) -> None:
        task = asyncio.create_task(operation, name=name)
        self._lifecycle_tasks.add(task)
        task.add_done_callback(self._lifecycle_done)

    def _start_participant_lifecycle_task(
        self,
        participant_id: str,
        operation: Callable[[], Awaitable[None]],
        *,
        name: str,
    ) -> None:
        previous = self._participant_lifecycle_tails.get(participant_id)

        async def run_in_order() -> None:
            if previous is not None:
                await asyncio.gather(previous, return_exceptions=True)
            await operation()

        task = asyncio.create_task(run_in_order(), name=name)
        self._participant_lifecycle_tails[participant_id] = task
        self._lifecycle_tasks.add(task)
        task.add_done_callback(
            lambda completed: self._participant_lifecycle_done(
                participant_id,
                completed,
            )
        )

    def _participant_lifecycle_done(
        self,
        participant_id: str,
        task: asyncio.Task[None],
    ) -> None:
        if self._participant_lifecycle_tails.get(participant_id) is task:
            self._participant_lifecycle_tails.pop(participant_id, None)
        self._lifecycle_done(task)

    def _lifecycle_done(self, task: asyncio.Task[None]) -> None:
        self._lifecycle_tasks.discard(task)
        if task.cancelled():
            return
        if error := task.exception():
            logger.error("voice lifecycle publication failed: {}", error)

    def _running_runtime(self) -> AgentRuntime:
        if self._runtime is None:
            raise RuntimeError("voice agent is not running")
        return self._runtime

    def _discard_stream(
        self,
        key: tuple[str, str, str],
        stream: _ResponseStream,
    ) -> None:
        self._remember_closed_stream(key)
        if self._streams.get(key) is stream:
            self._streams.pop(key, None)
        self._finish_response_trace(key, status="closed")

    def _remember_closed_stream(self, key: tuple[str, str, str]) -> None:
        self._closed_streams.pop(key, None)
        self._closed_streams[key] = None
        if len(self._closed_streams) > _CLOSED_STREAM_CAPACITY:
            oldest = next(iter(self._closed_streams))
            self._closed_streams.pop(oldest, None)

    def _finish_response_trace(
        self,
        key: tuple[str, str, str],
        *,
        status: str,
    ) -> None:
        trace = self._response_traces.pop(key, None)
        if trace is None:
            return
        with self._response_scope(
            participant_id=trace.participant_id,
            source=trace.source,
            response_id=trace.response_id,
            correlation_id=trace.correlation_id,
            text="".join(trace.fragments),
            fragment_count=len(trace.fragments),
            interrupt=trace.interrupt,
            timestamp_us=trace.timestamp_us,
            streaming=True,
            status=status,
            started_at=trace.started_at,
        ):
            pass

    @staticmethod
    def _response_scope(
        *,
        participant_id: str,
        source: str,
        response_id: str | None,
        correlation_id: str,
        text: str,
        fragment_count: int,
        interrupt: bool,
        timestamp_us: int,
        streaming: bool,
        status: str,
        started_at: datetime | None = None,
    ):
        return nemo_relay.scope.scope(
            "voice.response",
            nemo_relay.ScopeType.Agent,
            input={
                "text": text,
                "streaming": streaming,
                "fragment_count": fragment_count,
                "interrupt": interrupt,
            },
            metadata={
                "participant_id": participant_id,
                "source": source,
                "response_id": response_id,
                "correlation_id": correlation_id,
                "timestamp_us": timestamp_us,
                "status": status,
            },
            timestamp=started_at,
            end_timestamp=datetime.now(UTC) if started_at is not None else None,
        )

    async def _on_data(self, message: DataMessage) -> None:
        if message.topic:
            return
        text = (message.data or b"").decode("utf-8", errors="replace").strip()
        if not text or not self._session.is_running:
            return
        if not self._session.transport.target_participant:
            self._session.transport.set_target_participant(message.participant_id)
        logger.info("text input pid={!r} {!r}", message.participant_id, text[:80])
        await self._session.enqueue_query(
            message.participant_id,
            text,
            pts_us=message.pts_us,
        )


__all__ = [
    "VOICE_OUTPUT_TOPIC",
    "VOICE_TRANSCRIPT_TOPIC",
    "UserQuery",
    "VoiceAgent",
    "VoiceInterrupted",
    "VoiceOutput",
    "VoiceParticipantJoined",
    "VoiceParticipantLeft",
    "VoiceStreamClosedError",
    "VoiceTranscript",
]
