# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bidirectional agent-runtime boundary for participant-aware voice."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator
from xr_ai_hub import DataMessage
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, Topic, subscribe

from ._session import VoiceSession
from ._types import VoiceQuery

QueryTransform = Callable[[str], str]


class UserQuery(BaseModel):
    """One accepted user query emitted by the voice input boundary."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    timestamp_us: int = Field(ge=0)


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
    response_id: str | None = Field(default=None, min_length=1)
    final: bool = True
    interrupt: bool = False
    timestamp_us: int | None = Field(default=None, ge=0)

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


VOICE_OUTPUT_TOPIC = Topic("voice.output", VoiceOutput)


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
        session: VoiceSession,
        *,
        query_topic: Topic[UserQuery],
        response_capacity: int = 32,
        text_input: bool = True,
        text_ignore_topics: Iterable[str] = (),
        text_transform: QueryTransform | None = None,
        participant_left_topic: Topic[VoiceParticipantLeft] | None = None,
        interrupted_topic: Topic[VoiceInterrupted] | None = None,
        interrupt_on_supersede: bool = False,
    ) -> None:
        if response_capacity <= 0:
            raise ValueError("voice response capacity must be positive")
        super().__init__()
        self.session = session
        self.query_topic = query_topic
        self.response_capacity = response_capacity
        self.text_input = text_input
        self.text_ignore_topics = tuple(text_ignore_topics)
        self.text_transform = text_transform
        self.participant_left_topic = participant_left_topic
        self.interrupted_topic = interrupted_topic
        self.interrupt_on_supersede = interrupt_on_supersede
        self._runtime: AgentRuntime | None = None
        self._source = "voice"
        self._output_lock = asyncio.Lock()
        self._streams: dict[tuple[str, str, str], _ResponseStream] = {}

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
        try:
            await self.session.__aenter__()
            try:
                if self.text_input:
                    self.session.transport.endpoint.on_data(self._on_data)
                await self.session._run(
                    self._publish_input,
                    on_participant_left=(
                        self._publish_participant_left
                        if self.participant_left_topic is not None
                        else None
                    ),
                    on_interrupted=(
                        self._publish_interrupted
                        if self.interrupted_topic is not None
                        else None
                    ),
                    interrupt_on_supersede=self.interrupt_on_supersede,
                )
            finally:
                await asyncio.gather(
                    *(stream.aclose() for stream in tuple(self._streams.values()))
                )
                await self.session.close()
        finally:
            self._runtime = None
            self._source = "voice"

    @subscribe(VOICE_OUTPUT_TOPIC)
    async def output(self, output: VoiceOutput, ctx: RuntimeContext) -> None:
        """Send one voice message using participant and producer metadata."""

        async with self._output_lock:
            await self._output(output, ctx)

    async def _output(self, output: VoiceOutput, ctx: RuntimeContext) -> None:
        """Serialize access to response aggregation and the session FIFO."""

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
            await self.session._enqueue_response(
                participant_id,
                output.text,
                interrupt=output.interrupt,
                pts_us=timestamp_us,
            )
            return

        key = (participant_id, metadata.source, output.response_id)
        stream = self._streams.get(key)
        if stream is None:
            if output.final:
                if not output.text.strip():
                    raise ValueError("voice stream terminator has no open response")
                await self.session._enqueue_response(
                    participant_id,
                    output.text,
                    interrupt=output.interrupt,
                    pts_us=timestamp_us,
                )
                return
            stream = _ResponseStream(
                self.response_capacity,
                lambda closed: self._discard_stream(key, closed),
            )
            self._streams[key] = stream
            await self.session._enqueue_response(
                participant_id,
                stream,
                interrupt=output.interrupt,
                pts_us=timestamp_us,
            )
        await stream.send(output.text)
        if output.final:
            await stream.aclose()

    async def _publish_input(self, query: VoiceQuery) -> None:
        runtime = self._running_runtime()
        await runtime.publish(
            self.query_topic,
            UserQuery(text=query.text, timestamp_us=query.timestamp_us),
            participant_id=query.participant_id,
            source=self._source,
        )

    async def _publish_participant_left(self, participant_id: str) -> None:
        runtime = self._running_runtime()
        topic = self.participant_left_topic
        if topic is None:
            return
        await runtime.publish(
            topic,
            VoiceParticipantLeft(),
            participant_id=participant_id,
            source=self._source,
        )

    async def _publish_interrupted(self, participant_id: str | None) -> None:
        runtime = self._running_runtime()
        topic = self.interrupted_topic
        if topic is None:
            return
        await runtime.publish(
            topic,
            VoiceInterrupted(),
            participant_id=participant_id,
            source=self._source,
        )

    def _running_runtime(self) -> AgentRuntime:
        if self._runtime is None:
            raise RuntimeError("voice agent is not running")
        return self._runtime

    def _discard_stream(
        self,
        key: tuple[str, str, str],
        stream: _ResponseStream,
    ) -> None:
        if self._streams.get(key) is stream:
            self._streams.pop(key, None)

    async def _on_data(self, message: DataMessage) -> None:
        if message.topic in self.text_ignore_topics:
            return
        text = (message.data or b"").decode("utf-8", errors="replace").strip()
        if not text or not self.session.is_running:
            return
        if not self.session.transport.target_participant:
            self.session.transport.set_target_participant(message.participant_id)
        if self.text_transform is not None:
            text = self.text_transform(text)
        logger.info("text input pid={!r} {!r}", message.participant_id, text[:80])
        await self.session._enqueue_query(
            message.participant_id,
            text,
            pts_us=message.pts_us,
        )


__all__ = [
    "VOICE_OUTPUT_TOPIC",
    "UserQuery",
    "VoiceAgent",
    "VoiceInterrupted",
    "VoiceOutput",
    "VoiceParticipantLeft",
]
