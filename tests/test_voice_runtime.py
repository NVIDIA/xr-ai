# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the bidirectional voice runtime agent."""

from __future__ import annotations

import asyncio
from builtins import ExceptionGroup
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import nemo_relay
import pytest
from pydantic import ValidationError
from xr_ai_hub import DataMessage
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, Topic, subscribe
from xr_ai_voice import (
    VOICE_OUTPUT_TOPIC,
    UserQuery,
    VoiceAgent,
    VoiceInterrupted,
    VoiceOutput,
    VoiceParticipantLeft,
)
from xr_ai_voice._types import VoiceQuery

QUERY_TOPIC = Topic("test.user-query", UserQuery)
PARTICIPANT_LEFT_TOPIC = Topic("test.participant-left", VoiceParticipantLeft)
INTERRUPTED_TOPIC = Topic("test.interrupted", VoiceInterrupted)


class _Endpoint:
    def on_data(self, callback) -> None:
        self.data_callback = callback


class _Transport:
    def __init__(self) -> None:
        self.endpoint = _Endpoint()
        self.target_participant = ""

    def set_target_participant(self, participant_id: str) -> None:
        self.target_participant = participant_id


class _Session:
    def __init__(self) -> None:
        self.transport = _Transport()
        self.responses: list[tuple[str, str, bool, int | None]] = []
        self.response_tasks: list[asyncio.Task[None]] = []
        self.run_options = {}
        self.handler = None
        self.started = asyncio.Event()
        self.changed = asyncio.Event()
        self.closed = False
        self.queries: list[tuple[str, str, int | None]] = []

    @property
    def is_running(self) -> bool:
        return self.handler is not None

    async def __aenter__(self):
        return self

    async def _run(self, handler, **options) -> None:
        self.handler = handler
        self.run_options = options
        self.started.set()
        await asyncio.Event().wait()

    async def _enqueue_response(
        self,
        participant_id: str,
        response: str | AsyncIterator[str],
        *,
        interrupt: bool = False,
        pts_us: int | None = None,
    ) -> None:
        if isinstance(response, str):
            self.responses.append((participant_id, response, interrupt, pts_us))
            self.changed.set()
            return

        async def consume() -> None:
            chunks = [chunk async for chunk in response]
            self.responses.append((participant_id, "".join(chunks), interrupt, pts_us))
            self.changed.set()

        self.response_tasks.append(asyncio.create_task(consume()))

    async def _enqueue_query(
        self,
        participant_id: str,
        text: str,
        *,
        pts_us: int | None = None,
    ) -> None:
        self.queries.append((participant_id, text, pts_us))

    async def wait_for(self, count: int) -> None:
        while len(self.responses) < count:
            self.changed.clear()
            await self.changed.wait()

    async def close(self) -> None:
        self.closed = True
        if self.response_tasks:
            await asyncio.gather(*self.response_tasks, return_exceptions=True)


class _InputRecorder(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[tuple[str | None, str, UserQuery]] = []
        self.changed = asyncio.Event()

    @subscribe(QUERY_TOPIC)
    async def record(self, query: UserQuery, ctx: RuntimeContext) -> None:
        self.messages.append((ctx.metadata.participant_id, ctx.metadata.source, query))
        self.changed.set()


class _LifecycleRecorder(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, str | None]] = []
        self.changed = asyncio.Event()

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self,
        _event: VoiceParticipantLeft,
        ctx: RuntimeContext,
    ) -> None:
        self.events.append(("participant-left", ctx.metadata.participant_id))
        self.changed.set()

    @subscribe(INTERRUPTED_TOPIC)
    async def interrupted(
        self,
        _event: VoiceInterrupted,
        ctx: RuntimeContext,
    ) -> None:
        self.events.append(("interrupted", ctx.metadata.participant_id))
        self.changed.set()

    async def wait_for(self, count: int) -> None:
        while len(self.events) < count:
            self.changed.clear()
            await self.changed.wait()


@asynccontextmanager
async def _running_voice(
    runtime: AgentRuntime,
    voice: VoiceAgent,
    session: _Session,
) -> AsyncIterator[None]:
    async with runtime:
        task = asyncio.create_task(voice.run(runtime))
        await asyncio.wait_for(session.started.wait(), 1.0)
        try:
            yield
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_voice_agent_publishes_to_configured_query_topic() -> None:
    session = _Session()
    recorder = _InputRecorder()
    runtime = AgentRuntime()
    runtime.register("recorder", recorder)
    voice = VoiceAgent(  # type: ignore[arg-type]
        session,
        query_topic=QUERY_TOPIC,
        text_input=False,
    )
    runtime.register("voice", voice)

    async with _running_voice(runtime, voice, session):
        assert session.handler is not None
        assert await session.handler(
            VoiceQuery(
                participant_id="alice",
                text="start monitoring",
                timestamp_us=7,
            )
        ) is None
        await asyncio.wait_for(recorder.changed.wait(), 1.0)

    assert recorder.messages == [
        (
            "alice",
            "voice",
            UserQuery(text="start monitoring", timestamp_us=7),
        )
    ]
    assert session.closed is True


async def test_voice_agent_publishes_configured_lifecycle_topics() -> None:
    session = _Session()
    recorder = _LifecycleRecorder()
    runtime = AgentRuntime()
    runtime.register("recorder", recorder)
    voice = VoiceAgent(  # type: ignore[arg-type]
        session,
        query_topic=QUERY_TOPIC,
        participant_left_topic=PARTICIPANT_LEFT_TOPIC,
        interrupted_topic=INTERRUPTED_TOPIC,
        text_input=False,
    )
    runtime.register("voice", voice)

    async with _running_voice(runtime, voice, session):
        await session.run_options["on_participant_left"]("alice")
        await session.run_options["on_interrupted"](None)
        await asyncio.wait_for(recorder.wait_for(2), 1.0)

    assert recorder.events == [
        ("participant-left", "alice"),
        ("interrupted", None),
    ]


async def test_voice_agent_accepts_output_from_multiple_publishers() -> None:
    session = _Session()
    runtime = AgentRuntime()
    voice = VoiceAgent(  # type: ignore[arg-type]
        session,
        query_topic=QUERY_TOPIC,
        text_input=False,
    )
    runtime.register("voice", voice)

    async with _running_voice(runtime, voice, session):
        await runtime.publish(
            VOICE_OUTPUT_TOPIC,
            VoiceOutput(text="Careful.", interrupt=True),
            participant_id="alice",
            source="safety-monitor",
        )
        await runtime.publish(
            VOICE_OUTPUT_TOPIC,
            VoiceOutput(text="The timer is done."),
            participant_id="alice",
            source="tea-timer",
        )

    assert [(pid, text, interrupt) for pid, text, interrupt, _ in session.responses] == [
        ("alice", "Careful.", True),
        ("alice", "The timer is done.", False),
    ]


async def test_voice_agent_records_one_summary_for_finite_and_streamed_output() -> None:
    session = _Session()
    runtime = AgentRuntime()
    voice = VoiceAgent(  # type: ignore[arg-type]
        session,
        query_topic=QUERY_TOPIC,
        text_input=False,
    )
    runtime.register("voice", voice)
    events = []
    subscriber = "xr-ai-voice-response-summary"
    nemo_relay.subscribers.register(subscriber, events.append)
    try:
        async with _running_voice(runtime, voice, session):
            await runtime.publish(
                VOICE_OUTPUT_TOPIC,
                VoiceOutput(text="Finite answer."),
                participant_id="alice",
                source="finite-agent",
            )
            await runtime.publish(
                VOICE_OUTPUT_TOPIC,
                VoiceOutput(text="Streamed ", response_id="turn-1", final=False),
                participant_id="alice",
                source="stream-agent",
            )
            await runtime.publish(
                VOICE_OUTPUT_TOPIC,
                VoiceOutput(text="answer.", response_id="turn-1"),
                participant_id="alice",
                source="stream-agent",
            )
            await asyncio.wait_for(session.wait_for(2), 1.0)
        await nemo_relay.subscribers.flush_async()
    finally:
        nemo_relay.subscribers.deregister(subscriber)

    starts = [
        event.to_dict()
        for event in events
        if event.name == "voice.response"
        and event.to_dict().get("scope_category") == "start"
    ]
    assert len(starts) == 2
    by_text = {event["data"]["text"]: event for event in starts}
    assert by_text["Finite answer."]["data"] | {
        "streaming": False,
        "fragment_count": 1,
    } == by_text["Finite answer."]["data"]
    assert by_text["Streamed answer."]["data"] | {
        "streaming": True,
        "fragment_count": 2,
    } == by_text["Streamed answer."]["data"]
    assert by_text["Streamed answer."]["metadata"] | {
        "participant_id": "alice",
        "source": "stream-agent",
        "response_id": "turn-1",
        "status": "completed",
    } == by_text["Streamed answer."]["metadata"]
    assert by_text["Finite answer."]["metadata"]["correlation_id"]
    assert by_text["Streamed answer."]["metadata"]["correlation_id"]
    assert "publish:voice.output" not in {event.name for event in events}
    assert "agent:voice" not in {event.name for event in events}


async def test_voice_agent_routes_typed_input() -> None:
    session = _Session()
    runtime = AgentRuntime()
    voice = VoiceAgent(
        session,  # type: ignore[arg-type]
        query_topic=QUERY_TOPIC,
        text_ignore_topics={"control"},
        text_transform=str.upper,
    )
    runtime.register("voice", voice)

    async with _running_voice(runtime, voice, session):
        await session.transport.endpoint.data_callback(
            DataMessage(
                participant_id="alice",
                topic="control",
                pts_us=1,
                data=b"ignored",
            )
        )
        await session.transport.endpoint.data_callback(
            DataMessage(
                participant_id="alice",
                topic="",
                pts_us=2,
                data=b"hello",
            )
        )

    assert session.transport.target_participant == "alice"
    assert session.queries == [("alice", "HELLO", 2)]


async def test_incremental_responses_are_isolated_by_participant_and_source() -> None:
    session = _Session()
    runtime = AgentRuntime()
    voice = VoiceAgent(  # type: ignore[arg-type]
        session,
        query_topic=QUERY_TOPIC,
        text_input=False,
    )
    runtime.register("voice", voice)

    async with _running_voice(runtime, voice, session):
        for source, text in (("observer-a", "alpha "), ("observer-b", "beta ")):
            await runtime.publish(
                VOICE_OUTPUT_TOPIC,
                VoiceOutput(text=text, response_id="shared", final=False),
                participant_id="alice",
                source=source,
            )
        for source, text in (("observer-b", "two"), ("observer-a", "one")):
            await runtime.publish(
                VOICE_OUTPUT_TOPIC,
                VoiceOutput(text=text, response_id="shared"),
                participant_id="alice",
                source=source,
            )
        await asyncio.wait_for(session.wait_for(2), 1.0)

    assert sorted(text for _pid, text, _interrupt, _pts in session.responses) == [
        "alpha one",
        "beta two",
    ]


async def test_cancelled_response_stream_releases_blocked_publishers() -> None:
    session = _Session()
    runtime = AgentRuntime()
    agent = VoiceAgent(  # type: ignore[arg-type]
        session,
        query_topic=QUERY_TOPIC,
        response_capacity=1,
        text_input=False,
    )
    runtime.register("voice", agent)

    async with _running_voice(runtime, agent, session):
        await runtime.publish(
            VOICE_OUTPUT_TOPIC,
            VoiceOutput(text="one", response_id="turn", final=False),
            participant_id="alice",
            source="observer",
        )
        await asyncio.sleep(0)
        assert session.response_tasks
        session.response_tasks[0].cancel()
        await asyncio.gather(*session.response_tasks, return_exceptions=True)
        await asyncio.wait_for(
            runtime.publish(
                VOICE_OUTPUT_TOPIC,
                VoiceOutput(text="two", response_id="turn", final=False),
                participant_id="alice",
                source="observer",
            ),
            1.0,
        )

    assert agent._streams == {}  # noqa: SLF001


async def test_voice_output_preserves_originating_query_timestamp() -> None:
    session = _Session()
    runtime = AgentRuntime()
    voice = VoiceAgent(  # type: ignore[arg-type]
        session,
        query_topic=QUERY_TOPIC,
        text_input=False,
    )
    runtime.register("voice", voice)

    async with _running_voice(runtime, voice, session):
        await runtime.publish(
            VOICE_OUTPUT_TOPIC,
            VoiceOutput(text="answer", timestamp_us=123),
            participant_id="alice",
            source="observer",
        )

    assert session.responses == [("alice", "answer", False, 123)]


async def test_unknown_empty_stream_terminator_is_rejected() -> None:
    session = _Session()
    runtime = AgentRuntime()
    voice = VoiceAgent(  # type: ignore[arg-type]
        session,
        query_topic=QUERY_TOPIC,
        text_input=False,
    )
    runtime.register("voice", voice)

    async with _running_voice(runtime, voice, session):
        with pytest.raises(ExceptionGroup, match="event publication") as raised:
            await runtime.publish(
                VOICE_OUTPUT_TOPIC,
                VoiceOutput(response_id="missing"),
                participant_id="alice",
                source="observer",
            )

    assert len(raised.value.exceptions) == 1
    assert isinstance(raised.value.exceptions[0], ValueError)
    assert "no open response" in str(raised.value.exceptions[0])


def test_voice_output_rejects_ambiguous_empty_messages() -> None:
    with pytest.raises(ValidationError, match="response_id"):
        VoiceOutput(final=False)
    with pytest.raises(ValidationError, match="contain text"):
        VoiceOutput()
    with pytest.raises(ValidationError, match="cannot interrupt"):
        VoiceOutput(response_id="turn", interrupt=True)
