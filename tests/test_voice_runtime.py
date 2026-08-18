# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the bidirectional voice runtime agent."""

from __future__ import annotations

import asyncio
import inspect
import sys
from builtins import ExceptionGroup
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import patch

import nemo_relay
import pytest
from pydantic import ValidationError
from xr_ai_hub import DataMessage
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, Topic, subscribe
from xr_ai_voice import (
    VOICE_OUTPUT_TOPIC,
    VOICE_TRANSCRIPT_TOPIC,
    UserQuery,
    VoiceAgent,
    VoiceInterrupted,
    VoiceOutput,
    VoiceParticipantJoined,
    VoiceParticipantLeft,
    VoiceStreamClosedError,
    VoiceTranscript,
)
from xr_ai_voice import _runtime as voice_runtime_module
from xr_ai_voice._types import VoiceQuery

QUERY_TOPIC = Topic("test.user-query", UserQuery)
PARTICIPANT_JOINED_TOPIC = Topic("test.participant-joined", VoiceParticipantJoined)
PARTICIPANT_LEFT_TOPIC = Topic("test.participant-left", VoiceParticipantLeft)
INTERRUPTED_TOPIC = Topic("test.interrupted", VoiceInterrupted)

class _Endpoint:
    def on_audio(self, callback):
        self.audio_callback = callback

        def unsubscribe() -> None:
            self.audio_callback = None

        return unsubscribe

    def on_data(self, callback):
        self.data_callback = callback

        def unsubscribe() -> None:
            self.data_callback = None

        return unsubscribe


class _Transport:
    def __init__(self) -> None:
        self.endpoint = _Endpoint()
        self.target_participant = ""

    def set_target_participant(self, participant_id: str) -> None:
        self.target_participant = participant_id


class _Session:
    def __init__(self) -> None:
        self.transport = _Transport()
        self.text_topic = "agent.response"
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

    @property
    def endpoint(self):
        return self.transport.endpoint

    async def __aenter__(self):
        return self

    async def run(self, handler, **options) -> None:
        self.handler = handler
        self.run_options = options
        self.started.set()
        await asyncio.Event().wait()

    async def enqueue_response(
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

    async def enqueue_query(
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


class _TranscriptRecorder(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[tuple[str | None, str, VoiceTranscript]] = []
        self.changed = asyncio.Event()

    @subscribe(VOICE_TRANSCRIPT_TOPIC)
    async def record(self, transcript: VoiceTranscript, ctx: RuntimeContext) -> None:
        self.messages.append(
            (ctx.metadata.participant_id, ctx.metadata.source, transcript)
        )
        self.changed.set()


class _BlockingTranscript(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.changed = asyncio.Event()

    @subscribe(VOICE_TRANSCRIPT_TOPIC)
    async def record(
        self,
        transcript: VoiceTranscript,
        _ctx: RuntimeContext,
    ) -> None:
        self.messages.append(transcript.text)
        self.changed.set()
        if len(self.messages) == 1:
            self.started.set()
            await self.release.wait()


class _LifecycleRecorder(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, str | None]] = []
        self.changed = asyncio.Event()

    @subscribe(PARTICIPANT_JOINED_TOPIC)
    async def participant_joined(
        self,
        _event: VoiceParticipantJoined,
        ctx: RuntimeContext,
    ) -> None:
        self.events.append(("participant-joined", ctx.metadata.participant_id))
        self.changed.set()

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


class _BlockingParticipantLifecycle(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.participants: set[str] = set()
        self.events: list[tuple[str, str]] = []
        self.first_join_started = asyncio.Event()
        self.release_first_join = asyncio.Event()
        self.changed = asyncio.Event()
        self._join_count = 0

    @subscribe(PARTICIPANT_JOINED_TOPIC)
    async def participant_joined(
        self,
        _event: VoiceParticipantJoined,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = ctx.metadata.participant_id
        assert participant_id is not None
        self._join_count += 1
        if self._join_count == 1:
            self.first_join_started.set()
            await self.release_first_join.wait()
        self.participants.add(participant_id)
        self.events.append(("joined", participant_id))
        self.changed.set()

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self,
        _event: VoiceParticipantLeft,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = ctx.metadata.participant_id
        assert participant_id is not None
        self.participants.discard(participant_id)
        self.events.append(("left", participant_id))
        self.changed.set()

    async def wait_for(self, count: int) -> None:
        while len(self.events) < count:
            await self.changed.wait()
            self.changed.clear()


class _BlockingLifecycle(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    @subscribe(INTERRUPTED_TOPIC)
    async def interrupted(
        self,
        _event: VoiceInterrupted,
        _ctx: RuntimeContext,
    ) -> None:
        self.started.set()
        await self.release.wait()


class _OrderedRecorder(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []
        self.changed = asyncio.Event()

    @subscribe(INTERRUPTED_TOPIC)
    async def interrupted(
        self,
        _event: VoiceInterrupted,
        _ctx: RuntimeContext,
    ) -> None:
        self.events.append("interrupted")

    @subscribe(QUERY_TOPIC)
    async def query(self, _query: UserQuery, _ctx: RuntimeContext) -> None:
        self.events.append("query")
        self.changed.set()


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


def _voice_agent(session: _Session, **kwargs) -> VoiceAgent:
    with patch.object(voice_runtime_module, "_VoiceSession", return_value=session):
        return VoiceAgent(
            query_topic=kwargs.pop("query_topic", QUERY_TOPIC),
            stt=object(),
            tts=object(),
            vad=object(),
            voice_gate=object(),
            **kwargs,
        )  # type: ignore[arg-type]


async def test_voice_agent_publishes_to_configured_query_topic() -> None:
    session = _Session()
    recorder = _InputRecorder()
    runtime = AgentRuntime()
    runtime.register("recorder", recorder)
    voice = _voice_agent(
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


async def test_voice_agent_publishes_final_transcript_before_query_gating() -> None:
    session = _Session()
    transcripts = _TranscriptRecorder()
    queries = _InputRecorder()
    runtime = AgentRuntime()
    runtime.register("transcripts", transcripts)
    runtime.register("queries", queries)
    voice = _voice_agent(
        session,
        query_topic=QUERY_TOPIC,
        text_input=False,
    )
    runtime.register("voice", voice)

    async with _running_voice(runtime, voice, session):
        await session.run_options["on_transcript"](
            "alice",
            "background conversation",
            123,
        )
        await asyncio.wait_for(transcripts.changed.wait(), 1.0)

    assert transcripts.messages == [
        (
            "alice",
            "voice",
            VoiceTranscript(text="background conversation", timestamp_us=123),
        )
    ]
    assert queries.messages == []
    assert VOICE_TRANSCRIPT_TOPIC.telemetry == "full"


async def test_transcript_subscriber_failure_does_not_block_accepted_query() -> None:
    class FailingTranscriptAgent(Agent):
        @subscribe(VOICE_TRANSCRIPT_TOPIC)
        async def fail(
            self,
            _transcript: VoiceTranscript,
            _ctx: RuntimeContext,
        ) -> None:
            raise RuntimeError("subscriber failed")

    session = _Session()
    transcripts = _TranscriptRecorder()
    queries = _InputRecorder()
    runtime = AgentRuntime()
    runtime.register("failing-transcript", FailingTranscriptAgent())
    runtime.register("transcripts", transcripts)
    runtime.register("queries", queries)
    voice = _voice_agent(
        session,
        query_topic=QUERY_TOPIC,
        text_input=False,
    )
    runtime.register("voice", voice)

    async with _running_voice(runtime, voice, session):
        await session.run_options["on_transcript"]("alice", "hey agent listen", 7)
        assert session.handler is not None
        await session.handler(
            VoiceQuery(
                participant_id="alice",
                text="listen",
                timestamp_us=7,
            )
        )
        await session.run_options["on_transcript"]("alice", "second utterance", 8)
        await asyncio.wait_for(queries.changed.wait(), 1.0)
        for _ in range(20):
            if len(transcripts.messages) == 2:
                break
            await asyncio.sleep(0.05)

    assert [query.text for _pid, _source, query in queries.messages] == ["listen"]
    assert [message.text for _pid, _source, message in transcripts.messages] == [
        "hey agent listen",
        "second utterance",
    ]


async def test_blocked_transcript_subscriber_does_not_block_accepted_query() -> None:
    session = _Session()
    blocker = _BlockingTranscript()
    queries = _InputRecorder()
    runtime = AgentRuntime()
    runtime.register("blocked-transcript", blocker)
    runtime.register("queries", queries)
    voice = _voice_agent(session, text_input=False)
    runtime.register("voice", voice)

    async with _running_voice(runtime, voice, session):
        await asyncio.wait_for(
            session.run_options["on_transcript"]("alice", "hey agent listen", 7),
            0.1,
        )
        await asyncio.wait_for(blocker.started.wait(), 1.0)
        assert session.handler is not None
        await asyncio.wait_for(
            session.handler(
                VoiceQuery(
                    participant_id="alice",
                    text="listen",
                    timestamp_us=7,
                )
            ),
            0.1,
        )
        await asyncio.wait_for(queries.changed.wait(), 1.0)

    assert [query.text for _pid, _source, query in queries.messages] == ["listen"]
    assert voice._transcript_task is None  # noqa: SLF001
    assert voice._transcript_queue is None  # noqa: SLF001


async def test_transcript_queue_is_bounded_ordered_and_drops_oldest(
    monkeypatch,
) -> None:
    monkeypatch.setattr(voice_runtime_module, "_TRANSCRIPT_CAPACITY", 2)
    session = _Session()
    blocker = _BlockingTranscript()
    runtime = AgentRuntime()
    runtime.register("blocked-transcript", blocker)
    voice = _voice_agent(session, text_input=False)
    runtime.register("voice", voice)

    async with _running_voice(runtime, voice, session):
        publish = session.run_options["on_transcript"]
        await publish("alice", "first", 1)
        await asyncio.wait_for(blocker.started.wait(), 1.0)
        await publish("alice", "second", 2)
        await publish("alice", "third", 3)
        await publish("alice", "fourth", 4)
        assert voice._transcript_queue is not None  # noqa: SLF001
        assert voice._transcript_queue.qsize() == 2  # noqa: SLF001
        blocker.release.set()
        while len(blocker.messages) < 3:
            blocker.changed.clear()
            await asyncio.wait_for(blocker.changed.wait(), 1.0)

    assert blocker.messages == ["first", "third", "fourth"]


async def test_voice_agent_publishes_configured_lifecycle_topics() -> None:
    session = _Session()
    recorder = _LifecycleRecorder()
    runtime = AgentRuntime()
    runtime.register("recorder", recorder)
    voice = _voice_agent(
        session,
        query_topic=QUERY_TOPIC,
        participant_joined_topic=PARTICIPANT_JOINED_TOPIC,
        participant_left_topic=PARTICIPANT_LEFT_TOPIC,
        interrupted_topic=INTERRUPTED_TOPIC,
        text_input=False,
    )
    runtime.register("voice", voice)

    async with _running_voice(runtime, voice, session):
        await session.run_options["on_participant_joined"]("alice")
        await session.run_options["on_participant_left"]("alice")
        assert session.run_options["on_interrupted"](None) is None
        await asyncio.wait_for(recorder.wait_for(3), 1.0)

    assert [event for event in recorder.events if event[0] != "interrupted"] == [
        ("participant-joined", "alice"),
        ("participant-left", "alice"),
    ]
    assert ("interrupted", None) in recorder.events


async def test_voice_lifecycle_publication_does_not_block_media_callback() -> None:
    session = _Session()
    blocker = _BlockingLifecycle()
    runtime = AgentRuntime()
    runtime.register("blocker", blocker)
    voice = _voice_agent(
        session,
        query_topic=QUERY_TOPIC,
        interrupted_topic=INTERRUPTED_TOPIC,
        text_input=False,
    )
    runtime.register("voice", voice)

    async with _running_voice(runtime, voice, session):
        assert session.run_options["on_interrupted"]("alice") is None
        await asyncio.wait_for(blocker.started.wait(), 1.0)
        assert voice._lifecycle_tasks  # noqa: SLF001
        blocker.release.set()


async def test_participant_lifecycle_publications_preserve_order() -> None:
    session = _Session()
    recorder = _BlockingParticipantLifecycle()
    runtime = AgentRuntime()
    runtime.register("recorder", recorder)
    voice = _voice_agent(
        session,
        query_topic=QUERY_TOPIC,
        participant_joined_topic=PARTICIPANT_JOINED_TOPIC,
        participant_left_topic=PARTICIPANT_LEFT_TOPIC,
        text_input=False,
    )
    runtime.register("voice", voice)

    async with _running_voice(runtime, voice, session):
        joined = session.run_options["on_participant_joined"]
        left = session.run_options["on_participant_left"]

        await asyncio.wait_for(joined("alice"), 1.0)
        await asyncio.wait_for(recorder.first_join_started.wait(), 1.0)
        await asyncio.wait_for(left("alice"), 1.0)
        await asyncio.wait_for(joined("alice"), 1.0)
        await asyncio.sleep(0)
        assert recorder.events == []

        recorder.release_first_join.set()
        await asyncio.wait_for(recorder.wait_for(3), 1.0)

    assert recorder.events == [
        ("joined", "alice"),
        ("left", "alice"),
        ("joined", "alice"),
    ]
    assert recorder.participants == {"alice"}


async def test_replacement_query_publishes_interruption_before_query() -> None:
    session = _Session()
    recorder = _OrderedRecorder()
    runtime = AgentRuntime()
    runtime.register("recorder", recorder)
    voice = _voice_agent(
        session,
        query_topic=QUERY_TOPIC,
        interrupted_topic=INTERRUPTED_TOPIC,
        text_input=False,
    )
    runtime.register("voice", voice)

    async with _running_voice(runtime, voice, session):
        assert session.handler is not None
        await session.handler(
            VoiceQuery(
                participant_id="alice",
                text="replacement",
                timestamp_us=7,
                interrupted_output=True,
            )
        )
        await asyncio.wait_for(recorder.changed.wait(), 1.0)

    assert recorder.events == ["interrupted", "query"]


async def test_voice_agent_accepts_output_from_multiple_publishers() -> None:
    session = _Session()
    runtime = AgentRuntime()
    voice = _voice_agent(
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
    voice = _voice_agent(
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
    voice = _voice_agent(session)
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
    assert session.queries == [("alice", "hello", 2)]


async def test_voice_agent_drops_inactive_non_client_and_empty_text() -> None:
    session = _Session()
    runtime = AgentRuntime()
    voice = _voice_agent(session)
    runtime.register("voice", voice)
    message = DataMessage(
        participant_id="alice",
        topic="request",
        pts_us=3,
        data=b"   ",
    )

    await voice._on_data(message)  # noqa: SLF001
    async with _running_voice(runtime, voice, session):
        await session.transport.endpoint.data_callback(
            DataMessage(
                participant_id="alice",
                topic="agent.control",
                pts_us=4,
                data=b"ignored",
            )
        )
        await session.transport.endpoint.data_callback(message)

    assert session.queries == []


async def test_untopiced_hub_data_reaches_voice_as_typed_query(
    hub,
    make_connector,
    make_processor,
) -> None:
    endpoint = make_processor()
    session = _Session()
    session.transport.endpoint = endpoint
    runtime = AgentRuntime()
    voice = _voice_agent(session)
    runtime.register("voice", voice)
    connector = make_connector(connector_id="voice-text-client")

    async with _running_voice(runtime, voice, session):
        await connector.register()
        await asyncio.sleep(0.05)
        await connector.notify_participant_joined("alice", pts_us=1)
        for _ in range(20):
            if "alice" in endpoint.subscribed_participants:
                break
            await asyncio.sleep(0.05)
        assert "alice" in endpoint.subscribed_participants

        await connector.push_data(
            DataMessage("alice", "", 2, b"describe the room")
        )
        for _ in range(20):
            if session.queries:
                break
            await asyncio.sleep(0.05)
        await connector.push_data(
            DataMessage("alice", "agent.control", 3, b"ignored")
        )
        await asyncio.sleep(0.1)

    assert session.queries == [("alice", "describe the room", 2)]


async def test_voice_agent_unregisters_typed_input_callback() -> None:
    session = _Session()
    runtime = AgentRuntime()
    voice = _voice_agent(session)
    runtime.register("voice", voice)

    async with _running_voice(runtime, voice, session):
        assert session.transport.endpoint.data_callback is not None

    assert session.transport.endpoint.data_callback is None


async def test_incremental_responses_are_isolated_by_participant_and_source() -> None:
    session = _Session()
    runtime = AgentRuntime()
    voice = _voice_agent(
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
    agent = _voice_agent(
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
        await runtime.publish(
            VOICE_OUTPUT_TOPIC,
            VoiceOutput(response_id="turn"),
            participant_id="alice",
            source="observer",
        )
        assert len(session.response_tasks) == 1

        assert agent._closed_streams  # noqa: SLF001
    assert agent._streams == {}  # noqa: SLF001


async def test_voice_output_preserves_originating_query_timestamp() -> None:
    session = _Session()
    runtime = AgentRuntime()
    voice = _voice_agent(
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


async def test_blocked_stream_does_not_block_unrelated_output() -> None:
    class HeldSession(_Session):
        def __init__(self) -> None:
            super().__init__()
            self.held: list[AsyncIterator[str]] = []

        async def enqueue_response(
            self,
            participant_id: str,
            response: str | AsyncIterator[str],
            *,
            interrupt: bool = False,
            pts_us: int | None = None,
        ) -> None:
            if isinstance(response, str):
                await super().enqueue_response(
                    participant_id,
                    response,
                    interrupt=interrupt,
                    pts_us=pts_us,
                )
            else:
                self.held.append(response)

    session = HeldSession()
    runtime = AgentRuntime()
    voice = _voice_agent(
        session,
        query_topic=QUERY_TOPIC,
        response_capacity=1,
        text_input=False,
    )
    runtime.register("voice", voice)

    async with _running_voice(runtime, voice, session):
        await runtime.publish(
            VOICE_OUTPUT_TOPIC,
            VoiceOutput(text="one", response_id="held", final=False),
            participant_id="alice",
            source="observer",
        )
        blocked = asyncio.create_task(
            runtime.publish(
                VOICE_OUTPUT_TOPIC,
                VoiceOutput(text="two", response_id="held", final=False),
                participant_id="alice",
                source="observer",
            )
        )
        await asyncio.sleep(0)
        assert not blocked.done()
        await asyncio.wait_for(
            runtime.publish(
                VOICE_OUTPUT_TOPIC,
                VoiceOutput(text="independent"),
                participant_id="bob",
                source="other",
            ),
            1.0,
        )
        await session.held[0].aclose()  # type: ignore[attr-defined]
        await asyncio.wait_for(blocked, 1.0)

    assert [(pid, text, interrupt) for pid, text, interrupt, _ in session.responses] == [
        ("bob", "independent", False)
    ]


async def test_open_response_streams_are_bounded(monkeypatch) -> None:
    class HeldSession(_Session):
        def __init__(self) -> None:
            super().__init__()
            self.held: list[AsyncIterator[str]] = []

        async def enqueue_response(self, _participant_id, response, **_kwargs) -> None:
            self.held.append(response)

    monkeypatch.setattr(voice_runtime_module, "_OPEN_STREAM_CAPACITY", 1)
    session = HeldSession()
    runtime = AgentRuntime()
    voice = _voice_agent(session, text_input=False)
    runtime.register("voice", voice)

    async with _running_voice(runtime, voice, session):
        for response_id in ("first", "second"):
            await runtime.publish(
                VOICE_OUTPUT_TOPIC,
                VoiceOutput(text=response_id, response_id=response_id, final=False),
                participant_id="alice",
                source="observer",
            )

        assert len(voice._streams) == 1  # noqa: SLF001
        assert ("alice", "observer", "second") in voice._streams  # noqa: SLF001
        assert session.held[0].closed.is_set()  # type: ignore[attr-defined]


async def test_failed_stream_enqueue_does_not_register_response() -> None:
    class FailingSession(_Session):
        async def enqueue_response(self, *_args, **_kwargs) -> None:
            raise RuntimeError("session stopped")

    session = FailingSession()
    runtime = AgentRuntime()
    voice = _voice_agent(session, text_input=False)
    runtime.register("voice", voice)

    async with _running_voice(runtime, voice, session):
        with pytest.raises(ExceptionGroup, match="event publication"):
            await runtime.publish(
                VOICE_OUTPUT_TOPIC,
                VoiceOutput(text="first", response_id="turn", final=False),
                participant_id="alice",
                source="observer",
            )

    assert voice._streams == {}  # noqa: SLF001
    assert voice._response_traces == {}  # noqa: SLF001


async def test_midstream_interrupt_is_rejected() -> None:
    session = _Session()
    runtime = AgentRuntime()
    voice = _voice_agent(session, text_input=False)
    runtime.register("voice", voice)

    async with _running_voice(runtime, voice, session):
        await runtime.publish(
            VOICE_OUTPUT_TOPIC,
            VoiceOutput(text="first", response_id="turn", final=False),
            participant_id="alice",
            source="observer",
        )
        with pytest.raises(ExceptionGroup, match="event publication") as raised:
            await runtime.publish(
                VOICE_OUTPUT_TOPIC,
                VoiceOutput(
                    text="second",
                    response_id="turn",
                    final=False,
                    interrupt=True,
                ),
                participant_id="alice",
                source="observer",
            )

    assert "only the first chunk" in str(raised.value.exceptions[0])


async def test_unknown_empty_stream_terminator_is_rejected() -> None:
    session = _Session()
    runtime = AgentRuntime()
    voice = _voice_agent(
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
    assert isinstance(raised.value.exceptions[0], VoiceStreamClosedError)
    assert "no open response" in str(raised.value.exceptions[0])


def test_voice_output_rejects_ambiguous_empty_messages() -> None:
    with pytest.raises(ValidationError, match="response_id"):
        VoiceOutput(final=False)
    with pytest.raises(ValidationError, match="contain text"):
        VoiceOutput()
    with pytest.raises(ValidationError, match="cannot interrupt"):
        VoiceOutput(response_id="turn", interrupt=True)


def test_voice_session_is_not_part_of_the_public_api() -> None:
    voice_module = sys.modules["xr_ai_voice"]
    assert "VoiceSession" not in voice_module.__all__
    assert not hasattr(voice_module, "VoiceSession")


def test_raw_audio_is_not_part_of_the_voice_runtime_api() -> None:
    voice_module = sys.modules["xr_ai_voice"]
    assert "VOICE_AUDIO_TOPIC" not in voice_module.__all__
    assert "VoiceAudio" not in voice_module.__all__
    assert not hasattr(voice_module, "VOICE_AUDIO_TOPIC")
    assert not hasattr(voice_module, "VoiceAudio")


def test_voice_agent_does_not_expose_owned_media_internals() -> None:
    assert not hasattr(VoiceAgent, "endpoint")
    assert not hasattr(VoiceAgent, "transport")
    assert not hasattr(VoiceAgent, "close")
    parameters = inspect.signature(VoiceAgent).parameters
    assert "_session" not in parameters
    assert "text_transform" not in parameters
    assert "text_ignore_topics" not in parameters
