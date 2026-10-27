# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the packaged conversation-example worker."""

from __future__ import annotations

import asyncio
import io
import sys
import time
import tomllib
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import nemo_relay
import pytest
import yaml
from _conversation_output import CaptureRuntime
from _conversation_output import CapturingAgentRouter as AgentRouter
from _conversation_output import answer as _answer
from _conversation_output import handle as _handle
from loguru import logger
from xr_ai_hub import FrameData, FrameSignal, FrameUnavailable, PixelFormat, ProcessorEndpoint
from xr_ai_interaction import Interrupted, ParticipantJoined, ParticipantLeft, ResponseOutput, UserQuery
from xr_ai_models import (
    ChatResponse,
    ChatStreamEvent,
    DecisionAnswer,
    DecisionResponse,
    VLMService,
    load_models_config,
)
from xr_ai_runtime import Agent, AgentRuntime, Topic, subscribe
from xr_ai_voice import VOICE_OUTPUT_TOPIC, VoiceAgent
from xr_ai_voice import _runtime as voice_runtime_module
from xr_ai_voice._session import _VoiceSession as VoiceSession
from xr_ai_voicegate import VoiceGateConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE_DIR = _REPO_ROOT / "agent-samples" / "conversation-example"
_WORKER_DIR = _SAMPLE_DIR / "worker"
sys.path.insert(0, str(_WORKER_DIR))

from conversation_example_worker import __main__ as worker_main  # noqa: E402  # pyright: ignore[reportMissingImports]
from conversation_example_worker import app  # noqa: E402  # pyright: ignore[reportMissingImports]
from conversation_example_worker._note_taking import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    NoteTakingAgent,
)
from conversation_example_worker._topics import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    INTERRUPTED_TOPIC,
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    USER_QUERY_TOPIC,
)
from conversation_example_worker._visual_agents import VisualObservationAgent  # noqa: E402
from conversation_example_worker.app import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    _CONVERSATION_INSTRUCTIONS,
)
from conversation_example_worker.config import load_config  # noqa: E402  # pyright: ignore[reportMissingImports]
from xr_ai_conversation import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    ContextManager,
    Conversation,
    ConversationContextUpdate,
)
from xr_ai_routing import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    Admission,
    AgentTurn,
    DestinationResult,
    RoutableAgent,
    RoutingAid,
)
from xr_ai_tools import Tool  # noqa: E402
from xr_ai_tools.current_frame import (  # noqa: E402
    CurrentFrameRequest,
    CurrentFrameTool,
    ImageFrame,
)
from xr_ai_tools.image import ImageReference, ImageRegistry  # noqa: E402
from xr_ai_tools.vision import (  # noqa: E402
    StreamingImageQueryTool,
)


class _Service:
    def __init__(self) -> None:
        self.health_calls = 0
        self.close_calls = 0
        self.stream_calls = []

    async def health(self) -> bool:
        self.health_calls += 1
        return True

    async def stream_images(
        self,
        images,
        question,
        *,
        system_prompt="",
        max_tokens=None,
        timeout=None,
    ):
        self.stream_calls.append((images, question, system_prompt, max_tokens, timeout))
        yield "gray"

    async def close(self) -> None:
        self.close_calls += 1


class _ConversationService(_Service):
    def __init__(
        self,
        *,
        stream_gate=None,
        error: Exception | None = None,
    ) -> None:
        super().__init__()
        self.requests = []
        self.stream_gate = stream_gate
        self.error = error

    def stream_events(self, messages, *, tools=None, **_kwargs):
        self.requests.append((messages, tools))

        async def events():
            if self.error is not None:
                raise self.error
            yield ChatStreamEvent(kind="text", text="Hello ")
            if self.stream_gate is not None:
                await self.stream_gate.wait()
            yield ChatStreamEvent(kind="text", text="there.")
            yield ChatStreamEvent(kind="finish", finish_reason="stop")

        return events()


class _Transport:
    def __init__(self) -> None:
        self.endpoint = _DataEndpoint()
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _DataEndpoint:
    def __init__(self) -> None:
        self.statuses: list[tuple[str, str]] = []

    def on_audio(self, callback):
        self.audio_callback = callback

        def unsubscribe() -> None:
            self.audio_callback = None

        return unsubscribe

    def on_data(self, callback) -> None:
        self.data_callback = callback

    async def set_status(self, status: str, participant_id: str) -> None:
        self.statuses.append((status, participant_id))


class _CurrentFrameTool:
    instances: list["_CurrentFrameTool"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.requests = []
        self.released: list[str] = []
        self.instances.append(self)

    async def execute(self, request):
        self.requests.append(request)
        return ImageFrame(
            image=ImageReference(uri="https://example.com/frame.jpg"),
            timestamp_us=123,
            width=2,
            height=2,
            sequence=1,
            participant_id=request.participant_id,
        )

    def release(self, participant_id: str) -> None:
        self.released.append(participant_id)


class _StreamingImageQueryTool:
    instances: list["_StreamingImageQueryTool"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.requests = []
        self.instances.append(self)

    async def stream(self, request):
        self.requests.append(request)
        for text in ("a ", "blue square"):
            yield SimpleNamespace(text=text)


class _SelectedFrameTool:
    async def execute(self, _request):
        return SimpleNamespace(
            image=ImageReference(uri="https://example.com/frame.jpg"),
            timestamp_us=None,
        )

    def release(self, _participant_id: str) -> None:
        pass


class _ConversationFrames:
    def __init__(self) -> None:
        self.requests = []

    async def execute(self, request):
        self.requests.append(request)
        return ImageFrame(
            image=ImageReference(uri="https://example.com/frame.jpg"),
            timestamp_us=123,
            width=2,
            height=2,
            sequence=1,
            participant_id=request.participant_id,
        )

    def release(self, _participant_id: str) -> None:
        pass


async def _ignore_status(_status: str, _participant_id: str) -> None:
    pass


async def _ignore_output(_source: str, _text: str) -> None:
    pass


class _RouteDecisions:
    def __init__(self, destination: str | tuple[str, ...] = "observation") -> None:
        self.destinations = list(destination) if isinstance(destination, tuple) else [destination]
        self.health_calls = 0
        self.close_calls = 0
        self.states = []
        self.questions = []

    async def health(self) -> bool:
        self.health_calls += 1
        return True

    async def close(self) -> None:
        self.close_calls += 1

    async def decide(self, state, _questions, *, timeout=None):
        self.states.append(state)
        self.questions.append(_questions)
        choice = self.destinations.pop(0) if len(self.destinations) > 1 else self.destinations[0]
        return DecisionResponse(
            answers={
                "destination": DecisionAnswer(
                    choice,
                    1.0,
                    {name: float(name == choice) for name in _questions["destination"].criteria},
                )
            },
            model="fake",
            usage={},
            raw={},
        )


def _make_router(
    llm,
    frames,
    vision,
    *,
    destination: str | tuple[str, ...] = "observation",
    include_vision: bool = True,
    summary_context_topics: tuple[str, ...] = (),
):
    manager = ContextManager(llm, summary_context_topics=summary_context_topics)
    conversation = Conversation(
        llm,
        context=manager,
        instructions=_CONVERSATION_INSTRUCTIONS,
        runtime=CaptureRuntime(),
        output_topic=VOICE_OUTPUT_TOPIC,
    )  # type: ignore[arg-type]
    destinations = {"conversation": conversation}
    conversations = [conversation]
    if include_vision:
        destinations["observation"] = VisualObservationAgent(
            frames,  # type: ignore[arg-type]
            vision,  # type: ignore[arg-type]
            runtime=CaptureRuntime(),
            output_topic=VOICE_OUTPUT_TOPIC,
        )
    decisions = _RouteDecisions(destination)
    router = AgentRouter(
        destinations,
        decisions,  # type: ignore[arg-type]
    )
    return router, tuple(conversations), manager


def _make_agent(
    runtime,
    llm,
    frames,
    vision,
    *,
    destination="observation",
    include_vision=True,
    set_status=_ignore_status,
):
    manager = ContextManager(llm)
    conversation = Conversation(
        llm,
        context=manager,
        instructions=_CONVERSATION_INSTRUCTIONS,
        runtime=runtime,
        output_topic=VOICE_OUTPUT_TOPIC,
    )
    destinations = {"conversation": conversation}
    if include_vision:
        destinations["observation"] = VisualObservationAgent(
            frames, vision, runtime=runtime, output_topic=VOICE_OUTPUT_TOPIC
        )

    async def cleanup(participant_id):
        await manager.clear(participant_id)
        if frames is not None:
            frames.release(participant_id)

    return AgentRouter(
        destinations,
        _RouteDecisions(destination),
        query_topic=USER_QUERY_TOPIC,
        output_topic=VOICE_OUTPUT_TOPIC,
        participant_joined_topic=PARTICIPANT_JOINED_TOPIC,
        participant_left_topic=PARTICIPANT_LEFT_TOPIC,
        interrupted_topic=INTERRUPTED_TOPIC,
        cleanup_participant=cleanup,
        set_status=set_status,
    )


async def test_conversation_is_text_only_and_does_not_select_a_frame() -> None:
    llm = _ConversationService()
    manager = ContextManager(llm)
    conversation = Conversation(
        llm,
        context=manager,
        runtime=CaptureRuntime(),
        output_topic=VOICE_OUTPUT_TOPIC,
    )  # type: ignore[arg-type]

    answer = await _answer(conversation, AgentTurn("alice", "turn-1", "Say hello"), emit=_ignore_output)

    assert answer.text == "Hello there."
    assert llm.requests[0][1] is None
    assert "Application instructions:" not in llm.requests[0][0][0].content


async def test_application_instructions_and_routing_metadata_are_separate_inputs() -> None:
    llm = _ConversationService()
    instructions = (
        "Answer general questions about museum exhibits. If asked for a medical diagnosis, "
        "say you cannot diagnose and suggest consulting a clinician."
    )
    routing = RoutingAid(
        name="museum_guide",
        description="Questions about museum exhibits and visible exhibit labels",
        priority=3,
    )
    manager = ContextManager(llm)
    conversation = Conversation(
        llm,
        context=manager,
        instructions=instructions,
        routing=routing,
        runtime=CaptureRuntime(),
        output_topic=VOICE_OUTPUT_TOPIC,
    )

    await _answer(conversation, AgentTurn("alice", "turn-1", "What is this?"), emit=_ignore_output)
    aid = conversation.routing_aid("alice")

    assert f"Application instructions:\n{instructions}" in llm.requests[0][0][0].content
    assert "medical diagnosis" in llm.requests[0][0][0].content
    assert aid == routing


@pytest.mark.asyncio
async def test_conversation_route_uses_the_sdk_conversation_directly() -> None:
    llm = _ConversationService()
    manager = ContextManager(llm)
    destination = Conversation(
        llm,
        context=manager,
        routing=RoutingAid(name="conversation", description="General conversation and follow-up help."),
        runtime=CaptureRuntime(),
        output_topic=VOICE_OUTPUT_TOPIC,
    )
    emitted = []

    async def emit(source, text):
        emitted.append((source, text))

    result = await _answer(
        destination,
        AgentTurn("alice", "turn-1", "Can you explain that?"),
        emit=emit,
    )
    aid = destination.routing_aid("alice")

    assert result.disposition == "answer"
    assert aid.description == "General conversation and follow-up help."
    assert "".join(text for source, text in emitted if source == "conversation") == "Hello there."

    await destination.release("alice")
    assert destination.routing_aid("alice") == aid
    await destination.participant_left("alice")


@pytest.mark.asyncio
async def test_conversation_keeps_early_context_after_interleaved_router_history_eviction() -> None:
    llm = _ConversationService()
    manager = ContextManager(llm, dialogue_budget_bytes=16_000, compaction_input_budget_bytes=32_000)
    conversation = Conversation(
        llm,
        context=manager,
        routing=RoutingAid(name="conversation", description="General conversation."),
        runtime=CaptureRuntime(),
        output_topic=VOICE_OUTPUT_TOPIC,
    )  # type: ignore[arg-type]

    class Detour(RoutableAgent):
        def routing_aid(self, _participant_id):
            return RoutingAid(name="detour", description="Handle an unrelated request.")

        async def admit(self, _turn, *, focused):
            return Admission("accept")

        async def answer(self, _turn, *, started):
            return DestinationResult("answer", "Handled the detour.")

        async def release(self, _participant_id):
            pass

        async def participant_left(self, _participant_id):
            pass

    detour_count = 8
    decisions = _RouteDecisions(("conversation", *(["detour"] * detour_count), "conversation"))
    router = AgentRouter(
        {"conversation": conversation, "detour": Detour()},
        decisions,  # type: ignore[arg-type]
        context_budget_bytes=512,
    )

    async def emit(_source, _text):
        return None

    first_query = "Please remember that the project label is Harbor."
    await _handle(router, AgentTurn("alice", "first", first_query), emit=emit)
    for index in range(detour_count):
        await _handle(
            router,
            AgentTurn("alice", f"detour-{index}", f"Unrelated detour question number {index}."),
            emit=emit,
        )
    await _handle(
        router,
        AgentTurn("alice", "follow-up", "What project label did I ask you to remember?"),
        emit=emit,
    )

    recent_router_history = decisions.states[-1]["completed_history"]
    final_model_input = str(llm.requests[-1][0])
    assert first_query not in str(recent_router_history)
    assert first_query in final_model_input
    await conversation.close()


class _NoteDecisions:
    """Answer each note-action decision with the next scripted choice."""

    def __init__(self, *choices: str) -> None:
        self.choices = iter(choices)
        self.states = []

    async def decide(self, state, questions, *, timeout=None):
        self.states.append(state)
        choice = next(self.choices)
        assert choice in questions["note_action"].criteria
        return DecisionResponse(
            answers={"note_action": DecisionAnswer(choice, 1.0, {choice: 1.0})}, model="fake", usage={}, raw={}
        )


def _notes(decisions, publish_note=None) -> NoteTakingAgent:
    return NoteTakingAgent(
        decisions,  # type: ignore[arg-type]
        publish_note=publish_note,
        runtime=CaptureRuntime(),
        output_topic=VOICE_OUTPUT_TOPIC,
    )


async def _ignore(_source, _text):
    return None


_NOTE = "The demo showed an AR overlay."


@pytest.mark.asyncio
async def test_notes_capture_readback_and_finish_until_departure() -> None:
    decisions = _NoteDecisions("start", "capture", "readback", "finish")
    published = []

    async def publish_note(participant_id, update):
        published.append((participant_id, update.text))

    destination = _notes(decisions, publish_note)
    emitted = []

    async def emit(source, text):
        emitted.append((source, text))

    start = AgentTurn("alice", "start", "Take some notes.")
    assert (await destination.admit(start, focused=False)).disposition == "accept"
    assert (await _answer(destination, start, emit=emit)).disposition == "focus"

    capture = AgentTurn("alice", "capture", _NOTE)
    assert (await destination.admit(capture, focused=True)).disposition == "accept"
    assert (await _answer(destination, capture, emit=emit)).disposition == "focus"

    readback = AgentTurn("alice", "readback", "Read back my notes.")
    assert (await destination.admit(readback, focused=True)).disposition == "accept"
    reviewed = await _answer(destination, readback, emit=emit)
    assert reviewed.disposition == "focus"
    assert reviewed.text == f"Your notes: {_NOTE}"

    finish = AgentTurn("alice", "finish", "That's all for the notes.")
    assert (await destination.admit(finish, focused=True)).disposition == "accept"
    assert (await _answer(destination, finish, emit=emit)).disposition == "release"
    assert len(decisions.states) == 4
    assert [source for source, _ in emitted] == ["notes"] * 4
    # Only the saved note is shared; session controls publish nothing.
    assert published == [("alice", _NOTE)]
    await destination.participant_left("alice")
    assert destination._read_back("alice") == "There are no notes in this session yet."  # noqa: SLF001


@pytest.mark.asyncio
async def test_notes_read_back_after_the_session_finishes_without_refocusing() -> None:
    destination = _notes(_NoteDecisions("start", "capture", "finish", "readback"))
    for turn_id, query, focused in (
        ("start", "Take some notes.", False),
        ("capture", _NOTE, True),
        ("finish", "That's all for the notes.", True),
    ):
        turn = AgentTurn("alice", turn_id, query)
        assert (await destination.admit(turn, focused=focused)).disposition == "accept"
        await _answer(destination, turn, emit=_ignore)

    readback = AgentTurn("alice", "readback", "Read back my notes.")
    assert (await destination.admit(readback, focused=False)).disposition == "accept"
    result = await _answer(destination, readback, emit=_ignore)
    assert result == DestinationResult("answer", f"Your notes: {_NOTE}")


@pytest.mark.asyncio
async def test_note_capture_drains_context_publish_on_cancel() -> None:
    entered_publish = asyncio.Event()
    finish_publish = asyncio.Event()
    published = []

    async def publish_note(participant_id, update):
        entered_publish.set()
        await finish_publish.wait()
        published.append((participant_id, update.text))

    destination = _notes(_NoteDecisions("capture"), publish_note)
    turn = AgentTurn("alice", "capture", _NOTE)
    assert (await destination.admit(turn, focused=True)).disposition == "accept"

    task = asyncio.create_task(_answer(destination, turn, emit=_ignore))
    await entered_publish.wait()
    task.cancel()
    finish_publish.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert published == [("alice", _NOTE)]
    assert destination._read_back("alice") == f"Your notes: {_NOTE}"  # noqa: SLF001


@pytest.mark.asyncio
@pytest.mark.parametrize("release_before_cancel", [False, True])
async def test_note_capture_logs_child_publish_failure_and_preserves_caller_cancellation(
    release_before_cancel: bool,
) -> None:
    entered_publish = asyncio.Event()
    release_publisher = asyncio.Event()
    publisher_failed = asyncio.Event()
    log_stream = io.StringIO()

    async def failing_publisher(_participant_id, _update):
        entered_publish.set()
        await release_publisher.wait()
        publisher_failed.set()
        raise RuntimeError("child publisher failed")

    destination = _notes(_NoteDecisions("capture"), failing_publisher)
    turn = AgentTurn("alice", "capture", _NOTE)
    assert (await destination.admit(turn, focused=True)).disposition == "accept"

    sink_id = logger.add(log_stream, format="{message}\n{exception}")
    try:
        task = asyncio.create_task(_answer(destination, turn, emit=_ignore))
        await entered_publish.wait()
        if release_before_cancel:
            release_publisher.set()
            await publisher_failed.wait()
            task.cancel()
        else:
            task.cancel()
            release_publisher.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        logger.remove(sink_id)

    assert "child publisher failed" in log_stream.getvalue()
    assert "note context publication failed while draining cancellation" in log_stream.getvalue()
    assert destination._read_back("alice") == f"Your notes: {_NOTE}"  # noqa: SLF001


@pytest.mark.asyncio
async def test_note_capture_propagates_programming_error_from_context_publisher() -> None:
    async def broken_publisher(_participant_id, _update):
        raise RuntimeError("publisher programming error")

    destination = _notes(_NoteDecisions("capture"), broken_publisher)
    turn = AgentTurn("alice", "capture", _NOTE)
    assert (await destination.admit(turn, focused=True)).disposition == "accept"

    with pytest.raises(RuntimeError, match="publisher programming error"):
        await _answer(destination, turn, emit=_ignore)

    assert destination._read_back("alice") == f"Your notes: {_NOTE}"  # noqa: SLF001


@pytest.mark.asyncio
async def test_note_capture_logs_runtime_publisher_exception_group_and_keeps_note() -> None:
    note_topic = Topic("conversation-example.test-notes", ConversationContextUpdate)

    class FailingSubscriber(Agent):
        @subscribe(note_topic)
        async def fail(self, _update: ConversationContextUpdate, _ctx) -> None:
            raise RuntimeError("subscriber programming error")

    runtime = AgentRuntime()
    runtime.register("failing-context-subscriber", FailingSubscriber())

    async def publish_note(participant_id, update):
        await runtime.publish(note_topic, update, participant_id=participant_id)

    destination = _notes(_NoteDecisions("capture"), publish_note)
    turn = AgentTurn("alice", "capture", _NOTE)
    log_stream = io.StringIO()
    sink_id = logger.add(log_stream, format="{message}\n{exception}")
    try:
        async with runtime:
            assert (await destination.admit(turn, focused=True)).disposition == "accept"
            result = await _answer(destination, turn, emit=_ignore)
    finally:
        logger.remove(sink_id)

    assert result.disposition == "focus"
    assert "couldn't share it" in result.text
    assert destination._read_back("alice") == f"Your notes: {_NOTE}"  # noqa: SLF001
    assert "note context publication failed" in log_stream.getvalue()
    assert "subscriber programming error" in log_stream.getvalue()


@pytest.mark.asyncio
async def test_sample_departure_callback_clears_notes_after_router_release() -> None:
    manager = ContextManager(_ConversationService())
    conversation = Conversation(
        _ConversationService(),
        context=manager,
        runtime=CaptureRuntime(),
        output_topic=VOICE_OUTPUT_TOPIC,
    )
    notes = _notes(_NoteDecisions("capture"))
    capture = AgentTurn("alice", "capture", _NOTE)
    await notes.admit(capture, focused=True)
    await _answer(notes, capture, emit=_ignore)
    runtime = AgentRuntime()
    router = AgentRouter(
        {"conversation": conversation},
        _RouteDecisions("conversation"),
        query_topic=USER_QUERY_TOPIC,
        output_topic=VOICE_OUTPUT_TOPIC,
        participant_joined_topic=PARTICIPANT_JOINED_TOPIC,
        participant_left_topic=PARTICIPANT_LEFT_TOPIC,
        cleanup_participant=notes.participant_left,
    )
    runtime.register("router", router.runtime_subscriber())
    async with runtime:
        await runtime.publish(PARTICIPANT_LEFT_TOPIC, ParticipantLeft(), participant_id="alice")
        assert notes._read_back("alice") == "There are no notes in this session yet."  # noqa: SLF001
        await router.stop()
    await conversation.close()


@pytest.mark.asyncio
async def test_router_departure_error_does_not_skip_sample_cleanup() -> None:
    calls = []

    class Destination:
        def routing_aid(self, _participant):
            return RoutingAid("conversation", "General questions")

        async def admit(self, _turn, *, focused):
            return Admission("accept")

        async def answer(self, _turn, *, started):
            return DestinationResult("answer", "done")

        async def release(self, _participant):
            pass

        async def participant_left(self, _participant):
            calls.append("router")
            raise RuntimeError("release failed")

    async def cleanup(_participant):
        calls.extend(("notes", "conversation", "frames"))

    router = AgentRouter(
        {"conversation": Destination()},
        _RouteDecisions("conversation"),
        query_topic=USER_QUERY_TOPIC,
        output_topic=VOICE_OUTPUT_TOPIC,
        cleanup_participant=cleanup,
    )
    await router.handle(AgentTurn("alice", "turn", "Hello"))
    with pytest.raises(ExceptionGroup, match="failed to release destinations"):
        await router.participant_left("alice")
    assert calls == ["router", "notes", "conversation", "frames"]


async def test_domain_visual_request_routes_to_direct_observation_agent() -> None:
    class Decisions:
        def __init__(self) -> None:
            self.state = None

        async def decide(self, state, _questions, *, timeout=None):
            self.state = state
            return DecisionResponse(
                answers={"destination": DecisionAnswer("observation", 1.0, {"observation": 1.0})},
                model="fake",
                usage={},
                raw={},
            )

    decisions = Decisions()
    frames = _ConversationFrames()
    vision = _StreamingImageQueryTool()
    observation = VisualObservationAgent(
        frames,  # type: ignore[arg-type]
        vision,  # type: ignore[arg-type]
        runtime=CaptureRuntime(),
        output_topic=VOICE_OUTPUT_TOPIC,
    )
    manager = ContextManager(_ConversationService())
    general = Conversation(
        _ConversationService(),
        context=manager,
        runtime=CaptureRuntime(),
        output_topic=VOICE_OUTPUT_TOPIC,
    )  # type: ignore[arg-type]
    router = AgentRouter(
        {"conversation": general, "observation": observation},
        decisions,  # type: ignore[arg-type]
    )
    output = []

    async def emit(source, text):
        output.append((source, text))

    result = await _handle(
        router,
        AgentTurn("alice", "museum-1", "Read the label on this exhibit."),
        emit=emit,
    )

    assert result.text == "a blue square"
    assert output == [("observation", "a "), ("observation", "blue square")]
    assert frames.requests[0].participant_id == "alice"
    assert "Read the label on this exhibit." in vision.requests[0].query
    observation_aid = next(item for item in decisions.state["destinations"] if item["name"] == "observation")
    assert "current camera frame" in observation_aid["description"]


async def test_text_and_visual_destinations_share_router_history_and_output_sources() -> None:
    class LLM:
        def __init__(self) -> None:
            self.requests = []

        def stream_events(self, messages, *, tools=None, **_kwargs):
            self.requests.append((messages, tools))
            call = len(self.requests)

            async def events():
                if call == 1:
                    yield ChatStreamEvent(kind="text", text="Lisbon.")
                    yield ChatStreamEvent(kind="finish", finish_reason="stop")
                else:
                    yield ChatStreamEvent(kind="text", text="The sign is blue.")
                    yield ChatStreamEvent(kind="finish", finish_reason="stop")

            return events()

    llm = LLM()
    frames = _ConversationFrames()
    vision = _StreamingImageQueryTool()
    manager = ContextManager(llm)
    general = Conversation(
        llm,
        context=manager,
        instructions=_CONVERSATION_INSTRUCTIONS,
        routing=RoutingAid("conversation", "General conversation and recall"),
        runtime=CaptureRuntime(),
        output_topic=VOICE_OUTPUT_TOPIC,
    )
    visual = VisualObservationAgent(
        frames,  # type: ignore[arg-type]
        vision,  # type: ignore[arg-type]
        runtime=CaptureRuntime(),
        output_topic=VOICE_OUTPUT_TOPIC,
    )
    decisions = _RouteDecisions(("conversation", "observation", "conversation"))
    router = AgentRouter(
        {"conversation": general, "observation": visual},
        decisions,  # type: ignore[arg-type]
    )
    output = []

    async def emit(source, text):
        output.append((source, text))

    first = await _handle(
        router,
        AgentTurn("alice", "turn-1", "What is the capital of Portugal?"),
        emit=emit,
    )
    second = await _handle(
        router,
        AgentTurn("alice", "turn-2", "What color is the sign?"),
        emit=emit,
    )
    third = await _handle(
        router,
        AgentTurn("alice", "turn-3", "And what did I first ask?"),
        emit=emit,
    )

    assert first.text == "Lisbon."
    assert second.text == "a blue square"
    assert third.text == "The sign is blue."
    assert llm.requests[0][1] is None
    prompt = "\n".join(str(message.content) for message in llm.requests[1][0])
    assert "What is the capital of Portugal?" in prompt
    assert "What color is the sign?" in prompt
    assert "Assistant: “a blue square”" in prompt
    assert frames.requests[0].participant_id == "alice"
    assert "What color is the sign?" in vision.requests[0].query
    assert [source for source, _text in output] == ["conversation", "observation", "observation", "conversation"]


async def test_conversation_publishes_answer_before_model_stream_finishes() -> None:
    gate = asyncio.Event()
    llm = _ConversationService(stream_gate=gate)
    manager = ContextManager(llm)
    conversation = Conversation(
        llm,
        context=manager,
        runtime=CaptureRuntime(),
        output_topic=VOICE_OUTPUT_TOPIC,
    )  # type: ignore[arg-type]
    first_publication = asyncio.Event()
    published = []

    async def emit(source, text):
        published.append((source, text))
        first_publication.set()

    answer_task = asyncio.create_task(_answer(conversation, AgentTurn("alice", "turn-1", "Say hello"), emit=emit))
    await asyncio.wait_for(first_publication.wait(), timeout=1)

    assert published == [("conversation", "Hello ")]
    assert not gate.is_set()
    assert not answer_task.done()
    gate.set()
    answer = await answer_task
    assert answer.text == "Hello there."


async def test_conversation_example_agent_reports_one_safe_fallback_on_llm_failure() -> None:
    published = []
    control = []

    class Context:
        agent_name = "conversation-example"
        metadata = SimpleNamespace(
            message_id="turn-failed",
            correlation_id="turn-failed",
            participant_id="alice",
        )

        async def publish(self, topic, output, **_kwargs) -> None:
            published.append((topic, output))

    llm = _ConversationService(error=RuntimeError("offline"))
    agent = _make_agent(
        Context(),
        llm,
        _ConversationFrames(),
        _StreamingImageQueryTool(),
        destination="conversation",
    )

    async def capture(source, text):
        control.append((source, text))

    with pytest.raises(RuntimeError, match="offline"):
        await _handle(agent, AgentTurn("alice", "turn-failed", "Hello", timestamp_us=456), emit=capture)

    # The router reports a failure before any output; the destination publishes nothing.
    assert published == []
    assert control == [("conversation", "I couldn't complete that request. Please try again.")]
    assert "You are a fast voice conversational assistant" in llm.requests[0][0][0].content


class _LiveEndpoint:
    def __init__(self) -> None:
        self.frame_callback = None
        self.statuses: list[tuple[str, str]] = []

    def on_frame(self, callback) -> None:
        self.frame_callback = callback

    def _on_image_capture(self, _callback):
        return lambda: None

    @property
    def connected_participants(self):
        return frozenset({"alice"})

    def on_participant(self, _callback) -> None:
        return None

    async def request_frame(self, signal: FrameSignal) -> FrameData:
        return FrameData(
            seq=signal.seq,
            pts_us=signal.pts_us,
            width=2,
            height=2,
            fmt=PixelFormat.RGB24,
            data=bytes([20, 40, 60] * 4),
            participant_id=signal.participant_id,
            track_id=signal.track_id,
        )

    async def set_status(self, status: str, participant_id: str) -> None:
        self.statuses.append((status, participant_id))


class _StreamingVlm:
    def __init__(self) -> None:
        self.calls = []
        self.ask_calls = []

    async def ask_images(
        self,
        images,
        question: str,
        *,
        system_prompt: str = "",
        headers=None,
    ) -> ChatResponse:
        self.ask_calls.append((images, question, system_prompt, dict(headers or {})))
        return ChatResponse("a blue square", None, None, "stop", {})

    async def stream_images(
        self,
        images,
        question: str,
        *,
        system_prompt: str = "",
        headers=None,
    ):
        self.calls.append((images, question, system_prompt, dict(headers or {})))
        for token in ("a ", "blue ", "square"):
            yield token


def test_worker_distribution_declares_runtime_and_console_contracts() -> None:
    project = tomllib.loads((_WORKER_DIR / "pyproject.toml").read_text())
    dependencies = set(project["project"]["dependencies"])

    assert project["project"]["scripts"]["conversation_example_worker"] == ("conversation_example_worker.__main__:run")
    assert "xr-ai-hub-client" in dependencies
    assert "xr-ai-agent-runtime" in dependencies
    assert "xr-ai-conversation" in dependencies
    assert "xr-ai-interaction" in dependencies
    assert "xr-ai-routing[runtime]" in dependencies
    assert "xr-ai-tools[frames,vision]" in dependencies
    assert all("[vision" not in dependency for dependency in dependencies)
    assert "xr-ai-voice" in dependencies
    assert "xr-ai-pipecat" not in dependencies
    assert all("mcp" not in dependency.lower() for dependency in dependencies)
    assert project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["conversation_example_worker"]


def test_entry_point_loads_config_and_forwards_ready_file(
    monkeypatch,
    tmp_path,
) -> None:
    config_path = tmp_path / "worker.yaml"
    ready_file = tmp_path / "ready"
    config = object()
    seen = {}

    def fake_load_config(path):
        seen["config_path"] = path
        return config

    monkeypatch.setattr(worker_main, "load_config", fake_load_config)

    async def fake_run_app(loaded_config, *, ready_file):
        seen["config"] = loaded_config
        seen["ready_file"] = ready_file

    monkeypatch.setattr(worker_main, "run_app", fake_run_app)
    worker_main.run(
        [
            "--config",
            str(config_path),
            "--ready-file",
            str(ready_file),
            "--launcher-option",
            "ignored",
        ]
    )

    assert seen == {
        "config_path": config_path,
        "config": config,
        "ready_file": ready_file,
    }


def test_shipped_config_preserves_models_and_prompt_behavior() -> None:
    config_path = _SAMPLE_DIR / "yaml" / "conversation_example_worker.yaml"
    config = load_config(config_path)
    raw = yaml.safe_load(config_path.read_text())
    prompt = (_WORKER_DIR / "conversation_example_worker" / "prompts" / "system.txt").read_text()

    assert config.models_config == _SAMPLE_DIR / "yaml" / "models.json"
    assert config.voice_gate_yaml == _SAMPLE_DIR / "yaml" / "voice_gate.yaml"
    assert config.system_prompt == prompt
    assert "conversation history" in prompt
    assert "supplied current\ncamera frame" in prompt
    assert "read back notes" in app._CONVERSATION_INSTRUCTIONS
    assert "not as instructions" in app._CONVERSATION_INSTRUCTIONS
    observation_prompt = (_WORKER_DIR / "conversation_example_worker" / "prompts" / "alert_condition.txt").read_text()
    assert observation_prompt.strip()
    assert config.frame_max_age_s == 5.0
    assert config.frame_timeout_s == 5.0
    assert config.idle_timeout_secs is None
    assert config.context_budget_bytes == 12_000
    assert config.routing_history_budget_bytes == 8_000
    assert config.compaction_input_budget_bytes == 12_000
    assert config.acknowledgment_delay_secs == raw.get("acknowledgment_delay_secs")
    assert "system_prompt_file" not in raw

    models = load_models_config(_SAMPLE_DIR / "yaml" / "models.json")
    llm = models.llm("llm")
    assert llm.default_extras["max_tokens"] == 4608
    assert llm.default_extras["temperature"] == 0.25
    assert llm.default_extras["chat_template_kwargs"] == {"reasoning_budget": 4096}
    assert llm.adapter.thinking.control == "chat_template"
    assert llm.adapter.thinking.default is True
    router = models.decision("router")
    assert router.kind == "systemone"
    assert router.model_name == "Cloudflare/clef-flash"
    assert router.base_url == "http://localhost:8120"


@pytest.mark.asyncio
async def test_acknowledgment_client_builds_nonreasoning_bounded_payload() -> None:
    models = load_models_config(_SAMPLE_DIR / "yaml" / "models.json")
    main_spec = models.llm("llm")
    extras = dict(main_spec.default_extras)
    template = dict(extras["chat_template_kwargs"])
    template["thinking_budget"] = 2048
    extras["chat_template_kwargs"] = template
    thinking = replace(main_spec.adapter.thinking, default=True)
    customized_main_spec = replace(
        main_spec,
        adapter=replace(main_spec.adapter, default_extras=extras, thinking=thinking),
    )
    customized_models = replace(models, entries={**models.entries, "llm": customized_main_spec})
    acknowledgment_models = app._acknowledgment_models(customized_models)
    acknowledgment_spec = acknowledgment_models.llm("acknowledgment_llm")
    assert acknowledgment_spec.adapter.default_extras == {"temperature": 0.25}
    assert acknowledgment_spec.adapter.thinking.control == "chat_template"
    assert acknowledgment_spec.adapter.thinking.default is False
    assert "max_tokens" not in acknowledgment_spec.default_extras

    client = app.make_llm(acknowledgment_models, "acknowledgment_llm")
    try:
        payload = client._build_payload(
            [],
            tools=None,
            max_tokens=64,
            temperature=None,
            enable_thinking=None,
            thinking_budget=None,
            stream=False,
        )
    finally:
        await client.close()

    assert payload["max_tokens"] == 64
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_budget" not in payload["chat_template_kwargs"]
    assert customized_models.llm("llm").adapter.thinking.default is True
    assert customized_models.llm("llm").default_extras["chat_template_kwargs"] == {
        "reasoning_budget": 4096,
        "thinking_budget": 2048,
    }


def test_config_without_a_file_uses_packaged_defaults(tmp_path) -> None:
    prompt = (_WORKER_DIR / "conversation_example_worker" / "prompts" / "system.txt").read_text()

    for config_path in (None, tmp_path / "missing.yaml"):
        config = load_config(config_path)
        assert config.system_prompt == prompt
        expected_parent = Path() if config_path is None else tmp_path
        assert config.models_config == expected_parent / "models.json"


def test_blank_inline_prompt_uses_packaged_default(tmp_path) -> None:
    config_path = tmp_path / "worker.yaml"
    config_path.write_text("system_prompt:\n")
    prompt = (_WORKER_DIR / "conversation_example_worker" / "prompts" / "system.txt").read_text()

    assert load_config(config_path).system_prompt == prompt


def test_config_keeps_inline_prompt_compatibility(tmp_path) -> None:
    config_path = tmp_path / "worker.yaml"
    config_path.write_text("voice_gate_yaml: gate.yaml\nsystem_prompt: custom prompt\nidle_timeout_secs: 30\n")

    config = load_config(config_path)

    assert config.models_config == tmp_path / "models.json"
    assert config.voice_gate_yaml == tmp_path / "gate.yaml"
    assert config.system_prompt == "custom prompt"
    assert config.idle_timeout_secs == 30.0


@pytest.mark.parametrize("delay", [0, -1, "nan", ".inf"])
def test_config_rejects_invalid_acknowledgment_delay(tmp_path, delay) -> None:
    config_path = tmp_path / "worker.yaml"
    config_path.write_text(f"acknowledgment_delay_secs: {delay}\n")

    with pytest.raises(ValueError, match="acknowledgment_delay_secs"):
        load_config(config_path)


def test_config_accepts_positive_acknowledgment_delay(tmp_path) -> None:
    config_path = tmp_path / "worker.yaml"
    config_path.write_text("acknowledgment_delay_secs: 2.5\n")

    assert load_config(config_path).acknowledgment_delay_secs == 2.5


def test_alert_intervals_default_and_can_be_configured(tmp_path) -> None:
    default = load_config(None)
    assert (default.alert_check_interval_secs, default.alert_confirm_interval_secs) == (1.0, 3.0)

    config_path = tmp_path / "worker.yaml"
    config_path.write_text("alert_check_interval_secs: 0\nalert_confirm_interval_secs: 10\n")
    config = load_config(config_path)
    assert (config.alert_check_interval_secs, config.alert_confirm_interval_secs) == (0.0, 10.0)


@pytest.mark.parametrize(
    "settings",
    [
        "alert_check_interval_secs: -1",
        "alert_check_interval_secs: 5\nalert_confirm_interval_secs: 2",
        "alert_confirm_interval_secs: .inf",
        "alert_check_interval_secs: nan",
    ],
)
def test_config_rejects_invalid_alert_intervals(tmp_path, settings) -> None:
    config_path = tmp_path / "worker.yaml"
    config_path.write_text(settings + "\n")

    with pytest.raises(ValueError, match="alert intervals"):
        load_config(config_path)


def test_config_rejects_a_non_mapping_yaml_document(tmp_path) -> None:
    config_path = tmp_path / "worker.yaml"
    config_path.write_text("- not\n- a\n- mapping\n")

    with pytest.raises(ValueError, match="must be a YAML mapping"):
        load_config(config_path)


async def test_conversation_example_agent_closes_tool_stream_when_publication_fails() -> None:
    closed = asyncio.Event()

    class Stream:
        def __init__(self) -> None:
            self.sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                await asyncio.Event().wait()
            self.sent = True
            return SimpleNamespace(text="first")

        async def aclose(self) -> None:
            closed.set()

    class Vision:
        def stream(self, _request):
            return Stream()

    class Context:
        agent_name = "conversation-example"
        metadata = SimpleNamespace(
            message_id="turn-1",
            correlation_id="turn-1",
            participant_id="alice",
        )

        async def publish(self, *_args, **_kwargs) -> None:
            raise RuntimeError("runtime stopped")

    agent = _make_agent(Context(), _ConversationService(), _SelectedFrameTool(), Vision())
    with pytest.raises(RuntimeError, match="runtime stopped"):
        await agent.handle(AgentTurn("alice", "turn-1", "What is shown?", timestamp_us=123))

    assert closed.is_set()


async def test_conversation_example_agent_reports_typed_missing_camera_frame() -> None:
    published: list[ResponseOutput] = []

    unavailable = FrameUnavailable("No camera frame available — please try again.")

    async def missing_frame(_request: CurrentFrameRequest) -> ImageFrame:
        raise unavailable

    missing_frame_tool = Tool(
        "missing_camera_frame",
        "Reproduce a missing camera frame through the real Relay boundary.",
        CurrentFrameRequest,
        ImageFrame,
        missing_frame,
    )
    with pytest.raises(FrameUnavailable) as raised:
        await missing_frame_tool.execute(CurrentFrameRequest(participant_id="alice"))
    assert raised.value is unavailable

    class Context:
        agent_name = "conversation-example"
        metadata = SimpleNamespace(
            message_id="turn-1",
            correlation_id="turn-1",
            participant_id="alice",
        )

        async def publish(self, _topic, output, **_kwargs) -> None:
            published.append(output)

    agent = _make_agent(Context(), _ConversationService(), missing_frame_tool, _StreamingImageQueryTool())

    await agent.handle(AgentTurn("alice", "turn-1", "What is shown?", timestamp_us=123))

    assert [output.text for output in published] == [
        "I couldn't get a current camera frame. Please try again.",
        "",
    ]
    assert [output.final for output in published] == [False, True]


async def test_cancelled_vlm_turn_closes_its_stream_without_late_text() -> None:
    waiting = asyncio.Event()
    closed = asyncio.Event()
    published: list[ResponseOutput] = []

    class Stream:
        def __init__(self) -> None:
            self.sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.sent:
                self.sent = True
                return SimpleNamespace(text="first")
            waiting.set()
            await asyncio.Event().wait()
            raise StopAsyncIteration

        async def aclose(self) -> None:
            closed.set()

    class Vision:
        def stream(self, _request):
            return Stream()

    class Context:
        agent_name = "conversation-example"
        metadata = SimpleNamespace(
            message_id="turn-1",
            correlation_id="turn-1",
            participant_id="alice",
        )

        async def publish(self, _topic, output, **_kwargs) -> None:
            published.append(output)

    agent = _make_agent(Context(), _ConversationService(), _SelectedFrameTool(), Vision())

    task = asyncio.create_task(agent.handle(AgentTurn("alice", "turn-1", "What is shown?", timestamp_us=123)))
    await asyncio.wait_for(waiting.wait(), 1.0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert closed.is_set()
    assert [(output.text, output.final) for output in published] == [("first", False), ("", True)]


@pytest.mark.asyncio
async def test_context_updates_apply_after_participant_rejoin() -> None:
    llm = _ConversationService()
    router, conversations, manager = _make_router(
        llm,
        None,
        None,
        destination="conversation",
        summary_context_topics=("tea.status",),
    )
    (conversation,) = conversations

    async def set_status(_status: str, _participant_id: str) -> None:
        return None

    async def cleanup(participant_id):
        await manager.clear(participant_id)

    agent = AgentRouter(
        {"conversation": conversation},
        _RouteDecisions("conversation"),
        query_topic=USER_QUERY_TOPIC,
        output_topic=VOICE_OUTPUT_TOPIC,
        participant_joined_topic=PARTICIPANT_JOINED_TOPIC,
        participant_left_topic=PARTICIPANT_LEFT_TOPIC,
        cleanup_participant=cleanup,
        set_status=set_status,
    )
    context_topic = Topic("tea.status", ConversationContextUpdate)
    runtime = AgentRuntime()
    runtime.register("conversation-context", manager.subscriber())
    runtime.register("conversation-example", agent.runtime_subscriber())

    async with runtime:
        await runtime.publish(
            PARTICIPANT_JOINED_TOPIC,
            ParticipantJoined(),
            participant_id="alice",
        )
        await runtime.publish(
            context_topic,
            {"text": "Timer is at 4 minutes."},
            participant_id="alice",
        )
        first_answer = await _answer(
            conversation,
            AgentTurn("alice", "timer-1", "What is the timer?"),
            emit=_ignore_output,
        )
        first_prompt = "\n".join(str(message.content) for message in llm.requests[-1][0])
        assert first_answer.text == "Hello there."
        assert "Timer is at 4 minutes." in first_prompt

        await runtime.publish(
            PARTICIPANT_LEFT_TOPIC,
            ParticipantLeft(),
            participant_id="alice",
        )
        await runtime.publish(
            PARTICIPANT_JOINED_TOPIC,
            ParticipantJoined(),
            participant_id="alice",
        )
        await runtime.publish(
            context_topic,
            {"text": "Timer is at 6 minutes."},
            participant_id="alice",
        )
        await _answer(
            conversation,
            AgentTurn("alice", "timer-2", "What is the timer now?"),
            emit=_ignore_output,
        )
        second_prompt = "\n".join(str(message.content) for message in llm.requests[-1][0])
        assert "Timer is at 6 minutes." in second_prompt
        assert "Timer is at 4 minutes." not in second_prompt
    await conversation.close()


class _LifecycleDestination:
    def __init__(self, events):
        self.events = events
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    def routing_aid(self, _participant):
        return RoutingAid("conversation", "General questions")

    async def admit(self, _turn, *, focused):
        return Admission("accept")

    async def answer(self, _turn, *, started):
        self.events.append("answer.started")
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.events.append("answer.cancelled")
            self.cancelled.set()

    async def release(self, _participant):
        pass

    async def participant_left(self, _participant):
        self.events.append("destination.left")


def _lifecycle_router(destination, initialize, cleanup):
    return AgentRouter(
        {"conversation": destination},
        _RouteDecisions("conversation"),
        query_topic=USER_QUERY_TOPIC,
        output_topic=VOICE_OUTPUT_TOPIC,
        participant_joined_topic=PARTICIPANT_JOINED_TOPIC,
        participant_left_topic=PARTICIPANT_LEFT_TOPIC,
        interrupted_topic=INTERRUPTED_TOPIC,
        initialize_participant=initialize,
        cleanup_participant=cleanup,
    )


@pytest.mark.asyncio
async def test_query_initialization_and_response_admission_are_atomic_with_departure() -> None:
    initialization_started = asyncio.Event()
    release_initialization = asyncio.Event()
    events = []
    initialization_count = 0
    destination = _LifecycleDestination(events)

    async def initialize(_participant):
        nonlocal initialization_count
        initialization_count += 1
        events.append("context.initialized")
        initialization_started.set()
        await release_initialization.wait()

    async def cleanup(_participant):
        events.append("application.left")

    runtime = AgentRuntime()
    router = _lifecycle_router(destination, initialize, cleanup)
    runtime.register("router", router.runtime_subscriber())
    async with runtime:
        request = UserQuery(text="Hello", timestamp_us=1)
        query = asyncio.create_task(runtime.publish(USER_QUERY_TOPIC, request, participant_id="alice"))
        await initialization_started.wait()
        departure = asyncio.create_task(
            runtime.publish(PARTICIPANT_LEFT_TOPIC, ParticipantLeft(), participant_id="alice")
        )
        release_initialization.set()
        await asyncio.gather(query, departure)
        assert events.index("context.initialized") < events.index("application.left")
        if destination.started.is_set():
            assert events.index("answer.cancelled") < events.index("destination.left")
        await runtime.publish(USER_QUERY_TOPIC, request, participant_id="alice")
        assert initialization_count == 1
        await runtime.publish(PARTICIPANT_JOINED_TOPIC, ParticipantJoined(), participant_id="alice")
        assert initialization_count == 2
        await router.stop()


@pytest.mark.asyncio
async def test_interruption_cancels_query_admitted_after_blocked_context_initialization() -> None:
    initialization_started = asyncio.Event()
    release_initialization = asyncio.Event()
    destination = _LifecycleDestination([])

    async def initialize(_participant):
        initialization_started.set()
        await release_initialization.wait()

    runtime = AgentRuntime()
    router = _lifecycle_router(destination, initialize, None)
    runtime.register("router", router.runtime_subscriber())
    async with runtime:
        query = asyncio.create_task(
            runtime.publish(USER_QUERY_TOPIC, UserQuery(text="Hello", timestamp_us=1), participant_id="alice")
        )
        await initialization_started.wait()
        interruption = asyncio.create_task(runtime.publish(INTERRUPTED_TOPIC, Interrupted(), participant_id="alice"))
        release_initialization.set()
        await asyncio.gather(query, interruption)
        if destination.started.is_set():
            assert destination.cancelled.is_set()
        await router.stop()


@pytest.mark.asyncio
async def test_stop_closes_query_admission_while_context_initialization_is_in_flight() -> None:
    initialization_started = asyncio.Event()
    release_initialization = asyncio.Event()
    events = []
    destination = _LifecycleDestination(events)

    async def initialize(_participant):
        initialization_started.set()
        await release_initialization.wait()

    async def cleanup(_participant):
        events.append("application.left")

    runtime = AgentRuntime()
    router = _lifecycle_router(destination, initialize, cleanup)
    runtime.register("router", router.runtime_subscriber())
    async with runtime:
        query = asyncio.create_task(
            runtime.publish(USER_QUERY_TOPIC, UserQuery(text="Hello", timestamp_us=1), participant_id="alice")
        )
        await initialization_started.wait()
        shutdown = asyncio.create_task(router.stop())
        await asyncio.sleep(0)
        release_initialization.set()
        await asyncio.gather(query, shutdown)
        assert not destination.started.is_set()
        assert events == ["application.left"]


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_acknowledgment_publish", [False, True])
async def test_acknowledgment_publish_does_not_affect_the_answer(fail_acknowledgment_publish: bool) -> None:
    class Prediction:
        async def decide(self, *_args, **_kwargs):
            return DecisionResponse(
                answers={
                    "first_answer_timing": DecisionAnswer(
                        "slow", 0.9, {"slow": 0.9, "within_delay": 0.05, "uncertain": 0.05}
                    )
                },
                model="fake",
                usage={},
                raw={},
            )

    class AckLLM:
        async def chat(self, *_args, **_kwargs):
            return ChatResponse("Let me think through that.", None, None, "stop", {})

    runtime = AgentRuntime()
    release_answer = asyncio.Event()
    answer_sent = asyncio.Event()
    acknowledgment_started = asyncio.Event()
    release_acknowledgment = asyncio.Event()
    idle = asyncio.Event()
    outputs = []

    class Destination:
        def routing_aid(self, _participant):
            return RoutingAid("conversation", "General questions")

        async def admit(self, _turn, *, focused):
            return Admission("accept")

        async def answer(self, turn, *, started):
            await release_answer.wait()
            await runtime.publish(
                VOICE_OUTPUT_TOPIC,
                ResponseOutput(
                    kind="reply", text="Here is the answer.", request_id=turn.turn_id, timestamp_us=turn.timestamp_us
                ),
                participant_id=turn.participant_id,
                source="conversation",
            )
            answer_sent.set()
            return DestinationResult("answer", "Here is the answer.")

        async def release(self, _participant):
            pass

    class Output(Agent):
        @subscribe(VOICE_OUTPUT_TOPIC)
        async def output(self, message: ResponseOutput, ctx):
            if message.text == "Let me think through that. ":
                acknowledgment_started.set()
                await release_acknowledgment.wait()
                if fail_acknowledgment_publish:
                    raise RuntimeError("voice output rejected before delivery")
            outputs.append((ctx.metadata.source, message))

    async def status(state, _participant):
        if state == "idle":
            idle.set()

    router = AgentRouter(
        {"conversation": Destination()},
        Prediction(),
        acknowledgment_llm=AckLLM(),
        acknowledgment_delay_secs=10,
        query_topic=USER_QUERY_TOPIC,
        output_topic=VOICE_OUTPUT_TOPIC,
        set_status=status,
    )
    runtime.register("router", router.runtime_subscriber())
    runtime.register("output", Output())
    async with runtime:
        await runtime.publish(
            USER_QUERY_TOPIC, UserQuery(text="Explain this request.", timestamp_us=123), participant_id="alice"
        )
        await asyncio.wait_for(acknowledgment_started.wait(), 1)
        release_answer.set()
        await asyncio.wait_for(answer_sent.wait(), 1)
        release_acknowledgment.set()
        await asyncio.wait_for(idle.wait(), 1)
        await router.stop()

    body = [(source, message) for source, message in outputs if source == "conversation"]
    assert len(body) == 1
    assert body[0][1].text == "Here is the answer."
    assert body[0][1].timestamp_us == 123
    acknowledgments = [message for source, message in outputs if source == "router"]
    assert len(acknowledgments) == (0 if fail_acknowledgment_publish else 1)
    assert all(message.final and message.response_id is None for message in acknowledgments)


async def test_app_wires_text_voice_cleanup_readiness_and_shutdown(
    monkeypatch,
    tmp_path,
) -> None:
    config = replace(
        load_config(_SAMPLE_DIR / "yaml" / "conversation_example_worker.yaml"),
        acknowledgment_delay_secs=None,
    )
    ready_file = tmp_path / "ready"
    stt = _Service()
    shutdown_order = []

    class RecordingLLM(_ConversationService):
        async def close(self) -> None:
            shutdown_order.append("llm.close")
            await super().close()

    class RecordingConversation(Conversation):
        async def close(self) -> None:
            shutdown_order.append("conversation.close")
            await super().close()

    class RecordingContextManager(ContextManager):
        async def clear(self, participant_id: str) -> None:
            shutdown_order.append("context.clear")
            await super().clear(participant_id)

        async def close(self) -> None:
            shutdown_order.append("context.close")
            await super().close()

    llm = RecordingLLM()
    vlm = _Service()
    tts = _Service()
    decisions = _RouteDecisions(("observation", "conversation"))
    transport = _Transport()
    sessions: list[VoiceSession] = []
    run_options = {}
    responses = []
    response_tasks: list[asyncio.Task[None]] = []
    response_complete = asyncio.Event()
    runtimes = []
    wired_front_ends = []

    class RecordingRouter(AgentRouter):
        def __init__(self, destinations, decisions, **kwargs):
            wired_front_ends.append(destinations)
            super().__init__(destinations, decisions, **kwargs)

        async def stop(self) -> None:
            await super().stop()
            shutdown_order.append("router.stop")

    class RecordingRuntime(AgentRuntime):
        def __init__(self) -> None:
            super().__init__()
            runtimes.append(self)

    monkeypatch.setattr(app, "AgentRuntime", RecordingRuntime)
    monkeypatch.setattr(app, "AgentRouter", RecordingRouter)
    monkeypatch.setattr(app, "Conversation", RecordingConversation)
    monkeypatch.setattr(app, "ContextManager", RecordingContextManager)

    worker_log = tmp_path / "logs" / "worker.log"
    worker_log.parent.mkdir()
    monkeypatch.setattr(app, "setup_logging", lambda _name: worker_log)
    monkeypatch.setattr(app, "load_models_config", lambda path: path)
    monkeypatch.setattr(app, "load_voice_gate_config", lambda _path: VoiceGateConfig())
    monkeypatch.setattr(voice_runtime_module, "make_stt", lambda _models, _name: stt)
    monkeypatch.setattr(app, "make_llm", lambda _models, _name: llm)
    monkeypatch.setattr(app, "make_decision", lambda _models, _name: decisions)
    monkeypatch.setattr(app, "make_vlm", lambda _models, _name: vlm)
    monkeypatch.setattr(app, "make_tts", lambda _models, _name: tts)
    monkeypatch.setattr(app, "HubVoiceTransport", lambda: transport)
    monkeypatch.setattr(app, "CurrentFrameTool", _CurrentFrameTool)
    monkeypatch.setattr(app, "StreamingImageQueryTool", _StreamingImageQueryTool)

    def make_session(**kwargs):
        session = VoiceSession(**kwargs)  # type: ignore[arg-type]
        sessions.append(session)

        async def run(handler, **options) -> None:
            if session.ready_file:
                session.ready_file.touch()
            run_options.update(options)
            await runtimes[0].publish(
                PARTICIPANT_JOINED_TOPIC,
                ParticipantJoined(),
                participant_id="alice",
            )
            for text, participant_id in (("Pack the spare cable.", "alice"), ("Bob's private note.", "bob")):
                await runtimes[0].publish(
                    app.NOTES_CONTEXT_TOPIC, ConversationContextUpdate(text=text), participant_id=participant_id
                )
            assert (
                await handler(
                    SimpleNamespace(
                        participant_id="alice",
                        text="What is in front of me?",
                        timestamp_us=123,
                        interrupted_output=False,
                    )
                )
                is None
            )
            await asyncio.wait_for(response_complete.wait(), 1.0)
            response_complete.clear()
            await handler(
                SimpleNamespace(
                    participant_id="alice",
                    text="What should I remember?",
                    timestamp_us=124,
                    interrupted_output=False,
                )
            )
            await asyncio.wait_for(response_complete.wait(), 1.0)
            await options["on_participant_left"]("alice")

            async def wait_until_released() -> None:
                while not _CurrentFrameTool.instances[0].released:
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_until_released(), 1.0)

        async def enqueue_reply(
            participant_id: str,
            response: str | AsyncIterator[str],
            *,
            request_id: str,
            pts_us: int | None = None,
        ) -> None:
            async def consume() -> None:
                text = response if isinstance(response, str) else "".join([chunk async for chunk in response])
                responses.append((participant_id, text, request_id, pts_us))
                response_complete.set()

            response_tasks.append(asyncio.create_task(consume()))

        session.run = run  # type: ignore[method-assign]
        session.enqueue_reply = enqueue_reply  # type: ignore[method-assign]
        return session

    monkeypatch.setattr(voice_runtime_module, "_VoiceSession", make_session)
    _CurrentFrameTool.instances.clear()
    _StreamingImageQueryTool.instances.clear()

    await app.run_app(config, ready_file=ready_file)

    assert len(wired_front_ends) == 1
    assert set(wired_front_ends[0]) == {"conversation", "observation", "alerts", "notes"}
    assert isinstance(wired_front_ends[0]["conversation"], Conversation)
    assert wired_front_ends[0]["observation"].__class__.__name__ == "VisualObservationAgent"
    assert wired_front_ends[0]["alerts"].__class__.__name__ == "VisualAlertAgent"
    assert wired_front_ends[0]["notes"].__class__.__name__ == "NoteTakingAgent"

    assert ready_file.exists()
    relay_log = worker_log.parent / "relay-events.jsonl"
    assert relay_log.exists()
    relay_events = [yaml.safe_load(line) for line in relay_log.read_text().splitlines()]
    assert {event["name"] for event in relay_events} >= {
        "publish:conversation-example.user-query",
        "agent:conversation-example",
        "routing.turn",
        "voice.response",
    }
    assert "publish:voice.output" not in {event["name"] for event in relay_events}
    assert "agent:voice" not in {event["name"] for event in relay_events}
    assert stt.health_calls == tts.health_calls == 0
    assert vlm.health_calls >= 1
    assert llm.health_calls >= 1
    assert decisions.health_calls >= 1
    assert vlm.stream_calls == []
    assert stt.close_calls == tts.close_calls == vlm.close_calls == llm.close_calls == decisions.close_calls == 1
    assert transport.shutdown_calls == 1
    assert sessions[0].text_topic == "vlm.response"
    assert _CurrentFrameTool.instances[0].kwargs["endpoint"] is transport.endpoint
    assert _CurrentFrameTool.instances[0].kwargs["frame_max_age_s"] == (config.frame_max_age_s)
    assert _CurrentFrameTool.instances[0].kwargs["frame_timeout_s"] == (config.frame_timeout_s)
    assert _CurrentFrameTool.instances[0].released == ["alice"]
    assert _CurrentFrameTool.instances[0].requests[0].participant_id == "alice"
    assert _StreamingImageQueryTool.instances[0].kwargs["vlm"] is vlm
    assert _StreamingImageQueryTool.instances[0].kwargs["system_prompt"] == (config.system_prompt)
    assert _StreamingImageQueryTool.instances[0].requests[0].query == (
        "Frame source time: 1970-01-01 00:00 UTC.\nWhat is in front of me?"
    )
    assert _StreamingImageQueryTool.instances[0].requests[0].image.uri == ("https://example.com/frame.jpg")
    assert [(participant, text, pts_us) for participant, text, _, pts_us in responses] == [
        ("alice", "a blue square", 123),
        ("alice", "Hello there.", 124),
    ]
    assert len({request_id for _, _, request_id, _ in responses}) == 2
    conversation_prompt = "\n".join(message.content for message in llm.requests[-1][0])
    assert "current view" in conversation_prompt
    assert "Pack the spare cable." in conversation_prompt
    assert "Bob's private note." not in conversation_prompt
    assert all(task.done() for task in response_tasks)
    assert shutdown_order.index("router.stop") < shutdown_order.index("conversation.close")
    assert shutdown_order.index("context.clear") < shutdown_order.index("context.close")
    assert shutdown_order.index("conversation.close") < shutdown_order.index("context.close")
    assert shutdown_order.index("context.close") < shutdown_order.index("llm.close")
    assert run_options["interrupt_on_supersede"] is True
    assert callable(run_options["on_interrupted"])


async def test_relay_event_log_excludes_stream_chunks(tmp_path) -> None:
    worker_log = tmp_path / "worker.log"

    async with app._relay_event_log(worker_log):  # noqa: SLF001
        with nemo_relay.scope.scope("test-turn", nemo_relay.ScopeType.Agent):
            nemo_relay.scope.event("llm.chunk", data={"text": "fragment"})
            nemo_relay.scope.event("turn.summary", data={"text": "complete"})

    events = [yaml.safe_load(line) for line in (tmp_path / "relay-events.jsonl").read_text().splitlines()]
    names = [event["name"] for event in events]
    assert "llm.chunk" not in names
    assert "turn.summary" in names
    assert names.count("test-turn") == 2


async def test_sample_runtime_streams_vision_through_voice_agent(monkeypatch) -> None:
    query = "What is shown?"
    endpoint = _LiveEndpoint()
    vlm = _StreamingVlm()
    images = ImageRegistry()
    frames = CurrentFrameTool(
        endpoint=cast(ProcessorEndpoint, endpoint),
        images=images,
    )
    vision = StreamingImageQueryTool(
        images=images,
        vlm=cast(VLMService, vlm),
        system_prompt="Answer briefly.",
    )

    class Session:
        def __init__(self) -> None:
            self.text = ""
            self.complete = asyncio.Event()
            self.started = asyncio.Event()
            self.text_topic = "agent.response"

        async def __aenter__(self):
            return self

        async def run(self, _handler, **_options) -> None:
            self.started.set()
            await asyncio.Event().wait()

        async def enqueue_reply(
            self,
            participant_id: str,
            response: str | AsyncIterator[str],
            *,
            request_id: str,
            pts_us: int | None = None,
        ) -> None:
            assert participant_id == "alice"
            assert request_id
            assert pts_us == 123

            async def consume() -> None:
                self.text = response if isinstance(response, str) else "".join([chunk async for chunk in response])
                self.complete.set()

            asyncio.create_task(consume())

        async def close(self) -> None:
            pass

    session = Session()
    monkeypatch.setattr(
        voice_runtime_module,
        "_VoiceSession",
        lambda **_kwargs: session,
    )
    runtime = AgentRuntime()
    router = _make_agent(runtime, _ConversationService(), frames, vision, set_status=endpoint.set_status)
    runtime.register("conversation-example", router.runtime_subscriber())
    runtime.register(
        "voice",
        VoiceAgent(
            query_topic=USER_QUERY_TOPIC,
            stt=object(),
            tts=object(),
            vad=object(),
            voice_gate=object(),
            text_input=False,
        ),  # type: ignore[arg-type]
    )
    assert endpoint.frame_callback is not None
    await endpoint.frame_callback(
        FrameSignal(
            slot=0,
            seq=1,
            pts_us=time.time_ns() // 1_000,
            width=2,
            height=2,
            fmt=PixelFormat.RGB24,
            data_sz=12,
            participant_id="alice",
            track_id="camera",
        )
    )
    events = []
    subscriber = "conversation-example-agent-vision"
    intercept = "conversation-example-agent-vision-header"

    def add_header(_name, request, annotated):
        headers = dict(request.headers)
        headers["X-Relay-Session"] = "turn-9"
        return nemo_relay.LLMRequestInterceptOutcome(
            nemo_relay.LLMRequest(headers, request.content),
            annotated,
        )

    nemo_relay.subscribers.register(subscriber, events.append)
    nemo_relay.intercepts.register_llm_request(intercept, 0, False, add_header)
    try:
        async with runtime:
            await runtime.publish(
                USER_QUERY_TOPIC,
                UserQuery(text=query, timestamp_us=123),
                participant_id="alice",
                source="test-input",
            )
            await asyncio.wait_for(session.complete.wait(), 1.0)
        await nemo_relay.subscribers.flush_async()
    finally:
        nemo_relay.intercepts.deregister_llm_request(intercept)
        nemo_relay.subscribers.deregister(subscriber)

    assert session.text == "a blue square"
    images, question, system_prompt, headers = vlm.calls[0]
    assert len(images) == 1
    assert isinstance(images[0], bytes)
    assert question.endswith(query)
    assert question.startswith("Frame source time: ")
    assert system_prompt == "Answer briefly."
    assert headers["X-Relay-Session"] == "turn-9"
    assert endpoint.statuses == [("processing", "alice"), ("idle", "alice")]
    assert {"tool", "llm"} <= {getattr(event, "category", None) for event in events}
    starts = {
        event.name: event.to_dict()
        for event in events
        if event.kind == "scope" and event.to_dict().get("scope_category") == "start"
    }
    assert starts["routing.turn"]["parent_uuid"] != (starts["agent:conversation-example"]["uuid"])
    tool_starts = [
        event.to_dict()
        for event in events
        if event.kind == "scope"
        and event.to_dict().get("scope_category") == "start"
        and event.to_dict().get("category") == "tool"
    ]
    assert tool_starts
    assert tool_starts[0]["parent_uuid"] == starts["routing.turn"]["uuid"]
    llm_events = [event.to_json() for event in events if getattr(event, "category", None) == "llm"]
    assert llm_events
    assert any("<redacted:image>" in event for event in llm_events)


async def test_conversation_example_agent_handles_global_interruption_event() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class BlockingVision:
        async def stream(self, _request):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            yield SimpleNamespace(text="unreachable")

    runtime = AgentRuntime()
    runtime.register(
        "conversation-example",
        _make_agent(runtime, _ConversationService(), _SelectedFrameTool(), BlockingVision()).runtime_subscriber(),
    )

    async with runtime:
        await runtime.publish(
            USER_QUERY_TOPIC,
            UserQuery(text="What is shown?", timestamp_us=123),
            participant_id="alice",
            source="test-input",
        )
        await asyncio.wait_for(started.wait(), 1.0)
        await runtime.publish(
            INTERRUPTED_TOPIC,
            Interrupted(),
            source="voice.interruption",
        )
        await asyncio.wait_for(cancelled.wait(), 1.0)
