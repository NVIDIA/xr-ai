# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the packaged simple-vlm-example worker."""

from __future__ import annotations

import asyncio
import io
import json
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
from PIL import Image
from xr_ai_hub import FrameData, FrameSignal, FrameUnavailable, PixelFormat, ProcessorEndpoint
from xr_ai_models import (
    ChatResponse,
    ChatStreamEvent,
    DecisionAnswer,
    DecisionResponse,
    ToolCall,
    VLMService,
    load_models_config,
)
from xr_ai_runtime import AgentRuntime, Topic
from xr_ai_voice import (
    UserQuery,
    VoiceAgent,
    VoiceInterrupted,
    VoiceOutput,
    VoiceParticipantJoined,
    VoiceParticipantLeft,
)
from xr_ai_voice import _runtime as voice_runtime_module
from xr_ai_voice._readiness import wait_for_services
from xr_ai_voice._session import _VoiceSession as VoiceSession
from xr_ai_voicegate import VoiceGateConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE_DIR = _REPO_ROOT / "agent-samples" / "simple-vlm-example"
_WORKER_DIR = _SAMPLE_DIR / "worker"
sys.path.insert(0, str(_WORKER_DIR))
sys.path.insert(0, str(_REPO_ROOT / "agent-sdk" / "xr-ai-conversation"))

from simple_vlm_example_worker import __main__ as worker_main  # noqa: E402  # pyright: ignore[reportMissingImports]
from simple_vlm_example_worker import app  # noqa: E402  # pyright: ignore[reportMissingImports]
from simple_vlm_example_worker.agent import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    _CONVERSATION_INSTRUCTIONS,
    _VISION_INSTRUCTIONS,
    INTERRUPTED_TOPIC,
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    USER_QUERY_TOPIC,
    SimpleVlmAgent,
)
from simple_vlm_example_worker.config import load_config  # noqa: E402  # pyright: ignore[reportMissingImports]
from xr_ai_conversation import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    Conversation,
    ConversationContextUpdate,
    ConversationExchange,
    ConversationFrontEnd,
    ConversationTurn,
    RoutingAid,
)
from xr_ai_tools import Tool  # noqa: E402
from xr_ai_tools import current_frame as current_frame_module  # noqa: E402
from xr_ai_tools.current_frame import (  # noqa: E402
    CurrentFrameRequest,
    CurrentFrameTool,
    ImageFrame,
)
from xr_ai_tools.image import ImageReference, ImageRegistry  # noqa: E402
from xr_ai_tools.vision import (  # noqa: E402
    ImageQueryRequest,
    ImageQueryResult,
    ImageQueryTool,
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
        visual: bool = True,
        stream_gate=None,
        preface: bool = False,
        visual_query: str = "What color is the square?",
        error: Exception | None = None,
    ) -> None:
        super().__init__()
        self.visual = visual
        self.requests = []
        self.stream_gate = stream_gate
        self.preface = preface
        self.visual_query = visual_query
        self.error = error

    def stream_events(self, messages, *, tools=None, **_kwargs):
        self.requests.append((messages, tools))

        async def events():
            if self.error is not None:
                raise self.error
            if self.visual:
                if self.preface:
                    yield ChatStreamEvent(kind="text", text="I will inspect it. ")
                yield ChatStreamEvent(
                    kind="tool_call",
                    tool_call=ToolCall(
                        "call-1",
                        "current_view",
                        json.dumps({"query": self.visual_query}),
                    ),
                )
                yield ChatStreamEvent(kind="finish", finish_reason="tool_calls")
                return
            yield ChatStreamEvent(kind="text", text="Hello ")
            if self.stream_gate is not None:
                await self.stream_gate.wait()
            yield ChatStreamEvent(kind="text", text="there.")
            yield ChatStreamEvent(kind="finish", finish_reason="stop")

        return events()


class _FlakyWarmupService(_Service):
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
        if len(self.stream_calls) == 1:
            raise RuntimeError("still loading")
        yield "gray"


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
        return SimpleNamespace(image=ImageReference(uri="https://example.com/frame.jpg"))

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
        return SimpleNamespace(image=ImageReference(uri="https://example.com/frame.jpg"))

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


class _RouteDecisions:
    def __init__(
        self,
        destination: str | tuple[str, ...] = "vision",
        admission: str | tuple[str, ...] = "accept",
    ) -> None:
        self.destinations = list(destination) if isinstance(destination, tuple) else [destination]
        self.admissions = list(admission) if isinstance(admission, tuple) else [admission]
        self.health_calls = 0
        self.close_calls = 0
        self.states = []

    async def health(self) -> bool:
        self.health_calls += 1
        return True

    async def close(self) -> None:
        self.close_calls += 1

    async def decide(self, state, _questions, *, timeout=None):
        self.states.append(state)
        if "admission" in _questions:
            choice = self.admissions.pop(0) if len(self.admissions) > 1 else self.admissions[0]
            answer_name = "admission"
        else:
            choice = self.destinations.pop(0) if len(self.destinations) > 1 else self.destinations[0]
            answer_name = "destination"
        probability_names = ("accept", "decline", "clarify") if answer_name == "admission" else (
            "conversation",
            "vision",
        )
        return DecisionResponse(
            answers={
                answer_name: DecisionAnswer(
                    choice,
                    1.0,
                    {name: float(name == choice) for name in probability_names},
                )
            },
            model="fake",
            usage={},
            raw={},
        )


def _make_front_end(
    llm,
    frames,
    vision,
    *,
    destination: str | tuple[str, ...] = "vision",
    admission: str | tuple[str, ...] = "accept",
    include_vision: bool = True,
):
    conversation = Conversation(llm, instructions=_CONVERSATION_INSTRUCTIONS)  # type: ignore[arg-type]
    destinations = {"conversation": conversation}
    conversations = [conversation]
    if include_vision:
        visual = Conversation(
            llm,  # type: ignore[arg-type]
            instructions=_VISION_INSTRUCTIONS,
            routing=RoutingAid(
                name="vision",
                description="Questions requiring evidence from the current camera view.",
            ),
            frames=frames,  # type: ignore[arg-type]
            vision=vision,  # type: ignore[arg-type]
        )
        destinations["vision"] = visual
        conversations.append(visual)
    decisions = _RouteDecisions(destination, admission)
    front_end = ConversationFrontEnd(
        {
            name: app.ClefAdmissionDestination(item, decisions)
            for name, item in destinations.items()
        },
        decisions,  # type: ignore[arg-type]
    )
    return front_end, tuple(conversations)


def _make_agent(
    llm,
    frames,
    vision,
    *,
    destination: str | tuple[str, ...] = "vision",
    admission: str | tuple[str, ...] = "accept",
    include_vision: bool = True,
):
    front_end, conversations = _make_front_end(
        llm,
        frames,
        vision,
        destination=destination,
        admission=admission,
        include_vision=include_vision,
    )
    return SimpleVlmAgent(
        _ignore_status,
        front_end=front_end,
        conversations=conversations,
        frames=frames,  # type: ignore[arg-type]
    )


async def test_conversation_delegates_native_current_view_without_speaking_preamble() -> None:
    llm = _ConversationService(visual_query="Read the label on the bottle on the left.")
    frames = _ConversationFrames()
    vision = _StreamingImageQueryTool()
    conversation = Conversation(
        llm,  # type: ignore[arg-type]
        frames=frames,  # type: ignore[arg-type]
        vision=vision,  # type: ignore[arg-type]
    )

    answer = [
        chunk
        async for chunk in conversation.stream(
            "Read its label",
            "alice",
            history=(ConversationExchange("Which bottle is fullest?", "The bottle on the left."),),
        )
    ]

    assert answer == ["a ", "blue square"]
    assert frames.requests[0].participant_id == "alice"
    assert vision.requests[0].image.uri == "https://example.com/frame.jpg"
    assert vision.requests[0].query == "Read the label on the bottle on the left."
    assert len(llm.requests[0][1]) == 1
    assert llm.requests[0][1][0].name == "current_view"
    assert llm.requests[0][1][0].parameters["required"] == ["query"]
    assert "Turn 1: user asked: Which bottle is fullest?" in llm.requests[0][0][1].content


async def test_conversation_answers_directly_without_selecting_a_frame() -> None:
    llm = _ConversationService(visual=False)
    frames = _ConversationFrames()
    vision = _StreamingImageQueryTool()
    conversation = Conversation(
        llm,  # type: ignore[arg-type]
        frames=frames,  # type: ignore[arg-type]
        vision=vision,  # type: ignore[arg-type]
    )

    answer = [chunk async for chunk in conversation.stream("Say hello", "alice")]

    assert answer == ["Hello ", "there."]
    assert frames.requests == []
    assert vision.requests == []


async def test_text_only_conversation_has_no_visual_tool_and_advertises_no_camera_scope() -> None:
    llm = _ConversationService(visual=False)
    conversation = Conversation(llm)  # type: ignore[arg-type]

    answer = [chunk async for chunk in conversation.stream("Say hello", "alice")]
    aid = conversation.routing_aid("alice")

    assert answer == ["Hello ", "there."]
    assert llm.requests[0][1] == ()
    assert "no current-view visual tools are configured" in aid.description
    assert "No camera or visual-inspection tool is available" in llm.requests[0][0][0].content


async def test_application_instructions_and_routing_metadata_are_separate_inputs() -> None:
    llm = _ConversationService(visual=False)
    instructions = (
        "Answer general questions about museum exhibits. If asked for a medical diagnosis, "
        "say you cannot diagnose and suggest consulting a clinician."
    )
    routing = RoutingAid(
        name="museum_guide",
        description="Questions about museum exhibits and visible exhibit labels",
        priority=3,
    )
    conversation = Conversation(
        llm,  # type: ignore[arg-type]
        instructions=instructions,
        routing=routing,
    )

    [chunk async for chunk in conversation.stream("What is this?", "alice")]
    aid = conversation.routing_aid("alice")

    assert f"Application instructions:\n{instructions}" in llm.requests[0][0][0].content
    assert "medical diagnosis" in llm.requests[0][0][0].content
    assert aid == routing


async def test_domain_visual_request_routes_to_configured_conversation_destination() -> None:
    class Decisions:
        def __init__(self) -> None:
            self.state = None

        async def decide(self, state, _questions, *, timeout=None):
            self.state = state
            return DecisionResponse(
                answers={"destination": DecisionAnswer("museum_guide", 1.0, {"museum_guide": 1.0})},
                model="fake",
                usage={},
                raw={},
            )

    decisions = Decisions()
    llm = _ConversationService(visual_query="Read the label on the exhibit display.")
    frames = _ConversationFrames()
    vision = _StreamingImageQueryTool()
    museum = Conversation(
        llm,  # type: ignore[arg-type]
        instructions="Answer questions about museum exhibits.",
        routing=RoutingAid(
            name="museum_guide",
            description="Questions about museum exhibits and visible display labels",
        ),
        frames=frames,  # type: ignore[arg-type]
        vision=vision,  # type: ignore[arg-type]
    )
    general = Conversation(_ConversationService(visual=False))  # type: ignore[arg-type]
    front_end = ConversationFrontEnd(
        {"conversation": general, "museum_guide": museum},
        decisions,  # type: ignore[arg-type]
    )
    output = []

    async def emit(source, text):
        output.append((source, text))

    result = await front_end.handle(
        ConversationTurn("alice", "museum-1", "Read the label on this exhibit."),
        emit=emit,
    )

    assert result.text == "a blue square"
    assert output == [("museum_guide", "a "), ("museum_guide", "blue square")]
    assert llm.requests[0][1][0].name == "current_view"
    assert vision.requests[0].query == "Read the label on the exhibit display."
    assert (
        next(item for item in decisions.state["destinations"] if item["name"] == "museum_guide")["description"]
        == "Questions about museum exhibits and visible display labels"
    )


async def test_sample_destinations_share_cross_route_history_and_output_source() -> None:
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
                    yield ChatStreamEvent(
                        kind="tool_call",
                        tool_call=ToolCall(
                            "call-1",
                            "current_view",
                            json.dumps({"query": "What color is the sign?"}),
                        ),
                    )
                    yield ChatStreamEvent(kind="finish", finish_reason="tool_calls")

            return events()

    llm = LLM()
    frames = _ConversationFrames()
    vision = _StreamingImageQueryTool()
    general = Conversation(
        llm,  # type: ignore[arg-type]
        instructions=_CONVERSATION_INSTRUCTIONS,
        routing=RoutingAid("conversation", "General conversation and recall"),
    )
    visual = Conversation(
        llm,  # type: ignore[arg-type]
        instructions=_VISION_INSTRUCTIONS,
        routing=RoutingAid("vision", "Questions requiring current-camera evidence"),
        frames=frames,  # type: ignore[arg-type]
        vision=vision,  # type: ignore[arg-type]
    )
    decisions = _RouteDecisions(("conversation", "vision"))
    front_end = ConversationFrontEnd(
        {"conversation": general, "vision": visual},
        decisions,  # type: ignore[arg-type]
    )
    output = []

    async def emit(source, text):
        output.append((source, text))

    first = await front_end.handle(
        ConversationTurn("alice", "turn-1", "What is the capital of Portugal?"),
        emit=emit,
    )
    second = await front_end.handle(
        ConversationTurn("alice", "turn-2", "What color is the sign?"),
        emit=emit,
    )

    assert first.text == "Lisbon."
    assert second.text == "a blue square"
    assert llm.requests[0][1] == ()
    assert [tool.name for tool in llm.requests[1][1]] == ["current_view"]
    assert "Turn 1: user asked: What is the capital of Portugal?" in llm.requests[1][0][1].content
    assert frames.requests[0].participant_id == "alice"
    assert [source for source, _text in output] == ["conversation", "vision", "vision"]


@pytest.mark.parametrize(
    ("first_choice", "query", "with_history", "expected_destination"),
    [
        ("vision", "What was the capital again?", True, "conversation"),
        ("conversation", "What text is visible on the sign?", False, "vision"),
    ],
)
async def test_clef_admission_rejects_wrong_route_before_answer(
    first_choice: str,
    query: str,
    with_history: bool,
    expected_destination: str,
) -> None:
    llm = _ConversationService(visual=expected_destination == "vision")
    frames = _ConversationFrames()
    vision = _StreamingImageQueryTool()
    front_end, _conversations = _make_front_end(
        llm,
        frames,
        vision,
        destination=(first_choice, expected_destination),
        admission=("decline", "accept"),
    )
    emitted = []

    async def emit(source, text):
        emitted.append((source, text))

    history = (ConversationExchange("What is the capital of Portugal?", "Lisbon."),) if with_history else ()
    result = await front_end.handle(
        ConversationTurn("alice", "turn-1", query, history=history),
        emit=emit,
    )

    assert result.disposition == "answer"
    assert emitted
    assert {source for source, _text in emitted} == {expected_destination}
    assert len(llm.requests) == 1
    assert [exchange.user for exchange in front_end._history["alice"]] == [query]
    if expected_destination == "conversation":
        assert llm.requests[0][1] == ()
        assert "Lisbon." in llm.requests[0][0][1].content
        assert frames.requests == []
    else:
        assert [tool.name for tool in llm.requests[0][1]] == ["current_view"]
        assert frames.requests


async def test_declining_route_preserves_its_context_until_participant_leaves() -> None:
    llm = _ConversationService(visual=False)
    front_end, conversations = _make_front_end(
        llm,
        _ConversationFrames(),
        _StreamingImageQueryTool(),
        destination=("vision", "conversation", "conversation"),
        admission=("decline", "accept", "accept"),
    )
    general, _vision = conversations
    await general._update_context(  # noqa: SLF001
        "alice",
        "tea.status",
        ConversationContextUpdate(mode="summary", text="The timer is at 4 minutes."),
    )

    async def emit(_source: str, _text: str) -> None:
        return None

    first = await front_end.handle(
        ConversationTurn("alice", "turn-1", "What is the capital of France?"),
        emit=emit,
    )
    second = await front_end.handle(
        ConversationTurn("alice", "turn-2", "And what is the capital of Italy?"),
        emit=emit,
    )

    assert first.disposition == second.disposition == "answer"
    assert general._context.sections("alice") == (  # noqa: SLF001
        ("tea.status", "The timer is at 4 minutes."),
    )
    assert "The timer is at 4 minutes." in "\n".join(
        str(message.content) for message in llm.requests[1][0]
    )

    agent = SimpleVlmAgent(_ignore_status, front_end=front_end, conversations=conversations)
    await agent.participant_left(
        VoiceParticipantLeft(),
        SimpleNamespace(metadata=SimpleNamespace(participant_id="alice")),
    )
    assert general._context.sections("alice") == ()  # noqa: SLF001


async def test_clef_admission_clarifies_without_answering_or_using_camera() -> None:
    llm = _ConversationService()
    frames = _ConversationFrames()
    front_end, _conversations = _make_front_end(
        llm,
        frames,
        _StreamingImageQueryTool(),
        destination="vision",
        admission="clarify",
    )
    emitted = []

    async def emit(source, text):
        emitted.append((source, text))

    result = await front_end.handle(
        ConversationTurn("alice", "turn-ambiguous", "Could you check that one?"),
        emit=emit,
    )

    assert result.disposition == "clarify"
    assert emitted == [
        (
            "vision",
            "Would you like a general answer, or help with what is visible in your camera view?",
        )
    ]
    assert llm.requests == []
    assert frames.requests == []


async def test_clef_admission_all_declines_emit_one_clarification_without_history() -> None:
    llm = _ConversationService(visual=False)
    front_end, _conversations = _make_front_end(
        llm,
        _ConversationFrames(),
        _StreamingImageQueryTool(),
        destination=("vision", "conversation"),
        admission=("decline", "decline"),
    )
    emitted = []

    async def emit(source, text):
        emitted.append((source, text))

    result = await front_end.handle(
        ConversationTurn("alice", "turn-unmatched", "Please do the thing."),
        emit=emit,
    )

    assert result.disposition == "clarify"
    assert len(emitted) == 1
    assert emitted[0][1] == "Would you like a general answer, or help with what is visible in your camera view?"
    assert llm.requests == []
    assert front_end._history.get("alice", []) == []


async def test_interruption_during_clef_admission_emits_nothing() -> None:
    admission_started = asyncio.Event()
    release_admission = asyncio.Event()

    class BlockingDecisions:
        async def decide(self, _state, questions, *, timeout=None):
            if "destination" in questions:
                return DecisionResponse(
                    answers={
                        "destination": DecisionAnswer(
                            "conversation",
                            1.0,
                            {"conversation": 1.0, "vision": 0.0},
                        )
                    },
                    model="fake",
                    usage={},
                    raw={},
                )
            admission_started.set()
            await release_admission.wait()
            return DecisionResponse(
                answers={
                    "admission": DecisionAnswer("accept", 1.0, {"accept": 1.0, "decline": 0.0, "clarify": 0.0})
                },
                model="fake",
                usage={},
                raw={},
            )

    llm = _ConversationService(visual=False)
    decisions = BlockingDecisions()
    conversation = Conversation(
        llm,  # type: ignore[arg-type]
        instructions=_CONVERSATION_INSTRUCTIONS,
        routing=RoutingAid("conversation", "General conversation and recall"),
    )
    visual = Conversation(
        llm,  # type: ignore[arg-type]
        instructions=_VISION_INSTRUCTIONS,
        routing=RoutingAid("vision", "Current-camera evidence"),
        frames=_ConversationFrames(),  # type: ignore[arg-type]
        vision=_StreamingImageQueryTool(),  # type: ignore[arg-type]
    )
    front_end = ConversationFrontEnd(
        {
            "conversation": app.ClefAdmissionDestination(conversation, decisions),
            "vision": app.ClefAdmissionDestination(visual, decisions),
        },
        decisions,  # type: ignore[arg-type]
    )
    emitted = []

    async def emit(source, text):
        emitted.append((source, text))

    task = asyncio.create_task(
        front_end.handle(
            ConversationTurn("alice", "turn-cancel", "Explain why the sky is blue."),
            emit=emit,
        )
    )
    await asyncio.wait_for(admission_started.wait(), 1.0)
    await front_end.interrupt("alice")
    release_admission.set()
    await asyncio.gather(task, return_exceptions=True)

    assert emitted == []
    assert llm.requests == []


def test_conversation_rejects_half_configured_vision_tools() -> None:
    with pytest.raises(ValueError, match="configured together"):
        Conversation(_ConversationService(), frames=_ConversationFrames())  # type: ignore[arg-type]


async def test_conversation_streams_answer_before_model_stream_finishes() -> None:
    gate = asyncio.Event()
    llm = _ConversationService(visual=False, stream_gate=gate)
    conversation = Conversation(
        llm,  # type: ignore[arg-type]
        frames=_ConversationFrames(),  # type: ignore[arg-type]
        vision=_StreamingImageQueryTool(),  # type: ignore[arg-type]
    )
    stream = conversation.stream("Say hello", "alice")

    first = await asyncio.wait_for(anext(stream), timeout=1)

    assert first == "Hello "
    assert not gate.is_set()
    gate.set()
    assert await anext(stream) == "there."
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


async def test_conversation_keeps_early_text_once_if_tool_call_arrives_later() -> None:
    llm = _ConversationService(preface=True)
    conversation = Conversation(
        llm,  # type: ignore[arg-type]
        frames=_ConversationFrames(),  # type: ignore[arg-type]
        vision=_StreamingImageQueryTool(),  # type: ignore[arg-type]
    )

    answer = [chunk async for chunk in conversation.stream("What color is it?", "alice")]

    assert answer == ["I will inspect it. ", "a ", "blue square"]


async def test_simple_vlm_agent_publishes_one_safe_fallback_on_llm_failure() -> None:
    published = []

    class Context:
        agent_name = "simple-vlm"
        metadata = SimpleNamespace(
            message_id="turn-failed",
            correlation_id="turn-failed",
            participant_id="alice",
        )

        async def publish(self, topic, output) -> None:
            published.append((topic, output))

    llm = _ConversationService(error=RuntimeError("offline"))
    agent = _make_agent(
        llm,
        _ConversationFrames(),
        _StreamingImageQueryTool(),
        destination="conversation",
    )

    await agent._stream(  # noqa: SLF001
        UserQuery(text="Hello", timestamp_us=456),
        Context(),  # type: ignore[arg-type]
    )

    assert len(published) == 1
    assert published[0][1].text == "I couldn't complete that request. Please try again."
    assert published[0][1].timestamp_us == 456
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


def test_worker_is_a_package_with_module_and_console_entry_points() -> None:
    project = tomllib.loads((_WORKER_DIR / "pyproject.toml").read_text())
    package = _WORKER_DIR / "simple_vlm_example_worker"
    dependencies = set(project["project"]["dependencies"])

    assert project["project"]["scripts"]["simple_vlm_example_worker"] == ("simple_vlm_example_worker.__main__:run")
    assert "xr-ai-hub-client" in dependencies
    assert "xr-ai-agent-runtime" in dependencies
    assert "xr-ai-conversation" in dependencies
    assert project["tool"]["uv"]["sources"]["xr-ai-agent-runtime"]["path"] == ("../../../agent-sdk/xr-ai-runtime")
    assert project["tool"]["uv"]["sources"]["xr-ai-conversation"]["path"] == ("../../../agent-sdk/xr-ai-conversation")
    assert (_WORKER_DIR / project["tool"]["uv"]["sources"]["xr-ai-conversation"]["path"]).resolve().is_dir()
    assert (_WORKER_DIR / project["tool"]["uv"]["sources"]["xr-ai-agent-runtime"]["path"]).resolve().is_dir()
    assert project["tool"]["uv"]["sources"]["xr-ai-hub-client"]["path"] == ("../../../agent-sdk/xr-ai-hub")
    assert (_WORKER_DIR / project["tool"]["uv"]["sources"]["xr-ai-hub-client"]["path"]).resolve().is_dir()
    assert "xr-ai-tools[frames,vision]" in dependencies
    assert all("[vision" not in dependency and "[voice" not in dependency for dependency in dependencies)
    assert "xr-ai-voice" in dependencies
    assert "xr-ai-pipecat" not in dependencies
    assert all("mcp" not in dependency.lower() for dependency in dependencies)
    assert project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["simple_vlm_example_worker"]
    assert {
        "__init__.py",
        "__main__.py",
        "agent.py",
        "app.py",
        "config.py",
        "prompts/system.txt",
    } <= {str(path.relative_to(package)) for path in package.rglob("*") if path.is_file()}
    assert not (_WORKER_DIR / "agent.py").exists()
    assert not (_WORKER_DIR / "simple_vlm_example_worker.py").exists()


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
    config_path = _SAMPLE_DIR / "yaml" / "simple_vlm_example_worker.yaml"
    config = load_config(config_path)
    raw = yaml.safe_load(config_path.read_text())
    prompt = (_WORKER_DIR / "simple_vlm_example_worker" / "prompts" / "system.txt").read_text()

    assert config.models_config == _SAMPLE_DIR / "yaml" / "models.json"
    assert config.voice_gate_yaml == _SAMPLE_DIR / "yaml" / "voice_gate.yaml"
    assert config.system_prompt == prompt
    assert "only evidence visible in the image" in prompt
    assert "Do not infer facts that are not visible" in prompt
    vision_prompt = (_WORKER_DIR / "simple_vlm_example_worker" / "prompts" / "vision.txt").read_text()
    assert "current camera view" in vision_prompt
    assert "Do not answer general knowledge" in vision_prompt
    assert config.frame_max_age_s == 5.0
    assert config.frame_timeout_s == 5.0
    assert config.idle_timeout_secs is None
    assert config.context_topics == ()
    assert config.context_budget_bytes == 12_000
    assert config.compaction_input_budget_bytes == 12_000
    assert "system_prompt_file" not in raw

    models = load_models_config(_SAMPLE_DIR / "yaml" / "models.json")
    llm = models.llm("llm")
    assert llm.default_extras["max_tokens"] == 4608
    assert llm.default_extras["temperature"] == 0.25
    assert llm.default_extras["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_budget": 4096,
    }
    router = models.decision("router")
    assert router.kind == "systemone"
    assert router.model_name == "Cloudflare/clef-flash"
    assert router.base_url == "http://localhost:8120"


def test_conversation_eval_cases_keep_direct_visual_and_followup_coverage() -> None:
    cases = yaml.safe_load((_SAMPLE_DIR / "eval" / "conversation-cases.yaml").read_text(encoding="utf-8"))
    by_name = {case["name"]: case for case in cases}

    assert by_name["general_question_without_camera"]["destination"] == "conversation"
    assert by_name["general_question_without_camera"]["admission"] == "accept"
    assert by_name["general_question_without_camera"]["declined_alternative"] == "vision"
    assert by_name["general_question_without_camera"]["route"] == "direct"
    assert by_name["general_question_without_camera"]["image"] == "blank"
    unavailable_action = by_name["unavailable_external_action"]
    assert unavailable_action["route"] == "direct"
    assert unavailable_action["any_of"] == ["can't", "cannot", "unable"]
    assert by_name["read_visible_sign_text"]["route"] == "current_view"
    assert by_name["read_visible_sign_text"]["destination"] == "vision"
    assert by_name["read_visible_sign_text"]["admission"] == "accept"
    assert by_name["read_visible_sign_text"]["declined_alternative"] == "conversation"
    followup = by_name["resolve_visual_followup_referent"]
    assert followup["route"] == "current_view"
    assert followup["destination"] == "vision"
    assert followup["history"]
    assert followup["any_of"] == ["red"]
    stale_clarification = by_name["new_question_after_stale_clarification"]
    assert stale_clarification["route"] == "direct"
    assert stale_clarification["destination"] == "conversation"
    assert stale_clarification["image"] == "blank"
    assert stale_clarification["history"][0]["assistant"].startswith("Could you clarify")
    assert stale_clarification["any_of"] == ["63", "sixty-three"]
    explicit_recall = by_name["explicit_recall_uses_completed_answer"]
    assert explicit_recall["route"] == "direct"
    assert explicit_recall["destination"] == "conversation"
    assert explicit_recall["image"] == "blank"
    assert explicit_recall["history"]
    assert explicit_recall["any_of"] == ["Lisbon"]
    pasta_loop = by_name["direct_howto_after_repeated_visual_clarifications"]
    assert pasta_loop["route"] == "direct"
    assert pasta_loop["destination"] == "conversation"
    assert pasta_loop["image"] == "blank"
    assert len(pasta_loop["history"]) == 4
    assert pasta_loop["any_of"] == ["add the pasta", "cook according to the package directions"]
    assert "pasta" not in pasta_loop["any_of"]
    bowline = by_name["direct_howto_after_visual_clarification"]
    assert bowline["route"] == "direct"
    assert bowline["destination"] == "conversation"
    assert bowline["admission"] == "accept"
    assert bowline["declined_alternative"] == "vision"
    assert bowline["image"] == "blank"
    assert bowline["any_of"] == ["rope", "loop"]
    domain_visual = by_name["domain_scoped_visual_question"]
    assert domain_visual["route"] == "current_view"
    assert domain_visual["destination"] == "vision"
    assert domain_visual["any_of"] == ["blue"]
    ambiguous = by_name["ambiguous_reference_clarifies_before_camera_access"]
    assert ambiguous["route"] == "clarify"
    assert ambiguous["admission"] == "clarify"
    assert ambiguous["destination"] == "vision"
    assert ambiguous["any_of"] == ["general answer", "visible"]


def test_config_without_a_file_uses_packaged_defaults(tmp_path) -> None:
    prompt = (_WORKER_DIR / "simple_vlm_example_worker" / "prompts" / "system.txt").read_text()

    for config_path in (None, tmp_path / "missing.yaml"):
        config = load_config(config_path)
        assert config.system_prompt == prompt
        expected_parent = Path() if config_path is None else tmp_path
        assert config.models_config == expected_parent / "models.json"


def test_blank_inline_prompt_uses_packaged_default(tmp_path) -> None:
    config_path = tmp_path / "worker.yaml"
    config_path.write_text("system_prompt:\n")
    prompt = (_WORKER_DIR / "simple_vlm_example_worker" / "prompts" / "system.txt").read_text()

    assert load_config(config_path).system_prompt == prompt


def test_config_keeps_inline_prompt_compatibility(tmp_path) -> None:
    config_path = tmp_path / "worker.yaml"
    config_path.write_text("voice_gate_yaml: gate.yaml\nsystem_prompt: custom prompt\nidle_timeout_secs: 30\n")

    config = load_config(config_path)

    assert config.models_config == tmp_path / "models.json"
    assert config.voice_gate_yaml == tmp_path / "gate.yaml"
    assert config.system_prompt == "custom prompt"
    assert config.idle_timeout_secs == 30.0


def test_config_rejects_a_non_mapping_yaml_document(tmp_path) -> None:
    config_path = tmp_path / "worker.yaml"
    config_path.write_text("- not\n- a\n- mapping\n")

    with pytest.raises(ValueError, match="must be a YAML mapping"):
        load_config(config_path)


async def test_simple_vlm_agent_closes_tool_stream_when_publication_fails() -> None:
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
        agent_name = "simple-vlm"
        metadata = SimpleNamespace(
            message_id="turn-1",
            correlation_id="turn-1",
            participant_id="alice",
        )

        async def publish(self, *_args, **_kwargs) -> None:
            raise RuntimeError("runtime stopped")

    agent = _make_agent(_ConversationService(), _SelectedFrameTool(), Vision())
    with pytest.raises(RuntimeError, match="runtime stopped"):
        await agent._stream(  # noqa: SLF001
            UserQuery(text="What is shown?", timestamp_us=123),
            Context(),  # type: ignore[arg-type]
        )

    assert closed.is_set()


async def test_simple_vlm_agent_reports_relay_wrapped_missing_camera_frame() -> None:
    published: list[VoiceOutput] = []

    async def missing_frame(_request: CurrentFrameRequest) -> ImageFrame:
        raise FrameUnavailable("No camera frame available — please try again.")

    missing_frame_tool = Tool(
        "missing_camera_frame",
        "Reproduce a missing camera frame through the real Relay boundary.",
        CurrentFrameRequest,
        ImageFrame,
        missing_frame,
    )
    with pytest.raises(RuntimeError, match="internal error: FrameUnavailable") as wrapped:
        await missing_frame_tool.execute(CurrentFrameRequest(participant_id="alice"))
    assert wrapped.value.__cause__ is None
    assert wrapped.value.__context__ is None

    class Context:
        agent_name = "simple-vlm"
        metadata = SimpleNamespace(
            message_id="turn-1",
            correlation_id="turn-1",
            participant_id="alice",
        )

        async def publish(self, _topic, output) -> None:
            published.append(output)

    agent = _make_agent(_ConversationService(), missing_frame_tool, _StreamingImageQueryTool())

    await agent._stream(  # noqa: SLF001
        UserQuery(text="What is shown?", timestamp_us=123),
        Context(),  # type: ignore[arg-type]
    )

    assert [output.text for output in published] == [
        "No camera frame available — please try again.",
        "",
    ]
    assert [output.final for output in published] == [False, True]


async def test_cancelled_vlm_turn_does_not_publish_stream_terminator() -> None:
    waiting = asyncio.Event()
    closed = asyncio.Event()
    published: list[VoiceOutput] = []

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
        agent_name = "simple-vlm"
        metadata = SimpleNamespace(
            message_id="turn-1",
            correlation_id="turn-1",
            participant_id="alice",
        )

        async def publish(self, _topic, output) -> None:
            published.append(output)

    agent = _make_agent(_ConversationService(), _SelectedFrameTool(), Vision())

    task = asyncio.create_task(
        agent._stream(  # noqa: SLF001
            UserQuery(text="What is shown?", timestamp_us=123),
            Context(),  # type: ignore[arg-type]
        )
    )
    await asyncio.wait_for(waiting.wait(), 1.0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert closed.is_set()
    assert len(published) == 1
    assert published[0].final is False


async def test_vlm_warmup_failure_is_retried_by_readiness() -> None:
    vlm = _FlakyWarmupService()

    await asyncio.wait_for(
        wait_for_services(
            {"vlm": lambda: app._warm_vlm(cast(VLMService, vlm))},
            poll_interval=0,
        ),
        timeout=1,
    )

    assert vlm.health_calls == 0
    assert len(vlm.stream_calls) == 2


@pytest.mark.asyncio
async def test_participant_rejoin_reopens_context_before_topic_updates() -> None:
    llm = _ConversationService()
    front_end, conversations = _make_front_end(llm, None, None, destination="conversation")
    conversation, visual_conversation = conversations

    async def set_status(_status: str, _participant_id: str) -> None:
        return None

    agent = SimpleVlmAgent(
        set_status, front_end=front_end, conversations=conversations
    )
    context_topic = Topic("tea.status", ConversationContextUpdate)
    runtime = AgentRuntime()
    runtime.register("conversation-context", conversation.context_subscriber((context_topic,)))
    runtime.register("vision-context", visual_conversation.context_subscriber((context_topic,)))
    runtime.register("simple-vlm", agent)

    async with runtime:
        await runtime.publish(
            PARTICIPANT_JOINED_TOPIC,
            VoiceParticipantJoined(),
            participant_id="alice",
        )
        await runtime.publish(
            context_topic,
            {"mode": "summary", "text": "Timer is at 4 minutes."},
            participant_id="alice",
        )
        assert conversation._context.sections("alice") == (("tea.status", "Timer is at 4 minutes."),)
        assert visual_conversation._context.sections("alice") == (("tea.status", "Timer is at 4 minutes."),)

        await runtime.publish(
            PARTICIPANT_LEFT_TOPIC,
            VoiceParticipantLeft(),
            participant_id="alice",
        )
        assert conversation._context.sections("alice") == ()
        assert visual_conversation._context.sections("alice") == ()
        await runtime.publish(
            PARTICIPANT_JOINED_TOPIC,
            VoiceParticipantJoined(),
            participant_id="alice",
        )
        await runtime.publish(
            context_topic,
            {"mode": "summary", "text": "Timer is at 6 minutes."},
            participant_id="alice",
        )
        assert conversation._context.sections("alice") == (("tea.status", "Timer is at 6 minutes."),)
    await conversation.close()


async def test_app_wires_text_voice_cleanup_readiness_and_shutdown(
    monkeypatch,
    tmp_path,
) -> None:
    config = replace(
        load_config(_SAMPLE_DIR / "yaml" / "simple_vlm_example_worker.yaml"),
        context_topics=("tea.status",),
    )
    ready_file = tmp_path / "ready"
    stt = _Service()
    llm = _ConversationService()
    vlm = _Service()
    tts = _Service()
    router = _RouteDecisions()
    transport = _Transport()
    sessions: list[VoiceSession] = []
    run_options = {}
    responses = []
    response_tasks: list[asyncio.Task[None]] = []
    response_complete = asyncio.Event()
    runtimes = []

    class RecordingRuntime(AgentRuntime):
        def __init__(self) -> None:
            super().__init__()
            runtimes.append(self)

    monkeypatch.setattr(app, "AgentRuntime", RecordingRuntime)

    worker_log = tmp_path / "logs" / "worker.log"
    worker_log.parent.mkdir()
    monkeypatch.setattr(app, "setup_logging", lambda _name: worker_log)
    monkeypatch.setattr(app, "load_models_config", lambda path: path)
    monkeypatch.setattr(app, "load_voice_gate_config", lambda _path: VoiceGateConfig())
    monkeypatch.setattr(voice_runtime_module, "make_stt", lambda _models, _name: stt)
    monkeypatch.setattr(app, "make_llm", lambda _models, _name: llm)
    monkeypatch.setattr(app, "make_decision", lambda _models, _name: router)
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
            await options["on_participant_left"]("alice")

            async def wait_until_released() -> None:
                while not _CurrentFrameTool.instances[0].released:
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_until_released(), 1.0)

        async def enqueue_response(
            participant_id: str,
            response: str | AsyncIterator[str],
            *,
            interrupt: bool = False,
            pts_us: int | None = None,
        ) -> None:
            async def consume() -> None:
                text = response if isinstance(response, str) else "".join([chunk async for chunk in response])
                responses.append((participant_id, text, interrupt, pts_us))
                response_complete.set()

            response_tasks.append(asyncio.create_task(consume()))

        session.run = run  # type: ignore[method-assign]
        session.enqueue_response = enqueue_response  # type: ignore[method-assign]
        return session

    monkeypatch.setattr(voice_runtime_module, "_VoiceSession", make_session)
    _CurrentFrameTool.instances.clear()
    _StreamingImageQueryTool.instances.clear()

    await app.run_app(config, ready_file=ready_file)

    assert ready_file.exists()
    relay_log = worker_log.parent / "relay-events.jsonl"
    assert relay_log.exists()
    relay_events = [yaml.safe_load(line) for line in relay_log.read_text().splitlines()]
    assert {event["name"] for event in relay_events} >= {
        "publish:simple-vlm.user-query",
        "agent:simple-vlm",
        "simple-vlm.turn",
        "voice.response",
    }
    assert "publish:voice.output" not in {event["name"] for event in relay_events}
    assert "agent:voice" not in {event["name"] for event in relay_events}
    assert stt.health_calls == tts.health_calls == vlm.health_calls == 0
    assert llm.health_calls >= 1
    assert router.health_calls >= 1
    assert len(vlm.stream_calls) == 1
    warmup_images, question, system_prompt, max_tokens, timeout = vlm.stream_calls[0]
    assert len(warmup_images) == 1
    with Image.open(io.BytesIO(warmup_images[0])) as warmup_image:
        assert warmup_image.format == "JPEG"
        assert warmup_image.size == (1280, 720)
    assert question == "What is the dominant color?"
    assert system_prompt == "Answer with one word."
    assert max_tokens == 4
    assert timeout == 120.0
    assert stt.close_calls == tts.close_calls == vlm.close_calls == llm.close_calls == router.close_calls == 1
    assert transport.shutdown_calls == 1
    assert sessions[0].text_topic == "vlm.response"
    assert _CurrentFrameTool.instances[0].kwargs["endpoint"] is transport.endpoint
    assert {"conversation-context", "vision-context"} <= runtimes[0]._agents.keys()
    assert _CurrentFrameTool.instances[0].kwargs["frame_max_age_s"] == (config.frame_max_age_s)
    assert _CurrentFrameTool.instances[0].kwargs["frame_timeout_s"] == (config.frame_timeout_s)
    assert _CurrentFrameTool.instances[0].released == ["alice"]
    assert _CurrentFrameTool.instances[0].requests[0].participant_id == "alice"
    assert _StreamingImageQueryTool.instances[0].kwargs["vlm"] is vlm
    assert _StreamingImageQueryTool.instances[0].kwargs["system_prompt"] == (config.system_prompt)
    assert _StreamingImageQueryTool.instances[0].requests[0].query == ("What is in front of me?")
    assert _StreamingImageQueryTool.instances[0].requests[0].image.uri == ("https://example.com/frame.jpg")
    assert responses == [("alice", "a blue square", True, 123)]
    assert all(task.done() for task in response_tasks)
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


async def test_selected_frame_returns_a_complete_agent_observation() -> None:
    endpoint = _LiveEndpoint()
    vlm = _StreamingVlm()
    images = ImageRegistry()
    frames = CurrentFrameTool(
        endpoint=cast(ProcessorEndpoint, endpoint),
        images=images,
    )
    vision = ImageQueryTool(
        images=images,
        vlm=cast(VLMService, vlm),
        system_prompt="Answer briefly.",
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
    subscriber = "simple-vlm-finite-vision"
    intercept = "simple-vlm-finite-vision-header"

    def add_header(_name, request, annotated):
        headers = dict(request.headers)
        headers["X-Relay-Session"] = "turn-8"
        return nemo_relay.LLMRequestInterceptOutcome(
            nemo_relay.LLMRequest(headers, request.content),
            annotated,
        )

    nemo_relay.subscribers.register(subscriber, events.append)
    nemo_relay.intercepts.register_llm_request(intercept, 0, False, add_header)
    try:
        frame = await frames.execute(CurrentFrameRequest(participant_id="alice"))
        result = await vision.execute(
            ImageQueryRequest(image=frame.image, query="What is shown?"),
        )
        await nemo_relay.subscribers.flush_async()
    finally:
        nemo_relay.intercepts.deregister_llm_request(intercept)
        nemo_relay.subscribers.deregister(subscriber)

    assert result == ImageQueryResult(text="a blue square")
    images, question, system_prompt, headers = vlm.ask_calls[0]
    assert len(images) == 1
    assert isinstance(images[0], bytes)
    assert question == "What is shown?"
    assert system_prompt == "Answer briefly."
    assert headers["X-Relay-Session"] == "turn-8"
    assert {"tool", "llm"} <= {getattr(event, "category", None) for event in events}
    llm_events = [event.to_json() for event in events if getattr(event, "category", None) == "llm"]
    assert llm_events
    assert any("<redacted:image>" in event for event in llm_events)


async def test_selected_frame_streams_typed_image_query_chunks() -> None:
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
    subscriber = "simple-vlm-image-query"
    intercept = "simple-vlm-image-query-header"

    def add_header(_name, request, annotated):
        headers = dict(request.headers)
        headers["X-Relay-Session"] = "turn-7"
        return nemo_relay.LLMRequestInterceptOutcome(
            nemo_relay.LLMRequest(headers, request.content),
            annotated,
        )

    nemo_relay.subscribers.register(subscriber, events.append)
    nemo_relay.intercepts.register_llm_request(intercept, 0, False, add_header)
    try:
        frame = await frames.execute(CurrentFrameRequest(participant_id="alice"))
        chunks = [chunk async for chunk in vision.stream(ImageQueryRequest(image=frame.image, query="What is shown?"))]
        await nemo_relay.subscribers.flush_async()
    finally:
        nemo_relay.intercepts.deregister_llm_request(intercept)
        nemo_relay.subscribers.deregister(subscriber)

    assert [chunk.text for chunk in chunks] == ["a ", "blue ", "square"]
    images, question, system_prompt, headers = vlm.calls[0]
    assert len(images) == 1
    assert isinstance(images[0], bytes)
    assert question == "What is shown?"
    assert system_prompt == "Answer briefly."
    assert headers["X-Relay-Session"] == "turn-7"
    assert {"tool", "llm"} <= {getattr(event, "category", None) for event in events}
    llm_events = [event.to_json() for event in events if getattr(event, "category", None) == "llm"]
    assert llm_events
    assert any("<redacted:image>" in event for event in llm_events)


async def test_streaming_image_query_stops_after_a_partial_failure() -> None:
    class _PartialFailureVlm:
        def __init__(self) -> None:
            self.calls = []

        async def stream_images(
            self,
            images,
            question: str,
            *,
            system_prompt: str = "",
            headers=None,
        ):
            self.calls.append((images, question, system_prompt, dict(headers or {})))
            yield "The object is "
            raise RuntimeError("stream disconnected")

    images = ImageRegistry()
    vision = StreamingImageQueryTool(
        images=images,
        vlm=cast(VLMService, _PartialFailureVlm()),
    )

    chunks = [
        chunk.text
        async for chunk in vision.stream(
            ImageQueryRequest(
                image=images.put(b"frame"),
                query="What is shown?",
            ),
        )
    ]

    assert chunks == ["The object is "]


async def test_streaming_image_query_reports_failure_before_any_output() -> None:
    class _ImmediateFailureVlm:
        async def stream_images(self, *_args, **_kwargs):
            if False:
                yield ""
            raise RuntimeError("stream unavailable")

    images = ImageRegistry()
    vision = StreamingImageQueryTool(
        images=images,
        vlm=cast(VLMService, _ImmediateFailureVlm()),
    )

    chunks = [
        chunk.text
        async for chunk in vision.stream(
            ImageQueryRequest(
                image=images.put(b"frame"),
                query="What is shown?",
            ),
        )
    ]

    assert chunks == ["VLM server unavailable — please retry."]


async def test_current_frame_tool_propagates_conversion_errors(monkeypatch) -> None:
    endpoint = _LiveEndpoint()
    frames = CurrentFrameTool(
        endpoint=cast(ProcessorEndpoint, endpoint),
        images=ImageRegistry(),
    )
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

    def malformed_frame(_frame):
        raise ValueError("malformed pixels")

    monkeypatch.setattr(current_frame_module, "frame_to_pil", malformed_frame)

    with pytest.raises(RuntimeError, match="malformed pixels"):
        await frames.execute(CurrentFrameRequest(participant_id="alice"))


async def test_sample_runtime_streams_vision_through_voice_agent(monkeypatch) -> None:
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

        async def enqueue_response(
            self,
            participant_id: str,
            response: str | AsyncIterator[str],
            *,
            interrupt: bool = False,
            pts_us: int | None = None,
        ) -> None:
            assert participant_id == "alice"
            assert interrupt is True
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
    runtime.register(
        "simple-vlm",
        _make_agent(_ConversationService(), frames, vision),
    )
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
    subscriber = "simple-vlm-agent-vision"
    intercept = "simple-vlm-agent-vision-header"

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
                UserQuery(text="What is shown?", timestamp_us=123),
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
    assert question == "What is shown?"
    assert system_prompt == "Answer briefly."
    assert headers["X-Relay-Session"] == "turn-9"
    assert endpoint.statuses == [("processing", "alice"), ("idle", "alice")]
    assert {"tool", "llm"} <= {getattr(event, "category", None) for event in events}
    starts = {
        event.name: event.to_dict()
        for event in events
        if event.kind == "scope" and event.to_dict().get("scope_category") == "start"
    }
    assert starts["simple-vlm.turn"]["parent_uuid"] != (starts["agent:simple-vlm"]["uuid"])
    tool_starts = [
        event.to_dict()
        for event in events
        if event.kind == "scope"
        and event.to_dict().get("scope_category") == "start"
        and event.to_dict().get("category") == "tool"
    ]
    assert tool_starts
    assert tool_starts[0]["parent_uuid"] == starts["simple-vlm.turn"]["uuid"]
    llm_events = [event.to_json() for event in events if getattr(event, "category", None) == "llm"]
    assert llm_events
    assert any("<redacted:image>" in event for event in llm_events)


async def test_simple_vlm_agent_handles_global_interruption_event() -> None:
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
        "simple-vlm",
        _make_agent(_ConversationService(), _SelectedFrameTool(), BlockingVision()),
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
            VoiceInterrupted(),
            source="voice.interruption",
        )
        await asyncio.wait_for(cancelled.wait(), 1.0)
