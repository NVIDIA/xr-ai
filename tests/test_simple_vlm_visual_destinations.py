# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused behavior tests for the sample's split visual destinations."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

from xr_ai_conversation import (
    Conversation,
    ConversationContextUpdate,
    ConversationExchange,
    ConversationTurn,
    RoutingAid,
)
from xr_ai_models import ChatResponse, ChatStreamEvent, ToolCall
from xr_ai_runtime import AgentRuntime, Topic
from xr_ai_tools.image import ImageReference
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryResult

_ROOT = Path(__file__).resolve().parents[1]
_WORKER = _ROOT / "agent-samples" / "simple-vlm-example" / "worker"
sys.path.insert(0, str(_WORKER))

from simple_vlm_example_worker.visual_destinations import (  # noqa: E402
    EnvironmentReasoningDestination,
    VisualObservationDestination,
)


class _PlannerLLM:
    def __init__(self, *, block_final: bool = False, plan: str | None = None) -> None:
        self.requests = []
        self.final_started = asyncio.Event()
        self.final_gate = asyncio.Event()
        self.block_final = block_final
        self.plan = plan

    def stream_events(self, messages, *, tools=None, **_kwargs):
        self.requests.append((messages, tools))

        async def events():
            if tools:
                if self.plan is None:
                    yield ChatStreamEvent(kind="text", text="I can help with that.")
                    yield ChatStreamEvent(kind="finish", finish_reason="stop")
                    return
                yield ChatStreamEvent(
                    kind="tool_call",
                    tool_call=ToolCall(
                        "visual-call",
                        "inspect_current_view",
                        json.dumps({"query": self.plan}),
                    ),
                )
                yield ChatStreamEvent(kind="finish", finish_reason="tool_calls")
                return
            self.final_started.set()
            if self.block_final:
                await self.final_gate.wait()
            yield ChatStreamEvent(kind="text", text="Use the whiteboard for the reminder.")
            yield ChatStreamEvent(kind="finish", finish_reason="stop")

        return events()

    async def chat(self, messages, *, tools=None, **_kwargs) -> ChatResponse:
        self.requests.append((messages, tools))
        if tools and self.plan is not None:
            return ChatResponse(
                "",
                None,
                [ToolCall("visual-call", "inspect_current_view", json.dumps({"query": self.plan}))],
                "stop",
                {},
            )
        if tools:
            return ChatResponse("I can answer from context.", None, None, "stop", {})
        self.final_started.set()
        if self.block_final:
            await self.final_gate.wait()
        return ChatResponse("Use the whiteboard for the reminder.", None, None, "stop", {})


class _Frames:
    def __init__(self, *, error: Exception | None = None, timestamp_us: int | None = None) -> None:
        self.error = error
        self.timestamp_us = timestamp_us
        self.requests = []

    async def execute(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(image=ImageReference(uri="image://current"), timestamp_us=self.timestamp_us)


class _ImageQuery:
    def __init__(self, result: ImageQueryResult) -> None:
        self.result = result
        self.requests = []

    async def execute(self, request: ImageQueryRequest) -> ImageQueryResult:
        self.requests.append(request)
        return self.result


class _StreamingVision:
    def __init__(self) -> None:
        self.requests = []

    def stream(self, request: ImageQueryRequest) -> AsyncIterator[SimpleNamespace]:
        self.requests.append(request)

        async def chunks():
            yield SimpleNamespace(text="The monitor is off.")

        return chunks()


async def test_environment_plans_targeted_visual_tool_then_synthesizes_from_context() -> None:
    llm = _PlannerLLM(plan="What clearly visible flat surfaces are present? If none are visible, say none.")
    conversation = Conversation(
        llm,  # type: ignore[arg-type]
        instructions="Use visual evidence to answer the user's broader goal.",
        routing=RoutingAid("environment", "Visual environment reasoning"),
    )
    frames = _Frames(timestamp_us=1_790_856_000_000_000)
    image_query = _ImageQuery(ImageQueryResult(text="A whiteboard is visible.", available=True))
    topic = Topic("test.environment-evidence", ConversationContextUpdate)
    runtime = AgentRuntime()
    runtime.register("environment-context", conversation.context_subscriber((topic,)))

    async def publish_evidence(participant_id, update):
        await runtime.publish(topic, update, participant_id=participant_id)

    destination = EnvironmentReasoningDestination(
        llm,  # type: ignore[arg-type]
        frames,  # type: ignore[arg-type]
        image_query,  # type: ignore[arg-type]
        conversation,
        publish_evidence,
        context_budget_bytes=12_000,
    )
    turn = ConversationTurn(
        participant_id="alice",
        turn_id="turn-1",
        query="Where should I write a reminder?",
    )
    await destination.admit(turn, focused=False)
    emitted = []

    async def emit(source: str, text: str) -> None:
        emitted.append((source, text))

    async with runtime:
        result = await destination.answer(turn, emit=emit)

    assert result.text == "Use the whiteboard for the reminder."
    assert emitted == [("environment", "Use the whiteboard for the reminder.")]
    assert [request.participant_id for request in frames.requests] == ["alice"]
    assert image_query.requests == [
        ImageQueryRequest(
            image=ImageReference(uri="image://current"),
            query="What clearly visible flat surfaces are present? If none are visible, say none.",
        )
    ]
    assert len(llm.requests) == 2
    planner_messages, planner_tools = llm.requests[0]
    assert planner_tools[0].name == "inspect_current_view"
    assert "without presupposing an unseen object" in planner_tools[0].description
    assert planner_messages[-1].content.endswith("Where should I write a reminder?")
    synthesis_messages, synthesis_tools = llm.requests[1]
    assert synthesis_tools == ()
    assert synthesis_messages[-1].content == "Where should I write a reminder?"
    assert "A whiteboard is visible." in synthesis_messages[1].content
    assert "Frame source time: 2026-10-01 12:00 UTC" in synthesis_messages[1].content
    assert [(item.user, item.assistant) for item in conversation._context.transcript("alice")] == [
        ("Where should I write a reminder?", "Use the whiteboard for the reminder.")
    ]
    assert conversation._context.sections("alice") == ((topic.name, "No current-turn visual evidence is active."),)
    await conversation.close()


async def test_environment_can_answer_from_history_without_requesting_a_new_frame() -> None:
    llm = _PlannerLLM()
    conversation = Conversation(llm, routing=RoutingAid("environment", "Visual reasoning"))  # type: ignore[arg-type]
    frames = _Frames()
    image_query = _ImageQuery(ImageQueryResult(text="unused", available=True))
    destination = EnvironmentReasoningDestination(
        llm,  # type: ignore[arg-type]
        frames,  # type: ignore[arg-type]
        image_query,  # type: ignore[arg-type]
        conversation,
        lambda *_args: asyncio.sleep(0),
        context_budget_bytes=12_000,
    )
    await conversation._update_context(
        "alice",
        "room.notes",
        ConversationContextUpdate(mode="summary", text="A whiteboard is beside the table."),
        timestamp_us=1_790_856_000_000_000,
    )
    turn = ConversationTurn(
        "alice",
        "turn-history-answer",
        "Based on that earlier observation, where could I write a short reminder?",
        (ConversationExchange("What writing surface is visible?", "A whiteboard is visible beside the table."),),
    )
    emitted = []

    async def emit(source: str, text: str) -> None:
        emitted.append((source, text))

    result = await destination.answer(turn, emit=emit)

    assert result.text == "Use the whiteboard for the reminder."
    assert emitted == [("environment", "Use the whiteboard for the reminder.")]
    assert frames.requests == []
    assert image_query.requests == []
    assert any("Recorded time: unknown" in message.content for message in llm.requests[0][0])
    assert "A whiteboard is beside the table." in llm.requests[-1][0][1].content
    assert conversation._context.transcript("alice")[-1].assistant == result.text
    assert "2026-10-01 12:00 UTC" in llm.requests[-1][0][1].content
    await conversation.close()


async def test_environment_clears_temporary_evidence_after_interruption() -> None:
    llm = _PlannerLLM(
        block_final=True,
        plan="What visible surface could hold a note?",
    )
    conversation = Conversation(llm, routing=RoutingAid("environment", "Visual reasoning"))  # type: ignore[arg-type]
    frames = _Frames()
    image_query = _ImageQuery(ImageQueryResult(text="A whiteboard is visible.", available=True))
    topic = Topic("test.environment-evidence-cancel", ConversationContextUpdate)
    runtime = AgentRuntime()
    runtime.register("environment-context", conversation.context_subscriber((topic,)))

    async def publish_evidence(participant_id, update):
        await runtime.publish(topic, update, participant_id=participant_id)

    destination = EnvironmentReasoningDestination(
        llm,  # type: ignore[arg-type]
        frames,  # type: ignore[arg-type]
        image_query,  # type: ignore[arg-type]
        conversation,
        publish_evidence,
        context_budget_bytes=12_000,
    )
    turn = ConversationTurn("alice", "turn-cancel", "Where could I leave a note?")
    await destination.admit(turn, focused=False)

    async def emit(_source: str, _text: str) -> None:
        pass

    async with runtime:
        task = asyncio.create_task(destination.answer(turn, emit=emit))
        await asyncio.wait_for(llm.final_started.wait(), 1)
        assert "A whiteboard is visible." in conversation._context.sections("alice")[0][1]
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert conversation._context.sections("alice") == ((topic.name, "No current-turn visual evidence is active."),)
    await conversation.close()


async def test_observation_asks_cosmos_directly_without_an_omni_planner() -> None:
    frames = _Frames()
    vision = _StreamingVision()
    destination = VisualObservationDestination(
        frames,  # type: ignore[arg-type]
        vision,  # type: ignore[arg-type]
        context_budget_bytes=12_000,
    )
    turn = ConversationTurn("alice", "turn-observe", "Is the monitor screen illuminated?")
    emitted = []

    async def emit(source: str, text: str) -> None:
        emitted.append((source, text))

    result = await destination.answer(turn, emit=emit)

    assert frames.requests[0].participant_id == "alice"
    assert vision.requests == [
        ImageQueryRequest(
            image=ImageReference(uri="image://current"),
            query="Is the monitor screen illuminated?",
        )
    ]
    assert emitted == [("observation", "The monitor is off.")]
    assert result.text == "The monitor is off."


async def test_observation_labels_the_actual_frame_timestamp() -> None:
    frames = _Frames(timestamp_us=1_790_856_000_000_000)
    vision = _StreamingVision()
    destination = VisualObservationDestination(
        frames,  # type: ignore[arg-type]
        vision,  # type: ignore[arg-type]
        context_budget_bytes=12_000,
    )
    turn = ConversationTurn("alice", "turn-observe-time", "Is the screen illuminated?")

    async def emit(_source: str, _text: str) -> None:
        pass

    await destination.answer(turn, emit=emit)

    assert vision.requests[0].query.startswith("Frame source time: 2026-10-01 12:00 UTC.")


async def test_environment_invalid_visual_plan_fails_closed_without_exposing_planner_text() -> None:
    llm = _PlannerLLM(plan="")
    conversation = Conversation(llm, routing=RoutingAid("environment", "Visual reasoning"))  # type: ignore[arg-type]
    frames = _Frames()
    image_query = _ImageQuery(ImageQueryResult(text="A whiteboard is visible.", available=True))
    topic = Topic("test.environment-invalid-plan", ConversationContextUpdate)
    runtime = AgentRuntime()
    runtime.register("environment-context", conversation.context_subscriber((topic,)))

    async def publish_evidence(participant_id, update):
        await runtime.publish(topic, update, participant_id=participant_id)

    destination = EnvironmentReasoningDestination(
        llm,  # type: ignore[arg-type]
        frames,  # type: ignore[arg-type]
        image_query,  # type: ignore[arg-type]
        conversation,
        publish_evidence,
        context_budget_bytes=12_000,
    )
    turn = ConversationTurn("alice", "turn-invalid-plan", "Where should I write a reminder?")
    await destination.admit(turn, focused=False)
    emitted = []

    async def emit(source: str, text: str) -> None:
        emitted.append((source, text))

    async with runtime:
        result = await destination.answer(turn, emit=emit)

    assert result.text == "I couldn't identify a visual detail to check. Please ask about a specific thing in view."
    assert emitted == [("environment", result.text)]
    assert len(llm.requests) == 1
    assert llm.requests[0][1][0].name == "inspect_current_view"
    assert frames.requests == []
    assert image_query.requests == []
    assert conversation._context.transcript("alice") == ()
    await conversation.close()
