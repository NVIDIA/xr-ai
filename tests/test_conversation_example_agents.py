# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle, isolation, and context-sharing tests for notes and direct vision."""

from __future__ import annotations

import asyncio
import sys
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from _conversation_output import CaptureRuntime, capture_emit
from _conversation_output import CapturingAgentRouter as AgentRouter
from _conversation_output import handle as _handle
from xr_ai_models import ChatStreamEvent
from xr_ai_runtime import AgentRuntime, Topic
from xr_ai_voice import VOICE_OUTPUT_TOPIC

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "agent-samples/conversation-example/worker"))

from conversation_example_worker._note_taking import NoteTakingAgent  # noqa: E402
from conversation_example_worker._visual_agents import (  # noqa: E402
    VisualObservationAgent,
)
from xr_ai_conversation import (  # noqa: E402
    ContextManager,
    Conversation,
    ConversationContextUpdate,
)
from xr_ai_routing import (  # noqa: E402
    Admission,
    AgentExchange,
    AgentTurn,
    DestinationResult,
    RoutingAid,
)
from xr_ai_tools.image import ImageReference  # noqa: E402


class _Decisions:
    def __init__(self, *choices: str) -> None:
        self.choices = deque(choices)
        self.calls = []

    async def decide(self, state, questions, *, images=(), timeout=None):
        self.calls.append((state, questions, timeout))
        key = next(iter(questions))
        choice = self.choices.popleft()
        return SimpleNamespace(
            answers={key: SimpleNamespace(choice=choice, confidence=1.0, probabilities={choice: 1.0})},
            model="fake",
            usage={},
            raw={},
        )


class _RouteDecisions:
    def __init__(self, *choices: str) -> None:
        self.choices = deque(choices)
        self.calls = []

    async def decide(self, state, questions, *, images=(), timeout=None):
        self.calls.append((state, questions, timeout))
        choice = self.choices.popleft()
        return SimpleNamespace(
            answers={"destination": SimpleNamespace(choice=choice, confidence=1.0, probabilities={choice: 1.0})},
            model="fake",
            usage={},
            raw={},
        )


class _Conversation:
    def __init__(self, name="conversation", answer=None, admit_gate=None, admit_entered=None):
        self.name = name
        self.answer_text = answer or f"Answered by {name}."
        self.admissions = []
        self.answers = []
        self.releases = []
        self.admit_gate = admit_gate
        self.admit_entered = admit_entered

    def routing_aid(self, _participant_id):
        return RoutingAid(self.name, f"Requests for {self.name}")

    async def admit(self, turn, *, focused):
        self.admissions.append((turn.participant_id, turn.query, focused))
        if self.admit_entered is not None:
            self.admit_entered.set()
        if self.admit_gate is not None:
            await self.admit_gate.wait()
        return Admission("accept")

    async def answer(self, turn, *, started):
        self.answers.append(turn)
        await capture_emit(self.name, self.answer_text)
        return DestinationResult("answer", self.answer_text)

    async def stream(self, request, participant_id, *, history=()):
        self.answers.append(SimpleNamespace(query=request, participant_id=participant_id, history=history))
        yield self.answer_text

    async def release(self, participant_id):
        self.releases.append(participant_id)

    async def participant_left(self, participant_id):
        self.releases.append(participant_id)


class _GatedTools:
    """Tools block on per-call events so the hot sampler cannot spin in tests."""

    def __init__(self, *outcomes):
        self.outcomes = deque(outcomes)
        self.started = asyncio.Queue()
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.active_by_participant = {}
        self.max_active_by_participant = {}
        self.parked = asyncio.Event()

    async def _next(self, request):
        index = len(self.calls)
        self.calls.append(request)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        participant_id = getattr(request, "participant_id", "query")
        participant_active = self.active_by_participant.get(participant_id, 0) + 1
        self.active_by_participant[participant_id] = participant_active
        self.max_active_by_participant[participant_id] = max(
            self.max_active_by_participant.get(participant_id, 0), participant_active
        )
        await self.started.put(index)
        try:
            if self.outcomes:
                gate, outcome = self.outcomes.popleft()
                await gate.wait()
            else:
                await self.parked.wait()
                outcome = None
            if isinstance(outcome, BaseException):
                raise outcome
            return index, outcome
        finally:
            self.active -= 1
            self.active_by_participant[participant_id] -= 1


class _Frames(_GatedTools):
    async def execute(self, request):
        index, outcome = await self._next(request)
        if outcome is not None:
            return outcome
        timestamp_us = int(datetime(2026, 10, 15, 16, index, tzinfo=UTC).timestamp() * 1_000_000)
        return SimpleNamespace(
            image=ImageReference(uri=f"xr-image://test/frame-{index}"),
            timestamp_us=timestamp_us,
        )


class _ImageQuery(_GatedTools):
    async def execute(self, request):
        index, outcome = await self._next(request)
        if outcome is not None:
            return outcome
        return SimpleNamespace(available=True, text=f"Observation {index + 1}.")


class _StreamingVision:
    def __init__(self, text: str):
        self.text = text
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        yield SimpleNamespace(text=self.text)


async def _turn(router, query, *, participant="alice", turn_id=None, history=()):
    output = []

    async def emit(source, text):
        output.append((source, text))

    result = await _handle(
        router,
        AgentTurn(participant, turn_id or f"{participant}-{query}", query, history=tuple(history)),
        emit=emit,
    )
    return result, output


async def _next_call(tool):
    return await asyncio.wait_for(tool.started.get(), timeout=1)


@pytest.mark.asyncio
async def test_observation_native_admission_overrides_poisoned_history_without_generic_admission():
    frame_gate = asyncio.Event()
    frames = _Frames((frame_gate, None))
    vision = _StreamingVision("The marker is red.")
    decisions = _RouteDecisions("observation")
    observation = VisualObservationAgent(frames, vision, runtime=CaptureRuntime(), output_topic=VOICE_OUTPUT_TOPIC)
    conversation = _Conversation("conversation")
    notes = NoteTakingAgent(
        _Conversation("notes"),
        _Decisions(),
        context_budget_bytes=4096,
        runtime=CaptureRuntime(),
        output_topic=VOICE_OUTPUT_TOPIC,
    )
    router = AgentRouter(
        {
            "conversation": conversation,
            "observation": observation,
            "notes": notes,
        },
        decisions,
    )

    pending = asyncio.create_task(
        _turn(
            router,
            "What color is the marker on the display?",
            turn_id="current-observation",
            history=(AgentExchange("What do you see?", "No camera frame was available."),),
        )
    )
    await _next_call(frames)
    frame_gate.set()
    result, output = await pending
    # Exercise the public frontend history path with a stale assistant claim.
    # The current-view destination still accepts natively and queries one fresh frame.
    assert result.disposition == "answer"
    assert output == [("observation", "The marker is red.")]
    assert len(frames.calls) == 1
    assert frames.calls[0].participant_id == "alice"
    assert len(vision.requests) == 1
    assert vision.requests[0].query == (
        "Frame source time: 2026-10-15 16:00 UTC.\nWhat color is the marker on the display?"
    )
    assert len(decisions.calls) == 1
    assert list(decisions.calls[0][1]) == ["destination"]
    assert decisions.calls[0][0]["completed_history"] == [
        {
            "user": "What do you see?",
            "assistant": "No camera frame was available.",
            "recorded_around_utc": "unknown",
        }
    ]
    assert "No camera frame was available." not in vision.requests[0].query
    await router.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["start", "capture", "finish"])
async def test_cancelled_note_transition_keeps_reported_activity_aligned_with_focus(transition):
    status_publishing = asyncio.Event()
    release_publication = asyncio.Event()
    updates = []

    async def publish(_participant_id, update):
        updates.append(update.text)
        should_block = (
            (transition == "start" and "Session status: active" in update.text)
            or (transition == "capture" and "Cancellation should not discard this note." in update.text)
            or (transition == "finish" and "Session status: not active" in update.text)
        )
        if should_block:
            status_publishing.set()
            await release_publication.wait()

    note_action = {
        "start": "start",
        "capture": "capture",
        "finish": "finish",
    }[transition]
    actions = _Decisions("start", note_action) if transition != "start" else _Decisions("start")
    notes = NoteTakingAgent(
        _Conversation("notes"),
        actions,
        context_budget_bytes=4_096,
        publish_context=publish,
        runtime=CaptureRuntime(),
        output_topic=VOICE_OUTPUT_TOPIC,
    )
    router = AgentRouter(
        {"conversation": _Conversation(), "notes": notes},
        _RouteDecisions("notes"),
    )

    async def emit(_source, _text):
        return None

    if transition != "start":
        started = await _handle(router, AgentTurn("alice", "start", "Start taking notes"), emit=emit)
        assert started.disposition == "focus"

    query = {
        "start": "Start taking notes",
        "capture": "Cancellation should not discard this note.",
        "finish": "Finish note-taking",
    }[transition]
    response = asyncio.create_task(_handle(router, AgentTurn("alice", transition, query), emit=emit))
    await asyncio.wait_for(status_publishing.wait(), timeout=1)
    response.cancel()
    release_publication.set()
    await asyncio.gather(response, return_exceptions=True)

    expected_active = transition != "start"
    assert ("alice" in notes._active_participants) is expected_active  # noqa: SLF001
    assert (router._focus.get("alice") == "notes") is expected_active  # noqa: SLF001
    latest = updates[-1]
    assert ("Session status: active" in latest) is expected_active
    if transition == "capture":
        assert "Cancellation should not discard this note." in latest
        assert "Cancellation should not discard this note." in notes.routing_aid("alice").description
    await router.stop()


@pytest.mark.asyncio
async def test_note_question_handoff_preserves_focus_until_explicit_finish():
    note_conversation = _Conversation("notes")
    conversation = _Conversation("conversation")
    published = []

    async def publish_note(participant_id, update):
        published.append((participant_id, update))

    notes = NoteTakingAgent(
        note_conversation,
        _Decisions("start", "capture", "handoff", "capture", "finish"),
        context_budget_bytes=4096,
        publish_context=publish_note,
        runtime=CaptureRuntime(),
        output_topic=VOICE_OUTPUT_TOPIC,
    )
    decisions = _RouteDecisions("notes", "conversation")
    router = AgentRouter({"conversation": conversation, "notes": notes}, decisions)

    await _turn(router, "Take some notes.")
    captured, _ = await _turn(router, "Record that the headset text should stay legible.")
    assert captured.disposition == "focus"
    assert "headset text should stay legible" in notes.routing_aid("alice").description

    query = "How should I evaluate AR/VR text legibility?"
    result, output = await _turn(router, query)
    assert result.disposition == "answer"
    assert len(conversation.answers) == 1
    assert conversation.answers[0].query == query
    # A focused handoff admission selects the eligible general destination
    # directly; it does not ask the router to choose a second time.
    assert len(decisions.calls) == 1
    assert output == [("conversation", "Answered by conversation.")]
    assert "headset text should stay legible" in notes.routing_aid("alice").description
    assert len(published) == 2
    assert "Session status: active" in published[0][1].text
    assert "headset text should stay legible" in published[1][1].text
    assert "Session status: active" in published[-1][1].text
    resumed, _ = await _turn(router, "Record that the controls need larger targets.")
    assert resumed.disposition == "focus"
    assert "controls need larger targets" in notes.routing_aid("alice").description
    assert len(conversation.answers) == 1
    finished, _ = await _turn(router, "Finish taking notes.")
    assert finished.disposition == "release"
    assert "Session status: not active" in published[-1][1].text
    await _turn(router, "What is a rainbow?")
    assert conversation.answers[-1].query == "What is a rainbow?"
    await router.stop()


@pytest.mark.asyncio
async def test_note_context_is_participant_scoped_and_cleared_on_departure():
    notes = NoteTakingAgent(
        _Conversation("notes"),
        _Decisions("start", "capture", "start", "capture"),
        context_budget_bytes=4096,
        runtime=CaptureRuntime(),
        output_topic=VOICE_OUTPUT_TOPIC,
    )
    decisions = _RouteDecisions("notes", "notes", "notes", "notes")
    router = AgentRouter({"conversation": _Conversation(), "notes": notes}, decisions)

    await _turn(router, "Take some notes.", participant="alice")
    await _turn(router, "Record a short headset note.", participant="alice")
    await _turn(router, "Take some notes.", participant="bob")
    await _turn(router, "Record a separate room note.", participant="bob")

    alice_aid = notes.routing_aid("alice").description
    bob_aid = notes.routing_aid("bob").description
    assert "headset note" in alice_aid
    assert "room note" not in alice_aid
    assert "room note" in bob_aid
    assert "headset note" not in bob_aid

    await router.participant_left("alice")
    assert "Latest completed note" not in notes.routing_aid("alice").description
    assert "room note" in notes.routing_aid("bob").description
    await router.stop()


@pytest.mark.asyncio
async def test_note_topic_reaches_general_conversation_after_focus_finishes():
    class CapturingLLM:
        def __init__(self):
            self.messages = []

        def stream_events(self, messages, **_kwargs):
            self.messages.append(messages)

            async def events():
                yield ChatStreamEvent(kind="text", text="I found your note about the headset display.")
                yield ChatStreamEvent(kind="finish", finish_reason="stop")

            return events()

    note_topic = Topic("conversation-example.foreground-notes", ConversationContextUpdate)
    runtime = AgentRuntime()
    llm = CapturingLLM()
    manager = ContextManager(
        llm,
        summary_context_topics=(note_topic.name,),
        dialogue_budget_bytes=2048,
        topic_budget_bytes={note_topic.name: 2048},
        compaction_input_budget_bytes=4096,
    )
    general_conversation = Conversation(
        llm,
        context=manager,
        instructions="Answer the participant using their conversation and named context.",
        runtime=CaptureRuntime(),
        output_topic=VOICE_OUTPUT_TOPIC,
    )
    runtime.register("general-context", manager.subscriber())
    note_conversation = _Conversation("notes")

    published = []

    async def publish_note(participant_id, update):
        published.append(update)
        await runtime.publish(note_topic, update, participant_id=participant_id)

    notes = NoteTakingAgent(
        note_conversation,
        _Decisions("start", "capture", "capture", "finish"),
        context_budget_bytes=4096,
        publish_context=publish_note,
        runtime=CaptureRuntime(),
        output_topic=VOICE_OUTPUT_TOPIC,
    )
    decisions = _RouteDecisions("notes", "conversation")
    router = AgentRouter(
        {
            "conversation": general_conversation,
            "notes": notes,
        },
        decisions,
    )

    async with runtime:
        await _turn(router, "Take some notes.")
        note = "Record that the headset display is easiest to read from the left side."
        await _turn(router, note)
        second_note = "Record that the demo starts at the east entrance."
        await _turn(router, second_note)
        assert len(published) == 3
        assert "Session status: active" in published[0].text
        assert note in published[1].text
        assert note in published[2].text and second_note in published[2].text
        await _turn(router, "I'm finished taking notes.")
        result, _ = await _turn(router, "What did I note about reading the display?")

    assert result.disposition == "answer"
    assert len(llm.messages) == 1
    prompt = "\n".join(str(message.content) for message in llm.messages[0])
    assert "Source conversation-example.foreground-notes" in prompt
    assert note in prompt
    assert second_note in prompt
    await general_conversation.close()
    await router.stop()
