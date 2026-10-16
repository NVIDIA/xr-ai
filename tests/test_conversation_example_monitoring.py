# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle, isolation, and context-sharing tests for monitor and notes."""

from __future__ import annotations

import asyncio
import sys
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from xr_ai_hub import FrameUnavailable
from xr_ai_models import ChatStreamEvent
from xr_ai_runtime import AgentRuntime, Topic

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "agent-samples/conversation-example/worker"))

from conversation_example_worker.conversation_destinations import (  # noqa: E402
    ContextualConversationAgent,
    NoteTakingAgent,
)
from conversation_example_worker.monitoring import (  # noqa: E402
    VisualMonitorAgent,
)
from conversation_example_worker.visual_destinations import (  # noqa: E402
    VisualObservationAgent,
)
from xr_ai_conversation import (  # noqa: E402
    Admission,
    Conversation,
    ConversationContextUpdate,
    ConversationExchange,
    ConversationFrontEnd,
    ConversationTurn,
    DestinationResult,
    RoutingAid,
)
from xr_ai_tools.image import ImageReference  # noqa: E402


class _Decisions:
    def __init__(self, *choices: str) -> None:
        self.choices = deque(choices)
        self.calls = []

    async def decide(self, state, questions, *, timeout=None):
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

    async def decide(self, state, questions, *, timeout=None):
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

    async def answer(self, turn, *, emit):
        self.answers.append(turn)
        await emit(self.name, self.answer_text)
        return DestinationResult("answer", self.answer_text)

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


def _monitor(
    decisions,
    frames,
    image_query,
    *,
    publisher=None,
    window_size=20,
):
    updates = []

    async def publish(participant_id, update):
        updates.append((participant_id, update))
        if publisher is not None:
            await publisher(participant_id, update)

    destination = VisualMonitorAgent(
        frames,
        image_query,
        decisions,
        publish,
        window_size=window_size,
        context_budget_bytes=4096,
    )
    return destination, updates


async def _turn(front_end, query, *, participant="alice", turn_id=None, history=()):
    output = []

    async def emit(source, text):
        output.append((source, text))

    result = await front_end.handle(
        ConversationTurn(participant, turn_id or f"{participant}-{query}", query, history=tuple(history)),
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
    router = _RouteDecisions("observation")
    observation = VisualObservationAgent(frames, vision)
    conversation = _Conversation("conversation")
    notes = NoteTakingAgent(_Conversation("notes"), _Decisions(), context_budget_bytes=4096)
    monitor, _ = _monitor(_Decisions(), _Frames(), _ImageQuery())
    front_end = ConversationFrontEnd(
        {
            "conversation": conversation,
            "observation": observation,
            "monitor": monitor,
            "notes": notes,
        },
        router,
    )

    pending = asyncio.create_task(
        _turn(
            front_end,
            "What color is the marker on the display?",
            turn_id="current-observation",
            history=(ConversationExchange("What do you see?", "No camera frame was available."),),
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
    assert len(router.calls) == 1
    assert list(router.calls[0][1]) == ["destination"]
    assert router.calls[0][0]["completed_history"] == [
        {
            "user": "What do you see?",
            "assistant": "No camera frame was available.",
            "recorded_around_utc": "unknown",
        }
    ]
    assert "No camera frame was available." not in vision.requests[0].query
    await front_end.stop()


@pytest.mark.asyncio
async def test_monitor_start_uses_native_monitor_action_without_generic_admission_veto():
    frame_gate, query_gate = asyncio.Event(), asyncio.Event()
    frames = _Frames((frame_gate, None))
    image_query = _ImageQuery((query_gate, None))
    monitor_actions = _Decisions("start")
    monitor, _ = _monitor(monitor_actions, frames, image_query)
    router = _RouteDecisions("monitor")
    front_end = ConversationFrontEnd({"conversation": _Conversation(), "monitor": monitor}, router)

    result, output = await _turn(front_end, "Start tracking the current workspace.")
    assert result.disposition == "answer"
    assert output == [("monitor", "I'll start tracking the current view in the background.")]
    await _next_call(frames)
    frame_gate.set()
    await _next_call(image_query)
    query_gate.set()
    await _next_call(frames)

    assert len(router.calls) == 1
    assert list(router.calls[0][1]) == ["destination"]
    assert len(monitor_actions.calls) == 1
    assert list(monitor_actions.calls[0][1]) == ["monitor_action"]
    assert all("admission" not in questions for _state, questions, _timeout in monitor_actions.calls)
    await front_end.stop()
    assert frames.active == image_query.active == 0


@pytest.mark.asyncio
async def test_monitor_handoff_declines_once_and_preserves_background_sampling():
    frame_gate, query_gate = asyncio.Event(), asyncio.Event()
    frames = _Frames((frame_gate, None))
    image_query = _ImageQuery((query_gate, None))
    monitor_actions = _Decisions("start", "handoff")
    monitor, _ = _monitor(monitor_actions, frames, image_query)
    conversation = _Conversation("conversation")
    router = _RouteDecisions("monitor", "monitor")
    front_end = ConversationFrontEnd({"conversation": conversation, "monitor": monitor}, router)

    await _turn(front_end, "Track activity by the entrance.")
    await _next_call(frames)
    frame_gate.set()
    await _next_call(image_query)
    query_gate.set()
    # The next event-gated frame call proves the monitor remains active while
    # the content request is handed off to the general conversation.
    await _next_call(frames)

    query = "What did the monitor last see?"
    result, output = await _turn(front_end, query)

    assert result.disposition == "answer"
    assert output == [("conversation", "Answered by conversation.")]
    assert len(conversation.answers) == 1
    assert conversation.answers[0].query == query
    # The router handles the start and the initial content request. After the
    # monitor declines, Conversation is the sole remaining candidate, so FE
    # routes the same turn there without a second model decision.
    assert len(router.calls) == 2
    assert list(router.calls[0][1]) == ["destination"]
    assert list(router.calls[1][1]) == ["destination"]
    assert router.calls[1][0]["request"] == query
    assert [item["name"] for item in router.calls[1][0]["destinations"]] == ["conversation", "monitor"]
    assert [list(call[1]) for call in monitor_actions.calls] == [["monitor_action"], ["monitor_action"]]
    assert monitor_actions.calls[1][0]["request"] == query
    assert monitor.routing_aid("alice").eligibility == "active"
    assert frames.active == 1
    assert image_query.active == 0

    await front_end.stop()
    await monitor.stop()
    assert frames.active == image_query.active == 0


@pytest.mark.asyncio
async def test_active_monitor_task_augments_general_routing_aid_but_not_current_view_evidence():
    monitor_frame_gate, monitor_query_gate = asyncio.Event(), asyncio.Event()
    monitor_frames = _Frames((monitor_frame_gate, None))
    monitor_images = _ImageQuery((monitor_query_gate, None))
    monitor_actions = _Decisions("start", "status", "stop", "start")
    monitor, monitor_updates = _monitor(monitor_actions, monitor_frames, monitor_images)

    general_inner = _Conversation("conversation")
    general = ContextualConversationAgent(
        general_inner,
        static_description="Answer general text questions and use completed context.",
        context_budget_bytes=4096,
        routing_context=monitor.routing_context,
    )
    visual_gate = asyncio.Event()
    visual_frames = _Frames((visual_gate, None))
    vision = _StreamingVision("The exit sign is green.")
    observation = VisualObservationAgent(visual_frames, vision)
    router = _RouteDecisions(
        "monitor", "conversation", "conversation", "observation", "monitor", "monitor", "conversation", "monitor"
    )
    front_end = ConversationFrontEnd(
        {
            "conversation": general,
            "observation": observation,
            "monitor": monitor,
            "notes": NoteTakingAgent(_Conversation("notes"), _Decisions(), context_budget_bytes=4096),
        },
        router,
    )

    assert "Monitor task coverage" not in general.routing_aid("alice").description
    started, _ = await _turn(front_end, "Keep track of where I leave my pen.")
    assert started.disposition == "answer"
    await _turn(front_end, "Where is my pen?")
    no_observation_candidate = next(
        item for item in router.calls[1][0]["destinations"] if item["name"] == "conversation"
    )
    assert "no completed observation is available yet" in no_observation_candidate["description"]

    await _next_call(monitor_frames)
    monitor_frame_gate.set()
    await _next_call(monitor_images)
    monitor_query_gate.set()
    await _next_call(monitor_frames)

    active_aid = general.routing_aid("alice").description
    assert "Monitor task coverage" in active_aid
    assert "Active task, using the latest start/retarget instruction" in active_aid
    assert "retarget replaces the previous task" in active_aid
    assert "Keep track of where I leave my pen." in active_aid
    assert "Observation 1." not in active_aid

    await _turn(front_end, "Where did monitoring last see my pen?")
    conversation_candidate = next(item for item in router.calls[2][0]["destinations"] if item["name"] == "conversation")
    assert "Active task, using the latest start/retarget instruction" in conversation_candidate["description"]
    assert "Keep track of where I leave my pen." in conversation_candidate["description"]
    assert "through" in conversation_candidate["description"]
    assert general_inner.answers[-1].query == "Where did monitoring last see my pen?"

    visual_task = asyncio.create_task(
        _turn(front_end, "Look at the current camera frame: what color is the exit sign?")
    )
    await _next_call(visual_frames)
    visual_gate.set()
    visual_result, visual_output = await visual_task
    assert visual_result.disposition == "answer"
    assert visual_output == [("observation", "The exit sign is green.")]
    assert vision.requests[0].query == (
        "Frame source time: 2026-10-15 16:00 UTC.\nLook at the current camera frame: what color is the exit sign?"
    )

    await _turn(front_end, "Is background monitoring still active?")
    assert monitor.routing_context("alice") is not None
    await _turn(front_end, "Stop tracking my pen.")
    stopped_context = monitor.routing_context("alice")
    assert stopped_context is not None
    assert "stopped" in stopped_context.casefold()
    assert "Active task:" not in stopped_context
    assert len(monitor_updates) >= 2

    await _turn(front_end, "Where did the monitor last see my pen?")
    final_candidate = next(item for item in router.calls[-1][0]["destinations"] if item["name"] == "conversation")
    assert "stopped" in final_candidate["description"].casefold()
    assert "not current-view evidence" in final_candidate["description"]

    await _turn(front_end, "Start tracking the front door instead.")
    retargeted_context = monitor.routing_context("alice")
    assert retargeted_context is not None
    assert "front door instead" in retargeted_context
    assert "pen" not in retargeted_context.casefold()
    assert [state["request"] for state, _questions, _timeout in monitor_actions.calls] == [
        "Keep track of where I leave my pen.",
        "Is background monitoring still active?",
        "Stop tracking my pen.",
        "Start tracking the front door instead.",
    ]
    await front_end.stop()
    assert monitor_frames.active == monitor_images.active == visual_frames.active == 0


@pytest.mark.asyncio
async def test_monitor_retarget_drains_old_observation_before_replacing_scope_and_window():
    frame_gate = asyncio.Event()
    frames = _Frames((frame_gate, None))

    class _LateImageQuery:
        def __init__(self):
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.release = asyncio.Event()
            self.requests = []

        async def execute(self, request):
            self.requests.append(request)
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                await self.release.wait()
            return SimpleNamespace(available=True, text="Old pen observation.")

    image_query = _LateImageQuery()
    monitor, updates = _monitor(_Decisions("start", "retarget", "stop"), frames, image_query)
    router = _RouteDecisions("monitor", "monitor", "monitor")
    front_end = ConversationFrontEnd({"conversation": _Conversation(), "monitor": monitor}, router)

    await _turn(front_end, "Keep track of where I leave my pen.")
    await _next_call(frames)
    frame_gate.set()
    await image_query.started.wait()

    retarget = asyncio.create_task(_turn(front_end, "Retarget monitoring to the doorway instead."))
    await image_query.cancelled.wait()
    assert not retarget.done()
    assert len(frames.calls) == 1

    image_query.release.set()
    result, output = await retarget
    assert result.disposition == "answer"
    assert output == [("monitor", "I'll start tracking the current view in the background.")]
    await _next_call(frames)

    cue = monitor.routing_context("alice")
    assert cue is not None
    assert "doorway instead" in cue
    assert "pen" not in cue.casefold()
    latest_window = updates[-1][1].text
    assert "Old pen observation." not in latest_window
    assert "status: active" in latest_window

    await _turn(front_end, "Stop tracking the doorway.")
    await front_end.stop()
    assert frames.active == 0


@pytest.mark.asyncio
async def test_monitor_continues_during_notes_focus_and_router_resumes_after_finish():
    frame1, query1, frame2, query2 = (asyncio.Event() for _ in range(4))
    frames = _Frames((frame1, None), (frame2, None))
    image_query = _ImageQuery((query1, None), (query2, None))
    monitor_actions = _Decisions("start", "stop")
    monitor, monitor_updates = _monitor(monitor_actions, frames, image_query)
    general = _Conversation("conversation")
    note_actions = _Decisions("start", "capture", "readback", "other", "finish")
    note_conversation = _Conversation("notes", answer="I can help with the note request.")
    note_updates = []

    async def publish_note(participant_id, update):
        note_updates.append((participant_id, update))

    notes = NoteTakingAgent(
        note_conversation,
        note_actions,
        context_budget_bytes=4096,
        publish_context=publish_note,
    )
    router = _RouteDecisions("conversation", "monitor", "notes", "conversation", "monitor", "conversation")
    front_end = ConversationFrontEnd(
        {
            "conversation": general,
            "notes": notes,
            "monitor": monitor,
        },
        router,
    )

    await _turn(front_end, "What is refraction?")
    assert frames.calls == image_query.calls == []

    started, output = await _turn(front_end, "Start tracking the current room.")
    assert started.disposition == "answer"
    assert output == [("monitor", "I'll start tracking the current view in the background.")]
    assert monitor.routing_aid("alice").eligibility == "active"
    await _next_call(frames)
    frame1.set()
    await _next_call(image_query)
    query1.set()
    await _next_call(frames)

    # Note capture is participant-focused, but ordinary topic changes stay in
    # the note session and do not stop the independent background sampler.
    entered_notes, note_start_output = await _turn(front_end, "Take some notes.")
    assert entered_notes.disposition == "focus"
    assert note_start_output == [("notes", "I’m ready to take notes.")]
    captured, _ = await _turn(front_end, "Record this note: AR/VR labels should remain readable.")
    assert captured.disposition == "focus"
    assert "Record this note: AR/VR labels should remain readable." in note_updates[-1][1].text
    update_count = len(note_updates)
    read_back, readback_output = await _turn(front_end, "Read back my notes.")
    assert read_back.disposition == "focus"
    assert readback_output == [("notes", "Your notes: " + note_updates[-1][1].text)]
    assert len(note_updates) == update_count

    frame2.set()
    await _next_call(image_query)
    query2.set()
    await _next_call(frames)
    assert len(monitor_updates) >= 2

    topic_change, _ = await _turn(front_end, "What is 18 percent of 250?")
    assert topic_change.disposition == "focus"
    assert len(router.calls) == 3
    exited, finish_output = await _turn(front_end, "I'm finished taking notes.")
    assert exited.disposition == "release"
    assert finish_output == [("notes", "Note-taking is finished.")]
    assert len(note_updates) == update_count
    assert len(note_actions.calls) == 5
    assert "labels should remain readable" in notes.routing_aid("alice").description

    # After note focus exits, normal routing resumes without stopping the monitor.
    await _turn(front_end, "Help me troubleshoot this headset display.")
    assert len(router.calls) == 4
    assert len(general.answers) == 2
    assert general.answers[-1].query == "Help me troubleshoot this headset display."
    assert monitor.routing_aid("alice").eligibility == "active"
    assert len(frames.calls) == 3
    stopped, stop_output = await _turn(front_end, "Stop tracking the room.")
    assert stopped.disposition == "answer"
    assert stop_output == [("monitor", "Background visual monitoring has stopped.")]
    frame_count = len(frames.calls)
    assert monitor.routing_aid("alice").eligibility == "available"

    await _turn(front_end, "What is refraction?")
    assert len(router.calls) == 6
    assert len(frames.calls) == frame_count
    await front_end.stop()
    assert frames.max_active_by_participant == {"alice": 1}
    assert frames.max_active == image_query.max_active == 1


@pytest.mark.asyncio
async def test_monitor_start_cancelled_during_admission_or_ack_creates_no_sampler():
    frames, image_query = _Frames(), _ImageQuery()
    router = _RouteDecisions("monitor")
    admit_entered, admit_release = asyncio.Event(), asyncio.Event()
    conversation = _Conversation(admit_gate=admit_release, admit_entered=admit_entered)
    delegate = _Decisions("start")

    class _GatedMonitorAction:
        async def decide(self, state, questions, *, timeout=None):
            admit_entered.set()
            await admit_release.wait()
            return await delegate.decide(state, questions, timeout=timeout)

    monitor, _ = _monitor(_GatedMonitorAction(), frames, image_query)
    front_end = ConversationFrontEnd({"conversation": conversation, "monitor": monitor}, router)
    emitted = []

    async def emit(source, text):
        emitted.append((source, text))

    pending = asyncio.create_task(
        front_end.handle(
            ConversationTurn("alice", "start-1", "Start background monitoring."),
            emit=emit,
        )
    )
    await admit_entered.wait()
    await front_end.interrupt("alice")
    try:
        await pending
    except asyncio.CancelledError:
        pass
    assert emitted == []
    assert frames.calls == image_query.calls == []
    await monitor.participant_left("alice")
    assert monitor.routing_aid("alice").eligibility == "available"

    # Cancellation during the acknowledgment itself must not create a sampler
    # after the interrupted turn resumes or unwinds.
    frames2, image_query2 = _Frames(), _ImageQuery()
    monitor2, _ = _monitor(_Decisions("start"), frames2, image_query2)
    front_end2 = ConversationFrontEnd(
        {"conversation": _Conversation(), "monitor": monitor2}, _RouteDecisions("monitor")
    )
    ack_entered, ack_release = asyncio.Event(), asyncio.Event()
    emitted_during_ack = []

    async def blocking_emit(source, text):
        ack_entered.set()
        await ack_release.wait()
        emitted_during_ack.append((source, text))

    pending_ack = asyncio.create_task(
        front_end2.handle(
            ConversationTurn("alice", "start-ack", "Start background monitoring."),
            emit=blocking_emit,
        )
    )
    await ack_entered.wait()
    await front_end2.interrupt("alice")
    try:
        await pending_ack
    except asyncio.CancelledError:
        pass
    ack_release.set()
    assert emitted_during_ack == []
    assert frames2.calls == image_query2.calls == []
    assert monitor2.routing_aid("alice").eligibility == "available"


@pytest.mark.asyncio
async def test_monitor_leave_is_participant_scoped_and_worker_stop_drains_remaining_tasks():
    alice_gate, bob_gate = asyncio.Event(), asyncio.Event()
    frames = _Frames((alice_gate, None), (bob_gate, None))
    image_query = _ImageQuery()
    monitor_actions = _Decisions("start", "start")
    monitor, _ = _monitor(monitor_actions, frames, image_query)
    router = _RouteDecisions("monitor", "monitor")
    front_end = ConversationFrontEnd({"conversation": _Conversation(), "monitor": monitor}, router)

    await _turn(front_end, "Start monitoring my workspace.", participant="alice")
    await _next_call(frames)
    await _turn(front_end, "Start monitoring the doorway.", participant="bob")
    await _next_call(frames)

    await front_end.participant_left("alice")
    assert monitor.routing_aid("alice").eligibility == "available"
    assert monitor.routing_aid("bob").eligibility == "active"
    assert frames.active == 1

    await front_end.stop()
    assert monitor.routing_aid("bob").eligibility == "available"
    assert frames.active == image_query.active == 0
    assert frames.max_active_by_participant == {"alice": 1, "bob": 1}
    assert frames.max_active <= 2


@pytest.mark.asyncio
async def test_note_handoff_releases_focus_and_routes_same_query_to_conversation():
    note_conversation = _Conversation("notes")
    conversation = _Conversation("conversation")
    published = []

    async def publish_note(participant_id, update):
        published.append((participant_id, update))

    notes = NoteTakingAgent(
        note_conversation,
        _Decisions("start", "capture", "handoff"),
        context_budget_bytes=4096,
        publish_context=publish_note,
    )
    router = _RouteDecisions("notes", "conversation")
    front_end = ConversationFrontEnd({"conversation": conversation, "notes": notes}, router)

    await _turn(front_end, "Take some notes.")
    captured, _ = await _turn(front_end, "Record that the headset text should stay legible.")
    assert captured.disposition == "focus"
    assert "headset text should stay legible" in notes.routing_aid("alice").description

    query = "How should I evaluate AR/VR text legibility?"
    result, output = await _turn(front_end, query)
    assert result.disposition == "answer"
    assert len(conversation.answers) == 1
    assert conversation.answers[0].query == query
    # A focused handoff admission selects the eligible general destination
    # directly; it does not ask the router to choose a second time.
    assert len(router.calls) == 1
    assert output == [("conversation", "Answered by conversation.")]
    assert "headset text should stay legible" in notes.routing_aid("alice").description
    assert len(published) == 1
    assert "headset text should stay legible" in published[0][1].text
    await front_end.stop()


@pytest.mark.asyncio
async def test_note_context_is_participant_scoped_and_cleared_on_departure():
    notes = NoteTakingAgent(
        _Conversation("notes"),
        _Decisions("start", "capture", "start", "capture"),
        context_budget_bytes=4096,
    )
    router = _RouteDecisions("notes", "notes", "notes", "notes")
    front_end = ConversationFrontEnd({"conversation": _Conversation(), "notes": notes}, router)

    await _turn(front_end, "Take some notes.", participant="alice")
    await _turn(front_end, "Record a short headset note.", participant="alice")
    await _turn(front_end, "Take some notes.", participant="bob")
    await _turn(front_end, "Record a separate room note.", participant="bob")

    alice_aid = notes.routing_aid("alice").description
    bob_aid = notes.routing_aid("bob").description
    assert "headset note" in alice_aid
    assert "room note" not in alice_aid
    assert "room note" in bob_aid
    assert "headset note" not in bob_aid

    await front_end.participant_left("alice")
    assert "Latest completed note" not in notes.routing_aid("alice").description
    assert "room note" in notes.routing_aid("bob").description
    await front_end.stop()


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
    general_conversation = Conversation(
        llm,
        instructions="Answer the participant using their conversation and named context.",
        summary_context_topics=(note_topic.name,),
        context_budget_bytes=4096,
        compaction_input_budget_bytes=4096,
    )
    runtime.register("general-context", general_conversation.context_subscriber())
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
    )
    router = _RouteDecisions("notes", "conversation")
    front_end = ConversationFrontEnd(
        {
            "conversation": ContextualConversationAgent(
                general_conversation,
                static_description="General conversation and recall.",
                context_budget_bytes=4096,
            ),
            "notes": notes,
        },
        router,
    )

    async with runtime:
        await _turn(front_end, "Take some notes.")
        note = "Record that the headset display is easiest to read from the left side."
        await _turn(front_end, note)
        second_note = "Record that the demo starts at the east entrance."
        await _turn(front_end, second_note)
        assert len(published) == 2
        assert note in published[0].text
        assert note in published[1].text and second_note in published[1].text
        await _turn(front_end, "I'm finished taking notes.")
        result, _ = await _turn(front_end, "What did I note about reading the display?")

    assert result.disposition == "answer"
    assert len(llm.messages) == 1
    prompt = "\n".join(str(message.content) for message in llm.messages[0])
    assert "Source conversation-example.foreground-notes" in prompt
    assert note in prompt
    assert second_note in prompt
    await general_conversation.close()
    await front_end.stop()


@pytest.mark.asyncio
async def test_retarget_replaces_published_monitor_summary_before_new_sampling_and_keeps_transcript():
    class CapturingLLM:
        def __init__(self):
            self.messages = []

        def stream_events(self, messages, **_kwargs):
            self.messages.append(messages)

            async def events():
                yield ChatStreamEvent(kind="text", text="I’ll keep that in mind.")
                yield ChatStreamEvent(kind="finish", finish_reason="stop")

            return events()

    first_frame_gate = asyncio.Event()
    blocked_frame_gate = asyncio.Event()
    new_frame_gate = asyncio.Event()
    frames = _Frames((first_frame_gate, None), (blocked_frame_gate, None), (new_frame_gate, None))
    first_query_gate = asyncio.Event()
    blocked_query_gate = asyncio.Event()
    new_query_gate = asyncio.Event()
    image_query = _ImageQuery(
        (first_query_gate, SimpleNamespace(available=True, text="STALE ENTRANCE OBSERVATION.")),
        (blocked_query_gate, None),
        (new_query_gate, SimpleNamespace(available=True, text="NEW PEN OBSERVATION.")),
    )

    monitor_topic = Topic("conversation-example.monitor-window", ConversationContextUpdate)
    runtime = AgentRuntime()
    llm = CapturingLLM()
    general_conversation = Conversation(
        llm,
        instructions="Answer using the conversation and named context sources.",
        summary_context_topics=(monitor_topic.name,),
        context_budget_bytes=8192,
        compaction_input_budget_bytes=8192,
    )
    runtime.register("general-context", general_conversation.context_subscriber())
    updates = []

    async def publish_window(participant_id, update):
        updates.append((participant_id, update))
        await runtime.publish(monitor_topic, update, participant_id=participant_id)

    monitor, _ = _monitor(
        _Decisions("start", "retarget"),
        frames,
        image_query,
        publisher=publish_window,
    )
    conversation = ContextualConversationAgent(
        general_conversation,
        static_description="General conversation and current monitor context.",
        context_budget_bytes=8192,
        routing_context=monitor.routing_context,
    )
    router = _RouteDecisions("conversation", "monitor", "monitor", "conversation")
    front_end = ConversationFrontEnd(
        {"conversation": conversation, "monitor": monitor},
        router,
    )

    async with runtime:
        await _turn(front_end, "Remember that I left my umbrella by the entry table.")
        await _turn(front_end, "Monitor the entrance for arrivals.")
        await _next_call(frames)
        first_frame_gate.set()
        await _next_call(image_query)
        first_query_gate.set()
        await _next_call(frames)
        assert "STALE ENTRANCE OBSERVATION." in updates[-1][1].text

        # The old sampler is blocked waiting for another frame. Retarget must
        # replace its published source before the new sampler can request one.
        await _turn(front_end, "Retarget monitoring from entrance arrivals to my desk pen.")
        await _next_call(frames)

        # A general request immediately after the retarget sees the new empty
        # active summary, not the prior task's observation. The FE transcript is
        # still available as ordinary conversation history.
        await _turn(front_end, "What did I say about my umbrella, and what is being monitored?")
        assert llm.messages
        prompt = "\n".join(str(message.content) for message in llm.messages[-1])
        assert "STALE ENTRANCE OBSERVATION." not in prompt
        assert "status: active" in prompt
        assert "umbrella by the entry table" in prompt
        assert "conversation-example.monitor-window" in prompt
        assert any(
            "status: active" in update.text and "STALE ENTRANCE OBSERVATION." not in update.text
            for _pid, update in updates[-2:]
        )

        await front_end.stop()
        await monitor.stop()
        await general_conversation.close()

    assert frames.active == image_query.active == 0


@pytest.mark.asyncio
async def test_retarget_publication_failure_rolls_back_active_cue_without_starting_sampler():
    frame_gate = asyncio.Event()
    frames = _Frames((frame_gate, None))
    attempted_updates = []
    active_publications = 0

    async def failing_publisher(_participant_id, update):
        nonlocal active_publications
        attempted_updates.append(update)
        if "status: active" in update.text:
            active_publications += 1
            if active_publications == 2:
                raise RuntimeError("synthetic context subscriber failure")

    monitor, _ = _monitor(
        _Decisions("start", "retarget"),
        frames,
        _ImageQuery(),
        publisher=failing_publisher,
    )
    router = _RouteDecisions("monitor", "monitor")
    front_end = ConversationFrontEnd({"conversation": _Conversation(), "monitor": monitor}, router)

    await _turn(front_end, "Monitor the entrance for arrivals.")
    await _next_call(frames)
    with pytest.raises(RuntimeError, match="synthetic context subscriber failure"):
        await _turn(front_end, "Retarget monitoring from entrance arrivals to my desk pen.")

    cue = monitor.routing_context("alice")
    assert cue is not None
    assert "stopped after failure" in cue
    assert "Active task" not in cue
    assert not monitor._tasks
    assert len(frames.calls) == 1
    assert "status: stopped" in attempted_updates[-1].text
    assert "status: active" not in attempted_updates[-1].text
    await front_end.stop()


@pytest.mark.asyncio
async def test_hot_monitor_serializes_observations_and_retains_only_bounded_timestamped_window():
    frame_gates = [asyncio.Event() for _ in range(3)]
    query_gates = [asyncio.Event() for _ in range(3)]
    frames = _Frames(*((gate, None) for gate in frame_gates))
    image_query = _ImageQuery(*((gate, None) for gate in query_gates))
    monitor_actions = _Decisions("start", "stop")
    monitor, updates = _monitor(
        monitor_actions,
        frames,
        image_query,
        window_size=2,
    )
    router = _RouteDecisions("monitor", "monitor")
    front_end = ConversationFrontEnd({"conversation": _Conversation(), "monitor": monitor}, router)

    await _turn(front_end, "Track what is happening at the table.")
    for index in range(3):
        await _next_call(frames)
        assert len(image_query.calls) == index
        frame_gates[index].set()
        await _next_call(image_query)
        assert len(frames.calls) == index + 1
        query_gates[index].set()

    await _next_call(frames)
    assert frames.max_active == image_query.max_active == 1
    assert all(update.text for _, update in updates)
    final_window = updates[-1][1].text
    assert "Observation 1." not in final_window
    assert "Observation 2." in final_window
    assert "Observation 3." in final_window
    assert "2026-10-15 16:01 UTC (frame source time)" in final_window
    assert "2026-10-15 16:02 UTC (frame source time)" in final_window

    await _turn(front_end, "Stop tracking the table.")
    await front_end.stop()


@pytest.mark.asyncio
async def test_frame_failure_stops_monitor_without_retrying():
    frame_gate = asyncio.Event()
    frames = _Frames((frame_gate, FrameUnavailable("no current frame")))
    image_query = _ImageQuery()
    failed = asyncio.Event()

    async def observe_failure(_participant_id, update):
        if "Observation unavailable" in update.text:
            failed.set()

    monitor_actions = _Decisions("start")
    monitor, _ = _monitor(monitor_actions, frames, image_query, publisher=observe_failure)
    router = _RouteDecisions("monitor")
    front_end = ConversationFrontEnd({"conversation": _Conversation(), "monitor": monitor}, router)

    await _turn(front_end, "Start background monitoring.")
    await _next_call(frames)
    frame_gate.set()
    await failed.wait()
    assert len(frames.calls) == 1
    assert image_query.calls == []
    failed_context = monitor.routing_context("alice")
    assert failed_context is not None
    assert "stopped after failure" in failed_context
    assert "Active task:" not in failed_context
    await front_end.stop()
