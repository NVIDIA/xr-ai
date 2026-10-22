# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise destination-owned voice output through real router topic delivery."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from xr_ai_conversation import Conversation
from xr_ai_models import ChatStreamEvent
from xr_ai_routing import AgentRouter, RoutingAid
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, Topic, subscribe
from xr_ai_tools.image import ImageReference
from xr_ai_voice import UserQuery, VoiceOutput

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "agent-samples/conversation-example/worker"))

from conversation_example_worker._monitoring import VisualMonitorAgent  # noqa: E402
from conversation_example_worker._note_taking import NoteTakingAgent  # noqa: E402
from conversation_example_worker._topics import USER_QUERY_TOPIC  # noqa: E402
from conversation_example_worker._visual_agents import VisualObservationAgent  # noqa: E402

_OUTPUT = Topic("conversation-example.test-output", VoiceOutput)


class _Decisions:
    def __init__(self, route):
        self.route = route
        self.note_actions = 0

    async def decide(self, _state, questions, **_kwargs):
        key = next(iter(questions))
        choice = self.route
        if key == "monitor_action":
            choice = "status"
        if key == "note_action":
            choice = "start" if self.note_actions == 0 else "other"
            self.note_actions += 1
        return SimpleNamespace(
            answers={key: SimpleNamespace(choice=choice, confidence=1.0, probabilities={choice: 1.0})}
        )


class _Model:
    def __init__(self, mode):
        self.mode = mode
        self.calls = 0
        self.waiting = asyncio.Event()
        self.closed = asyncio.Event()

    async def _text(self):
        self.calls += 1
        call = self.calls
        try:
            if self.mode == "before":
                raise RuntimeError("model failed before output")
            if call > 1:
                yield "replacement"
                return
            yield "first"
            if self.mode == "partial":
                raise RuntimeError("model failed after output")
            if self.mode.startswith("cancel"):
                self.waiting.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    # Deliberately violate cooperative cancellation at the model boundary.
                    if self.mode == "cancel_return":
                        return
                    if self.mode == "cancel_error":
                        raise RuntimeError("late model error") from None
                    yield "stale"
        finally:
            self.closed.set()

    def stream_events(self, _messages, **_kwargs):
        async def events():
            async for text in self._text():
                yield ChatStreamEvent(kind="text", text=text)
            yield ChatStreamEvent(kind="finish", finish_reason="stop")

        return events()

    async def stream(self, _request):
        async for text in self._text():
            yield SimpleNamespace(text=text)


class _Frames:
    async def execute(self, _request):
        return SimpleNamespace(image=ImageReference(uri="image://test"), timestamp_us=None)


class _Output(Agent):
    def __init__(self):
        super().__init__()
        self.messages = []
        self.completed = asyncio.Event()

    @subscribe(_OUTPUT)
    async def output(self, message: VoiceOutput, ctx: RuntimeContext):
        self.messages.append((ctx.metadata, message))
        if message.final:
            self.completed.set()


def _stack(runtime, route, mode):
    model = _Model(mode)
    general = Conversation(model, runtime=runtime, output_topic=_OUTPUT)
    notes = Conversation(
        model,
        routing=RoutingAid(name="notes", description="Focused note-taking destination."),
        runtime=runtime,
        output_topic=_OUTPUT,
    )
    decisions = _Decisions(route)
    destinations = {
        "conversation": general,
        "observation": VisualObservationAgent(_Frames(), model, runtime=runtime, output_topic=_OUTPUT),
        "notes": NoteTakingAgent(notes, decisions, runtime=runtime, output_topic=_OUTPUT, context_budget_bytes=4096),
    }
    idle = asyncio.Event()

    async def status(value, _participant):
        if value == "idle":
            idle.set()

    router = AgentRouter(destinations, decisions, query_topic=USER_QUERY_TOPIC, output_topic=_OUTPUT, set_status=status)
    runtime.register("router", router.runtime_subscriber())
    output = runtime.register("output", _Output())
    return router, model, output, idle, (general, notes)


async def _query(runtime, idle, text, timestamp):
    idle.clear()
    await runtime.publish(USER_QUERY_TOPIC, UserQuery(text=text, timestamp_us=timestamp), participant_id="alice")


async def _start_notes(runtime, route, output, idle):
    if route == "notes":
        await _query(runtime, idle, "Start taking notes.", 100)
        await asyncio.wait_for(idle.wait(), 1)
        assert len(output.messages) == 1
        metadata, message = output.messages[0]
        assert metadata.source == "notes"
        assert message.final and message.response_id and message.timestamp_us == 100
        output.messages.clear()
        output.completed.clear()


@pytest.mark.parametrize("route", ["conversation", "observation", "notes"])
@pytest.mark.parametrize("mode", ["before", "partial"])
async def test_destination_failure_preserves_source_and_never_duplicates_partial_output(route, mode):
    runtime = AgentRuntime()
    router, model, output, idle, conversations = _stack(runtime, route, mode)
    try:
        async with runtime:
            await _start_notes(runtime, route, output, idle)
            await _query(runtime, idle, "Answer this request.", 123)
            await asyncio.wait_for(idle.wait(), 1)
            await router.stop()
        assert model.closed.is_set()
        # A failure before any output is reported by the router; after output starts, the destination owns it.
        source = "router" if mode == "before" else route
        assert all(metadata.source == source and metadata.participant_id == "alice" for metadata, _ in output.messages)
        assert all(message.timestamp_us == 123 for _, message in output.messages)
        messages = [message for _, message in output.messages]
        if mode == "before":
            assert len(messages) == 1
            assert messages[0].text == "I couldn't complete that request. Please try again."
            assert messages[0].final and messages[0].response_id is None
        else:
            assert [message.text for message in messages] == ["first", ""]
            assert [message.final for message in messages] == [False, True]
            assert [message.interrupt for message in messages] == [True, False]
            assert messages[0].response_id == messages[1].response_id
    finally:
        for conversation in conversations:
            await conversation.close()


@pytest.mark.parametrize("route", ["conversation", "observation", "notes"])
@pytest.mark.parametrize("mode", ["cancel", "cancel_return", "cancel_error"])
async def test_replacement_suppresses_model_chunk_after_swallowed_cancellation(route, mode):
    runtime = AgentRuntime()
    router, model, output, idle, conversations = _stack(runtime, route, mode)
    try:
        async with runtime:
            await _start_notes(runtime, route, output, idle)
            await _query(runtime, idle, "First request.", 123)
            await asyncio.wait_for(model.waiting.wait(), 1)
            await _query(runtime, idle, "Replacement request.", 456)

            async def replacement_finished():
                while len(output.messages) < 4:
                    await asyncio.sleep(0)

            await asyncio.wait_for(replacement_finished(), 1)
            await asyncio.wait_for(router.stop(), 1)
        assert model.closed.is_set()
        messages = [message for _, message in output.messages]
        # The canceled stream is closed without stale text before the replacement streams.
        assert [message.text for message in messages] == ["first", "", "replacement", ""]
        assert [message.timestamp_us for message in messages] == [123, 123, 456, 456]
        assert messages[0].response_id == messages[1].response_id != messages[2].response_id
        assert messages[2].response_id == messages[3].response_id
        assert [message.final for message in messages] == [False, True, False, True]
        assert {metadata.source for metadata, _ in output.messages} == {route}
    finally:
        for conversation in conversations:
            await conversation.close()


async def test_monitor_control_publishes_one_finite_attributed_response():
    runtime = AgentRuntime()
    decisions = _Decisions("monitor")

    async def publish_window(_participant, _update):
        pass

    general = Conversation(_Model("done"), runtime=runtime, output_topic=_OUTPUT)
    monitor = VisualMonitorAgent(
        _Frames(),
        object(),
        decisions,
        publish_window,
        runtime=runtime,
        output_topic=_OUTPUT,
        window_size=4,
        context_budget_bytes=4096,
    )
    idle = asyncio.Event()

    async def status(value, _participant):
        if value == "idle":
            idle.set()

    router = AgentRouter(
        {"conversation": general, "monitor": monitor},
        decisions,
        query_topic=USER_QUERY_TOPIC,
        output_topic=_OUTPUT,
        set_status=status,
    )
    runtime.register("router", router.runtime_subscriber())
    output = runtime.register("output", _Output())
    try:
        async with runtime:
            await _query(runtime, idle, "Is monitoring active?", 789)
            await asyncio.wait_for(idle.wait(), 1)
            await router.stop()
        assert len(output.messages) == 1
        metadata, message = output.messages[0]
        assert metadata.source == "monitor" and metadata.participant_id == "alice"
        assert message.final and message.response_id
        assert message.interrupt and message.timestamp_us == 789
        assert "has not been started" in message.text
    finally:
        await general.close()
