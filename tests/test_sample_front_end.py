# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete-turn contracts for the reusable sample conversation front end."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from xr_ai_models import ChatResponse, ToolCall
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, Topic, subscribe
from xr_ai_sample_agents.front_end import (
    FRONT_END_QUERY_TOPIC,
    ConversationApplication,
    ConversationFrontEnd,
)
from xr_ai_voice import (
    VOICE_CONTRIBUTION_TOPIC,
    VOICE_OUTPUT_TOPIC,
    UserQuery,
    VoiceAggregationAgent,
    VoiceOutput,
)

_APP_QUERY_TOPIC = Topic("test-conversation.app-query", UserQuery)
_SECOND_APP_QUERY_TOPIC = Topic("test-conversation.second-app-query", UserQuery)


class _Conversation:
    def __init__(self, routes: list[str]) -> None:
        self.routes = routes
        self.decisions: list[tuple[str, tuple, str, tuple[str, ...]]] = []

    async def decide(self, request, *, history, app_context, applications):
        self.decisions.append((request, history, app_context, tuple(app.name for app in applications)))
        route = self.routes.pop(0)
        calls = [ToolCall(id="app-call", name=route, arguments="{}")] if route != "direct" else None
        return ChatResponse(
            content="A direct answer." if route == "direct" else "",
            reasoning=None,
            tool_calls=calls,
            finish_reason="tool_calls" if calls else "stop",
            raw={},
        )

    async def stream(self, _request, _participant_id, **_kwargs):
        if _kwargs["decision"].tool_calls:
            yield "I see "
            yield "a blue mug."
        else:
            yield "A direct answer."


class _Application(Agent):
    def __init__(self, *, focus: dict[str, bool]) -> None:
        super().__init__()
        self.focus = focus
        self.queries: list[tuple[str, str]] = []

    @subscribe(_APP_QUERY_TOPIC)
    async def answer(self, query: UserQuery, ctx: RuntimeContext) -> None:
        participant_id = ctx.metadata.participant_id
        assert participant_id is not None
        self.queries.append((participant_id, query.text))
        self.focus[participant_id] = query.text != "Leave the guide."
        await ctx.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(
                text=f"App answered: {query.text}",
                kind="result",
                turn_id=ctx.metadata.correlation_id,
            ),
        )


class _SpeechBridge(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.outputs: list[VoiceOutput] = []

    @subscribe(VOICE_CONTRIBUTION_TOPIC)
    async def deliver(self, output: VoiceOutput, ctx: RuntimeContext) -> None:
        self.outputs.append(output)
        await ctx.publish(VOICE_OUTPUT_TOPIC, output)


class _SecondApplication(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.queries: list[tuple[str, str]] = []

    @subscribe(_SECOND_APP_QUERY_TOPIC)
    async def answer(self, query: UserQuery, ctx: RuntimeContext) -> None:
        participant_id = ctx.metadata.participant_id
        assert participant_id is not None
        self.queries.append((participant_id, query.text))
        await ctx.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text=f"Second app answered: {query.text}", kind="result", turn_id=ctx.metadata.correlation_id),
        )


class _ProgressApplication(Agent):
    @subscribe(_APP_QUERY_TOPIC)
    async def answer(self, query: UserQuery, ctx: RuntimeContext) -> None:
        await ctx.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="I will check.", kind="acknowledgement", turn_id=ctx.metadata.correlation_id),
        )
        await ctx.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text=f"Finished: {query.text}", kind="result", turn_id=ctx.metadata.correlation_id),
        )


async def _wait_until(predicate) -> None:
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


@pytest.mark.parametrize("wait_for_delivery", [False, True])
async def test_failed_visual_stream_interrupts_and_remembers_only_recovery(wait_for_delivery) -> None:
    partial_delivered = asyncio.Event()
    outputs = []

    class Speech(Agent):
        @subscribe(VOICE_OUTPUT_TOPIC)
        async def receive(self, output: VoiceOutput, ctx: RuntimeContext):
            outputs.append(output)
            if output.text == "An unfinished visual claim":
                partial_delivered.set()

    class FailingConversation(_Conversation):
        async def stream(self, request, participant_id, **kwargs):
            if kwargs["decision"].tool_calls:
                yield "An unfinished visual claim"
                if wait_for_delivery:
                    await partial_delivered.wait()
                raise RuntimeError("visual inference disconnected")
            yield "A direct answer."

    runtime = AgentRuntime()
    conversation = FailingConversation(["current_view", "direct"])
    front = runtime.register("conversation", ConversationFrontEnd(conversation))
    aggregation = runtime.register("aggregation", VoiceAggregationAgent(
        llm=Mock(chat=AsyncMock()), coalesce_window_s=0,
        minimum_playback_s=0, maximum_playback_s=0.001,
        # Keep the real 15-second stream timeout: recovery must not wait for it.
    ))
    runtime.register("speech", Speech())
    async with runtime:
        try:
            await runtime.publish(
                FRONT_END_QUERY_TOPIC, UserQuery(text="What is visible?", timestamp_us=1), participant_id="alice",
            )
            recovery = "I couldn't complete that request. Please try again."
            await _wait_until(lambda: any(output.text == recovery for output in outputs))
            await _wait_until(lambda: bool(front._history.get("alice")))
            assert [exchange.assistant for exchange in front._history["alice"]] == [recovery]
            assert outputs[-1].interrupt and outputs[-1].final
            assert not front._pending and not front._spoken
            await runtime.publish(
                FRONT_END_QUERY_TOPIC, UserQuery(text="Try something else", timestamp_us=2), participant_id="alice",
            )
            await _wait_until(lambda: len(front._history["alice"]) == 2)
            assert conversation.decisions[-1][1][-1].assistant == recovery
        finally:
            await front.stop()
            await aggregation.stop()


@pytest.mark.parametrize("cancel", ["supersede", "interrupt", "leave", "stop"])
async def test_cancelled_visual_stream_does_not_block_next_direct_answer(cancel) -> None:
    outputs = []

    class Speech(Agent):
        @subscribe(VOICE_OUTPUT_TOPIC)
        async def receive(self, output: VoiceOutput, ctx: RuntimeContext):
            outputs.append(output)

    class PausedConversation(_Conversation):
        async def stream(self, request, participant_id, **kwargs):
            if kwargs["decision"].tool_calls:
                yield "An unfinished visual answer"
                await asyncio.Event().wait()
            else:
                yield "A direct answer."

    runtime = AgentRuntime()
    conversation = PausedConversation(["current_view", "direct"])
    front = runtime.register("conversation", ConversationFrontEnd(conversation))
    aggregation = runtime.register("aggregation", VoiceAggregationAgent(
        llm=Mock(chat=AsyncMock()), coalesce_window_s=0,
        minimum_playback_s=0, maximum_playback_s=0.001,
        # Keep the default 15-second timeout to detect abandoned streams.
    ))
    runtime.register("speech", Speech())
    async with runtime:
        try:
            await runtime.publish(
                FRONT_END_QUERY_TOPIC, UserQuery(text="What is visible?", timestamp_us=1), participant_id="alice",
            )
            await _wait_until(lambda: bool(outputs))
            old_turn = outputs[0].turn_id
            if cancel == "interrupt":
                await front.interrupted("alice")
            elif cancel == "leave":
                await front.participant_left("alice")
            elif cancel == "stop":
                await front.stop()
            await runtime.publish(
                FRONT_END_QUERY_TOPIC, UserQuery(text="Hello", timestamp_us=2), participant_id="alice",
            )
            await _wait_until(lambda: any(output.text == "A direct answer." for output in outputs))
            await _wait_until(lambda: bool(front._history.get("alice")))
            assert any(output.turn_id == old_turn and output.final for output in outputs)
            assert [exchange.assistant for exchange in front._history["alice"]] == ["A direct answer."]
            assert not front._pending and not front._spoken
        finally:
            await front.stop()
            await aggregation.stop()


async def test_focused_app_bypasses_router_and_releases_on_exit() -> None:
    focus: dict[str, bool] = {}
    conversation = _Conversation(["tea_guide", "direct"])
    runtime = AgentRuntime()
    front = runtime.register(
        "conversation",
        ConversationFrontEnd(
            conversation,  # type: ignore[arg-type]
            applications=(
                ConversationApplication(
                    name="tea_guide",
                    description="Guide a tea-making workflow.",
                    query_topic=_APP_QUERY_TOPIC,
                    has_focus=lambda pid: focus.get(pid, False),
                    context=lambda _pid: "Tea guide is available.",
                ),
            ),
        ),
    )
    app = runtime.register("app", _Application(focus=focus))
    speech = runtime.register("speech", _SpeechBridge())
    async with runtime:
        await runtime.publish(
            FRONT_END_QUERY_TOPIC,
            UserQuery(text="Start the guide.", timestamp_us=1),
            participant_id="alice",
        )
        await _wait_until(lambda: len(front._history.get("alice", ())) == 1)
        await runtime.publish(
            FRONT_END_QUERY_TOPIC,
            UserQuery(text="Leave the guide.", timestamp_us=2),
            participant_id="alice",
        )
        await _wait_until(lambda: len(front._history.get("alice", ())) == 2)
        assert len(conversation.decisions) == 1
        await runtime.publish(
            FRONT_END_QUERY_TOPIC,
            UserQuery(text="What did we just do?", timestamp_us=3),
            participant_id="alice",
        )
        await _wait_until(lambda: len(front._history.get("alice", ())) == 3)
        assert len(conversation.decisions) == 2
        assert [query for _pid, query in app.queries] == [
            "Start the guide.",
            "Leave the guide.",
        ]
        assert conversation.decisions[1][1][-1].assistant == ("App answered: Leave the guide.")
        assert all(output.kind == "result" for output in speech.outputs)
    await front.stop()


async def test_stateless_app_routes_every_turn() -> None:
    conversation = _Conversation(["xr_scene", "xr_scene"])
    runtime = AgentRuntime()
    front = runtime.register(
        "conversation",
        ConversationFrontEnd(
            conversation,  # type: ignore[arg-type]
            applications=(
                ConversationApplication(
                    name="xr_scene",
                    description="Handle the virtual XR scene.",
                    query_topic=_APP_QUERY_TOPIC,
                    has_focus=lambda _pid: False,
                    context=lambda _pid: "XR scene is available.",
                ),
            ),
        ),
    )
    app = runtime.register("app", _Application(focus={}))
    runtime.register("speech", _SpeechBridge())
    async with runtime:
        for index, text in enumerate(("Add a sphere.", "Now color it blue."), 1):
            await runtime.publish(
                FRONT_END_QUERY_TOPIC,
                UserQuery(text=text, timestamp_us=index),
                participant_id="alice",
            )
            await _wait_until(lambda: len(app.queries) == index)
            await _wait_until(lambda: len(front._history.get("alice", ())) == index)
        assert len(conversation.decisions) == 2
        assert conversation.decisions[1][1][0].assistant == "App answered: Add a sphere."
    await front.stop()


async def test_current_view_stream_has_no_acknowledgement() -> None:
    conversation = _Conversation(["current_view"])
    runtime = AgentRuntime()
    front = runtime.register(
        "conversation",
        ConversationFrontEnd(conversation),  # type: ignore[arg-type]
    )
    speech = runtime.register("speech", _SpeechBridge())
    async with runtime:
        await runtime.publish(
            FRONT_END_QUERY_TOPIC,
            UserQuery(text="What do you see?", timestamp_us=1),
            participant_id="alice",
        )
        await _wait_until(lambda: len(front._history.get("alice", ())) == 1)
        assert [output.text for output in speech.outputs] == [
            "I see ",
            "a blue mug.",
            "",
        ]
        assert all(output.kind == "result" for output in speech.outputs)
        assert front._history["alice"][0].assistant == "I see a blue mug."
    await front.stop()


async def test_focused_app_owns_cross_domain_turns_until_it_releases_focus() -> None:
    focus: dict[str, bool] = {}
    conversation = _Conversation(["tea_guide", "xr_scene", "direct"])
    runtime = AgentRuntime()
    front = runtime.register(
        "conversation",
        ConversationFrontEnd(
            conversation,  # type: ignore[arg-type]
            applications=(
                ConversationApplication(
                    name="tea_guide",
                    description="Guide the tea workflow.",
                    query_topic=_APP_QUERY_TOPIC,
                    has_focus=lambda pid: focus.get(pid, False),
                    context=lambda pid: f"Tea guide {'active' if focus.get(pid) else 'idle'} for {pid}.",
                ),
                ConversationApplication(
                    name="xr_scene",
                    description="Edit the XR scene.",
                    query_topic=_SECOND_APP_QUERY_TOPIC,
                    has_focus=lambda _pid: False,
                    context=lambda _pid: "XR scene is available.",
                ),
            ),
        ),
    )
    tea = runtime.register("tea", _Application(focus=focus))
    xr = runtime.register("xr", _SecondApplication())
    runtime.register("speech", _SpeechBridge())
    async with runtime:
        async def send(text: str, participant_id: str, expected_history: int) -> None:
            await runtime.publish(
                FRONT_END_QUERY_TOPIC,
                UserQuery(text=text, timestamp_us=expected_history),
                participant_id=participant_id,
            )
            await _wait_until(lambda: (
                len(front._history.get(participant_id, ())) == min(expected_history, 4)
                and front._history[participant_id][-1].user == text
            ))

        await send("Start the tea guide.", "alice", 1)
        await send("What do you see?", "alice", 2)
        await send("Add a virtual cube.", "alice", 3)
        await send("Add a virtual cube.", "bob", 1)
        await send("Leave the guide.", "alice", 4)
        await send("What did we make?", "alice", 5)
        assert [query for _pid, query in tea.queries] == [
            "Start the tea guide.",
            "What do you see?",
            "Add a virtual cube.",
            "Leave the guide.",
        ]
        assert xr.queries == [("bob", "Add a virtual cube.")]
        assert [request for request, *_ in conversation.decisions] == [
            "Start the tea guide.",
            "Add a virtual cube.",
            "What did we make?",
        ]
        assert conversation.decisions[1][1] == ()
        assert conversation.decisions[2][1][-1].assistant == "App answered: Leave the guide."
        assert conversation.decisions[2][2] == "Tea guide idle for alice.\nXR scene is available."
        assert conversation.decisions[2][3] == ("tea_guide", "xr_scene")
    await front.stop()


async def test_stateless_app_returns_to_router_for_unrelated_followup() -> None:
    conversation = _Conversation(["xr_scene", "direct", "xr_scene"])
    runtime = AgentRuntime()
    front = runtime.register(
        "conversation",
        ConversationFrontEnd(
            conversation,  # type: ignore[arg-type]
            applications=(
                ConversationApplication(
                    name="xr_scene",
                    description="Edit the XR scene.",
                    query_topic=_SECOND_APP_QUERY_TOPIC,
                    has_focus=lambda _pid: False,
                    context=lambda _pid: "XR scene is available.",
                ),
            ),
        ),
    )
    xr = runtime.register("xr", _SecondApplication())
    runtime.register("speech", _SpeechBridge())
    async with runtime:
        for index, query in enumerate(("Add a cube.", "What is a cube?", "Delete the cube."), 1):
            await runtime.publish(
                FRONT_END_QUERY_TOPIC,
                UserQuery(text=query, timestamp_us=index),
                participant_id="alice",
            )
            await _wait_until(lambda: len(front._history.get("alice", ())) == index)
        assert [query for _pid, query in xr.queries] == ["Add a cube.", "Delete the cube."]
        assert len(conversation.decisions) == 3
        assert conversation.decisions[2][1][-1].assistant == "A direct answer."
    await front.stop()


async def test_participant_departure_clears_only_that_participants_memory() -> None:
    conversation = _Conversation(["direct", "direct", "direct"])
    runtime = AgentRuntime()
    front = runtime.register("conversation", ConversationFrontEnd(conversation))  # type: ignore[arg-type]
    runtime.register("speech", _SpeechBridge())
    async with runtime:
        for participant_id in ("alice", "bob"):
            await runtime.publish(
                FRONT_END_QUERY_TOPIC,
                UserQuery(text="Hello.", timestamp_us=1),
                participant_id=participant_id,
            )
            await _wait_until(lambda pid=participant_id: len(front._history.get(pid, ())) == 1)
        await front.participant_left("alice")
        assert "alice" not in front._history
        assert len(front._history["bob"]) == 1
        await runtime.publish(
            FRONT_END_QUERY_TOPIC,
            UserQuery(text="Do you remember?", timestamp_us=2),
            participant_id="alice",
        )
        await _wait_until(lambda: len(front._history.get("alice", ())) == 1)
        assert conversation.decisions[2][1] == ()
    await front.stop()


async def test_spoken_acknowledgement_does_not_enter_conversation_history() -> None:
    conversation = _Conversation(["task_app", "direct"])
    runtime = AgentRuntime()
    front = runtime.register(
        "conversation",
        ConversationFrontEnd(
            conversation,  # type: ignore[arg-type]
            applications=(
                ConversationApplication(
                    name="task_app",
                    description="Handle a task.",
                    query_topic=_APP_QUERY_TOPIC,
                    has_focus=lambda _pid: False,
                    context=lambda _pid: "",
                ),
            ),
        ),
    )
    runtime.register("task", _ProgressApplication())
    speech = runtime.register("speech", _SpeechBridge())
    async with runtime:
        await runtime.publish(
            FRONT_END_QUERY_TOPIC,
            UserQuery(text="Do the task.", timestamp_us=1),
            participant_id="alice",
        )
        await _wait_until(lambda: len(front._history.get("alice", ())) == 1)
        assert [output.kind for output in speech.outputs] == ["acknowledgement", "result"]
        assert front._history["alice"][0].assistant == "Finished: Do the task."
        await runtime.publish(
            FRONT_END_QUERY_TOPIC,
            UserQuery(text="What happened?", timestamp_us=2),
            participant_id="alice",
        )
        await _wait_until(lambda: len(front._history.get("alice", ())) == 2)
        assert conversation.decisions[1][1][0].assistant == "Finished: Do the task."
    await front.stop()


async def test_interruption_preserves_application_focus() -> None:
    focus = {"alice": True}
    conversation = _Conversation([])
    runtime = AgentRuntime()
    front = runtime.register(
        "conversation",
        ConversationFrontEnd(
            conversation,  # type: ignore[arg-type]
            applications=(
                ConversationApplication(
                    name="tea_guide",
                    description="Guide tea making.",
                    query_topic=_APP_QUERY_TOPIC,
                    has_focus=lambda pid: focus.get(pid, False),
                    context=lambda _pid: "",
                ),
            ),
        ),
    )
    app = runtime.register("app", _Application(focus=focus))
    runtime.register("speech", _SpeechBridge())
    async with runtime:
        await front.interrupted("alice")
        await runtime.publish(
            FRONT_END_QUERY_TOPIC,
            UserQuery(text="Continue the tea.", timestamp_us=1),
            participant_id="alice",
        )
        await _wait_until(lambda: len(front._history.get("alice", ())) == 1)
        assert app.queries == [("alice", "Continue the tea.")]
        assert conversation.decisions == []
    await front.stop()
