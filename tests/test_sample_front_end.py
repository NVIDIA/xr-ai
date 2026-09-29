# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete-turn contracts for the reusable sample conversation front end."""

from __future__ import annotations

import asyncio

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
    VoiceOutput,
)

_APP_QUERY_TOPIC = Topic("test-conversation.app-query", UserQuery)


class _Conversation:
    def __init__(self, routes: list[str]) -> None:
        self.routes = routes
        self.decisions: list[tuple[str, tuple]] = []
        self.routing: list[tuple[str, tuple]] = []
        self.handbacks: list[str] = []

    async def _route(self, request, history, _app_context, _applications, *, handback=""):
        self.routing.append((request, history))
        self.handbacks.append(handback)
        route = self.routes.pop(0)
        return "conversation" if route == "direct" else route

    async def _decide_with_handoff(self, request, history, app_context):
        return await self.decide(request, history=history, app_context=app_context, applications=())

    async def decide(self, request, *, history, app_context, applications=()):
        self.decisions.append((request, history))
        route = self.routes.pop(0) if self.routes else "direct"
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


async def _wait_until(predicate) -> None:
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


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
        assert len(conversation.routing) == 1
        assert conversation.decisions == []
        await runtime.publish(
            FRONT_END_QUERY_TOPIC,
            UserQuery(text="What did we just do?", timestamp_us=3),
            participant_id="alice",
        )
        await _wait_until(lambda: len(front._history.get("alice", ())) == 3)
        assert len(conversation.routing) == 2
        assert len(conversation.decisions) == 1
        assert [query for _pid, query in app.queries] == [
            "Start the guide.",
            "Leave the guide.",
        ]
        assert conversation.routing[1][1][-1].assistant == ("App answered: Leave the guide.")
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
        assert len(conversation.routing) == 2
        assert conversation.decisions == []
        assert conversation.routing[1][1][0].assistant == "App answered: Add a sphere."
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


async def test_conversation_returns_app_request_to_router() -> None:
    class _HandbackConversation(_Conversation):
        async def _decide_with_handoff(self, request, history, app_context):
            self.decisions.append((request, history))
            return ChatResponse(
                content="",
                reasoning=None,
                tool_calls=[
                    ToolCall(
                        id="handoff",
                        name="application_handoff",
                        arguments='{"reason":"The prior virtual object needs recoloring."}',
                    )
                ],
                finish_reason="tool_calls",
                raw={},
            )

    conversation = _HandbackConversation(["conversation", "xr_scene"])
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
    speech = runtime.register("speech", _SpeechBridge())
    async with runtime:
        await runtime.publish(
            FRONT_END_QUERY_TOPIC,
            UserQuery(text="Make it blue.", timestamp_us=1),
            participant_id="alice",
        )
        await _wait_until(lambda: len(app.queries) == 1)
        assert conversation.handbacks == ["", "The prior virtual object needs recoloring."]
        assert len(conversation.decisions) == 1
        assert [text for _pid, text in app.queries] == ["Make it blue."]
        await _wait_until(lambda: len(speech.outputs) == 1)
        assert speech.outputs[0].text == "App answered: Make it blue."
    await front.stop()
