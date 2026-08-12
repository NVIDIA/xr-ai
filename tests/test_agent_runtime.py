# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for agent-owned tools, resource lifetimes, and typed pub/sub."""

from __future__ import annotations

import asyncio
from builtins import BaseExceptionGroup, ExceptionGroup
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import assert_type

import pytest
from pydantic import BaseModel
from xr_ai_models import ChatMessage, ToolCall
from xr_ai_runtime import (
    Agent,
    AgentRuntime,
    RuntimeClosedError,
    RuntimeContext,
    RuntimeFailedError,
    Topic,
    subscribe,
)
from xr_ai_tools import AsyncTool, Tool, ToolSet
from xr_ai_tools.tool_calling import handle_tool_call


class _Echo(BaseModel):
    text: str


class _Count(BaseModel):
    amount: int


class _Counted(BaseModel):
    count: int


class _Observation(BaseModel):
    labels: list[str]


class _Speak(BaseModel):
    text: str


OBSERVATIONS = Topic("vision.observation", _Observation)
VOICE_OUTPUT = Topic("voice.output", _Speak)
TRANSACTIONAL_ECHO = Topic("transactional", _Echo)
TRANSACTIONAL_COUNT = Topic("transactional", _Count)


class _TextAgent(Agent):
    def __init__(self) -> None:
        self.recorded: list[str] = []
        self.echo = Tool("echo", "Echo text.", _Echo, _Echo, self._echo)
        self.uppercase = Tool(
            "uppercase",
            "Uppercase text.",
            _Echo,
            _Echo,
            self._uppercase,
        )
        self.record = Tool("record", "Record text.", _Echo, None, self._record)
        self.stream_echo = AsyncTool(
            "stream_echo",
            "Stream echoed text.",
            _Echo,
            _Echo,
            self._stream_echo,
        )
        super().__init__((self.echo, self.uppercase, self.record, self.stream_echo))

    async def _echo(self, request: _Echo) -> _Echo:
        return request

    async def _uppercase(self, request: _Echo) -> _Echo:
        return _Echo(text=request.text.upper())

    async def _record(self, request: _Echo) -> None:
        self.recorded.append(request.text)

    async def _stream_echo(self, request: _Echo) -> AsyncIterator[_Echo]:
        yield _Echo(text=f"{request.text}-one")
        yield _Echo(text=f"{request.text}-two")


async def test_agent_exposes_existing_unary_and_streaming_tools() -> None:
    agent = _TextAgent()
    runtime = AgentRuntime()

    registered = runtime.register("text", agent)
    echoed = await agent.echo.execute(_Echo(text="hello"))
    chunks = [
        chunk async for chunk in agent.stream_echo.stream({"text": "chunk"})
    ]

    assert registered is agent
    assert registered.tools == (
        agent.echo,
        agent.uppercase,
        agent.record,
        agent.stream_echo,
    )
    assert_type(echoed, _Echo)
    assert echoed == _Echo(text="hello")
    assert chunks == [_Echo(text="chunk-one"), _Echo(text="chunk-two")]


async def test_agent_tool_uses_normal_model_tool_calling() -> None:
    agent = _TextAgent()
    tools = ToolSet.namespaced({"text": (agent.uppercase,)})

    result = await handle_tool_call(
        ToolCall(
            id="uppercase-call",
            name="text__uppercase",
            arguments='{"text":"hello"}',
        ),
        tools,
    )

    assert result.message == ChatMessage(
        role="tool",
        content='{"text":"HELLO"}',
        tool_call_id="uppercase-call",
    )


async def test_side_effect_tool_returns_none_through_direct_and_model_calls() -> None:
    agent = _TextAgent()

    direct = await agent.record.execute(_Echo(text="direct"))
    model = await agent.record.invoke('{"text":"model"}')

    assert direct is None
    assert model.content == "null"
    assert agent.recorded == ["direct", "model"]


class _ForwardingAgent(Agent):
    def __init__(self, target: Tool[_Echo, _Echo]) -> None:
        self._target = target
        self.forward = Tool("forward", "Forward text.", _Echo, _Echo, self._forward)
        super().__init__((self.forward,))

    async def _forward(self, request: _Echo) -> _Echo:
        return await self._target.execute(request)


async def test_agent_uses_another_agents_tool_without_runtime_dispatch() -> None:
    target = _TextAgent()
    caller = _ForwardingAgent(target.echo)
    runtime = AgentRuntime()
    runtime.register("target", target)
    runtime.register("caller", caller)

    result = await caller.forward.execute(_Echo(text="hello"))

    assert result == _Echo(text="hello")


class _VoiceOutput(Agent):
    def __init__(self, *, mutate: bool = False) -> None:
        super().__init__()
        self.mutate = mutate
        self.received: list[tuple[str, str, list[str]]] = []

    @subscribe(OBSERVATIONS)
    async def observe(self, event: _Observation, ctx: RuntimeContext) -> None:
        self.received.append(
            (ctx.metadata.participant_id, ctx.metadata.source, list(event.labels))
        )
        if self.mutate:
            event.labels.append("mutated")


async def test_publish_fans_out_isolated_typed_payloads() -> None:
    first = _VoiceOutput(mutate=True)
    second = _VoiceOutput()
    runtime = AgentRuntime()
    runtime.register("first", first)
    runtime.register("second", second)
    original = _Observation(labels=["kettle"])

    async with runtime:
        result = await runtime.publish(
            OBSERVATIONS,
            original,
            participant_id="alice",
            source="camera",
        )

    assert result is None
    assert original == _Observation(labels=["kettle"])
    assert first.received == [("alice", "camera", ["kettle"])]
    assert second.received == [("alice", "camera", ["kettle"])]


class _FailingSubscriber(Agent):
    def __init__(self, *, failure: Exception | None = None) -> None:
        super().__init__()
        self.failure = failure
        self.completed = False

    @subscribe(OBSERVATIONS)
    async def observe(self, _event: _Observation, _ctx: RuntimeContext) -> None:
        await asyncio.sleep(0)
        self.completed = True
        if self.failure is not None:
            raise self.failure


async def test_publish_settles_every_delivery_then_propagates_failures() -> None:
    failed = _FailingSubscriber(failure=RuntimeError("subscriber failed"))
    completed = _FailingSubscriber()
    runtime = AgentRuntime()
    runtime.register("failed", failed)
    runtime.register("completed", completed)
    await runtime.start()

    with pytest.raises(ExceptionGroup, match="event publication") as raised:
        await runtime.publish(
            OBSERVATIONS,
            _Observation(labels=[]),
            participant_id="alice",
        )

    await runtime.stop()
    assert failed.completed
    assert completed.completed
    assert [str(error) for error in raised.value.exceptions] == ["subscriber failed"]


class _EventForwarder(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.input_message_id = ""

    @subscribe(OBSERVATIONS)
    async def forward(self, _event: _Observation, ctx: RuntimeContext) -> None:
        self.input_message_id = ctx.metadata.message_id
        await ctx.publish(VOICE_OUTPUT, _Speak(text="Seen."))


class _Speaker(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.metadata = None

    @subscribe(VOICE_OUTPUT)
    async def speak(self, _event: _Speak, ctx: RuntimeContext) -> None:
        self.metadata = ctx.metadata


async def test_nested_publish_preserves_participant_and_trace_context() -> None:
    forwarder = _EventForwarder()
    speaker = _Speaker()
    runtime = AgentRuntime()
    runtime.register("forwarder", forwarder)
    runtime.register("speaker", speaker)

    async with runtime:
        await runtime.publish(
            OBSERVATIONS,
            _Observation(labels=["cup"]),
            participant_id="alice",
            source="camera",
        )

    assert speaker.metadata is not None
    assert speaker.metadata.participant_id == "alice"
    assert speaker.metadata.source == "forwarder"
    assert speaker.metadata.parent_message_id == forwarder.input_message_id
    assert speaker.metadata.correlation_id == forwarder.input_message_id


class _SerializedAgent(Agent):
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._release = asyncio.Event()
        self.tool_entered = asyncio.Event()
        self.active = 0
        self.max_active = 0
        self.count = 0
        self.increment = Tool(
            "increment",
            "Increment shared state.",
            _Count,
            _Counted,
            self._increment,
        )
        super().__init__((self.increment,))

    async def _mutate(self, amount: int, *, wait: bool = False) -> None:
        async with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if wait:
                self.tool_entered.set()
                await self._release.wait()
            self.count += amount
            self.active -= 1

    async def _increment(self, request: _Count) -> _Counted:
        await self._mutate(request.amount, wait=True)
        return _Counted(count=self.count)

    @subscribe(OBSERVATIONS)
    async def observe(self, _event: _Observation, _ctx: RuntimeContext) -> None:
        await self._mutate(1)


async def test_agent_can_serialize_tools_and_subscriptions_internally() -> None:
    agent = _SerializedAgent()
    runtime = AgentRuntime()
    runtime.register("serialized", agent)

    async with runtime:
        tool_call = asyncio.create_task(agent.increment.execute(_Count(amount=2)))
        await agent.tool_entered.wait()
        publication = asyncio.create_task(
            runtime.publish(
                OBSERVATIONS,
                _Observation(labels=["kettle"]),
                participant_id="alice",
            )
        )
        await asyncio.sleep(0)
        assert agent.max_active == 1
        agent._release.set()
        assert await tool_call == _Counted(count=2)
        await publication

    assert agent.count == 3
    assert agent.max_active == 1


class _BackgroundAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.stopped = asyncio.Event()

    @asynccontextmanager
    async def lifespan(self, ctx: RuntimeContext) -> AsyncIterator[None]:
        ctx.start_task(self._run(), name="background-agent")
        yield

    async def _run(self) -> None:
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.stopped.set()


async def test_lifespan_owns_background_work() -> None:
    agent = _BackgroundAgent()
    runtime = AgentRuntime()
    runtime.register("background", agent)

    async with runtime:
        await agent.started.wait()

    assert agent.stopped.is_set()


class _FailingBackgroundAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()
        self.companion_started = asyncio.Event()
        self.companion_stopped = asyncio.Event()

    @asynccontextmanager
    async def lifespan(self, ctx: RuntimeContext) -> AsyncIterator[None]:
        ctx.start_task(self._fail(), name="failing-background")
        ctx.start_task(self._companion(), name="background-companion")
        yield

    async def _fail(self) -> None:
        await self.release.wait()
        raise RuntimeError("background failed")

    async def _companion(self) -> None:
        self.companion_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.companion_stopped.set()


async def test_background_failure_transitions_runtime_and_surfaces_on_stop() -> None:
    agent = _FailingBackgroundAgent()
    runtime = AgentRuntime()
    runtime.register("background", agent)
    await runtime.start()
    await agent.companion_started.wait()

    agent.release.set()
    for _ in range(10):
        if not runtime.running:
            break
        await asyncio.sleep(0)

    assert not runtime.running
    await asyncio.wait_for(agent.companion_stopped.wait(), timeout=1)
    with pytest.raises(RuntimeFailedError) as raised:
        await runtime.publish(
            OBSERVATIONS,
            _Observation(labels=[]),
            participant_id="alice",
        )
    assert isinstance(raised.value.__cause__, RuntimeError)
    with pytest.raises(ExceptionGroup, match="agent runtime") as stopped:
        await runtime.stop()
    assert [str(error) for error in stopped.value.exceptions] == ["background failed"]


class _SlowSubscriber(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.stopped = asyncio.Event()
        self.cleanup_saw_stopped_delivery = False

    @asynccontextmanager
    async def lifespan(self, _ctx: RuntimeContext) -> AsyncIterator[None]:
        try:
            yield
        finally:
            self.cleanup_saw_stopped_delivery = self.stopped.is_set()

    @subscribe(OBSERVATIONS)
    async def observe(self, _event: _Observation, _ctx: RuntimeContext) -> None:
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.stopped.set()


async def test_stop_cancels_deliveries_before_exiting_lifespans() -> None:
    agent = _SlowSubscriber()
    runtime = AgentRuntime()
    runtime.register("slow", agent)
    await runtime.start()
    publication = asyncio.create_task(
        runtime.publish(
            OBSERVATIONS,
            _Observation(labels=[]),
            participant_id="alice",
        )
    )
    await agent.started.wait()

    await runtime.stop()
    await asyncio.gather(publication, return_exceptions=True)

    assert agent.cleanup_saw_stopped_delivery


class _LifecycleAgent(Agent):
    def __init__(
        self,
        *,
        enter_error: BaseException | None = None,
        exit_error: BaseException | None = None,
    ) -> None:
        super().__init__()
        self.enter_error = enter_error
        self.exit_error = exit_error
        self.entries = 0
        self.exits = 0

    @asynccontextmanager
    async def lifespan(self, _ctx: RuntimeContext) -> AsyncIterator[None]:
        self.entries += 1
        if self.enter_error is not None:
            raise self.enter_error
        try:
            yield
        finally:
            self.exits += 1
            if self.exit_error is not None:
                raise self.exit_error


async def test_lifespan_entry_failure_exits_only_entered_agents() -> None:
    entered = _LifecycleAgent()
    failed = _LifecycleAgent(enter_error=RuntimeError("enter failed"))
    not_entered = _LifecycleAgent()
    runtime = AgentRuntime()
    runtime.register("entered", entered)
    runtime.register("failed", failed)
    runtime.register("not-entered", not_entered)

    with pytest.raises(RuntimeError, match="enter failed"):
        await runtime.start()

    assert (entered.entries, entered.exits) == (1, 1)
    assert (failed.entries, failed.exits) == (1, 0)
    assert (not_entered.entries, not_entered.exits) == (0, 0)


async def test_stop_reports_every_lifespan_exit_failure() -> None:
    first = _LifecycleAgent(exit_error=ValueError("first exit failed"))
    second = _LifecycleAgent(exit_error=RuntimeError("second exit failed"))
    runtime = AgentRuntime()
    runtime.register("first", first)
    runtime.register("second", second)
    await runtime.start()

    with pytest.raises(ExceptionGroup) as raised:
        await runtime.stop()

    assert [str(error) for error in raised.value.exceptions] == [
        "second exit failed",
        "first exit failed",
    ]


async def test_cancelled_lifespan_exit_does_not_skip_remaining_cleanup() -> None:
    first = _LifecycleAgent()
    second = _LifecycleAgent(exit_error=asyncio.CancelledError())
    runtime = AgentRuntime()
    runtime.register("first", first)
    runtime.register("second", second)
    await runtime.start()

    with pytest.raises(BaseExceptionGroup) as raised:
        await runtime.stop()

    assert isinstance(raised.value.exceptions[0], asyncio.CancelledError)
    assert (first.exits, second.exits) == (1, 1)


class _PartiallyInvalidSubscriber(Agent):
    def __init__(self) -> None:
        super().__init__()

    @subscribe(TRANSACTIONAL_ECHO)
    async def a_valid(self, _message: _Echo, _ctx: RuntimeContext) -> None:
        return None

    @subscribe(TRANSACTIONAL_ECHO)
    async def z_invalid(self, _message: str, _ctx: RuntimeContext) -> None:
        return None


class _TransactionalSubscriber(Agent):
    def __init__(self) -> None:
        super().__init__()

    @subscribe(TRANSACTIONAL_COUNT)
    async def receive(self, _message: _Count, _ctx: RuntimeContext) -> None:
        return None


def test_registration_failure_does_not_mutate_topic_state() -> None:
    runtime = AgentRuntime()

    with pytest.raises(TypeError, match="Pydantic models"):
        runtime.register("invalid", _PartiallyInvalidSubscriber())

    runtime.register("valid", _TransactionalSubscriber())


def test_agent_rejects_duplicate_or_non_tool_members() -> None:
    tool = Tool("echo", "Echo text.", _Echo, _Echo, lambda request: request)

    with pytest.raises(ValueError, match="duplicate"):
        Agent((tool, tool))
    with pytest.raises(TypeError, match="Tool or AsyncTool"):
        Agent((object(),))  # type: ignore[arg-type]


class _MissingSuperAgent(Agent):
    def __init__(self) -> None:
        pass


async def test_runtime_rejects_invalid_registration_and_publication() -> None:
    runtime = AgentRuntime()
    runtime.register("agent", Agent())

    with pytest.raises(TypeError, match=r"super\(\)"):
        AgentRuntime().register("invalid", _MissingSuperAgent())
    with pytest.raises(ValueError, match="already registered"):
        runtime.register("agent", Agent())
    with pytest.raises(RuntimeClosedError):
        await runtime.publish(
            OBSERVATIONS,
            _Observation(labels=[]),
            participant_id="alice",
        )

    async with runtime:
        with pytest.raises(ValueError, match="must not be empty"):
            await runtime.publish(
                OBSERVATIONS,
                _Observation(labels=[]),
                participant_id="",
            )

    with pytest.raises(RuntimeClosedError):
        await runtime.publish(
            OBSERVATIONS,
            _Observation(labels=[]),
            participant_id="alice",
        )
