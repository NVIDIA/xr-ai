# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for Relay-managed native tools and model-selected tool calls."""

from __future__ import annotations

import asyncio
import json
import traceback
from builtins import BaseExceptionGroup
from collections.abc import AsyncIterator

import nemo_relay
import pytest
from pydantic import BaseModel
from xr_ai_hub import FrameUnavailable
from xr_ai_models import ChatMessage, ToolCall, ToolDef
from xr_ai_tools import AsyncTool, Tool, ToolSet
from xr_ai_tools.tool_calling import handle_tool_call, tool_definitions


class AddRequest(BaseModel):
    """Two integers to add."""

    left: int
    right: int


class AddResult(BaseModel):
    """The computed total."""

    total: int


async def add(request: AddRequest) -> AddResult:
    return AddResult(total=request.left + request.right)


async def add_stream(request: AddRequest) -> AsyncIterator[AddResult]:
    yield AddResult(total=request.left)
    yield AddResult(total=request.left + request.right)


async def acknowledge(_request: AddRequest) -> None:
    return None


async def test_async_tool_validates_and_yields_typed_chunks() -> None:
    tool = AsyncTool(
        "stream_add",
        "Stream a running total.",
        AddRequest,
        AddResult,
        add_stream,
    )

    chunks = [chunk async for chunk in tool.stream({"left": 2, "right": 3})]

    assert chunks == [AddResult(total=2), AddResult(total=5)]


async def test_async_tool_yields_chunk_buffered_before_producer_completion() -> None:
    async def single_chunk(request: AddRequest) -> AsyncIterator[AddResult]:
        yield AddResult(total=request.left)

    tool = AsyncTool(
        "stream_add",
        "Stream a running total.",
        AddRequest,
        AddResult,
        single_chunk,
    )

    chunks = [chunk async for chunk in tool.stream({"left": 2, "right": 3})]

    assert chunks == [AddResult(total=2)]


async def test_async_tool_propagates_handler_failure_after_buffered_chunks() -> None:
    emitted = []

    async def failing_stream(request: AddRequest) -> AsyncIterator[AddResult]:
        yield AddResult(total=request.left)
        raise RuntimeError("stream failed")

    tool = AsyncTool(
        "stream_add",
        "Stream a running total.",
        AddRequest,
        AddResult,
        failing_stream,
    )

    with pytest.raises(RuntimeError, match="stream failed"):
        async for chunk in tool.stream({"left": 2, "right": 3}):
            emitted.append(chunk)

    assert emitted == [AddResult(total=2)]


async def test_async_tool_propagates_base_exception_group_without_hanging() -> None:
    class FatalStreamError(BaseException):
        pass

    async def fatal_stream(_request: AddRequest) -> AsyncIterator[AddResult]:
        if False:
            yield AddResult(total=0)
        raise BaseExceptionGroup("stream failed", [FatalStreamError()])

    tool = AsyncTool(
        "stream_add",
        "Stream a running total.",
        AddRequest,
        AddResult,
        fatal_stream,
    )

    async def consume() -> None:
        async for _chunk in tool.stream({"left": 2, "right": 3}):
            pass

    with pytest.raises(BaseExceptionGroup, match="stream failed"):
        await asyncio.wait_for(consume(), timeout=1.0)


async def test_async_tool_consumer_cancellation_closes_handler() -> None:
    started = asyncio.Event()
    closed = asyncio.Event()
    blocked = asyncio.Event()

    async def blocking_stream(_request: AddRequest) -> AsyncIterator[AddResult]:
        started.set()
        try:
            await blocked.wait()
            if False:
                yield AddResult(total=0)
        finally:
            closed.set()

    tool = AsyncTool(
        "stream_add",
        "Stream a running total.",
        AddRequest,
        AddResult,
        blocking_stream,
    )

    async def consume() -> None:
        async for _chunk in tool.stream({"left": 2, "right": 3}):
            pass

    consumer = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), timeout=1.0)
    consumer.cancel()

    with pytest.raises(asyncio.CancelledError):
        _ = await consumer

    assert closed.is_set()


async def test_async_tool_closes_an_abandoned_stream_in_its_relay_context() -> None:
    closed = asyncio.Event()
    blocked = asyncio.Event()

    async def blocking_stream(request: AddRequest) -> AsyncIterator[AddResult]:
        try:
            yield AddResult(total=request.left)
            await blocked.wait()
        finally:
            closed.set()

    tool = AsyncTool(
        "stream_add",
        "Stream a running total.",
        AddRequest,
        AddResult,
        blocking_stream,
    )
    consumer_scope_unchanged: list[bool] = []

    async def abandon_stream() -> None:
        consumer_scope = nemo_relay.scope.get_handle()
        try:
            async for chunk in tool.stream({"left": 2, "right": 3}):
                assert chunk == AddResult(total=2)
                raise RuntimeError("consumer failed")
        except RuntimeError:
            consumer_scope_unchanged.append(
                nemo_relay.scope.get_handle().uuid == consumer_scope.uuid
            )

    # A forked consumer reproduces finalization outside the caller's Relay context.
    await asyncio.create_task(
        abandon_stream(),
        context=nemo_relay.fork_asyncio_context(),
    )
    await asyncio.wait_for(closed.wait(), timeout=1.0)

    # This is the assertion that fails when the tool scope leaks across a yield.
    assert consumer_scope_unchanged == [True]


def test_tool_definitions_adapt_native_tools_for_model_services() -> None:
    tool = Tool("add", "Add two integers.", AddRequest, AddResult, add)

    assert tool_definitions((tool,)) == (
        ToolDef(
            name="add",
            description="Add two integers.",
            parameters=AddRequest.model_json_schema(),
        ),
    )


async def test_handle_tool_call_returns_a_model_ready_tool_message() -> None:
    tool = Tool("add", "Add two integers.", AddRequest, AddResult, add)

    result = await handle_tool_call(
        ToolCall(id="add-call", name="add", arguments='{"left":2,"right":3}'),
        ToolSet((tool,)),
    )

    assert result.message == ChatMessage(
        role="tool",
        content='{"total":5}',
        tool_call_id="add-call",
    )
    assert result.return_direct is False


async def test_side_effect_tool_returns_none_and_renders_null() -> None:
    tool = Tool("acknowledge", "Acknowledge input.", AddRequest, None, acknowledge)

    direct = await tool.execute(AddRequest(left=2, right=3))
    model = await tool.invoke('{"left":2,"right":3}')

    assert direct is None
    assert model.content == "null"


async def test_handled_tool_calls_use_the_relay_tool_lifecycle() -> None:
    tool = Tool("add", "Add two integers.", AddRequest, AddResult, add)
    events = []
    subscriber = "xr-ai-native-tool-call-lifecycle"
    nemo_relay.subscribers.register(subscriber, events.append)
    try:
        await handle_tool_call(
            ToolCall(id="add-call", name="add", arguments='{"left":2,"right":3}'),
            ToolSet((tool,)),
        )
        await nemo_relay.subscribers.flush_async()
    finally:
        nemo_relay.subscribers.deregister(subscriber)

    lifecycle = [event for event in events if event.category == "tool" and event.name == "add"]
    assert [event.scope_category for event in lifecycle] == ["start", "end"]
    assert lifecycle[1].data == {"total": 5}
    assert lifecycle[1].metadata["otel.status_code"] == "OK"


async def test_invalid_tool_arguments_are_returned_to_the_model_for_repair() -> None:
    tool = Tool("add", "Add two integers.", AddRequest, AddResult, add)

    result = await handle_tool_call(
        ToolCall(id="add-call", name="add", arguments='{"left":"not-an-int"}'),
        ToolSet((tool,)),
    )

    assert result.return_direct is False
    assert isinstance(result.message.content, str)
    payload = json.loads(result.message.content)
    assert payload["error"] == "invalid_tool_arguments"
    assert "right" in payload["detail"]


async def test_unknown_tool_is_returned_to_the_model_for_repair() -> None:
    result = await handle_tool_call(
        ToolCall(id="missing-call", name="missing", arguments="{}"),
        ToolSet(()),
    )

    assert result.message == ChatMessage(
        role="tool",
        content='{"error": "unknown_tool", "tool": "missing"}',
        tool_call_id="missing-call",
    )
    assert result.return_direct is False


async def test_handle_tool_call_preserves_return_direct() -> None:
    tool = Tool(
        "add",
        "Add two integers.",
        AddRequest,
        AddResult,
        add,
        return_direct=True,
    )

    result = await handle_tool_call(
        ToolCall(id="add-call", name="add", arguments='{"left":2,"right":3}'),
        ToolSet((tool,)),
    )

    assert result.message.content == '{"total":5}'
    assert result.return_direct is True


def test_tool_sets_reject_duplicate_names() -> None:
    tool = Tool("add", "Add two integers.", AddRequest, AddResult, add)

    with pytest.raises(ValueError, match="duplicate tool name"):
        ToolSet((tool, tool))


async def test_tool_set_aliases_remap_model_definition_and_dispatch_names() -> None:
    tool = Tool("add", "Add two integers.", AddRequest, AddResult, add)
    tools = ToolSet({"sum": tool})

    assert tool_definitions(tools) == (
        ToolDef(
            name="sum",
            description="Add two integers.",
            parameters=AddRequest.model_json_schema(),
        ),
    )
    result = await handle_tool_call(
        ToolCall(id="sum-call", name="sum", arguments='{"left":2,"right":3}'),
        tools,
    )

    assert result.message.content == '{"total":5}'
    assert tool.name == "add"


def test_tool_set_namespaces_similarly_named_tool_groups() -> None:
    vision_status = Tool("status", "Vision status.", AddRequest, AddResult, add)
    planner_status = Tool("status", "Planner status.", AddRequest, AddResult, add)
    tools = ToolSet.namespaced(
        {
            "vision": (vision_status,),
            "planner": (planner_status,),
        }
    )

    assert [definition.name for definition in tool_definitions(tools)] == [
        "vision__status",
        "planner__status",
    ]
    assert tools.get("vision__status") is vision_status
    assert tools.get("planner__status") is planner_status


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("error_type", [FrameUnavailable, ValueError])
async def test_tool_preserves_handler_exception_and_failed_relay_trace(
    asynchronous: bool, error_type: type[Exception],
) -> None:
    cause = OSError("source failed")
    error = error_type("handler failed")

    def fail(_request: AddRequest) -> AddResult:
        raise error from cause

    async def async_fail(request: AddRequest) -> AddResult:
        return fail(request)

    tool = Tool(
        "typed_failure", "Fail with a typed exception.", AddRequest, AddResult,
        async_fail if asynchronous else fail,
    )
    events = []
    subscriber = "xr-ai-typed-failure"
    nemo_relay.subscribers.register(subscriber, events.append)
    try:
        with pytest.raises(error_type) as raised:
            await tool.execute(AddRequest(left=2, right=3))
        await nemo_relay.subscribers.flush_async()
    finally:
        nemo_relay.subscribers.deregister(subscriber)

    assert raised.value is error
    assert raised.value.__cause__ is cause
    assert any(frame.name == "fail" for frame in traceback.extract_tb(error.__traceback__))
    lifecycle = [event for event in events if event.name == tool.name]
    assert [event.scope_category for event in lifecycle] == ["start", "end"]
    assert lifecycle[0].uuid == lifecycle[1].uuid
    assert lifecycle[1].metadata["otel.status_code"] == "ERROR"
    assert lifecycle[1].metadata["exception.type"] == error_type.__name__


async def test_tool_concurrent_handler_failures_are_invocation_local() -> None:
    both_started = asyncio.Event()
    release = asyncio.Event()
    errors = {1: FrameUnavailable("first"), 2: ValueError("second")}
    started = 0

    async def fail(request: AddRequest) -> AddResult:
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        await release.wait()
        raise errors[request.left]

    tool = Tool("concurrent_failure", "Fail independently.", AddRequest, AddResult, fail)
    first = asyncio.create_task(tool.execute(AddRequest(left=1, right=0)))
    second = asyncio.create_task(tool.execute(AddRequest(left=2, right=0)))
    try:
        await asyncio.wait_for(both_started.wait(), timeout=1.0)
    finally:
        release.set()
    results = await asyncio.gather(first, second, return_exceptions=True)

    assert results[0] is errors[1]
    assert results[1] is errors[2]
    tool.handler = add
    assert await tool.execute(AddRequest(left=1, right=2)) == AddResult(total=3)


async def test_tool_cancellation_propagates_and_closes_handler() -> None:
    started = asyncio.Event()
    closed = asyncio.Event()

    async def block(_request: AddRequest) -> AddResult:
        started.set()
        try:
            await asyncio.Future()
        finally:
            closed.set()

    tool = Tool("cancel_finite", "Wait for cancellation.", AddRequest, AddResult, block)
    task = asyncio.create_task(tool.execute(AddRequest(left=1, right=2)))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(closed.wait(), timeout=1.0)


async def test_tool_result_codec_failure_retains_relay_error() -> None:
    def invalid_result(_request: AddRequest) -> None:
        return "invalid"  # type: ignore[return-value]

    tool = Tool("invalid_result", "Return an invalid result.", AddRequest, None, invalid_result)
    with pytest.raises(TypeError, match="side-effect tools must return None"):
        await tool.execute(AddRequest(left=1, right=2))


async def test_async_tool_preserves_typed_failure_after_buffered_chunk() -> None:
    error = FrameUnavailable("stream source unavailable")
    chunks = []

    async def stream(request: AddRequest) -> AsyncIterator[AddResult]:
        yield AddResult(total=request.left)
        raise error

    tool = AsyncTool("typed_stream", "Stream then fail.", AddRequest, AddResult, stream)
    with pytest.raises(FrameUnavailable) as raised:
        async for chunk in tool.stream(AddRequest(left=1, right=2)):
            chunks.append(chunk)
    assert raised.value is error
    assert chunks == [AddResult(total=1)]


@pytest.mark.parametrize("recover", [False, True])
async def test_relay_middleware_owns_recovery_or_replacement(recover: bool) -> None:
    source = FrameUnavailable("camera unavailable")
    replacement = LookupError("middleware replacement")
    observed = []
    events = []

    async def fail(_request: AddRequest) -> AddResult:
        raise source

    async def intercept(context, next_call):
        try:
            result = await next_call(context.args)
        except FrameUnavailable as exc:
            observed.append(exc)
            if not recover:
                raise replacement from exc
            return nemo_relay.ToolExecutionInterceptOutcome({"total": 9})
        return nemo_relay.ToolExecutionInterceptOutcome(result.result)

    tool = Tool("intercepted_failure", "Let middleware handle failure.", AddRequest, AddResult, fail)
    name = "xr-ai-native-tool-recovery"
    nemo_relay.intercepts.register_tool_execution(name, 0, intercept)
    nemo_relay.subscribers.register(name, events.append)
    try:
        if recover:
            assert await tool.execute(AddRequest(left=1, right=2)) == AddResult(total=9)
        else:
            with pytest.raises(LookupError) as raised:
                await tool.execute(AddRequest(left=1, right=2))
            assert raised.value is replacement
            assert raised.value.__cause__ is source
        await nemo_relay.subscribers.flush_async()
    finally:
        nemo_relay.intercepts.deregister_tool_execution(name)
        nemo_relay.subscribers.deregister(name)
    assert observed == [source]
    end = next(event for event in events if event.name == tool.name and event.scope_category == "end")
    assert end.metadata["otel.status_code"] == ("OK" if recover else "ERROR")
    if not recover:
        assert end.metadata["exception.type"] == "LookupError"
