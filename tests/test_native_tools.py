# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for Relay-managed native tools and model-selected tool calls."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import nemo_relay
import pytest
from pydantic import BaseModel
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

    assert "tool" in {getattr(event, "category", None) for event in events}


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
