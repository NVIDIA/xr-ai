# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for dispatching model-selected native tool calls."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from xr_ai_models import ChatMessage, ToolCall, ToolDef

from .tools import Tool, ToolSet


@dataclass(frozen=True, slots=True)
class ToolCallResult:
    """One model-ready tool response and its control-flow hint."""

    message: ChatMessage
    return_direct: bool


def tool_definitions(tools: Iterable[Tool[Any, Any]]) -> tuple[ToolDef, ...]:
    """Return model-service definitions for native tools."""

    return tuple(
        ToolDef(
            name=tool.name,
            description=tool.description,
            parameters=tool.request_model.model_json_schema(),
        )
        for tool in tools
    )


async def handle_tool_call(call: ToolCall, tools: ToolSet) -> ToolCallResult:
    """Invoke one model-produced call and return its tool-role message."""

    tool = tools.get(call.name)
    if tool is None:
        content = json.dumps({"error": "unknown_tool", "tool": call.name})
        return_direct = False
    else:
        invocation = await tool.invoke(call.arguments)
        content = invocation.content
        return_direct = invocation.return_direct
    return ToolCallResult(
        message=ChatMessage(
            role="tool",
            content=content,
            tool_call_id=call.id,
        ),
        return_direct=return_direct,
    )


__all__ = ["ToolCallResult", "handle_tool_call", "tool_definitions"]
