# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for dispatching model-selected native tool calls."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from xr_ai_models import ChatMessage, ChatResponse, ToolCall, ToolDef

from .tools import Tool, ToolSet


@dataclass(frozen=True, slots=True)
class ToolCallResult:
    """One model-ready tool response and its control-flow hint."""

    message: ChatMessage
    return_direct: bool


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    """One model-requested call and the model-visible result it produced."""

    call: ToolCall
    message: ChatMessage
    return_direct: bool


@dataclass(frozen=True, slots=True)
class ToolLoopResult:
    """A completed turn with its full transcript and tool-call audit."""

    content: str
    messages: tuple[ChatMessage, ...]
    tool_calls: tuple[ToolCallRecord, ...]
    iterations: int
    return_direct: bool


class ToolLoopError(RuntimeError):
    """Base error for a turn that cannot safely produce a final answer."""

    def __init__(
        self,
        message: str,
        *,
        messages: Sequence[ChatMessage],
        tool_calls: Sequence[ToolCallRecord],
        iterations: int,
    ) -> None:
        super().__init__(message)
        self.messages = tuple(messages)
        self.tool_calls = tuple(tool_calls)
        self.iterations = iterations


class ToolLoopIterationLimitError(ToolLoopError):
    """The model used every permitted iteration without a final answer."""


class ToolLoopToolCallLimitError(ToolLoopError):
    """A model batch would exceed the turn's total tool-call budget."""


class ToolLoopTruncatedResponseError(ToolLoopError):
    """The model reported a truncated response that is unsafe to execute."""


class ToolLoopEmptyResponseError(ToolLoopError):
    """The model returned neither a tool call nor a non-empty answer."""


class ToolLoopDirectReturnError(ToolLoopError):
    """A model batch mixed a direct-return tool with other calls."""


class ToolLoopMalformedResponseError(ToolLoopError):
    """The model returned tool calls that cannot form a valid transcript."""


ModelCallback = Callable[
    [tuple[ChatMessage, ...], tuple[ToolDef, ...]],
    Awaitable[ChatResponse],
]


def tool_definitions(
    tools: Iterable[Tool[Any, Any]] | ToolSet,
) -> tuple[ToolDef, ...]:
    """Return model-service definitions for native tools."""

    entries = (
        tools.items()
        if isinstance(tools, ToolSet)
        else tuple((tool.name, tool) for tool in tools)
    )
    return tuple(
        ToolDef(
            name=name,
            description=tool.description,
            parameters=tool.request_model.model_json_schema(),
        )
        for name, tool in entries
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


async def run_tool_loop(
    messages: Sequence[ChatMessage],
    tools: ToolSet,
    call_model: ModelCallback,
    *,
    max_iterations: int = 4,
    max_tool_calls: int = 16,
) -> ToolLoopResult:
    """Run one stateless, bounded model turn over native tools.

    The callback owns all model parameters and retry policy. The loop only
    supplies the current immutable transcript and the current turn's tool
    definitions. Calls in a model batch execute sequentially in emitted order.
    """

    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if max_tool_calls < 0:
        raise ValueError("max_tool_calls must not be negative")

    transcript = list(messages)
    definitions = tool_definitions(tools)
    records: list[ToolCallRecord] = []
    seen_call_ids: set[str] = set()

    for iteration in range(1, max_iterations + 1):
        response = await call_model(tuple(transcript), definitions)
        calls = tuple(response.tool_calls or ())
        transcript.append(
            ChatMessage(
                role="assistant",
                content=response.content,
                tool_calls=list(calls) if response.tool_calls is not None else None,
            )
        )

        if response.finish_reason in {"length", "max_tokens"}:
            raise ToolLoopTruncatedResponseError(
                "model response was truncated",
                messages=transcript,
                tool_calls=records,
                iterations=iteration,
            )

        if response.finish_reason == "tool_calls" and not calls:
            raise ToolLoopMalformedResponseError(
                "model reported tool calls without returning any calls",
                messages=transcript,
                tool_calls=records,
                iterations=iteration,
            )

        call_ids = tuple(call.id for call in calls)
        if any(not call_id.strip() for call_id in call_ids):
            raise ToolLoopMalformedResponseError(
                "tool-call IDs must not be blank",
                messages=transcript,
                tool_calls=records,
                iterations=iteration,
            )
        if len(set(call_ids)) != len(call_ids) or seen_call_ids.intersection(call_ids):
            raise ToolLoopMalformedResponseError(
                "tool-call IDs must be unique within a turn",
                messages=transcript,
                tool_calls=records,
                iterations=iteration,
            )

        direct_calls = tuple(
            call
            for call in calls
            if (tool := tools.get(call.name)) is not None and tool.return_direct
        )
        if direct_calls and len(calls) != 1:
            raise ToolLoopDirectReturnError(
                "a direct-return tool must be the only call in its model batch",
                messages=transcript,
                tool_calls=records,
                iterations=iteration,
            )

        if len(records) + len(calls) > max_tool_calls:
            raise ToolLoopToolCallLimitError(
                f"model batch would exceed the {max_tool_calls} tool-call limit",
                messages=transcript,
                tool_calls=records,
                iterations=iteration,
            )

        seen_call_ids.update(call_ids)
        if not calls:
            content = response.content.strip()
            if not content:
                raise ToolLoopEmptyResponseError(
                    "model returned neither tool calls nor a final answer",
                    messages=transcript,
                    tool_calls=records,
                    iterations=iteration,
                )
            return ToolLoopResult(
                content=content,
                messages=tuple(transcript),
                tool_calls=tuple(records),
                iterations=iteration,
                return_direct=False,
            )

        for call in calls:
            result = await handle_tool_call(call, tools)
            transcript.append(result.message)
            records.append(
                ToolCallRecord(
                    call=call,
                    message=result.message,
                    return_direct=result.return_direct,
                )
            )
            if result.return_direct:
                return ToolLoopResult(
                    content=str(result.message.content),
                    messages=tuple(transcript),
                    tool_calls=tuple(records),
                    iterations=iteration,
                    return_direct=True,
                )

    raise ToolLoopIterationLimitError(
        f"model did not produce a final answer within {max_iterations} iterations",
        messages=transcript,
        tool_calls=records,
        iterations=max_iterations,
    )


__all__ = [
    "ModelCallback",
    "ToolCallRecord",
    "ToolCallResult",
    "ToolLoopDirectReturnError",
    "ToolLoopEmptyResponseError",
    "ToolLoopError",
    "ToolLoopIterationLimitError",
    "ToolLoopMalformedResponseError",
    "ToolLoopResult",
    "ToolLoopToolCallLimitError",
    "ToolLoopTruncatedResponseError",
    "handle_tool_call",
    "run_tool_loop",
    "tool_definitions",
]
