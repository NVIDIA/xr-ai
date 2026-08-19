# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct LLM tool-calling loop shared by the supervisor and every subagent."""

from __future__ import annotations

import itertools
import json

from loguru import logger
from xr_ai_models import ChatMessage, LLMService, ToolCall, ToolDef
from xr_ai_tools import ToolSet
from xr_ai_tools.tool_calling import ToolCallResult, handle_tool_call

_MAX_ITERATIONS = 12
_recovery_counter = itertools.count()


async def tool_loop(
    llm: LLMService,
    messages: list[ChatMessage],
    tool_defs: tuple[ToolDef, ...],
    toolset: ToolSet,
    *,
    max_iterations: int = _MAX_ITERATIONS,
    max_tokens: int = 2048,
) -> str:
    """Run an agentic loop: call LLM, dispatch tool calls, repeat until text.

    Returns the final text response, or an apology if the iteration cap is hit.
    Tool exceptions are converted to error tool-result messages so the model can
    reason about and degrade gracefully from failures. Named tool calls that
    appear as bare JSON in content are recovered before dispatching.
    """
    for _ in range(max_iterations):
        response = await llm.chat(
            messages,
            tools=list(tool_defs) or None,
            max_tokens=max_tokens,
            temperature=0.0,
        )

        tool_calls = list(response.tool_calls or ())

        # Recover a named tool call the model emitted as bare JSON in content.
        if not tool_calls and response.content and tool_defs:
            if recovered := _recover(response.content, tool_defs):
                tool_calls = [recovered]
                response = type(response)(
                    content="",
                    reasoning=None,
                    tool_calls=tool_calls,
                    finish_reason=response.finish_reason,
                    raw=response.raw,
                )

        if not tool_calls:
            return response.content or ""

        messages.append(ChatMessage(
            role="assistant",
            content=response.content or "",
            tool_calls=tool_calls,
        ))

        for call in tool_calls:
            try:
                result = await handle_tool_call(call, toolset)
            except ValueError as exc:
                logger.debug("tool {} rejected input: {!r}", call.name, exc)
                result = ToolCallResult(
                    message=ChatMessage(
                        role="tool",
                        content=json.dumps({"error": type(exc).__name__, "detail": str(exc)}),
                        tool_call_id=call.id,
                    ),
                    return_direct=False,
                )
            except Exception as exc:
                logger.exception("tool {} failed unexpectedly", call.name)
                result = ToolCallResult(
                    message=ChatMessage(
                        role="tool",
                        content=json.dumps({"error": type(exc).__name__, "detail": str(exc)}),
                        tool_call_id=call.id,
                    ),
                    return_direct=False,
                )
            messages.append(result.message)
            if result.return_direct:
                return result.message.content

    return "I'm sorry — I reached the maximum number of steps. Please try again."


def _recover(content: str, tool_defs: tuple[ToolDef, ...]) -> ToolCall | None:
    """Recover an explicitly named tool call from bare JSON in content."""
    text = content.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    for wrapper in ("command", "function", "tool_call"):
        wrapped = data.get(wrapper)
        if isinstance(wrapped, dict) and (wrapped.get("name") or wrapped.get("tool") or wrapped.get("action")):
            data = wrapped
            break
    name = data.get("name") or data.get("tool") or data.get("action")
    offered = {d.name: d for d in tool_defs}
    if isinstance(name, str) and name not in offered:
        matches = [n for n in offered if n.endswith(f"__{name}")]
        name = matches[0] if len(matches) == 1 else name
    if not (isinstance(name, str) and name in offered):
        return None
    arguments = data.get("arguments", data.get("args", data.get("parameters")))
    if arguments is None:
        arguments = {k: v for k, v in data.items() if k not in ("name", "tool", "action", "id", "type")}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    return ToolCall(id=f"recovered-{name}-{next(_recovery_counter)}", name=name, arguments=json.dumps(arguments))


__all__ = ["tool_loop"]
