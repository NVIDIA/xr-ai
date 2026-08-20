# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct LLM tool-calling loop shared by the supervisor and every subagent."""

from __future__ import annotations

import json

from loguru import logger
from xr_ai_models import ChatMessage, LLMService, ToolDef
from xr_ai_tools import ToolSet
from xr_ai_tools.tool_calling import ToolCallResult, handle_tool_call

_MAX_ITERATIONS = 12


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
    reason about and degrade gracefully from failures.
    """
    for _ in range(max_iterations):
        response = await llm.chat(
            messages,
            tools=list(tool_defs) or None,
            max_tokens=max_tokens,
            temperature=0.0,
        )

        tool_calls = list(response.tool_calls or ())

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


__all__ = ["tool_loop"]
