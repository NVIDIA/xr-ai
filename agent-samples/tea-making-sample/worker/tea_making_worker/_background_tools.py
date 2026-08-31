# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private required-tool loop for background model decisions."""

from __future__ import annotations

from collections.abc import Sequence

from xr_ai_models import ChatMessage
from xr_ai_tools import ToolSet
from xr_ai_tools.tool_calling import ModelCallback, ToolLoopError, run_tool_loop


class RequiredToolCallError(RuntimeError):
    """The background model did not produce its required typed commit."""


async def run_required_tool(
    messages: Sequence[ChatMessage],
    tools: ToolSet,
    call_model: ModelCallback,
    *,
    required_tool: str,
) -> str:
    """Return one validated direct-tool result, retrying one missed commit."""

    transcript = tuple(messages)
    for _attempt in range(2):
        try:
            result = await run_tool_loop(
                transcript,
                tools,
                call_model,
                max_iterations=3,
                max_tool_calls=2,
            )
        except ToolLoopError as exc:
            transcript = _retry_transcript(exc.messages, required_tool)
            continue
        if (
            result.return_direct
            and result.tool_calls
            and result.tool_calls[-1].call.name == required_tool
        ):
            return result.content
        transcript = _retry_transcript(result.messages, required_tool)
    raise RequiredToolCallError(
        f"model did not call required tool {required_tool!r}"
    )


def _retry_transcript(
    messages: Sequence[ChatMessage],
    required_tool: str,
) -> tuple[ChatMessage, ...]:
    return (
        *messages,
        ChatMessage(
            role="user",
            content=(
                f"Call {required_tool} exactly once with valid arguments. "
                "Do not answer directly."
            ),
        ),
    )
