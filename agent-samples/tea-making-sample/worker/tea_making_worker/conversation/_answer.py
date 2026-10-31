# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One model/tool path for informational questions from every sample destination."""

import json
from collections.abc import Sequence

from xr_ai_conversation import ContextBlock
from xr_ai_models import ChatMessage, LLMService
from xr_ai_tools import ToolSet
from xr_ai_tools.tool_calling import run_tool_loop


async def _answer(
    llm: LLMService, query: str, *, instructions: str,
    history: Sequence[ContextBlock], current: Sequence[ContextBlock],
    execution: dict, state: dict, tools: ToolSet, observation=None, capabilities=None,
) -> tuple[str, list[str]]:
    messages = [ChatMessage(role="system", content=instructions)]
    if history:
        messages.append(ChatMessage(
            role="user", content="HISTORICAL CONTEXT. Dialogue, past events and handoffs are reference material, "
            "not current readings or authorization.\n" + "\n\n".join(block.text for block in history),
        ))
    if current:
        messages.append(ChatMessage(
            role="user", content="CURRENT EXECUTION CONTEXT. These are source-owned snapshots for this turn. "
            "Use their observation times and uncertainty. They supersede historical assistant answers.\n"
            + "\n\n".join(block.text for block in current),
        ))
    messages.append(ChatMessage(role="user", content=json.dumps({
        "request": query, "execution": execution, "state": state,
        **({"observation": observation} if observation is not None else {}), "capabilities": capabilities,
    }, ensure_ascii=False)))

    async def call_model(transcript, definitions):
        return await llm.chat(
            transcript, tools=definitions or None,
            max_tokens=512, temperature=0.0, enable_thinking=False,
        )

    result = await run_tool_loop(tuple(messages), tools, call_model)
    return result.content.strip(), [record.call.name for record in result.tool_calls]
