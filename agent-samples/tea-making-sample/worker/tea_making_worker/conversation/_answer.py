# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate read-only answers through the SDK tool loop.

The caller supplies historical context, fresh source-owned context and a restricted
tool catalog. This module labels their authority, assembles the actual live request
and delegates tool execution to ``run_tool_loop``. It owns no routing or SOP state.
"""

import json
from collections.abc import Sequence

from xr_ai_conversation import ContextBlock
from xr_ai_models import ChatMessage, LLMService
from xr_ai_tools import ToolSet
from xr_ai_tools.tool_calling import run_tool_loop


async def _answer(
    llm: LLMService,
    query: str,
    *,
    instructions: str,
    history: Sequence[ContextBlock],
    current: Sequence[ContextBlock],
    execution: dict,
    state: dict,
    tools: ToolSet,
    observation=None,
    capabilities=None,
) -> tuple[str, list[str]]:
    """Build the live request and run only the caller's read-only tool catalog.

    Args:
        llm: Language service used by the SDK tool loop.
        query: Current user question.
        instructions: Application/workflow policy selected by the caller.
        history: Historical dialogue, events and closed-run handoffs.
        current: Fresh source-owned execution/evidence context blocks.
        execution: Current run identity and active status.
        state: Read-only state projection allowed for this answer.
        tools: Restricted tool catalog; does not expose workflow mutations.
        observation: Optional fresh observation for an informational question.
        capabilities: Optional descriptions of available sample destinations.

    Returns:
        Final answer text and the names of tools actually called by the SDK loop.
    """
    messages = [ChatMessage(role="system", content=instructions)]
    # Historical answers explain what happened; fresh source-owned blocks below
    # supply current state/readings. Keep both labels in the actual model request.
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
            transcript,
            tools=definitions or None,
            max_tokens=512,
            temperature=0.0,
            enable_thinking=False,
        )

    result = await run_tool_loop(tuple(messages), tools, call_model)
    return result.content.strip(), [record.call.name for record in result.tool_calls]
