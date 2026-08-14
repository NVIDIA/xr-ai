# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Memory subagent: recall earlier conversation turns."""

from pathlib import Path

from loguru import logger
from xr_ai_models import ChatMessage, LLMService
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.text_memory import TextMemoryTools
from xr_ai_tools.tool_calling import tool_definitions

from ..._loop import tool_loop
from ...models import SubagentResult, SubagentTask

_PROMPT = Path(__file__).with_name("prompt.txt")
DESCRIPTION = "Recall earlier conversation turns; never mutates the XR scene."


_prompt_text = _PROMPT.read_text(encoding="utf-8").strip()

def make_memory_agent(llm: LLMService, text_memory: TextMemoryTools) -> Tool:
    recall_tool = text_memory.recall_conversation

    async def handle(request: SubagentTask) -> SubagentResult:
        logger.debug("memory agent instruction={!r}", request.instruction[:200])
        toolset = ToolSet([recall_tool])
        prompt = _prompt_text
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", content=(
                f"Active participant: {request.participant_id}\n"
                f"Utterance timestamp: {request.reference_time_us}\n\n"
                f"Focused instruction: {request.instruction}"
            )),
        ]
        result = await tool_loop(llm, messages, tool_definitions(toolset), toolset)
        return SubagentResult(result=result or "Done.")

    return Tool(name="memory_agent", description=DESCRIPTION,
                request_model=SubagentTask, result_model=SubagentResult, handler=handle)


__all__ = ["make_memory_agent"]
