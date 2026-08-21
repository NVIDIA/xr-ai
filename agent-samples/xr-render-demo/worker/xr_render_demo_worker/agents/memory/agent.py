# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Memory subagent: recall earlier conversation turns."""

from pathlib import Path

from loguru import logger
from xr_ai_models import ChatMessage, LLMService
from xr_ai_tools import Tool
from xr_ai_tools.text_memory import TextMemoryTools
from xr_ai_tools.tool_calling import ToolLoopError, run_tool_loop

from ..._tolerant import tolerant_toolset
from ..._trace import current_participant_id, current_reference_time_us, current_trace_id
from ...models import SubagentResult, SubagentTask

_PROMPT = Path(__file__).with_name("prompt.txt")
DESCRIPTION = "Recall earlier conversation turns; never mutates the XR scene."


_prompt_text = _PROMPT.read_text(encoding="utf-8").strip()

def make_memory_agent(llm: LLMService, text_memory: TextMemoryTools) -> Tool:
    recall_tool = text_memory.recall_conversation

    async def handle(request: SubagentTask) -> SubagentResult:
        logger.debug("memory agent instruction={!r} trace={}", request.instruction[:200], current_trace_id.get())
        toolset = tolerant_toolset([recall_tool])
        prompt = _prompt_text
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", content=(
                f"Active participant: {current_participant_id.get()}\n"
                f"Utterance timestamp: {current_reference_time_us.get()}\n\n"
                f"Focused instruction: {request.instruction}"
            )),
        ]
        async def _call_model(transcript, definitions):
            return await llm.chat(transcript, tools=list(definitions) or None, max_tokens=2048, temperature=0.0)
        try:
            loop_result = await run_tool_loop(messages, toolset, _call_model)
        except ToolLoopError:
            return SubagentResult(result="I couldn't complete that. Please try again.")
        return SubagentResult(result=loop_result.content or "Done.")

    return Tool(name="memory_agent", description=DESCRIPTION,
                request_model=SubagentTask, result_model=SubagentResult, handler=handle)


__all__ = ["make_memory_agent"]
