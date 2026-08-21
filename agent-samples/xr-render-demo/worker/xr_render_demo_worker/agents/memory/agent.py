# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Memory subagent: recall earlier conversation turns."""

from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field
from xr_ai_models import ChatMessage, LLMService
from xr_ai_tools import Tool
from xr_ai_tools.text_memory import (
    RecallConversationRequest,
    RecallConversationResult,
    TextMemoryTools,
)
from xr_ai_tools.tool_calling import ToolLoopError, run_tool_loop

from ..._tolerant import tolerant_toolset
from ..._trace import current_participant_id, current_reference_time_us, current_trace_id
from ...models import SubagentResult, SubagentTask

_PROMPT = Path(__file__).with_name("prompt.txt")
DESCRIPTION = ("Recall what was said in earlier conversation turns; knows nothing about "
               "the physical world, the camera, or any object's color.")

_MAX_END_US = RecallConversationRequest.model_fields["end_us"].default


class _RecallWindow(BaseModel):
    """Model-visible recall window; participant identity is runtime-bound."""

    start_us: int = Field(default=0, description="Inclusive window start in Unix microseconds.")
    end_us: int = Field(default=_MAX_END_US, description="Inclusive window end in Unix microseconds.")


_prompt_text = _PROMPT.read_text(encoding="utf-8").strip()

def make_memory_agent(llm: LLMService, text_memory: TextMemoryTools) -> Tool:
    async def recall(req: _RecallWindow) -> RecallConversationResult:
        return await text_memory.recall_conversation.execute(
            RecallConversationRequest(
                participant_id=current_participant_id.get(),
                start_us=req.start_us,
                end_us=req.end_us,
            )
        )

    recall_tool = Tool(
        "recall_conversation",
        "Recall the active participant's timestamped user and agent turns for a time window.",
        _RecallWindow,
        RecallConversationResult,
        recall,
    )

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
