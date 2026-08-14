# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vision subagent: answer questions from the live or recorded camera."""

from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field
from xr_ai_models import ChatMessage, LLMService
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.historical_vision import HistoricalVisionTool, VisionResult
from xr_ai_tools.live_vision import LiveVisionTool, VisionResponse
from xr_ai_tools.tool_calling import tool_definitions

from ..._loop import tool_loop
from ...models import SubagentResult, SubagentTask
from ...scene import SceneContext

_PROMPT = Path(__file__).with_name("prompt.txt")
DESCRIPTION = (
    "Answer a question about the physical world from the live or recorded camera; "
    "never for XR scene state or placement."
)


class _LiveQuestion(BaseModel):
    question: str = Field(min_length=1, description="Specific question about the live camera frame.")


class _PastQuestion(BaseModel):
    question: str = Field(min_length=1, description="Specific question about the recorded camera frame.")
    second_ago: int = Field(gt=0, description="Positive offset from the utterance time in seconds.")


_prompt_text = _PROMPT.read_text(encoding="utf-8").strip()

def make_vision_agent(
    llm: LLMService,
    live_vision: LiveVisionTool,
    past_vision: HistoricalVisionTool,
    context: SceneContext | None = None,
) -> Tool:
    async def handle(request: SubagentTask) -> SubagentResult:
        logger.debug("vision agent instruction={!r}", request.instruction[:200])
        if context is not None:
            context.mark_delegated(request.participant_id)

        participant_id = request.participant_id
        reference_time_us = request.reference_time_us

        async def look(req: _LiveQuestion) -> BaseModel:
            from xr_ai_tools._vision import VisionRequest
            return await live_vision.execute(VisionRequest(
                participant_id=participant_id,
                query=req.question,
            ))

        async def look_past(req: _PastQuestion) -> BaseModel:
            from xr_ai_tools.historical_vision import HistoricalVisionRequest
            return await past_vision.execute(HistoricalVisionRequest(
                participant_id=participant_id,
                query=req.question,
                second_ago=req.second_ago,
                reference_time_us=reference_time_us,
            ))

        tools = [
            Tool(
                "look_at_current_frame",
                "Inspect the user's present physical view when a request explicitly requires a visible "
                "fact. Do not use this tool to interpret conversation or inspect the virtual XR scene.",
                _LiveQuestion, VisionResponse, look,
            ),
            Tool(
                "look_at_past_frame",
                "Inspect a recorded camera frame only for an explicitly historical question, using a "
                "positive seconds offset from the user's utterance time.",
                _PastQuestion, VisionResult, look_past,
            ),
        ]
        toolset = ToolSet(tools)
        prompt = _prompt_text
        scene_block = ""
        if context is not None:
            scene_block = f"{await context.describe(request.participant_id)}\n\n"
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", content=(
                f"Active participant: {participant_id}\n"
                f"Utterance timestamp: {reference_time_us}\n"
                f"{scene_block}"
                f"Focused instruction: {request.instruction}"
            )),
        ]
        result = await tool_loop(llm, messages, tool_definitions(toolset), toolset)
        return SubagentResult(result=result or "Done.")

    return Tool(name="vision_agent", description=DESCRIPTION,
                request_model=SubagentTask, result_model=SubagentResult, handler=handle)


__all__ = ["make_vision_agent"]
