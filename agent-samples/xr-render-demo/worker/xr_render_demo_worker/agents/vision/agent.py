# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vision subagent: answer questions from the live or recorded camera."""

from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field
from xr_ai_models import ChatMessage, LLMService
from xr_ai_tools import Tool
from xr_ai_tools.current_frame import CurrentFrameTool
from xr_ai_tools.tool_calling import ToolLoopError, run_tool_loop
from xr_ai_tools.video_memory import HistoricalFrameRequest, VideoMemoryTools
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryResult, ImageQueryTool

from ..._tolerant import reraise_unavailable, tolerant_toolset
from ..._trace import (
    current_participant_id,
    current_reference_time_us,
    current_trace_id,
    record_evidence,
)
from ...models import SubagentResult, SubagentTask
from ...scene import SceneContext

_PROMPT = Path(__file__).with_name("prompt.txt")
DESCRIPTION = (
    "Answer a question about the physical world from the live or recorded camera; "
    "never for XR scene state or placement."
)


class _LiveQuestion(BaseModel):
    question: str = Field(min_length=1, description="Specific question about the live camera frame.")


_prompt_text = _PROMPT.read_text(encoding="utf-8").strip()


def make_vision_agent(
    llm: LLMService,
    current_frame: CurrentFrameTool,
    image_query: ImageQueryTool,
    context: SceneContext | None = None,
    video: VideoMemoryTools | None = None,
) -> Tool:
    async def handle(request: SubagentTask) -> SubagentResult:
        logger.debug("vision agent instruction={!r} trace={}", request.instruction[:200], current_trace_id.get())

        participant_id = current_participant_id.get()
        reference_time_us = current_reference_time_us.get()

        async def look(req: _LiveQuestion) -> ImageQueryResult:
            from xr_ai_tools.current_frame import CurrentFrameRequest
            try:
                frame = await current_frame.execute(CurrentFrameRequest(participant_id=participant_id))
            except Exception as error:
                reraise_unavailable(error, "the current camera view")
            try:
                result = await image_query.execute(
                    ImageQueryRequest(image=frame.image, query=req.question)
                )
            except Exception as error:
                reraise_unavailable(error, "image analysis")
            if result.available:
                record_evidence("observed")
            return result

        tools = [
            Tool(
                "look_at_current_frame",
                "Inspect the user's present physical view when a request explicitly requires a visible "
                "fact. Do not use this tool to interpret conversation or inspect the virtual XR scene.",
                _LiveQuestion, ImageQueryResult, look,
            ),
        ]
        if video is not None:
            class _PastQuestion(BaseModel):
                question: str = Field(min_length=1, description="Specific question about the recorded frame.")
                seconds_ago: int = Field(gt=0, le=300, description="Whole seconds before the utterance timestamp.")

            async def look_past(req: _PastQuestion) -> ImageQueryResult:
                start_us = reference_time_us - req.seconds_ago * 1_000_000
                try:
                    frame = await video.get_historical_frame.execute(
                        HistoricalFrameRequest(participant_id=participant_id, start_us=start_us)
                    )
                except Exception as error:
                    reraise_unavailable(error, "recorded video")
                # A recorded observation never licenses a present-tense claim;
                # the supervisor's gate uses it only to steer the reply toward
                # the recorded moment.
                try:
                    result = await image_query.execute(
                        ImageQueryRequest(image=frame.image, query=req.question)
                    )
                except Exception as error:
                    reraise_unavailable(error, "image analysis")
                if result.available:
                    record_evidence("observed_recorded")
                return result

            tools.append(Tool(
                "look_at_past_frame",
                "Inspect a recorded camera frame from seconds_ago seconds before the utterance timestamp.",
                _PastQuestion, ImageQueryResult, look_past,
            ))
        toolset = tolerant_toolset(tools)
        scene_block = ""
        if context is not None:
            scene_block = f"{await context.describe(current_participant_id.get())}\n\n"
        messages = [
            ChatMessage(role="system", content=_prompt_text),
            ChatMessage(role="user", content=(
                f"Active participant: {participant_id}\n"
                f"Utterance timestamp: {reference_time_us}\n"
                f"{scene_block}"
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

    return Tool(name="vision_agent", description=DESCRIPTION,
                request_model=SubagentTask, result_model=SubagentResult, handler=handle)


__all__ = ["make_vision_agent"]
