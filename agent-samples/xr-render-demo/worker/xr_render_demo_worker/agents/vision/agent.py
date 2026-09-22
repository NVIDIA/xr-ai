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
from ..._trace import current_participant_id, current_reference_time_us, current_trace_id
from ...models import SubagentResult, SubagentTask
from ...scene import SceneContext
from .._adaptive import adaptive_result, refusal_toolset

_PROMPT = Path(__file__).with_name("prompt.txt")
# _SHARED_RULES ends mid-sentence: each description completes it differently.
_SHARED_RULES = (
    "Any request to describe or survey what is visible means the physical view. Never use for "
    "facts about XR objects: SCENE OBJECTS is always current and complete. Never use as a "
    "prerequisite for a mutation; mutating agents read physical color sources themselves. A new "
    "question about what the user holds, wears, or sees always needs fresh evidence. If live "
    "vision is unavailable this agent reports so"
)

DESCRIPTION = (
    "Owns every unqualified request to describe what the assistant or user can see, survey what "
    "is visible, or inspect the user's physical surroundings. Use the live camera for the present "
    "and recorded video for a past moment. Never use as a prerequisite for an XR mutation or for "
    "a fact whose subject is listed in SCENE OBJECTS. " + _SHARED_RULES + "; never substitute "
    "recorded video for the present."
)

_LIVE_ONLY_DESCRIPTION = (
    "Answer standalone questions about the user's present physical surroundings from the live "
    "camera. Recorded video is unavailable, so report that for past moments. " + _SHARED_RULES + "."
)

_EXAMPLES = (
    "'What am I carrying at present?' uses live physical evidence.",
    "'What item was in my hand a moment earlier?' uses recorded physical evidence.",
    "'Survey what is visible around me' uses live physical evidence, never SCENE OBJECTS.",
    "A physical color mentioned inside a requested scene mutation stays inside that mutation; "
    "do not observe it separately.",
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
            return await image_query.execute(ImageQueryRequest(image=frame.image, query=req.question))

        tools = [
            Tool(
                "look_at_current_frame",
                "Required exactly once before answering any focused question about the user's "
                "present physical view, including what they hold, wear, or see. Call even when the "
                "instruction claims vision is unavailable; only this tool establishes that. Do not "
                "use to interpret conversation or inspect the virtual XR scene.",
                _LiveQuestion,
                ImageQueryResult,
                look,
                examples=(
                    "For a present question about what the user wears or carries, call exactly "
                    "once with that specific question.",
                ),
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
                return await image_query.execute(ImageQueryRequest(image=frame.image, query=req.question))

            tools.append(
                Tool(
                    "look_at_past_frame",
                    "Required exactly once for an explicit question about the past physical view. "
                    "Inspect a recorded camera frame from seconds_ago seconds before the utterance "
                    "timestamp and preserve recorded-video failure wording.",
                    _PastQuestion,
                    ImageQueryResult,
                    look_past,
                    examples=(
                        "For a question about twelve seconds earlier, call with seconds_ago=12 "
                        "and preserve the requested physical fact in question.",
                    ),
                )
            )
        toolset = tolerant_toolset(tools)
        toolset = refusal_toolset(toolset)
        scene_block = ""
        if context is not None:
            scene_block = f"{await context.describe(current_participant_id.get())}\n\n"
        messages = [
            ChatMessage(role="system", content=_prompt_text),
            ChatMessage(
                role="user",
                content=(
                    f"Active participant: {participant_id}\n"
                    f"Utterance timestamp: {reference_time_us}\n"
                    f"{scene_block}"
                    f"Focused instruction: {request.instruction}"
                ),
            ),
        ]

        async def _call_model(transcript, definitions):
            return await llm.chat(
                transcript,
                tools=list(definitions) or None,
                max_tokens=2048,
                temperature=0.0,
                enable_thinking=False,
            )

        try:
            loop_result = await run_tool_loop(
                messages,
                toolset,
                _call_model,
                max_iterations=6,
            )
        except ToolLoopError:
            return SubagentResult(result="I couldn't complete that. Please try again.")
        return await adaptive_result(
            llm,
            instruction=request.instruction,
            responsibility=(
                f"{description} When directly delegated a question about an XR object, answer "
                "from the supplied complete SCENE OBJECTS context without a perception call."
            ),
            result=loop_result,
        )

    description = DESCRIPTION if video is not None else _LIVE_ONLY_DESCRIPTION
    return Tool(
        name="vision_agent",
        description=description,
        request_model=SubagentTask,
        result_model=SubagentResult,
        handler=handle,
        examples=_EXAMPLES,
    )


__all__ = ["make_vision_agent"]
