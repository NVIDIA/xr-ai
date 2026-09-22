# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Appearance subagent: change the color of existing XR objects."""

import asyncio
from pathlib import Path

from loguru import logger
from xr_ai_models import ChatMessage, LLMService
from xr_ai_tools import Tool
from xr_ai_tools.tool_calling import ToolLoopError, run_tool_loop
from xr_render_scene import EmptyRequest, SceneState, SceneTools

from ..._tolerant import tolerant_toolset
from ..._trace import current_participant_id, current_reference_time_us, current_trace_id
from ...models import SubagentResult, SubagentTask
from ...scene import SceneContext
from ...spatial_ops import TurnGuard, make_appearance_tools
from .._adaptive import adaptive_result, reasoning_messages

_PROMPT = Path(__file__).with_name("prompt.txt")
_RESPONSIBILITY = (
    "Owns requested color changes to existing XR objects, including copying color from another "
    "scene object or a physical source. Does not own movement, creation, deletion, shape, or size."
)
DESCRIPTION = (
    "Use for every requested end state that changes only the color of an existing XR object, "
    "whatever verb expresses it. "
    f"{_RESPONSIBILITY} Pass the target and the user's complete color-source words. This agent "
    "reads a physical color source itself, so route the recolor directly here without a separate "
    "camera lookup."
)
_EXAMPLES = (
    "'Give the existing capsule the color of my backpack' belongs here as one complete recolor; "
    "do not inspect the backpack first.",
    "'Copy the cube's color to the ring' copies a scene object's color here.",
)


_prompt_text = _PROMPT.read_text(encoding="utf-8").strip()


def make_appearance_agent(
    llm: LLMService,
    scene: SceneTools,
    context: SceneContext,
    physical_color: Tool | None = None,
) -> Tool:
    delegation_lock = asyncio.Lock()

    async def handle(request: SubagentTask) -> SubagentResult:
        logger.debug("appearance agent instruction={!r} trace={}", request.instruction[:200], current_trace_id.get())
        async with delegation_lock:
            guard = TurnGuard()
            tools = make_appearance_tools(scene, guard=guard, physical_color=physical_color)
            tools.append(
                Tool(
                    "get_scene_state",
                    "Return every current XR object with its ID, type, world position, color, and size.",
                    EmptyRequest,
                    SceneState,
                    lambda _: scene.get_scene_state.execute(EmptyRequest()),
                )
            )
            toolset = tolerant_toolset(tools)
            messages = [
                ChatMessage(role="system", content=_prompt_text),
                ChatMessage(
                    role="user",
                    content=(
                        f"Active participant: {current_participant_id.get()}\n"
                        f"Utterance timestamp: {current_reference_time_us.get()}\n"
                        f"{await context.describe(current_participant_id.get())}\n\n"
                        f"Focused instruction: {request.instruction}"
                    ),
                ),
            ]

            async def _call_model(transcript, definitions):
                return await llm.chat(
                    reasoning_messages(transcript, enabled=True),
                    tools=list(definitions) or None,
                    max_tokens=2048,
                    temperature=0.0,
                    enable_thinking=True,
                    thinking_budget=1024,
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
                responsibility=_RESPONSIBILITY,
                result=loop_result,
            )

    return Tool(
        name="appearance_agent",
        description=DESCRIPTION,
        request_model=SubagentTask,
        result_model=SubagentResult,
        handler=handle,
        examples=_EXAMPLES,
    )


__all__ = ["make_appearance_agent"]
