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
from ..._trace import current_trace_id
from ...models import SubagentResult, SubagentTask
from ...scene import SceneContext
from ...spatial_ops import TurnGuard, make_appearance_tools

_PROMPT = Path(__file__).with_name("prompt.txt")
DESCRIPTION = "Change only the color of existing XR objects."


_prompt_text = _PROMPT.read_text(encoding="utf-8").strip()

def make_appearance_agent(
    llm: LLMService,
    scene: SceneTools,
    context: SceneContext,
) -> Tool:
    delegation_lock = asyncio.Lock()

    async def handle(request: SubagentTask) -> SubagentResult:
        logger.debug("appearance agent instruction={!r} trace={}", request.instruction[:200], current_trace_id.get())
        async with delegation_lock:
            guard = TurnGuard()
            tools = make_appearance_tools(scene, guard=guard)
            tools.append(Tool(
                "get_scene_state",
                "Return every current XR object with its ID, type, world position, color, and size.",
                EmptyRequest,
                SceneState,
                lambda _: scene.get_scene_state.execute(EmptyRequest()),
            ))
            toolset = tolerant_toolset(tools)
            prompt = _prompt_text
            messages = [
                ChatMessage(role="system", content=prompt),
                ChatMessage(role="user", content=(
                    f"Active participant: {request.participant_id}\n"
                    f"Utterance timestamp: {request.reference_time_us}\n"
                    f"{await context.describe(request.participant_id)}\n\n"
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

    return Tool(name="appearance_agent", description=DESCRIPTION,
                request_model=SubagentTask, result_model=SubagentResult, handler=handle)


__all__ = ["make_appearance_agent"]
