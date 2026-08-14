# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Object subagent: create, remove, resize, and reshape XR objects."""

import asyncio
from pathlib import Path

from loguru import logger
from xr_ai_models import ChatMessage, LLMService
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.tool_calling import tool_definitions
from xr_ai_tools.tracking import TrackingTools
from xr_render_scene import EmptyRequest, SceneState, SceneTools

from ..._loop import tool_loop
from ...models import SubagentResult, SubagentTask
from ...scene import SceneContext
from ...spatial_ops import CreationLedger, TurnGuard, make_object_tools

_PROMPT = Path(__file__).with_name("prompt.txt")
DESCRIPTION = (
    "Create new XR objects at their requested initial positions, and remove, resize, duplicate, or "
    "reshape existing ones; never moves an existing object, including putting one in, on, or next to another."
)


_prompt_text = _PROMPT.read_text(encoding="utf-8").strip()

def make_object_agent(
    llm: LLMService,
    scene: SceneTools,
    tracking: TrackingTools,
    context: SceneContext,
) -> Tool:
    delegation_lock = asyncio.Lock()

    async def handle(request: SubagentTask) -> SubagentResult:
        logger.debug("object agent instruction={!r}", request.instruction[:200])
        context.mark_mutating(request.participant_id)
        async with delegation_lock:
            guard = TurnGuard()
            ledger = CreationLedger()
            tools = make_object_tools(scene, tracking, ledger=ledger, guard=guard)
            tools.append(Tool(
                "get_scene_state",
                "Return every current XR object with its ID, type, world position, color, and size.",
                EmptyRequest,
                SceneState,
                lambda _: scene.get_scene_state.execute(EmptyRequest()),
            ))
            toolset = ToolSet(tools)
            prompt = _prompt_text
            messages = [
                ChatMessage(role="system", content=prompt),
                ChatMessage(role="user", content=(
                    f"Active participant: {request.participant_id}\n"
                    f"Utterance timestamp: {request.reference_time_us}\n"
                    f"{await context.describe(request.participant_id, bearings=True)}\n\n"
                    f"Focused instruction: {request.instruction}"
                )),
            ]
            result = await tool_loop(llm, messages, tool_definitions(toolset), toolset)
            return SubagentResult(result=result or "Done.")

    return Tool(name="object_agent", description=DESCRIPTION,
                request_model=SubagentTask, result_model=SubagentResult, handler=handle)


__all__ = ["make_object_agent"]
