# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Object subagent: create, remove, resize, and reshape XR objects."""

import asyncio
from pathlib import Path

from loguru import logger
from xr_ai_models import ChatMessage, LLMService
from xr_ai_tools import Tool
from xr_ai_tools.tool_calling import ToolLoopError, run_tool_loop
from xr_ai_tools.tracking import TrackingTools
from xr_render_scene import EmptyRequest, SceneState, SceneTools

from ..._tolerant import tolerant_toolset
from ..._trace import current_participant_id, current_reference_time_us, current_trace_id
from ...models import SubagentResult, SubagentTask
from ...scene import SceneContext
from ...spatial_ops import CreationLedger, TurnGuard, make_object_tools

_PROMPT = Path(__file__).with_name("prompt.txt")
# The creation-always-new rule deliberately restates a supervisor_prompt.txt rule:
# tool descriptions alone are under-weighted mid-loop, so both copies are load-bearing.
DESCRIPTION = (
    "Create new XR objects at their requested initial positions, and remove, resize, duplicate, "
    "or reshape existing ones; never moves or recolors an existing object. Moving includes "
    "putting one in, on, or next to another, and any color change of an existing object belongs "
    "to appearance_agent. A creation request always makes a new object, even when SCENE OBJECTS "
    "already lists an identical one. Creation carries its position in the same single call, "
    "whether user-relative, anchored on scene objects, or between two of them; this agent "
    "resolves tracking and anchor geometry itself. Phrase creation as: Create <count, when more "
    "than one> <color> <shape>(s) <the user's own spatial words>, with the user's color words "
    "verbatim (a color word, \"same as capsule-8\", or a physical phrase like \"the color of my "
    "apron\") and every requested copy in the one instruction. When the user named no position "
    "at all, end with \"no position stated\"; never add that phrase when spatial words are "
    "present, and never invent a position. Removal words with a side (\"the X on the left\") "
    "stay a removal; a resize instruction carries the resolved id (\"Resize capsule-8 to half "
    "its current size\")."
)


_prompt_text = _PROMPT.read_text(encoding="utf-8").strip()

def make_object_agent(
    llm: LLMService,
    scene: SceneTools,
    tracking: TrackingTools,
    context: SceneContext,
    physical_color: Tool | None = None,
) -> Tool:
    delegation_lock = asyncio.Lock()

    async def handle(request: SubagentTask) -> SubagentResult:
        logger.debug("object agent instruction={!r} trace={}", request.instruction[:200], current_trace_id.get())
        async with delegation_lock:
            guard = TurnGuard()
            ledger = CreationLedger()
            tools = make_object_tools(scene, tracking, ledger=ledger, guard=guard,
                                      physical_color=physical_color)
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
                    f"Active participant: {current_participant_id.get()}\n"
                    f"Utterance timestamp: {current_reference_time_us.get()}\n"
                    f"{await context.describe(current_participant_id.get(), bearings=True)}\n\n"
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

    return Tool(name="object_agent", description=DESCRIPTION,
                request_model=SubagentTask, result_model=SubagentResult, handler=handle)


__all__ = ["make_object_agent"]
