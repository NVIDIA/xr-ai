# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Placement subagent: move, swap, and contain existing XR objects."""

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
from ...spatial_ops import TurnGuard, make_placement_tools
from .._adaptive import adaptive_result, reasoning_messages

_PROMPT = Path(__file__).with_name("prompt.txt")
_RESPONSIBILITY = (
    "Owns requested spatial changes to existing XR objects: move, nudge, swap, contain, stack, "
    "or restore. A move or restore remains this agent's responsibility when its named target "
    "cannot be found; report the failed lookup without creating anything. Does not own creation, "
    "deletion, duplication, recoloring, reshaping, or resizing. Changing color is not a spatial "
    "change, regardless of verbs such as turn, make, paint, or match."
)
DESCRIPTION = (
    "Use only when every target being repositioned already exists in SCENE OBJECTS; a verb such "
    "as put or place does not by itself make a placement task. "
    f"{_RESPONSIBILITY} Examples: 'move ring-alpha left', 'put the "
    "existing ring inside capsule-beta', and 'swap the cone and box'. A request to put or place a "
    "new target is a creation task for object_agent. An explicitly requested move or restore of a "
    "named target is still a placement attempt if lookup fails: report that it is missing and do "
    "not reinterpret it as creation. Never use for creation, deletion, duplication, recoloring, "
    "reshaping, or resizing."
)


_prompt_text = _PROMPT.read_text(encoding="utf-8").strip()

def make_placement_agent(
    llm: LLMService,
    scene: SceneTools,
    tracking: TrackingTools,
    context: SceneContext,
) -> Tool:
    delegation_lock = asyncio.Lock()

    async def handle(request: SubagentTask) -> SubagentResult:
        logger.debug("placement agent instruction={!r} trace={}", request.instruction[:200], current_trace_id.get())
        async with delegation_lock:
            guard = TurnGuard()
            tools = make_placement_tools(scene, tracking, guard=guard)
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
                return await llm.chat(
                    reasoning_messages(transcript, enabled=request.reasoning_mode == "deliberate"),
                    tools=list(definitions) or None,
                    max_tokens=2048,
                    temperature=0.0,
                    enable_thinking=request.reasoning_mode == "deliberate",
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
        name="placement_agent",
        description=DESCRIPTION,
        request_model=SubagentTask,
        result_model=SubagentResult,
        handler=handle,
        examples=(
            "For 'Move X, make Y orange, and create Z', receive the focused instruction "
            "'Move X' with reasoning_mode='fast'.",
            "For a novel arrangement that must reconcile multiple interacting relative "
            "constraints before its first movement call, use reasoning_mode='deliberate'.",
            "For example, arranging several existing shapes into a collision-free pattern while "
            "preserving a separate spatial order uses reasoning_mode='deliberate'.",
        ),
    )


__all__ = ["make_placement_agent"]
