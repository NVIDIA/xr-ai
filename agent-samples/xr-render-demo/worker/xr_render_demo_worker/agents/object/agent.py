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
from .._adaptive import adaptive_result, reasoning_messages, refusal_toolset

_PROMPT = Path(__file__).with_name("prompt.txt")
DESCRIPTION = (
    "Use whenever the requested target is new or absent from SCENE OBJECTS, even with verbs such "
    "as put or place. Also owns object existence, shape, and size: create any new XR object at "
    "its requested initial position; remove or delete one; duplicate or copy one; reshape one; "
    "or resize one. Examples: "
    "'put a new ring inside the capsule', 'erase the left cone', and 'make ring-alpha smaller'. A "
    "creation remains new even if an identical object exists, and its initial position stays in "
    "the same instruction. This agent reads a physical color source for a new object itself, so "
    "route that creation directly here without vision_agent. Preserve the user's shape, color, "
    "count, source, and spatial words; include resolved ids for existing targets. Never use for "
    "moving or recoloring an existing object."
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
            toolset = refusal_toolset(toolset)
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
                responsibility=DESCRIPTION,
                result=loop_result,
            )

    return Tool(
        name="object_agent",
        description=DESCRIPTION,
        request_model=SubagentTask,
        result_model=SubagentResult,
        handler=handle,
        examples=(
            "For 'Move X, make Y orange, and create Z', receive the focused instruction "
            "'Create Z' with reasoning_mode='fast'.",
            "If a cone already exists, 'Make a cone beside the capsule' still means create a "
            "new cone beside the existing capsule with reasoning_mode='fast'.",
            "Direct deletion, duplication, reshaping, and resizing use reasoning_mode='fast'.",
            "For a novel creation that must reconcile multiple interacting relative constraints "
            "before its first creation call, use reasoning_mode='deliberate'.",
        ),
    )


__all__ = ["make_object_agent"]
