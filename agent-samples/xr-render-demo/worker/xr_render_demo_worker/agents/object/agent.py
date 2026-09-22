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
from .._adaptive import (
    adaptive_result,
    reasoning_messages,
    refusal_needs_retry,
    refusal_toolset,
)

_PROMPT = Path(__file__).with_name("prompt.txt")
DESCRIPTION = (
    "Use whenever the requested target is new or absent from SCENE OBJECTS, even with verbs such "
    "as put or place. Also owns object existence, shape, and size: create any new XR object at "
    "its requested initial position; remove or delete one; duplicate or copy one; reshape one; "
    "or resize one. It owns complete arrangements made only of new objects, including rows and "
    "stacks; do not split their creation from their initial arrangement. A creation remains new "
    "even if an identical object exists, and its initial position stays in the same instruction. "
    "This agent reads a physical color source for a new "
    "object itself. Preserve the user's shape, color, count, source, and spatial words; include "
    "resolved ids for existing targets. Never use for moving or recoloring an existing object."
)
_EXAMPLES = (
    "'Place a new ring inside the capsule' creates and positions the ring here.",
    "'Make the existing ring smaller' resizes it here.",
    "'Create a pyramid using the color of my shirt' belongs here as one complete creation; do "
    "not inspect the shirt first.",
    "'Build four new rings in a vertical stack' belongs here as one complete creation arrangement; "
    "do not create them first and delegate their initial layout separately. Conventional rows "
    "and stacks are routine tool operations and stay fast.",
    "'Change the existing capsule into a gold cone' changes shape here, while the color facet "
    "remains a separate color operation owned elsewhere.",
    "Preserve interacting spatial constraints as one complete creation instruction.",
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
            tools = make_object_tools(scene, tracking, ledger=ledger, guard=guard, physical_color=physical_color)
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
            toolset = refusal_toolset(
                toolset,
                examples=(
                    "For movement of an existing object, decline with operation placement and "
                    "suggest the placement owner.",
                    "For a color-only change to an existing object, decline with operation recolor "
                    "and suggest the appearance owner.",
                ),
            )
            messages = [
                ChatMessage(role="system", content=_prompt_text),
                ChatMessage(
                    role="user",
                    content=(
                        f"Active participant: {current_participant_id.get()}\n"
                        f"Utterance timestamp: {current_reference_time_us.get()}\n"
                        f"{await context.describe(current_participant_id.get(), bearings=True)}\n\n"
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
                if await refusal_needs_retry(
                    llm,
                    instruction=request.instruction,
                    responsibility=DESCRIPTION,
                    result=loop_result,
                ):
                    async def _retry_model(transcript, definitions):
                        return await llm.chat(
                            reasoning_messages(transcript, enabled=True),
                            tools=list(definitions) or None,
                            max_tokens=2048,
                            temperature=0.0,
                            enable_thinking=True,
                            thinking_budget=1024,
                        )

                    loop_result = await run_tool_loop(
                        messages,
                        toolset,
                        _retry_model,
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
        examples=_EXAMPLES,
    )


__all__ = ["make_object_agent"]
