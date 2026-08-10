# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define the placement subagent and its immediate NAT dependencies."""

import asyncio
from pathlib import Path

from loguru import logger
from nat.plugin_api import (
    Builder,
    FunctionBaseConfig,
    FunctionGroupRef,
    FunctionInfo,
    LLMRef,
    register_function,
)
from nat.plugins.langchain.agent.tool_calling_agent.register import ToolCallAgentWorkflowConfig
from pydantic import ConfigDict, Field

from ...models import SubagentResult, SubagentTask
from ...scene import SceneContext
from ...spatial_ops import PlacementOpsConfig, TurnGuard

_PROMPT = Path(__file__).with_name("prompt.txt")
DESCRIPTION = (
            "Move, swap, contain, stack, or restore existing XR objects; never creates, recolors, or removes them."
)


class PlacementAgentConfig(FunctionBaseConfig, name="xr_render_placement_agent"):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm_name: LLMRef = LLMRef("scene_llm")
    scene_state: FunctionGroupRef = FunctionGroupRef("scene_state")
    scene_updates: FunctionGroupRef = FunctionGroupRef("scene_updates")
    tracking: FunctionGroupRef = FunctionGroupRef("tracking")
    spatial: FunctionGroupRef = FunctionGroupRef("spatial")
    context: SceneContext = Field(exclude=True, repr=False)


@register_function(config_type=PlacementAgentConfig)
async def placement_agent(config: PlacementAgentConfig, builder: Builder):
    guard = TurnGuard()
    delegation_lock = asyncio.Lock()
    ops = FunctionGroupRef("placement_ops")
    await builder.add_function_group(
        ops,
        PlacementOpsConfig(
            guard=guard,
            scene_state=config.scene_state,
            scene_updates=config.scene_updates,
            tracking=config.tracking,
            spatial=config.spatial,
        ),
    )
    reasoning = await builder.add_function(
        "placement_reasoning",
        ToolCallAgentWorkflowConfig(
            llm_name=config.llm_name,
            tool_names=[ops],
            system_prompt=_PROMPT.read_text(encoding="utf-8").strip(),
            handle_tool_errors=True,
            max_iterations=12,
            max_empty_response_retries=1,
        ),
    )

    async def place(request: SubagentTask) -> SubagentResult:
        logger.debug("placement agent instruction={!r}", request.instruction[:200])
        config.context.mark_mutating(request.participant_id)
        # The guard is agent-scoped; serialize delegations.
        async with delegation_lock:
            guard.reset()
            return await _run(request)

    async def _run(request: SubagentTask) -> SubagentResult:
        message = (
            f"Active participant: {request.participant_id}\n"
            f"Utterance timestamp: {request.reference_time_us}\n"
            f"{await config.context.describe(request.participant_id, bearings=True)}\n\n"
            f"Focused instruction: {request.instruction}"
        )
        output = await reasoning.ainvoke(message, to_type=str)
        return SubagentResult(result=str(output or "Done."))

    yield FunctionInfo.from_fn(
        place,
        description=DESCRIPTION,
    )


__all__ = ["PlacementAgentConfig"]
