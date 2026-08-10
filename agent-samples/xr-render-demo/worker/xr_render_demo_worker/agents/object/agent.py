# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define the object subagent and its immediate NAT dependencies."""

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
from ...spatial_ops import CreationLedger, ObjectOpsConfig, TurnGuard

_PROMPT = Path(__file__).with_name("prompt.txt")
DESCRIPTION = (
            "Create new XR objects at their requested initial positions, and remove, resize, duplicate, or "
            "reshape existing ones; never moves an existing object, including putting one in, on, or next to another."
)


class ObjectAgentConfig(FunctionBaseConfig, name="xr_render_object_agent"):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm_name: LLMRef = LLMRef("scene_llm")
    scene_state: FunctionGroupRef = FunctionGroupRef("scene_state")
    scene_updates: FunctionGroupRef = FunctionGroupRef("scene_updates")
    scene_objects: FunctionGroupRef = FunctionGroupRef("scene_objects")
    tracking: FunctionGroupRef = FunctionGroupRef("tracking")
    spatial: FunctionGroupRef = FunctionGroupRef("spatial")
    context: SceneContext = Field(exclude=True, repr=False)
    # A supervisor-owned ledger spans the whole user turn, including the
    # verification pass; without one, dedup only covers a single delegation.
    ledger: CreationLedger | None = Field(default=None, exclude=True, repr=False)


@register_function(config_type=ObjectAgentConfig)
async def object_agent(config: ObjectAgentConfig, builder: Builder):
    ledger = config.ledger if config.ledger is not None else CreationLedger()
    owns_ledger = config.ledger is None
    guard = TurnGuard()
    delegation_lock = asyncio.Lock()
    ops = FunctionGroupRef("object_ops")
    await builder.add_function_group(
        ops,
        ObjectOpsConfig(
            ledger=ledger,
            guard=guard,
            scene_state=config.scene_state,
            scene_updates=config.scene_updates,
            scene_objects=config.scene_objects,
            tracking=config.tracking,
            spatial=config.spatial,
        ),
    )
    reasoning = await builder.add_function(
        "object_reasoning",
        ToolCallAgentWorkflowConfig(
            llm_name=config.llm_name,
            tool_names=[config.scene_state, ops],
            system_prompt=_PROMPT.read_text(encoding="utf-8").strip(),
            handle_tool_errors=True,
            max_iterations=12,
            max_empty_response_retries=1,
        ),
    )

    async def change_object(request: SubagentTask) -> SubagentResult:
        logger.debug("object agent instruction={!r}", request.instruction[:200])
        config.context.mark_mutating(request.participant_id)
        # One delegation at a time: the guard and ledger are agent-scoped,
        # and interleaved delegations would clobber each other's state.
        async with delegation_lock:
            if owns_ledger:
                ledger.reset()
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
        change_object,
        description=DESCRIPTION,
    )


__all__ = ["ObjectAgentConfig"]
