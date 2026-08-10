# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define the appearance subagent and its immediate NAT dependencies."""

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
from ...spatial_ops import AppearanceOpsConfig, TurnGuard

_PROMPT = Path(__file__).with_name("prompt.txt")
DESCRIPTION = "Change only the color of existing XR objects."


class AppearanceAgentConfig(FunctionBaseConfig, name="xr_render_appearance_agent"):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm_name: LLMRef = LLMRef("scene_llm")
    scene_state: FunctionGroupRef = FunctionGroupRef("scene_state")
    scene_updates: FunctionGroupRef = FunctionGroupRef("scene_updates")
    context: SceneContext = Field(exclude=True, repr=False)


@register_function(config_type=AppearanceAgentConfig)
async def appearance_agent(config: AppearanceAgentConfig, builder: Builder):
    guard = TurnGuard()
    ops = FunctionGroupRef("appearance_ops")
    await builder.add_function_group(
        ops,
        AppearanceOpsConfig(guard=guard, scene_state=config.scene_state, scene_updates=config.scene_updates),
    )
    reasoning = await builder.add_function(
        "appearance_reasoning",
        ToolCallAgentWorkflowConfig(
            llm_name=config.llm_name,
            tool_names=[config.scene_state, ops],
            system_prompt=_PROMPT.read_text(encoding="utf-8").strip(),
            handle_tool_errors=True,
            max_iterations=12,
            max_empty_response_retries=1,
        ),
    )

    async def change_appearance(request: SubagentTask) -> SubagentResult:
        logger.debug("appearance agent instruction={!r}", request.instruction[:200])
        config.context.mark_mutating(request.participant_id)
        guard.reset()
        message = (
            f"Active participant: {request.participant_id}\n"
            f"Utterance timestamp: {request.reference_time_us}\n"
            f"{await config.context.describe(request.participant_id)}\n\n"
            f"Focused instruction: {request.instruction}"
        )
        output = await reasoning.ainvoke(message, to_type=str)
        return SubagentResult(result=str(output or "Done."))

    yield FunctionInfo.from_fn(
        change_appearance,
        description=DESCRIPTION,
    )


__all__ = ["AppearanceAgentConfig"]
