# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define the vision subagent and its immediate NAT dependencies."""

from pathlib import Path

from loguru import logger
from nat.plugin_api import Builder, FunctionBaseConfig, FunctionGroupRef, FunctionInfo, LLMRef, register_function
from nat.plugins.langchain.agent.tool_calling_agent.register import ToolCallAgentWorkflowConfig
from pydantic import ConfigDict, Field

from ...models import SubagentResult, SubagentTask
from ...scene import SceneContext

_PROMPT = Path(__file__).with_name("prompt.txt")
DESCRIPTION = (
            "Answer a question about the physical world from the live or recorded camera; never for XR scene "
            "state or placement."
)


class VisionAgentConfig(FunctionBaseConfig, name="xr_render_vision_agent"):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm_name: LLMRef = LLMRef("scene_llm")
    vision: FunctionGroupRef = FunctionGroupRef("vision")
    context: SceneContext | None = Field(default=None, exclude=True, repr=False)


@register_function(config_type=VisionAgentConfig)
async def vision_agent(config: VisionAgentConfig, builder: Builder):
    reasoning = await builder.add_function(
        "vision_reasoning",
        ToolCallAgentWorkflowConfig(
            llm_name=config.llm_name,
            tool_names=[config.vision],
            system_prompt=_PROMPT.read_text(encoding="utf-8").strip(),
            # A missing live frame must return to the supervisor instead of
            # letting this agent improvise a recorded-frame fallback.
            handle_tool_errors=False,
            max_iterations=12,
            max_empty_response_retries=1,
        ),
    )

    async def observe(request: SubagentTask) -> SubagentResult:
        logger.debug("vision agent instruction={!r}", request.instruction[:200])
        if config.context is not None:
            config.context.mark_delegated(request.participant_id)
        scene_block = ""
        if config.context is not None:
            scene_block = f"{await config.context.describe(request.participant_id)}\n\n"
        message = (
            f"Active participant: {request.participant_id}\n"
            f"Utterance timestamp: {request.reference_time_us}\n"
            f"{scene_block}"
            f"Focused instruction: {request.instruction}"
        )
        # A dead camera is an answer, not a workflow failure: retry once
        # against scene data alone, then degrade explicitly.
        try:
            output = await reasoning.ainvoke(message, to_type=str)
        except Exception as error:
            retry = (
                f"{message}\n\n(The camera is unavailable: {error}. If SCENE OBJECTS"
                " answers the instruction, answer from it with no tool call;"
                " otherwise state that no visual fact is available.)"
            )
            try:
                output = await reasoning.ainvoke(retry, to_type=str)
            except Exception:
                return SubagentResult(result=f"No visual fact available ({error}); proceed without vision.")
        return SubagentResult(result=str(output or "Done."))

    yield FunctionInfo.from_fn(
        observe,
        description=DESCRIPTION,
    )


__all__ = ["VisionAgentConfig"]
