# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define the memory subagent and its immediate NAT dependencies."""

from pathlib import Path

from loguru import logger
from nat.plugin_api import Builder, FunctionBaseConfig, FunctionGroupRef, FunctionInfo, LLMRef, register_function
from nat.plugins.langchain.agent.tool_calling_agent.register import ToolCallAgentWorkflowConfig

from ...models import SubagentResult, SubagentTask

_PROMPT = Path(__file__).with_name("prompt.txt")
DESCRIPTION = "Recall earlier conversation turns; never mutates the XR scene."


class MemoryAgentConfig(FunctionBaseConfig, name="xr_render_memory_agent"):
    llm_name: LLMRef = LLMRef("scene_llm")
    conversations: FunctionGroupRef = FunctionGroupRef("conversations")


@register_function(config_type=MemoryAgentConfig)
async def memory_agent(config: MemoryAgentConfig, builder: Builder):
    reasoning = await builder.add_function(
        "memory_reasoning",
        ToolCallAgentWorkflowConfig(
            llm_name=config.llm_name,
            tool_names=[config.conversations],
            system_prompt=_PROMPT.read_text(encoding="utf-8").strip(),
            handle_tool_errors=True,
            max_iterations=12,
            max_empty_response_retries=1,
        ),
    )

    async def recall(request: SubagentTask) -> SubagentResult:
        logger.debug("memory agent instruction={!r}", request.instruction[:200])
        message = (
            f"Active participant: {request.participant_id}\n"
            f"Utterance timestamp: {request.reference_time_us}\n\n"
            f"Focused instruction: {request.instruction}"
        )
        output = await reasoning.ainvoke(message, to_type=str)
        return SubagentResult(result=str(output or "Done."))

    yield FunctionInfo.from_fn(
        recall,
        description=DESCRIPTION,
    )


__all__ = ["MemoryAgentConfig"]
