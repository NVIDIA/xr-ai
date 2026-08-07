# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small shared builders for NAT tool-calling agents."""

from __future__ import annotations

from nat.builder.function import Function
from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugin_api import FunctionRef, LLMRef
from nat.plugins.langchain.agent.tool_calling_agent.register import ToolCallAgentWorkflowConfig
from xr_ai_models import LLMService
from xr_ai_nat.llm import ModelsLLMConfig

from ..desktop.types import RoutedFunction


async def add_guidance_llm(builder: WorkflowBuilder, llm: LLMService) -> LLMRef:
    llm_ref = LLMRef("guide_llm")
    await builder.add_llm(
        llm_ref,
        ModelsLLMConfig(
            service=llm,
            model_name="guidance-llm",
            max_tokens=512,
            temperature=0,
            enable_thinking=False,
        ),
    )
    return llm_ref


async def build_agent(
    builder: WorkflowBuilder,
    *,
    name: str,
    llm_ref: LLMRef,
    prompt: str,
    tools: tuple[FunctionRef, ...],
    return_direct: tuple[FunctionRef, ...] = (),
) -> Function:
    if not tools:
        raise ValueError(f"NAT tool-calling agent {name!r} needs at least one tool")
    return await builder.add_function(
        name,
        ToolCallAgentWorkflowConfig(
            llm_name=llm_ref,
            tool_names=list(tools),
            return_direct=list(return_direct) or None,
            system_prompt=prompt,
            max_iterations=4,
            max_history=6,
            handle_tool_errors=False,
            verbose=True,
            log_response_max_chars=4_000,
            description=f"Guidance agent {name}",
        ),
    )


async def build_routed_agent(
    builder: WorkflowBuilder,
    *,
    name: str,
    llm_ref: LLMRef,
    prompt: str,
    functions: tuple[RoutedFunction, ...],
) -> Function:
    return await build_agent(
        builder,
        name=name,
        llm_ref=llm_ref,
        prompt=prompt,
        tools=tuple(function.ref for function in functions),
        return_direct=tuple(function.ref for function in functions if function.return_direct),
    )


__all__ = ["add_guidance_llm", "build_agent", "build_routed_agent"]
