# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the xr-ai-models NAT LLM provider."""

from __future__ import annotations

import json

import pytest
from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugin_api import FunctionGroupRef, LLMRef
from nat.plugins.langchain.agent.tool_calling_agent.register import ToolCallAgentWorkflowConfig
from pydantic import ValidationError
from xr_ai_models import Capabilities, ChatMessage, ChatResponse, ToolCall, ToolDef
from xr_ai_nat.functions.spatial_math import SpatialMathFunctionsConfig
from xr_ai_nat.llm import ModelsLLMConfig


class _ToolCallingLLM:
    capabilities = Capabilities(tool_calls=True)

    def __init__(self) -> None:
        self.calls: list[tuple[list[ChatMessage], list[ToolDef] | None]] = []
        self.closed = False

    async def chat(self, messages, *, tools=None, **_kwargs) -> ChatResponse:
        self.calls.append((list(messages), list(tools) if tools else None))
        if len(self.calls) == 1:
            return ChatResponse(
                content="",
                reasoning=None,
                tool_calls=[
                    ToolCall(
                        id="midpoint-call",
                        name="spatial_math__compute_midpoint",
                        arguments=json.dumps(
                            {
                                "first_position": {"x": 0.0, "y": 1.0, "z": 2.0},
                                "second_position": {"x": 2.0, "y": 3.0, "z": 4.0},
                            }
                        ),
                    )
                ],
                finish_reason="tool_calls",
                raw={"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            )
        return ChatResponse(
            content="The midpoint is (1, 2, 3).",
            reasoning=None,
            tool_calls=None,
            finish_reason="stop",
            raw={"usage": {"prompt_tokens": 20, "completion_tokens": 7}},
        )

    async def close(self) -> None:
        self.closed = True


def test_models_llm_config_requires_one_service_source() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        ModelsLLMConfig()
    with pytest.raises(ValidationError, match="exactly one"):
        ModelsLLMConfig(profile_path="models.yaml", service=object())


async def test_models_llm_runs_a_nat_tool_calling_agent() -> None:
    service = _ToolCallingLLM()
    llm_ref = LLMRef("test_models_llm")

    async with WorkflowBuilder() as builder:
        await builder.add_function_group("spatial_math", SpatialMathFunctionsConfig())
        await builder.add_llm(
            llm_ref,
            ModelsLLMConfig(
                service=service,
                model_name="test-model",
                max_tokens=80,
                temperature=0.25,
            ),
        )
        agent = await builder.add_function(
            "midpoint_agent",
            ToolCallAgentWorkflowConfig(
                llm_name=llm_ref,
                tool_names=[FunctionGroupRef("spatial_math")],
                system_prompt="Use the midpoint tool, then answer briefly.",
                max_iterations=2,
            ),
        )

        result = await agent.ainvoke("Find the midpoint.", to_type=str)

    assert result == "The midpoint is (1, 2, 3)."
    assert len(service.calls) == 2
    first_messages, first_tools = service.calls[0]
    assert first_messages[0].role == "system"
    assert first_messages[-1].role == "user"
    assert first_tools is not None
    assert "spatial_math__compute_midpoint" in [tool.name for tool in first_tools]
    second_messages, _ = service.calls[1]
    assert any(message.role == "assistant" and message.tool_calls for message in second_messages)
    assert any(message.role == "tool" and "1.0" in str(message.content) for message in second_messages)
    assert service.closed is False
