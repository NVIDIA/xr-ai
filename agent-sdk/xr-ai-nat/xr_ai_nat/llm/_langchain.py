# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate LangChain messages while leaving model I/O in ``xr-ai-models``."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any, cast

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field
from xr_ai_models import ChatMessage, LLMService, ToolCall, ToolDef


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part if isinstance(part, str) else str(part.get("text", ""))
            for part in content
            if isinstance(part, (str, dict))
        )
    return str(content or "")


def _message(message: BaseMessage) -> ChatMessage:
    if isinstance(message, SystemMessage):
        return ChatMessage(role="system", content=_text(message.content))
    if isinstance(message, HumanMessage):
        return ChatMessage(role="user", content=_text(message.content))
    if isinstance(message, ToolMessage):
        return ChatMessage(role="tool", content=_text(message.content), tool_call_id=message.tool_call_id)
    if isinstance(message, AIMessage):
        calls = [
            ToolCall(
                id=str(call.get("id", "")),
                name=str(call.get("name", "")),
                arguments=json.dumps(call.get("args", {})),
            )
            for call in message.tool_calls
        ]
        return ChatMessage(role="assistant", content=_text(message.content), tool_calls=calls or None)
    raise TypeError(f"unsupported message type: {type(message).__name__}")


def _tool(tool: BaseTool | dict[str, Any]) -> ToolDef:
    if isinstance(tool, dict):
        function = tool.get("function", tool)
        return ToolDef(
            name=str(function["name"]),
            description=str(function.get("description", "")),
            parameters=dict(function.get("parameters", {"type": "object"})),
        )
    schema = tool.args_schema
    if isinstance(schema, dict):
        parameters = schema
    elif schema is None:
        parameters = {"type": "object", "properties": {}}
    else:
        parameters = cast(type[BaseModel], schema).model_json_schema()
    return ToolDef(name=tool.name, description=tool.description, parameters=parameters)


class LangChainChatModel(BaseChatModel):
    """Present an ``LLMService`` through LangChain's async chat-model contract."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    service: Any = Field(exclude=True)
    model_name: str = "xr-ai-model"
    temperature: float = 0.0
    max_tokens: int = 1024
    enable_thinking: bool = False
    thinking_budget: int | None = None

    @property
    def _llm_type(self) -> str:
        return "xr-ai-models"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        del tool_choice
        return self.bind(tools=list(tools), **kwargs)

    def _generate(self, *_args: Any, **_kwargs: Any) -> ChatResult:
        raise NotImplementedError("LangChainChatModel supports async generation only")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager
        raw_tools = kwargs.pop("tools", None)
        response = await cast(LLMService, self.service).chat(
            [_message(message) for message in messages],
            tools=[_tool(tool) for tool in raw_tools] if raw_tools else None,
            max_tokens=int(kwargs.pop("max_tokens", self.max_tokens)),
            temperature=float(kwargs.pop("temperature", self.temperature)),
            enable_thinking=bool(kwargs.pop("enable_thinking", self.enable_thinking)),
            thinking_budget=kwargs.pop("thinking_budget", self.thinking_budget),
        )
        tool_calls = []
        for call in response.tool_calls or ():
            try:
                arguments = json.loads(call.arguments)
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append({"name": call.name, "args": arguments, "id": call.id, "type": "tool_call"})
        usage = response.raw.get("usage", {}) if isinstance(response.raw, dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        message = AIMessage(
            content=response.content,
            tool_calls=tool_calls,
            usage_metadata={
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            response_metadata={"finish_reason": response.finish_reason, "model_name": self.model_name},
        )
        return ChatResult(generations=[ChatGeneration(message=message)], llm_output=response.raw)


__all__ = ["LangChainChatModel"]
