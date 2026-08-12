# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool-driven agents whose model calls remain internal to the tools layer."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import nemo_relay
from nemo_relay.codecs import OpenAIChatCodec
from xr_ai_models import ChatMessage, ChatResponse, LLMService, ToolCall, ToolDef

from ._relay import headers_from_relay
from .agent_runner import AgentRunner, as_agent_tool
from .tools import Tool, ToolSet


class ToolLoopLimitError(RuntimeError):
    """Raised when a model has not produced a final answer within the configured budget."""


@dataclass(frozen=True, slots=True)
class AgentResult:
    """The final text and messages generated during one tool-driven agent turn."""

    text: str
    messages: tuple[ChatMessage, ...]


class Agent(AgentRunner[str, AgentResult]):
    """Run one small stateless tool-calling turn over a catalog of native tools."""

    def __init__(
        self,
        *,
        name: str,
        llm: LLMService,
        system_prompt: str,
        tools: Sequence[Tool[Any, Any]],
        model_name: str = "xr-ai-model",
        max_iterations: int = 4,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        thinking_budget: int | None = None,
    ) -> None:
        if not name:
            raise ValueError("agent name must not be empty")
        if not system_prompt:
            raise ValueError(f"agent {name!r} needs a system prompt")
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least one")
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = ToolSet(tools)
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget

    async def run(self, request: str) -> AgentResult:
        """Execute one user request without carrying hidden conversation history."""

        if not request.strip():
            raise ValueError("agent request must not be blank")
        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=request),
        ]
        with nemo_relay.scope.scope(
            self.name,
            nemo_relay.ScopeType.Agent,
            input={"request": request},
        ):
            for _ in range(self.max_iterations):
                response = await self._chat(messages)
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )
                if not response.tool_calls:
                    return AgentResult(response.content.strip(), tuple(messages))
                for call in response.tool_calls:
                    tool = self.tools.get(call.name)
                    if tool is None:
                        outcome = json.dumps({"error": "unknown_tool", "tool": call.name})
                        return_direct = False
                    else:
                        invocation = await tool.invoke(call.arguments)
                        outcome = invocation.content
                        return_direct = invocation.return_direct
                    messages.append(
                        ChatMessage(
                            role="tool",
                            content=outcome,
                            tool_call_id=call.id,
                        )
                    )
                    if return_direct:
                        return AgentResult(outcome, tuple(messages))
        raise ToolLoopLimitError(
            f"agent {self.name!r} exhausted {self.max_iterations} model iterations",
        )

    async def _chat(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        tool_definitions = list(self.tools.definitions)
        content: dict[str, Any] = {
            "model": self.model_name,
            "messages": [_message_to_openai(message) for message in messages],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "enable_thinking": self.enable_thinking,
            "thinking_budget": self.thinking_budget,
        }
        if tool_definitions:
            content["tools"] = tool_definitions
        relay_request = nemo_relay.LLMRequest(
            {},
            content,
        )

        async def invoke(request: nemo_relay.LLMRequest) -> dict[str, Any]:
            content = request.content
            response = await self.llm.chat(
                _messages_from_openai(content.get("messages")),
                tools=_tools_from_openai(content.get("tools")),
                max_tokens=_optional_int(content.get("max_tokens")),
                temperature=_optional_float(content.get("temperature")),
                enable_thinking=bool(content.get("enable_thinking", False)),
                thinking_budget=_optional_int(content.get("thinking_budget")),
                headers=headers_from_relay(request.headers),
            )
            return _response_to_openai(response)

        raw_response = await nemo_relay.llm.execute(
            self.name,
            relay_request,
            invoke,
            model_name=self.model_name,
            codec=OpenAIChatCodec(),
            response_codec=OpenAIChatCodec(),
        )
        return _response_from_openai(raw_response)


def _message_to_openai(message: ChatMessage) -> dict[str, Any]:
    if not isinstance(message.content, str):
        raise TypeError("the native agent accepts text-only LLM messages")
    content: str | None = message.content
    if message.role == "assistant" and not content and message.tool_calls:
        content = None
    result: dict[str, Any] = {"role": message.role, "content": content}
    if message.tool_calls:
        result["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in message.tool_calls
        ]
    if message.tool_call_id:
        result["tool_call_id"] = message.tool_call_id
    return result


def _messages_from_openai(raw: object) -> list[ChatMessage]:
    if not isinstance(raw, list):
        raise TypeError("Relay LLM request must contain a message array")
    messages: list[ChatMessage] = []
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError("Relay LLM messages must be objects")
        role = item.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported Relay LLM role: {role!r}")
        content = item.get("content", "")
        tool_calls = _tool_calls_from_openai(item.get("tool_calls")) or None
        if content is None and tool_calls:
            content = ""
        if not isinstance(content, str):
            raise TypeError("the native agent accepts text-only LLM messages")
        tool_call_id = item.get("tool_call_id")
        if tool_call_id is not None and not isinstance(tool_call_id, str):
            raise TypeError("tool_call_id must be a string")
        messages.append(
            ChatMessage(
                role=role,
                content=content,
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
            )
        )
    return messages


def _tool_calls_from_openai(raw: object) -> list[ToolCall]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise TypeError("tool_calls must be an array")
    calls: list[ToolCall] = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("function"), dict):
            raise TypeError("tool call must contain a function object")
        function = item["function"]
        name = function.get("name")
        arguments = function.get("arguments")
        identifier = item.get("id")
        if not isinstance(identifier, str) or not isinstance(name, str) or not isinstance(arguments, str):
            raise TypeError("tool call id, name, and arguments must be strings")
        calls.append(ToolCall(id=identifier, name=name, arguments=arguments))
    return calls


def _tools_from_openai(raw: object) -> list[ToolDef] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise TypeError("tools must be an array")
    definitions: list[ToolDef] = []
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError("tool definition must be an object")
        function = item.get("function", item)
        if not isinstance(function, dict):
            raise TypeError("tool definition function must be an object")
        name = function.get("name")
        description = function.get("description", "")
        parameters = function.get("parameters", {"type": "object"})
        if not isinstance(name, str) or not isinstance(description, str) or not isinstance(parameters, dict):
            raise TypeError("tool definition has invalid fields")
        definitions.append(ToolDef(name=name, description=description, parameters=parameters))
    return definitions


def _response_to_openai(response: ChatResponse) -> dict[str, Any]:
    message = _message_to_openai(
        ChatMessage(role="assistant", content=response.content, tool_calls=response.tool_calls),
    )
    return {
        "model": response.raw.get("model", "xr-ai-model") if isinstance(response.raw, dict) else "xr-ai-model",
        "choices": [{"message": message, "finish_reason": response.finish_reason}],
        "usage": response.raw.get("usage", {}) if isinstance(response.raw, dict) else {},
        "xr_ai_reasoning": response.reasoning,
    }


def _response_from_openai(raw: object) -> ChatResponse:
    if not isinstance(raw, dict):
        raise TypeError("Relay LLM response must be an object")
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("Relay LLM response must contain one choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise TypeError("Relay LLM response choice must contain a message")
    content = message.get("content", "")
    tool_calls = _tool_calls_from_openai(message.get("tool_calls")) or None
    if content is None and tool_calls:
        content = ""
    if not isinstance(content, str):
        raise TypeError("Relay LLM response content must be a string")
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise TypeError("Relay LLM finish_reason must be a string")
    reasoning = raw.get("xr_ai_reasoning")
    if reasoning is not None and not isinstance(reasoning, str):
        raise TypeError("Relay LLM reasoning must be a string")
    return ChatResponse(
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        raw=raw,
    )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("expected an integer or null")
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("expected a number or null")
    return float(value)


__all__ = ["Agent", "AgentResult", "AgentRunner", "ToolLoopLimitError", "as_agent_tool"]
