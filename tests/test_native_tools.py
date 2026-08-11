# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for Relay-managed native tools and tool-driven agents."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import nemo_relay
import pytest
from nemo_relay.codecs import OpenAIChatCodec
from pydantic import BaseModel
from xr_ai_models import Capabilities, ChatMessage, ChatResponse, ToolCall, ToolDef
from xr_ai_nat import AgentRunner, Tool, ToolSet, as_agent_tool
from xr_ai_nat.agents import Agent, ToolLoopLimitError, _response_from_openai


class AddRequest(BaseModel):
    """Two integers to add."""

    left: int
    right: int


class AddResult(BaseModel):
    """The computed total."""

    total: int


class AskRequest(BaseModel):
    """One text request delegated to an agent tool."""

    text: str


class AskResult(BaseModel):
    """The agent tool's text result."""

    text: str


async def add(request: AddRequest) -> AddResult:
    return AddResult(total=request.left + request.right)


class _ToolCallingLLM:
    capabilities = Capabilities(tool_calls=True)

    def __init__(self) -> None:
        self.calls: list[tuple[list[ChatMessage], list[ToolDef] | None]] = []

    async def chat(self, messages, *, tools=None, **_kwargs) -> ChatResponse:
        self.calls.append((list(messages), list(tools) if tools else None))
        if len(self.calls) == 1:
            return ChatResponse(
                content="",
                reasoning=None,
                tool_calls=[
                    ToolCall(
                        id="add-call",
                        name="add",
                        arguments=json.dumps({"left": 2, "right": 3}),
                    )
                ],
                finish_reason="tool_calls",
                raw={"usage": {"prompt_tokens": 10, "completion_tokens": 3}},
            )
        return ChatResponse(
            content="The answer is five.",
            reasoning=None,
            tool_calls=None,
            finish_reason="stop",
            raw={"usage": {"prompt_tokens": 20, "completion_tokens": 5}},
        )

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None

    async def stream(self, *_args, **_kwargs) -> AsyncIterator[str]:
        if False:
            yield ""


async def test_agent_executes_a_native_tool_and_returns_the_final_model_answer() -> None:
    llm = _ToolCallingLLM()
    agent = Agent(
        name="calculator",
        llm=llm,
        system_prompt="Use the add tool.",
        tools=(Tool("add", "Add two integers.", AddRequest, AddResult, add),),
        max_iterations=2,
    )

    result = await agent.run("What is two plus three?")

    assert result.text == "The answer is five."
    assert len(llm.calls) == 2
    first_messages, first_tools = llm.calls[0]
    assert [message.role for message in first_messages] == ["system", "user"]
    assert first_tools is not None
    assert first_tools[0].name == "add"
    second_messages, _ = llm.calls[1]
    assert [(message.role, message.content) for message in second_messages[-2:]] == [
        ("assistant", ""),
        ("tool", '{"total":5}'),
    ]


async def test_tools_are_relay_managed_for_agent_and_direct_invocation() -> None:
    tool = Tool("add", "Add two integers.", AddRequest, AddResult, add)
    llm = _ToolCallingLLM()
    agent = Agent(
        name="calculator-lifecycle",
        llm=llm,
        system_prompt="Use the add tool.",
        tools=(tool,),
        max_iterations=2,
    )
    events = []
    subscriber = "xr-ai-native-tools-lifecycle"
    nemo_relay.subscribers.register(subscriber, events.append)
    try:
        assert await tool.execute(AddRequest(left=1, right=2)) == AddResult(total=3)
        await agent.run("What is two plus three?")
        await nemo_relay.subscribers.flush_async()
    finally:
        nemo_relay.subscribers.deregister(subscriber)

    categories = {getattr(event, "category", None) for event in events}
    assert {"llm", "tool"} <= categories


async def test_agent_can_be_exposed_as_a_normal_native_tool() -> None:
    class _FinalAnswerLLM(_ToolCallingLLM):
        async def chat(self, messages, *, tools=None, **_kwargs) -> ChatResponse:
            self.calls.append((list(messages), list(tools) if tools else None))
            return ChatResponse(
                content="Handled by the agent tool.",
                reasoning=None,
                tool_calls=None,
                finish_reason="stop",
                raw={},
            )

    agent = Agent(
        name="delegate",
        llm=_FinalAnswerLLM(),
        system_prompt="Answer directly.",
        tools=(),
    )
    tool = as_agent_tool(
        name="delegate",
        description="Delegate one request to the agent.",
        agent=agent,
        request_model=AskRequest,
        result_model=AskResult,
        request=lambda value: value.text,
        response=lambda result: AskResult(text=result.text),
    )

    assert await tool.execute(AskRequest(text="Hello")) == AskResult(
        text="Handled by the agent tool.",
    )


async def test_custom_agent_runner_can_be_exposed_as_a_normal_native_tool() -> None:
    class _CustomRunner:
        async def run(self, request: str) -> AskResult:
            return AskResult(text=f"Custom runner: {request}")

    runner: AgentRunner[str, AskResult] = _CustomRunner()
    tool = as_agent_tool(
        name="custom_delegate",
        description="Delegate one request to a custom agent runner.",
        agent=runner,
        request_model=AskRequest,
        result_model=AskResult,
        request=lambda value: value.text,
        response=lambda result: result,
    )

    assert await tool.execute(AskRequest(text="Hello")) == AskResult(
        text="Custom runner: Hello",
    )


async def test_agent_forwards_relay_rewritten_request_to_the_model(monkeypatch) -> None:
    class _RecordingLLM:
        capabilities = Capabilities()

        def __init__(self) -> None:
            self.messages: list[ChatMessage] = []
            self.headers: dict[str, str] = {}

        async def chat(self, messages, *, headers=None, **_kwargs) -> ChatResponse:
            self.messages = list(messages)
            self.headers = dict(headers or {})
            return ChatResponse("Rewritten.", None, None, "stop", {})

        async def health(self) -> bool:
            return True

        async def close(self) -> None:
            return None

        async def stream(self, *_args, **_kwargs) -> AsyncIterator[str]:
            if False:
                yield ""

    llm = _RecordingLLM()
    observed: dict[str, object] = {}

    async def execute(_name, request, invoke, **kwargs):
        observed["content"] = request.content
        observed["codec"] = kwargs["codec"]
        observed["response_codec"] = kwargs["response_codec"]
        rewritten = dict(request.content)
        rewritten["messages"] = [
            {"role": "system", "content": "Answer directly."},
            {"role": "user", "content": "rewritten request"},
        ]
        return await invoke(nemo_relay.LLMRequest({"X-Relay-Session": "turn-7"}, rewritten))

    monkeypatch.setattr(nemo_relay.llm, "execute", execute)
    result = await Agent(
        name="relay-boundary",
        llm=llm,
        system_prompt="Answer directly.",
        tools=(),
    ).run("original request")

    assert result.text == "Rewritten."
    initial_content = observed["content"]
    assert isinstance(initial_content, dict)
    assert "tools" not in initial_content
    assert isinstance(observed["codec"], OpenAIChatCodec)
    assert isinstance(observed["response_codec"], OpenAIChatCodec)
    assert llm.headers == {"X-Relay-Session": "turn-7"}
    assert llm.messages[-1].content == "rewritten request"


def test_agent_accepts_null_content_for_tool_call_only_response() -> None:
    response = _response_from_openai(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "lookup-1",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )

    assert response.content == ""
    assert response.tool_calls == [ToolCall(id="lookup-1", name="lookup", arguments="{}")]


async def test_invalid_tool_arguments_are_returned_to_the_model_for_repair() -> None:
    tool = Tool("add", "Add two integers.", AddRequest, AddResult, add)

    result = await tool.invoke('{"left":"not-an-int"}')

    assert result.return_direct is False
    payload = json.loads(result.content)
    assert payload["error"] == "invalid_tool_arguments"
    assert "right" in payload["detail"]


async def test_return_direct_tool_finishes_without_an_extra_model_call() -> None:
    llm = _ToolCallingLLM()
    agent = Agent(
        name="calculator",
        llm=llm,
        system_prompt="Use the add tool.",
        tools=(
            Tool(
                "add",
                "Add two integers.",
                AddRequest,
                AddResult,
                add,
                return_direct=True,
            ),
        ),
    )

    result = await agent.run("What is two plus three?")

    assert result.text == '{"total":5}'
    assert len(llm.calls) == 1


async def test_unknown_tool_is_returned_to_the_model_for_repair() -> None:
    class _UnknownToolLLM(_ToolCallingLLM):
        async def chat(self, messages, *, tools=None, **kwargs) -> ChatResponse:
            response = await super().chat(messages, tools=tools, **kwargs)
            if len(self.calls) == 1:
                return ChatResponse(
                    content="",
                    reasoning=None,
                    tool_calls=[ToolCall(id="missing", name="missing", arguments="{}")],
                    finish_reason="tool_calls",
                    raw={},
                )
            return response

    llm = _UnknownToolLLM()
    agent = Agent(
        name="calculator",
        llm=llm,
        system_prompt="Use the available tools.",
        tools=(Tool("add", "Add two integers.", AddRequest, AddResult, add),),
        max_iterations=2,
    )

    result = await agent.run("Calculate.")

    assert result.text == "The answer is five."
    content = llm.calls[1][0][-1].content
    assert isinstance(content, str)
    assert json.loads(content) == {"error": "unknown_tool", "tool": "missing"}


async def test_agent_enforces_its_model_iteration_budget() -> None:
    class _LoopingLLM(_ToolCallingLLM):
        async def chat(self, messages, *, tools=None, **kwargs) -> ChatResponse:
            self.calls.append((list(messages), list(tools) if tools else None))
            return ChatResponse(
                content="",
                reasoning=None,
                tool_calls=[ToolCall(id=str(len(self.calls)), name="add", arguments='{"left":1,"right":1}')],
                finish_reason="tool_calls",
                raw={},
            )

    agent = Agent(
        name="calculator",
        llm=_LoopingLLM(),
        system_prompt="Use the add tool.",
        tools=(Tool("add", "Add two integers.", AddRequest, AddResult, add),),
        max_iterations=2,
    )

    with pytest.raises(ToolLoopLimitError, match="exhausted 2"):
        await agent.run("Loop forever.")


def test_tool_sets_reject_duplicate_names() -> None:
    tool = Tool("add", "Add two integers.", AddRequest, AddResult, add)

    with pytest.raises(ValueError, match="duplicate tool name"):
        ToolSet((tool, tool))
