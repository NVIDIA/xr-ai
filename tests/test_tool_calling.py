# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded model-loop contracts for Relay-managed native tools."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import BaseModel
from xr_ai_models import ChatMessage, ChatResponse, ToolCall, ToolDef
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.tool_calling import (
    ToolLoopDirectReturnError,
    ToolLoopEmptyResponseError,
    ToolLoopIterationLimitError,
    ToolLoopMalformedResponseError,
    ToolLoopToolCallLimitError,
    ToolLoopTruncatedResponseError,
    run_tool_loop,
)


class ValueRequest(BaseModel):
    """One integer tool argument."""

    value: int


class ValueResult(BaseModel):
    """One integer tool result."""

    value: int


@pytest.fixture(autouse=True)
def _run_tool_handlers_without_relay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep loop tests focused; native-tool tests cover the Relay boundary."""

    async def execute(
        _name: str,
        arguments: Any,
        handler: Any,
        *_args: Any,
        **_kwargs: Any,
    ) -> Any:
        result = handler(arguments)
        return await result if inspect.isawaitable(result) else result

    monkeypatch.setattr("xr_ai_tools.tools.typed.tool_execute", execute)


def _response(
    content: str = "",
    *,
    tool_calls: list[ToolCall] | None = None,
    finish_reason: str = "stop",
) -> ChatResponse:
    return ChatResponse(
        content=content,
        reasoning=None,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        raw={},
    )


class _Model:
    def __init__(self, responses: Sequence[ChatResponse]) -> None:
        self._responses = iter(responses)
        self.requests: list[tuple[tuple[ChatMessage, ...], tuple[ToolDef, ...]]] = []

    async def __call__(
        self,
        messages: tuple[ChatMessage, ...],
        tools: tuple[ToolDef, ...],
    ) -> ChatResponse:
        self.requests.append((messages, tools))
        return next(self._responses)


def _call(call_id: str, name: str, arguments: str) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


async def test_run_tool_loop_returns_final_answer_and_complete_transcript() -> None:
    initial = (ChatMessage(role="user", content="hello"),)
    model = _Model((_response("  Hello.  "),))

    result = await run_tool_loop(initial, ToolSet(()), model)

    assert result.content == "Hello."
    assert result.messages == (
        *initial,
        ChatMessage(role="assistant", content="  Hello.  "),
    )
    assert result.tool_calls == ()
    assert result.iterations == 1
    assert result.return_direct is False
    assert model.requests == [(initial, ())]


async def test_run_tool_loop_executes_batches_sequentially_in_emitted_order() -> None:
    invoked: list[int] = []

    async def record(request: ValueRequest) -> ValueResult:
        invoked.append(request.value)
        return ValueResult(value=request.value)

    tool = Tool("record", "Record a value.", ValueRequest, ValueResult, record)
    calls = [
        _call("first", "record", '{"value":1}'),
        _call("second", "record", '{"value":2}'),
    ]
    model = _Model(
        (
            _response(tool_calls=calls, finish_reason="tool_calls"),
            _response("Recorded both."),
        )
    )

    result = await run_tool_loop(
        (ChatMessage(role="user", content="record two values"),),
        ToolSet((tool,)),
        model,
    )

    assert invoked == [1, 2]
    assert [record.call.id for record in result.tool_calls] == ["first", "second"]
    assert [message.role for message in result.messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    assert model.requests[1][0] == result.messages[:-1]


async def test_structured_calls_execute_even_when_finish_reason_is_stop() -> None:
    invoked: list[int] = []

    async def record(request: ValueRequest) -> ValueResult:
        invoked.append(request.value)
        return ValueResult(value=request.value)

    tool = Tool("record", "Record a value.", ValueRequest, ValueResult, record)
    model = _Model(
        (
            _response(
                "I will record it.",
                tool_calls=[_call("record", "record", '{"value":4}')],
                finish_reason="stop",
            ),
            _response("Recorded."),
        )
    )

    result = await run_tool_loop(
        (ChatMessage(role="user", content="record four"),),
        ToolSet((tool,)),
        model,
    )

    assert invoked == [4]
    assert result.content == "Recorded."
    assert result.messages[1].content == "I will record it."


async def test_unknown_and_invalid_calls_are_returned_for_model_repair() -> None:
    invoked: list[int] = []

    async def record(request: ValueRequest) -> ValueResult:
        invoked.append(request.value)
        return ValueResult(value=request.value)

    tool = Tool("record", "Record a value.", ValueRequest, ValueResult, record)
    model = _Model(
        (
            _response(
                tool_calls=[_call("missing", "not_a_tool", "{}")],
                finish_reason="tool_calls",
            ),
            _response(
                tool_calls=[_call("invalid", "record", '{"value":"bad"}')],
                finish_reason="tool_calls",
            ),
            _response(
                tool_calls=[_call("valid", "record", '{"value":3}')],
                finish_reason="tool_calls",
            ),
            _response("Repaired."),
        )
    )

    result = await run_tool_loop(
        (ChatMessage(role="user", content="record"),),
        ToolSet((tool,)),
        model,
    )

    assert invoked == [3]
    errors = [json.loads(record.message.content) for record in result.tool_calls[:2]]
    assert [error["error"] for error in errors] == [
        "unknown_tool",
        "invalid_tool_arguments",
    ]
    assert [record.call.id for record in result.tool_calls] == [
        "missing",
        "invalid",
        "valid",
    ]


async def test_direct_return_finishes_without_another_model_call() -> None:
    async def direct(request: ValueRequest) -> ValueResult:
        return ValueResult(value=request.value)

    tool = Tool(
        "direct",
        "Return a value directly.",
        ValueRequest,
        ValueResult,
        direct,
        return_direct=True,
    )
    model = _Model(
        (
            _response(
                tool_calls=[_call("direct-call", "direct", '{"value":7}')],
                finish_reason="tool_calls",
            ),
        )
    )

    result = await run_tool_loop(
        (ChatMessage(role="user", content="direct"),),
        ToolSet((tool,)),
        model,
    )

    assert result.content == '{"value":7}'
    assert result.return_direct is True
    assert result.iterations == 1
    assert len(model.requests) == 1


async def test_invalid_direct_call_can_be_repaired_without_returning() -> None:
    invoked: list[int] = []

    async def direct(request: ValueRequest) -> ValueResult:
        invoked.append(request.value)
        return ValueResult(value=request.value)

    tool = Tool(
        "direct",
        "Return a value directly.",
        ValueRequest,
        ValueResult,
        direct,
        return_direct=True,
    )
    model = _Model(
        (
            _response(
                tool_calls=[_call("invalid", "direct", "not-json")],
                finish_reason="tool_calls",
            ),
            _response("Could not run that tool."),
        )
    )

    result = await run_tool_loop(
        (ChatMessage(role="user", content="direct"),),
        ToolSet((tool,)),
        model,
    )

    assert result.content == "Could not run that tool."
    assert result.return_direct is False
    assert invoked == []
    assert result.tool_calls[0].return_direct is False


async def test_mixed_direct_return_batch_is_rejected_before_any_effect() -> None:
    invoked: list[str] = []

    async def regular(request: ValueRequest) -> ValueResult:
        invoked.append(f"regular:{request.value}")
        return ValueResult(value=request.value)

    async def direct(request: ValueRequest) -> ValueResult:
        invoked.append(f"direct:{request.value}")
        return ValueResult(value=request.value)

    regular_tool = Tool(
        "regular", "Record normally.", ValueRequest, ValueResult, regular
    )
    direct_tool = Tool(
        "direct",
        "Return directly.",
        ValueRequest,
        ValueResult,
        direct,
        return_direct=True,
    )
    calls = [
        _call("regular", "regular", '{"value":1}'),
        _call("direct", "direct", '{"value":2}'),
    ]

    with pytest.raises(ToolLoopDirectReturnError) as raised:
        await run_tool_loop(
            (ChatMessage(role="user", content="both"),),
            ToolSet((regular_tool, direct_tool)),
            _Model((_response(tool_calls=calls, finish_reason="tool_calls"),)),
        )

    assert invoked == []
    assert raised.value.tool_calls == ()
    assert raised.value.messages[-1].tool_calls == calls


async def test_multiple_direct_return_calls_are_rejected_before_any_effect() -> None:
    invoked: list[int] = []

    async def direct(request: ValueRequest) -> ValueResult:
        invoked.append(request.value)
        return ValueResult(value=request.value)

    tool = Tool(
        "direct",
        "Return directly.",
        ValueRequest,
        ValueResult,
        direct,
        return_direct=True,
    )
    calls = [
        _call("first", "direct", '{"value":1}'),
        _call("second", "direct", '{"value":2}'),
    ]

    with pytest.raises(ToolLoopDirectReturnError):
        await run_tool_loop(
            (ChatMessage(role="user", content="both"),),
            ToolSet((tool,)),
            _Model((_response(tool_calls=calls, finish_reason="tool_calls"),)),
        )

    assert invoked == []


@pytest.mark.parametrize(
    "calls",
    [
        [_call("", "record", '{"value":1}')],
        [_call("  ", "record", '{"value":1}')],
        [
            _call("duplicate", "record", '{"value":1}'),
            _call("duplicate", "record", '{"value":2}'),
        ],
    ],
)
async def test_malformed_tool_call_ids_are_rejected_before_any_effect(
    calls: list[ToolCall],
) -> None:
    invoked: list[int] = []

    async def record(request: ValueRequest) -> ValueResult:
        invoked.append(request.value)
        return ValueResult(value=request.value)

    tool = Tool("record", "Record a value.", ValueRequest, ValueResult, record)

    with pytest.raises(ToolLoopMalformedResponseError) as raised:
        await run_tool_loop(
            (ChatMessage(role="user", content="record"),),
            ToolSet((tool,)),
            _Model((_response(tool_calls=calls, finish_reason="tool_calls"),)),
        )

    assert invoked == []
    assert raised.value.tool_calls == ()


async def test_reused_tool_call_id_is_rejected_before_repeated_effect() -> None:
    invoked: list[int] = []

    async def record(request: ValueRequest) -> ValueResult:
        invoked.append(request.value)
        return ValueResult(value=request.value)

    tool = Tool("record", "Record a value.", ValueRequest, ValueResult, record)
    model = _Model(
        (
            _response(
                tool_calls=[_call("reused", "record", '{"value":1}')],
                finish_reason="tool_calls",
            ),
            _response(
                tool_calls=[_call("reused", "record", '{"value":2}')],
                finish_reason="tool_calls",
            ),
        )
    )

    with pytest.raises(ToolLoopMalformedResponseError) as raised:
        await run_tool_loop(
            (ChatMessage(role="user", content="record"),),
            ToolSet((tool,)),
            model,
        )

    assert invoked == [1]
    assert len(raised.value.tool_calls) == 1


async def test_tool_calls_finish_reason_requires_a_nonempty_call_batch() -> None:
    with pytest.raises(ToolLoopMalformedResponseError):
        await run_tool_loop(
            (ChatMessage(role="user", content="answer"),),
            ToolSet(()),
            _Model((_response("", tool_calls=[], finish_reason="tool_calls"),)),
        )


async def test_tool_call_budget_rejects_whole_batch_before_effects() -> None:
    invoked: list[int] = []

    async def record(request: ValueRequest) -> ValueResult:
        invoked.append(request.value)
        return ValueResult(value=request.value)

    tool = Tool("record", "Record a value.", ValueRequest, ValueResult, record)
    calls = [
        _call("first", "record", '{"value":1}'),
        _call("second", "record", '{"value":2}'),
    ]

    with pytest.raises(ToolLoopToolCallLimitError) as raised:
        await run_tool_loop(
            (ChatMessage(role="user", content="record"),),
            ToolSet((tool,)),
            _Model((_response(tool_calls=calls, finish_reason="tool_calls"),)),
            max_tool_calls=1,
        )

    assert invoked == []
    assert raised.value.iterations == 1
    assert raised.value.tool_calls == ()


async def test_iteration_limit_keeps_partial_transcript_and_audit() -> None:
    async def record(request: ValueRequest) -> ValueResult:
        return ValueResult(value=request.value)

    tool = Tool("record", "Record a value.", ValueRequest, ValueResult, record)
    model = _Model(
        (
            _response(
                tool_calls=[_call("first", "record", '{"value":1}')],
                finish_reason="tool_calls",
            ),
            _response(
                tool_calls=[_call("second", "record", '{"value":1}')],
                finish_reason="tool_calls",
            ),
        )
    )

    with pytest.raises(ToolLoopIterationLimitError) as raised:
        await run_tool_loop(
            (ChatMessage(role="user", content="never finish"),),
            ToolSet((tool,)),
            model,
            max_iterations=2,
        )

    assert raised.value.iterations == 2
    assert len(raised.value.tool_calls) == 2
    assert [message.role for message in raised.value.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens"])
async def test_truncated_batch_is_rejected_before_tool_execution(
    finish_reason: str,
) -> None:
    invoked: list[int] = []

    async def record(request: ValueRequest) -> ValueResult:
        invoked.append(request.value)
        return ValueResult(value=request.value)

    tool = Tool("record", "Record a value.", ValueRequest, ValueResult, record)
    model = _Model(
        (
            _response(
                "partial",
                tool_calls=[_call("record", "record", '{"value":1}')],
                finish_reason=finish_reason,
            ),
        )
    )

    with pytest.raises(ToolLoopTruncatedResponseError) as raised:
        await run_tool_loop(
            (ChatMessage(role="user", content="record"),),
            ToolSet((tool,)),
            model,
        )

    assert invoked == []
    assert raised.value.messages[-1].content == "partial"


async def test_empty_model_response_is_not_treated_as_success() -> None:
    initial = (ChatMessage(role="user", content="answer"),)

    with pytest.raises(ToolLoopEmptyResponseError) as raised:
        await run_tool_loop(initial, ToolSet(()), _Model((_response("  "),)))

    assert raised.value.messages == (
        *initial,
        ChatMessage(role="assistant", content="  "),
    )


async def test_cancellation_propagates_from_tool_execution() -> None:
    started = asyncio.Event()
    invoked_later = False

    async def cancelled(_request: ValueRequest) -> ValueResult:
        started.set()
        raise asyncio.CancelledError

    async def later(request: ValueRequest) -> ValueResult:
        nonlocal invoked_later
        invoked_later = True
        return ValueResult(value=request.value)

    tool = Tool(
        "cancelled", "Cancel execution.", ValueRequest, ValueResult, cancelled
    )
    later_tool = Tool("later", "Run later.", ValueRequest, ValueResult, later)
    model = _Model(
        (
            _response(
                tool_calls=[
                    _call("cancel", "cancelled", '{"value":1}'),
                    _call("later", "later", '{"value":2}'),
                ],
                finish_reason="tool_calls",
            ),
        )
    )

    with pytest.raises(asyncio.CancelledError):
        await run_tool_loop(
            (ChatMessage(role="user", content="cancel"),),
            ToolSet((tool, later_tool)),
            model,
        )

    assert started.is_set()
    assert invoked_later is False


async def test_cancellation_propagates_from_model_call() -> None:
    async def cancelled_model(
        _messages: tuple[ChatMessage, ...],
        _tools: tuple[ToolDef, ...],
    ) -> ChatResponse:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await run_tool_loop(
            (ChatMessage(role="user", content="cancel"),),
            ToolSet(()),
            cancelled_model,
        )


async def test_tool_handler_failure_propagates_without_later_calls() -> None:
    invoked_later = False

    async def failed(_request: ValueRequest) -> ValueResult:
        raise RuntimeError("tool failed")

    async def later(request: ValueRequest) -> ValueResult:
        nonlocal invoked_later
        invoked_later = True
        return ValueResult(value=request.value)

    failed_tool = Tool("failed", "Fail.", ValueRequest, ValueResult, failed)
    later_tool = Tool("later", "Run later.", ValueRequest, ValueResult, later)
    model = _Model(
        (
            _response(
                tool_calls=[
                    _call("failed", "failed", '{"value":1}'),
                    _call("later", "later", '{"value":2}'),
                ],
                finish_reason="tool_calls",
            ),
        )
    )

    with pytest.raises(RuntimeError, match="tool failed"):
        await run_tool_loop(
            (ChatMessage(role="user", content="run"),),
            ToolSet((failed_tool, later_tool)),
            model,
        )

    assert invoked_later is False


@pytest.mark.parametrize(
    ("max_iterations", "max_tool_calls", "message"),
    [
        (0, 1, "max_iterations must be at least 1"),
        (1, -1, "max_tool_calls must not be negative"),
    ],
)
async def test_run_tool_loop_rejects_invalid_limits(
    max_iterations: int,
    max_tool_calls: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        await run_tool_loop(
            (),
            ToolSet(()),
            _Model(()),
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
        )
