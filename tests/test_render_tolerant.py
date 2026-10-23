# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the expected-degradation classifier and tolerant toolset."""
from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ValidationError
from xr_ai_hub import FrameUnavailable
from xr_ai_tools import Tool
from xr_render_demo_worker._tolerant import as_unavailable, tolerant_toolset


def test_direct_transport_types_classify() -> None:
    assert as_unavailable(TimeoutError("slow"), "the feed") is not None
    assert as_unavailable(ConnectionError("refused"), "the feed") is not None
    assert as_unavailable(FrameUnavailable("No camera frame available"), "the feed") is not None


def test_nested_cause_walk_classifies() -> None:
    outer = RuntimeError("wrapper")
    outer.__cause__ = TimeoutError("slow")
    assert as_unavailable(outer, "the feed") is not None
    contextual = RuntimeError("wrapper")
    contextual.__context__ = ConnectionError("refused")
    assert as_unavailable(contextual, "the feed") is not None


def test_cause_cycle_terminates() -> None:
    first = RuntimeError("a")
    second = RuntimeError("b")
    first.__cause__ = second
    second.__context__ = first
    assert as_unavailable(first, "the feed") is None


@pytest.mark.parametrize("message", [
    "internal error: FrameUnavailable: no camera", "RPCError: no feed",
    "connection refused by peer", "StatusCode.UNAVAILABLE",
])
def test_exception_text_does_not_establish_unavailability(message: str) -> None:
    assert as_unavailable(RuntimeError(message), "the camera") is None


def test_ordinary_english_does_not_classify() -> None:
    assert as_unavailable(RuntimeError("boom"), "the feed") is None
    assert as_unavailable(RuntimeError("the operator is unavailable today"), "the feed") is None
    assert as_unavailable(RuntimeError("feature disabled by config"), "the feed") is None
    assert as_unavailable(FileNotFoundError("missing.yaml"), "the feed") is None
    assert as_unavailable(PermissionError("denied"), "the feed") is None


def test_classified_message_names_the_feed() -> None:
    degraded = as_unavailable(TimeoutError("slow"), "the current camera view")
    assert isinstance(degraded, ValueError)
    assert "the current camera view is unavailable" in str(degraded)


class _Empty(BaseModel):
    pass


async def test_tolerant_tool_converts_value_error_and_propagates_bugs() -> None:
    async def reject(req: _Empty) -> None:
        raise ValueError("no such object")

    async def explode(req: _Empty) -> None:
        raise TypeError("wiring bug")

    toolset = tolerant_toolset([
        Tool("reject", "Reject.", _Empty, None, reject),
        Tool("explode", "Explode.", _Empty, None, explode),
    ])
    result = await toolset.get("reject").invoke("{}")
    assert json.loads(result.content)["detail"] == "no such object"
    with pytest.raises(TypeError, match="wiring bug"):
        await toolset.get("explode").invoke("{}")


@pytest.mark.parametrize("chaining", ["cause", "context", "text", "validation"])
async def test_tolerant_tool_does_not_recover_internal_failures(chaining: str) -> None:
    if chaining == "validation":
        class InternalResult(BaseModel):
            total: int

        with pytest.raises(ValidationError) as raised:
            InternalResult.model_validate({"total": "not a number"})
        error = raised.value
    else:
        error = TypeError("ValueError: old rejection" if chaining == "text" else "internal bug")
        if chaining == "cause":
            error.__cause__ = ValueError("old rejection")
        elif chaining == "context":
            error.__context__ = ValueError("old rejection")

    async def fail(_request: _Empty) -> None:
        raise error

    toolset = tolerant_toolset([Tool("fail", "Fail internally.", _Empty, None, fail)])
    with pytest.raises(type(error)) as raised:
        await toolset.get("fail").invoke("{}")
    assert raised.value is error


async def test_tolerant_tool_preserves_model_argument_validation() -> None:
    class Request(BaseModel):
        count: int

    async def apply(_request: Request) -> None:
        pytest.fail("invalid model arguments must not reach the handler")

    toolset = tolerant_toolset([Tool("apply", "Validate model input.", Request, None, apply)])
    result = await toolset.get("apply").invoke('{"count":"not a number"}')
    assert json.loads(result.content)["error"] == "invalid_tool_arguments"
