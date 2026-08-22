# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the expected-degradation classifier and tolerant toolset."""
from __future__ import annotations

import json

import pytest
from pydantic import BaseModel
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


def test_relay_wrapped_type_name_classifies_from_text() -> None:
    # Relay re-raises with the cause chain erased; only the message keeps
    # the original class name.
    wrapped = RuntimeError(
        "internal error: FrameUnavailable: No camera frame available — please try again."
    )
    assert as_unavailable(wrapped, "the camera") is not None
    assert as_unavailable(RuntimeError("RPCError: no feed"), "the camera") is not None
    assert as_unavailable(RuntimeError("connection refused by peer"), "the camera") is not None
    assert as_unavailable(RuntimeError("StatusCode.UNAVAILABLE"), "the camera") is not None


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
    # The Relay Tool boundary rewraps handler bugs as RuntimeError; the
    # tolerant layer must let that propagate, not convert it.
    with pytest.raises(RuntimeError, match="TypeError: wiring bug"):
        await toolset.get("explode").invoke("{}")
