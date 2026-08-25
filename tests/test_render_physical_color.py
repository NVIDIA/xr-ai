# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for camera-backed physical color resolution."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from xr_ai_tools.image import ImageReference
from xr_render_demo_worker._physical_color import (
    ResolvePhysicalColorRequest,
    make_physical_color_tool,
    parse_color_answer,
)
from xr_render_demo_worker._trace import current_participant_id, current_trace_id


def test_parse_accepts_only_the_closed_grammar() -> None:
    assert parse_color_answer("VISIBLE 0.0 0.4 1.0") == (0.0, 0.4, 1.0)
    assert parse_color_answer("visible 1, 0, 0") == (1.0, 0.0, 0.0)
    assert parse_color_answer("VISIBLE: .5 .5 .5") == (0.5, 0.5, 0.5)
    assert parse_color_answer("**VISIBLE 0.1 0.2 0.3**") == (0.1, 0.2, 0.3)
    assert parse_color_answer("UNKNOWN") is None
    assert parse_color_answer("unknown.") is None


def test_parse_rejects_everything_else() -> None:
    # Out-of-range, prose, hedges, and bare triples all fail closed.
    assert parse_color_answer("VISIBLE 255 0 0") is None
    assert parse_color_answer("VISIBLE 2 0 0") is None
    assert parse_color_answer("0.0 0.4 1.0") is None
    assert parse_color_answer("The lid is blue.") is None
    assert parse_color_answer("The wall behind the red couch is white.") is None
    assert parse_color_answer("The object may be occluded; likely VISIBLE 1 0 0") is None
    assert parse_color_answer("The requested item is outside the frame; the couch is blue") is None
    assert parse_color_answer("I cannot tell from this image.") is None
    assert parse_color_answer("") is None


@pytest.fixture(autouse=True)
def _bind_participant():
    token = current_participant_id.set("test-user")
    trace_token = current_trace_id.set("trace-1")
    yield
    current_participant_id.reset(token)
    current_trace_id.reset(trace_token)


def _frame_tool(error: Exception | None = None):
    calls = []

    async def execute(request):
        calls.append(request)
        if error is not None:
            raise error
        return SimpleNamespace(image=ImageReference(uri="fake://frame"))
    return SimpleNamespace(execute=execute, calls=calls)


def _query_tool(text: str, available: bool = True, error: Exception | None = None):
    calls = []

    async def execute(request):
        calls.append(request)
        if error is not None:
            raise error
        return SimpleNamespace(text=text, available=available)
    return SimpleNamespace(execute=execute, calls=calls)


async def _expect_rejection(tool, source: str, needle: str) -> None:
    # The Relay Tool boundary re-raises handler errors as RuntimeError with
    # the original class name in the text; the tolerant toolset recovers the
    # rejection from exactly that shape.
    with pytest.raises(RuntimeError) as excinfo:
        await tool.execute(ResolvePhysicalColorRequest(source_words=source))
    assert "ValueError: " in str(excinfo.value)
    assert needle in str(excinfo.value)


async def test_resolver_returns_typed_color() -> None:
    tool = make_physical_color_tool(_frame_tool(), _query_tool("VISIBLE 0.0 0.4 1.0"))
    resolved = await tool.execute(ResolvePhysicalColorRequest(source_words="the lid"))
    assert (resolved.r, resolved.g, resolved.b) == (0.0, 0.4, 1.0)


async def test_resolver_caches_within_turn() -> None:
    query = _query_tool("VISIBLE 0.0 0.4 1.0")
    tool = make_physical_color_tool(_frame_tool(), query)
    first = await tool.execute(ResolvePhysicalColorRequest(source_words="the lid"))
    second = await tool.execute(ResolvePhysicalColorRequest(source_words="The lid"))
    assert (first.r, first.g, first.b) == (second.r, second.g, second.b)
    assert len(query.calls) == 1
    token = current_trace_id.set("trace-2")
    try:
        await tool.execute(ResolvePhysicalColorRequest(source_words="the lid"))
    finally:
        current_trace_id.reset(token)
    assert len(query.calls) == 2


async def test_resolver_unavailable_view_is_value_error() -> None:
    tool = make_physical_color_tool(_frame_tool(), _query_tool("no signal", available=False))
    await _expect_rejection(tool, "the lid", "cannot currently see")


async def test_resolver_unknown_and_malformed_cannot_mutate() -> None:
    for answer in ("UNKNOWN", "The shirt is blue.", "VISIBLE 300 0 0", "likely VISIBLE 1 0 0"):
        tool = make_physical_color_tool(_frame_tool(), _query_tool(answer))
        await _expect_rejection(tool, "my shirt", "did not yield an observation")


async def test_resolver_frame_transport_failure_degrades() -> None:
    tool = make_physical_color_tool(
        _frame_tool(error=RuntimeError("RPCError: no feed")), _query_tool("x"))
    await _expect_rejection(tool, "the lid", "unavailable")


async def test_resolver_query_transport_failure_degrades() -> None:
    tool = make_physical_color_tool(
        _frame_tool(), _query_tool("x", error=TimeoutError("vlm timed out")))
    await _expect_rejection(tool, "the lid", "unavailable")


async def test_resolver_genuine_bug_propagates() -> None:
    tool = make_physical_color_tool(_frame_tool(error=TypeError("wiring bug")), _query_tool("x"))
    with pytest.raises(RuntimeError) as excinfo:
        await tool.execute(ResolvePhysicalColorRequest(source_words="the lid"))
    assert "TypeError: wiring bug" in str(excinfo.value)
    assert "ValueError" not in str(excinfo.value)
    assert "unavailable" not in str(excinfo.value)


async def test_resolver_truncates_long_source_at_word_boundary() -> None:
    query = _query_tool("VISIBLE 0.0 0.4 1.0")
    tool = make_physical_color_tool(_frame_tool(), query)
    await tool.execute(ResolvePhysicalColorRequest(source_words="the thing " * 20))
    quoted = query.calls[0].query.split('"')[1]
    assert len(quoted) <= 80
    assert not quoted.endswith("thin")


async def test_color_of_prefix_is_stripped_from_query() -> None:
    query = _query_tool("VISIBLE 0.0 0.4 1.0")
    tool = make_physical_color_tool(_frame_tool(), query)
    await tool.execute(ResolvePhysicalColorRequest(source_words="the color of my apron"))
    assert query.calls[0].query.split('"')[1] == "my apron"


async def test_no_caching_without_a_trace_id() -> None:
    query = _query_tool("VISIBLE 0.0 0.4 1.0")
    tool = make_physical_color_tool(_frame_tool(), query)
    token = current_trace_id.set("")
    try:
        await tool.execute(ResolvePhysicalColorRequest(source_words="the lid"))
        await tool.execute(ResolvePhysicalColorRequest(source_words="the lid"))
    finally:
        current_trace_id.reset(token)
    assert len(query.calls) == 2
