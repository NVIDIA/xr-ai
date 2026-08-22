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
from xr_render_demo_worker._trace import current_participant_id, current_reference_time_us
from xr_render_demo_worker.spatial_ops import COLOR_WORDS


def test_parse_bare_triples() -> None:
    assert parse_color_answer("0.0 0.4 1.0", COLOR_WORDS) == (0.0, 0.4, 1.0)
    assert parse_color_answer("0, 0.4, 1", COLOR_WORDS) == (0.0, 0.4, 1.0)
    assert parse_color_answer("1 0 0", COLOR_WORDS) == (1.0, 0.0, 0.0)
    assert parse_color_answer(".5 .5 .5", COLOR_WORDS) == (0.5, 0.5, 0.5)


def test_parse_decorated_triples() -> None:
    assert parse_color_answer("The color is 0.1 0.2 0.3", COLOR_WORDS) == (0.1, 0.2, 0.3)
    assert parse_color_answer("0.1 0.2 0.3.", COLOR_WORDS) == (0.1, 0.2, 0.3)
    assert parse_color_answer("rgb(0.1, 0.2, 0.3)", COLOR_WORDS) == (0.1, 0.2, 0.3)
    assert parse_color_answer("r=0.0 g=0.4 b=1.0", COLOR_WORDS) == (0.0, 0.4, 1.0)
    assert parse_color_answer("**0.1 0.2 0.3**", COLOR_WORDS) == (0.1, 0.2, 0.3)
    assert parse_color_answer("(0.0, 0.4, 1.0)", COLOR_WORDS) == (0.0, 0.4, 1.0)


def test_parse_rejects_out_of_range_triples() -> None:
    # "255, 0, 0" matches the triple shape and must die at the range guard.
    assert parse_color_answer("255, 0, 0", COLOR_WORDS) is None
    assert parse_color_answer("2 0 0", COLOR_WORDS) is None
    assert parse_color_answer("1.5 0.2 0.9", COLOR_WORDS) is None
    assert parse_color_answer("0.5 meters, 0 lights, 1 fan", COLOR_WORDS) is None
    assert parse_color_answer("0.5 0.5", COLOR_WORDS) is None


def test_parse_color_word_last_mention_wins() -> None:
    assert parse_color_answer("The lid is blue.", COLOR_WORDS) == COLOR_WORDS["blue"]
    assert (
        parse_color_answer("The wall behind the red couch is white.", COLOR_WORDS)
        == COLOR_WORDS["white"]
    )


def test_parse_refusals_never_become_colors() -> None:
    assert parse_color_answer("UNKNOWN", COLOR_WORDS) is None
    assert parse_color_answer("I cannot determine whether the wall is white.", COLOR_WORDS) is None
    assert parse_color_answer("I am unable to see; the frame is black.", COLOR_WORDS) is None
    assert parse_color_answer("Not visible: 0.1 0.2 0.3", COLOR_WORDS) is None


def test_parse_garbage_is_none() -> None:
    assert parse_color_answer("", COLOR_WORDS) is None
    assert parse_color_answer("I cannot tell from this image.", COLOR_WORDS) is None
    assert parse_color_answer("hard to say from here", COLOR_WORDS) is None


@pytest.fixture(autouse=True)
def _bind_participant():
    token = current_participant_id.set("test-user")
    time_token = current_reference_time_us.set(1_700_000_000_000_000)
    yield
    current_participant_id.reset(token)
    current_reference_time_us.reset(time_token)


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


async def test_resolver_returns_typed_color() -> None:
    tool = make_physical_color_tool(_frame_tool(), _query_tool("0.0 0.4 1.0"), COLOR_WORDS)
    resolved = await tool.execute(ResolvePhysicalColorRequest(source_words="the lid"))
    assert (resolved.r, resolved.g, resolved.b) == (0.0, 0.4, 1.0)


async def test_resolver_caches_within_turn() -> None:
    query = _query_tool("0.0 0.4 1.0")
    tool = make_physical_color_tool(_frame_tool(), query, COLOR_WORDS)
    first = await tool.execute(ResolvePhysicalColorRequest(source_words="the lid"))
    second = await tool.execute(ResolvePhysicalColorRequest(source_words="The lid"))
    assert (first.r, first.g, first.b) == (second.r, second.g, second.b)
    assert len(query.calls) == 1
    token = current_reference_time_us.set(1_700_000_001_000_000)
    try:
        await tool.execute(ResolvePhysicalColorRequest(source_words="the lid"))
    finally:
        current_reference_time_us.reset(token)
    assert len(query.calls) == 2


async def _expect_rejection(tool, source: str, needle: str) -> None:
    # The Relay Tool boundary re-raises handler errors as RuntimeError with
    # the original class name in the text; the tolerant toolset recovers the
    # rejection from exactly that shape.
    with pytest.raises(RuntimeError) as excinfo:
        await tool.execute(ResolvePhysicalColorRequest(source_words=source))
    assert "ValueError: " in str(excinfo.value)
    assert needle in str(excinfo.value)


async def test_resolver_unavailable_view_is_value_error() -> None:
    tool = make_physical_color_tool(
        _frame_tool(), _query_tool("no signal", available=False), COLOR_WORDS)
    await _expect_rejection(tool, "the lid", "cannot currently see")


async def test_resolver_unparseable_answer_is_value_error() -> None:
    tool = make_physical_color_tool(
        _frame_tool(), _query_tool("hard to say from here"), COLOR_WORDS)
    await _expect_rejection(tool, "the lid", "did not yield a color")


async def test_resolver_not_visible_is_value_error_without_mutation_value() -> None:
    tool = make_physical_color_tool(_frame_tool(), _query_tool("UNKNOWN"), COLOR_WORDS)
    await _expect_rejection(tool, "my shirt", "did not yield a color")


async def test_resolver_frame_transport_failure_degrades() -> None:
    tool = make_physical_color_tool(
        _frame_tool(error=RuntimeError("RPCError: no feed")), _query_tool("x"), COLOR_WORDS)
    await _expect_rejection(tool, "the lid", "unavailable")


async def test_resolver_query_transport_failure_degrades() -> None:
    tool = make_physical_color_tool(
        _frame_tool(), _query_tool("x", error=TimeoutError("vlm timed out")), COLOR_WORDS)
    await _expect_rejection(tool, "the lid", "unavailable")


async def test_resolver_genuine_bug_propagates() -> None:
    tool = make_physical_color_tool(
        _frame_tool(error=TypeError("wiring bug")), _query_tool("x"), COLOR_WORDS)
    with pytest.raises(RuntimeError) as excinfo:
        await tool.execute(ResolvePhysicalColorRequest(source_words="the lid"))
    assert "TypeError: wiring bug" in str(excinfo.value)
    assert "ValueError" not in str(excinfo.value)
    assert "unavailable" not in str(excinfo.value)


async def test_resolver_query_genuine_bug_propagates() -> None:
    tool = make_physical_color_tool(
        _frame_tool(), _query_tool("x", error=TypeError("bad request model")), COLOR_WORDS)
    with pytest.raises(RuntimeError) as excinfo:
        await tool.execute(ResolvePhysicalColorRequest(source_words="the lid"))
    assert "TypeError: bad request model" in str(excinfo.value)
    assert "ValueError" not in str(excinfo.value)
    assert "unavailable" not in str(excinfo.value)


async def test_resolver_truncates_long_source_at_word_boundary() -> None:
    query = _query_tool("0.0 0.4 1.0")
    tool = make_physical_color_tool(_frame_tool(), query, COLOR_WORDS)
    await tool.execute(ResolvePhysicalColorRequest(source_words="the thing " * 20))
    quoted = query.calls[0].query.split('"')[1]
    assert len(quoted) <= 80
    assert not quoted.endswith("thin")
