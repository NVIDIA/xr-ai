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
from xr_render_demo_worker._trace import current_participant_id
from xr_render_demo_worker.spatial_ops import _COLOR_WORDS


def test_parse_strict_triple() -> None:
    assert parse_color_answer("0.0 0.4 1.0", _COLOR_WORDS) == (0.0, 0.4, 1.0)
    assert parse_color_answer("0, 0.4, 1", _COLOR_WORDS) == (0.0, 0.4, 1.0)


def test_parse_rejects_255_scale_and_stray_numbers() -> None:
    assert parse_color_answer("255, 0, 0", _COLOR_WORDS) is None
    assert parse_color_answer("0.5 meters, 0 lights, 1 fan", _COLOR_WORDS) is None


def test_parse_color_word_last_mention_wins() -> None:
    assert parse_color_answer("The lid is blue.", _COLOR_WORDS) == _COLOR_WORDS["blue"]
    assert (
        parse_color_answer("The wall behind the red couch is white.", _COLOR_WORDS)
        == _COLOR_WORDS["white"]
    )


def test_parse_garbage_is_none() -> None:
    assert parse_color_answer("", _COLOR_WORDS) is None
    assert parse_color_answer("I cannot tell from this image.", _COLOR_WORDS) is None


@pytest.fixture(autouse=True)
def _bind_participant():
    token = current_participant_id.set("test-user")
    yield
    current_participant_id.reset(token)


def _frame_tool(error: Exception | None = None):
    async def execute(request):
        if error is not None:
            raise error
        return SimpleNamespace(image=ImageReference(uri="fake://frame"))
    return SimpleNamespace(execute=execute)


def _query_tool(text: str, available: bool = True):
    async def execute(request):
        return SimpleNamespace(text=text, available=available)
    return SimpleNamespace(execute=execute)


async def test_resolver_returns_typed_color() -> None:
    tool = make_physical_color_tool(_frame_tool(), _query_tool("0.0 0.4 1.0"), _COLOR_WORDS)
    resolved = await tool.execute(ResolvePhysicalColorRequest(source_words="the lid"))
    assert (resolved.r, resolved.g, resolved.b) == (0.0, 0.4, 1.0)


async def test_resolver_unavailable_view_is_value_error() -> None:
    tool = make_physical_color_tool(
        _frame_tool(), _query_tool("no signal", available=False), _COLOR_WORDS)
    with pytest.raises(Exception) as excinfo:
        await tool.execute(ResolvePhysicalColorRequest(source_words="the lid"))
    assert "cannot currently see" in str(excinfo.value)


async def test_resolver_unparseable_answer_is_value_error() -> None:
    tool = make_physical_color_tool(
        _frame_tool(), _query_tool("hard to say from here"), _COLOR_WORDS)
    with pytest.raises(Exception) as excinfo:
        await tool.execute(ResolvePhysicalColorRequest(source_words="the lid"))
    assert "did not yield a color" in str(excinfo.value)


async def test_resolver_transport_failure_degrades() -> None:
    tool = make_physical_color_tool(
        _frame_tool(error=RuntimeError("RPCError: no feed")), _query_tool("x"), _COLOR_WORDS)
    with pytest.raises(Exception) as excinfo:
        await tool.execute(ResolvePhysicalColorRequest(source_words="the lid"))
    assert "unavailable" in str(excinfo.value)


async def test_resolver_genuine_bug_propagates() -> None:
    tool = make_physical_color_tool(
        _frame_tool(error=TypeError("wiring bug")), _query_tool("x"), _COLOR_WORDS)
    with pytest.raises(Exception) as excinfo:
        await tool.execute(ResolvePhysicalColorRequest(source_words="the lid"))
    assert "TypeError" in str(excinfo.value) or "wiring bug" in str(excinfo.value)
    assert "unavailable" not in str(excinfo.value)
