# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming and materialization tests for ``VisionModule``."""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest
from xr_ai_skills import VisionModule, VisionUnavailable

_AGENT_PATH = (
    Path(__file__).resolve().parents[1]
    / "agent-samples"
    / "simple-vlm-example"
    / "worker"
    / "agent.py"
)
_AGENT_SPEC = importlib.util.spec_from_file_location("simple_vlm_agent", _AGENT_PATH)
assert _AGENT_SPEC is not None and _AGENT_SPEC.loader is not None
_AGENT_MODULE = importlib.util.module_from_spec(_AGENT_SPEC)
_AGENT_SPEC.loader.exec_module(_AGENT_MODULE)
SimpleVlmBrain = _AGENT_MODULE.SimpleVlmBrain


class _Endpoint:
    pass


class _ControlledVlm:
    def __init__(self) -> None:
        self.release = asyncio.Event()

    async def stream(self, image, question, *, system_prompt="", **kwargs):
        yield "First sentence."
        await self.release.wait()
        yield " Second sentence."

    async def ask_image(self, *args, **kwargs):
        raise AssertionError("VisionModule must materialize the canonical stream")


async def _vision(vlm) -> VisionModule:
    vision = VisionModule(_Endpoint(), vlm)  # type: ignore[arg-type]

    async def acquire(pid: str) -> str:
        return "data:image/jpeg;base64,stub"

    vision._acquire_image_url = acquire  # type: ignore[method-assign]
    return vision


async def test_stream_delivers_first_chunk_before_generation_finishes() -> None:
    vlm = _ControlledVlm()
    vision = await _vision(vlm)
    stream = vision.stream("pid-1", "What do you see?")

    assert await anext(stream) == "First sentence."

    second = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    assert not second.done()

    vlm.release.set()
    assert await second == " Second sentence."
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


async def test_perceive_collects_the_canonical_stream() -> None:
    vlm = _ControlledVlm()
    vlm.release.set()
    vision = await _vision(vlm)

    answer = await vision.perceive("pid-1", "What do you see?")

    assert answer == "First sentence. Second sentence."


class _EmptyVlm:
    async def stream(self, image, question, *, system_prompt="", **kwargs):
        yield "  "


async def test_empty_stream_raises_speakable_error() -> None:
    vision = await _vision(_EmptyVlm())

    with pytest.raises(VisionUnavailable, match="couldn't make out"):
        await vision.perceive("pid-1", "What do you see?")


class _StatusEndpoint:
    def __init__(self) -> None:
        self.statuses: list[tuple[str, str]] = []

    async def set_status(self, status: str, pid: str) -> None:
        self.statuses.append((status, pid))


class _Transport:
    def __init__(self, endpoint: _StatusEndpoint) -> None:
        self.endpoint = endpoint


async def test_simple_vlm_status_spans_stream_iteration() -> None:
    endpoint = _StatusEndpoint()
    vision = _ControlledVlm()
    brain = object.__new__(SimpleVlmBrain)
    brain._transport = _Transport(endpoint)
    brain._vision = vision
    stream = brain._stream_answer("pid-1", "What do you see?")

    assert endpoint.statuses == []
    assert await anext(stream) == "First sentence."
    assert endpoint.statuses == [("processing", "pid-1")]

    second = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    assert endpoint.statuses == [("processing", "pid-1")]

    vision.release.set()
    assert await second == " Second sentence."
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert endpoint.statuses == [("processing", "pid-1"), ("idle", "pid-1")]
