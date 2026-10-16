# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused behavior tests for direct current-view observation."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

from xr_ai_routing import (
    AgentTurn,
)
from xr_ai_tools.image import ImageReference
from xr_ai_tools.vision import ImageQueryRequest

_ROOT = Path(__file__).resolve().parents[1]
_WORKER = _ROOT / "agent-samples" / "conversation-example" / "worker"
sys.path.insert(0, str(_WORKER))

from conversation_example_worker.visual_agents import (  # noqa: E402
    VisualObservationAgent,
)


class _Frames:
    def __init__(self, *, error: Exception | None = None, timestamp_us: int | None = None) -> None:
        self.error = error
        self.timestamp_us = timestamp_us
        self.requests = []

    async def execute(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(image=ImageReference(uri="image://current"), timestamp_us=self.timestamp_us)


class _StreamingVision:
    def __init__(self) -> None:
        self.requests = []

    def stream(self, request: ImageQueryRequest) -> AsyncIterator[SimpleNamespace]:
        self.requests.append(request)

        async def chunks():
            yield SimpleNamespace(text="The monitor is off.")

        return chunks()


async def test_observation_asks_cosmos_directly_without_an_omni_planner() -> None:
    frames = _Frames()
    vision = _StreamingVision()
    destination = VisualObservationAgent(
        frames,  # type: ignore[arg-type]
        vision,  # type: ignore[arg-type]
    )
    turn = AgentTurn("alice", "turn-observe", "Is the monitor screen illuminated?")
    emitted = []

    async def emit(source: str, text: str) -> None:
        emitted.append((source, text))

    result = await destination.answer(turn, emit=emit)

    assert frames.requests[0].participant_id == "alice"
    assert vision.requests == [
        ImageQueryRequest(
            image=ImageReference(uri="image://current"),
            query="Is the monitor screen illuminated?",
        )
    ]
    assert emitted == [("observation", "The monitor is off.")]
    assert result.text == "The monitor is off."


async def test_observation_labels_the_actual_frame_timestamp() -> None:
    frames = _Frames(timestamp_us=1_790_856_000_000_000)
    vision = _StreamingVision()
    destination = VisualObservationAgent(
        frames,  # type: ignore[arg-type]
        vision,  # type: ignore[arg-type]
    )
    turn = AgentTurn("alice", "turn-observe-time", "Is the screen illuminated?")

    async def emit(_source: str, _text: str) -> None:
        pass

    await destination.answer(turn, emit=emit)

    assert vision.requests[0].query.startswith("Frame source time: 2026-10-01 12:00 UTC.")
