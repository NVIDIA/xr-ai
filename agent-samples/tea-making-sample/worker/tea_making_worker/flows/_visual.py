# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable visual checks with caller-owned state and frame acquisition."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from xr_ai_models import DecisionService, LLMService
from xr_ai_tools import ToolSet
from xr_ai_tools.current_frame import ImageFrame
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryTool

from ._observations import Observation, ObservationPolicy, _observe


@dataclass(frozen=True)
class VisualCheck:
    """A source frame identity, visible description and normalized evidence result."""

    frame_key: tuple[str, int, int]
    description: str
    outcome: Observation


class VisualValidationStep:
    """Observe and evaluate evidence without writing workflow or confirmation state.

    Acquisition cadence, duplicate-frame rejection, revision checks and commits
    belong to the caller. The same check can therefore serve read-only questions.
    Numeric evaluation uses configured native tools; semantic choices use Clef.

    ``question`` selects visible facts; ``policy`` supplies external instructions,
    fields and criteria. ``observer`` owns its visual system prompt and image
    registry. ``tools`` supplies configured native checks or retrieval. Optional
    ``on_event(name, **payload)`` records identity-resolution events; no recording
    storage or workflow state is owned here.
    """

    def __init__(
        self,
        *,
        question: str,
        policy: ObservationPolicy,
        observer: ImageQueryTool,
        decisions: DecisionService,
        llm: LLMService,
        tools: ToolSet,
        timeout_s: float,
        on_event: Callable[..., Awaitable[None]] | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("visual validation timeout must be positive")
        self._on_event = on_event
        self.question, self.policy, self.observer = question, policy, observer
        self.decisions, self.llm, self.tools, self.timeout_s = decisions, llm, tools, timeout_s

    async def observe(self, frame: ImageFrame) -> str | None:
        """Describe a caller-selected frame; return None when it is unanswerable."""

        async with asyncio.timeout(self.timeout_s):
            response = await self.observer.execute(
                ImageQueryRequest(image=frame.image, query=self.policy.extraction_question(self.question))
            )
        if not response.available or not response.text.strip():
            return None
        if self.policy.kind == "choice" and self.policy.structured:
            result = json.loads(response.text)
            if (
                not isinstance(result, dict)
                or set(result) != {"observable", "description"}
                or type(result["observable"]) is not bool
                or not isinstance(result["description"], str)
            ):
                raise ValueError("visual observation requires observable and description")
            return result["description"] if result["observable"] and result["description"].strip() else None
        return response.text

    async def evaluate(self, observation: Any, *, expected: dict[str, Any]) -> Observation:
        """Judge or normalize evidence against caller-supplied established values."""

        if observation is None or observation == "":
            return Observation("unknown", {}, {})
        return await _observe(
            self.policy, observation, expected,
            decisions=self.decisions, llm=self.llm, tools=self.tools, timeout_s=self.timeout_s,
            on_event=self._on_event,
        )

    async def check(self, frame: ImageFrame, *, expected: dict[str, Any]) -> VisualCheck:
        """Observe then evaluate once, without acquiring frames or committing state."""

        description = await self.observe(frame)
        outcome = await self.evaluate(description, expected=expected)
        return VisualCheck((frame.track_id, frame.sequence, frame.timestamp_us), description or "", outcome)
