# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe and validate one caller-selected frame without owning workflow state.

``VisualValidationStep.check`` combines ``observe`` and ``evaluate``. Call them
separately when acquisition and evaluation need different scheduling. The caller
owns frame freshness, duplicate rejection, confirmation counts and guarded commits;
this module only describes evidence and returns a normalized result.
"""

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
    ``on_event(name, **payload)`` records evidence/provenance events; no recording
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
        """Configure a stateless, caller-scheduled visual validation flow.

        Args:
            question: External wording selecting facts to observe in the frame.
            policy: Bounded choices or typed extraction/evaluation policy.
            observer: Injected visual observer with its system instructions and images.
            decisions: Semantic-choice service used by choice policies.
            llm: Language service used for configured settings extraction.
            tools: Restricted native evaluation/retrieval tool catalog.
            timeout_s: Positive timeout for each visual/model operation.
            on_event: Optional asynchronous recording callback; receives event payloads.
        """
        if timeout_s <= 0:
            raise ValueError("visual validation timeout must be positive")
        self._on_event = on_event
        self.question = question
        self.policy = policy
        self.observer = observer
        self.decisions = decisions
        self.llm = llm
        self.tools = tools
        self.timeout_s = timeout_s

    async def observe(self, frame: ImageFrame) -> str | None:
        """Describe the supplied frame without acquiring another one.

        Args:
            frame: Caller-selected image and its source identity.

        Returns:
            Visible evidence text, or None when the requested view is unanswerable.

        Raises:
            TimeoutError: Visual inference exceeds the configured timeout.
            ValueError: Structured evidence does not match the required response shape.
        """

        async with asyncio.timeout(self.timeout_s):
            response = await self.observer.execute(
                ImageQueryRequest(image=frame.image, query=self.policy.extraction_question(self.question))
            )
        if not response.available or not response.text.strip():
            return None
        if self.policy.kind == "choice" and self.policy.structured:
            structured_observation = json.loads(response.text)
            if (
                not isinstance(structured_observation, dict)
                or set(structured_observation) != {"observable", "description"}
                or type(structured_observation["observable"]) is not bool
                or not isinstance(structured_observation["description"], str)
            ):
                raise ValueError("visual observation requires observable and description")
            return (
                structured_observation["description"]
                if structured_observation["observable"] and structured_observation["description"].strip()
                else None
            )
        return response.text

    async def evaluate(self, observation: Any, *, expected: dict[str, Any]) -> Observation:
        """Normalize or judge evidence against established caller-owned values.

        Args:
            observation: Visual text, structured extraction or native trigger result.
            expected: Established values needed for configured comparison/retrieval.

        Returns:
            Evidence and proposed values; no state or confirmation counters are mutated.
        """

        if observation is None or observation == "":
            return Observation("unknown", {}, {})
        return await _observe(
            self.policy,
            observation,
            expected,
            decisions=self.decisions,
            llm=self.llm,
            tools=self.tools,
            timeout_s=self.timeout_s,
            on_event=self._on_event,
        )

    async def check(self, frame: ImageFrame, *, expected: dict[str, Any]) -> VisualCheck:
        """Observe and evaluate once while preserving the supplied frame identity.

        Args:
            frame: Frame whose evidence the caller may later deduplicate or confirm.
            expected: Established values used by the configured evidence evaluation.

        Returns:
            A frame-correlated result without acquiring frames or committing state.
        """

        description = await self.observe(frame)
        outcome = await self.evaluate(description, expected=expected)
        return VisualCheck((frame.track_id, frame.sequence, frame.timestamp_us), description or "", outcome)
