# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify a selected frame with CLEF and request a focused caption when uncertain.

Image judgment, caption generation, and caption judgment remain separate calls.
This module composes them into the sample's uniform decision rule; the backend
owns frame selection, consecutive evidence, and progress updates.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Literal

from loguru import logger
from xr_ai_models import DecisionAnswer, DecisionService, ImageInput
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryResult, ImageQueryTool

from .measurements import MeasurementPolicy, MeasurementResult, compare_measurements
from .policy import load_policy
from .verification import judge_image, judge_observation, verdict


@dataclass(slots=True)
class VisualCheckTrace:
    """Retain completed stages even when a later inference fails."""

    prepared_image: bytes | None = field(default=None, repr=False)
    latency_s: dict[str, float] = field(default_factory=dict)

    def prepared(self, image):
        self.prepared_image = image

    @contextmanager
    def stage(self, name):
        started = time.monotonic()
        try:
            yield
        finally:
            self.latency_s[name] = time.monotonic() - started

    source: Literal["image", "caption", "measurement"] = "image"
    image_answer: DecisionAnswer | None = None
    caption: ImageQueryResult | None = None
    answer: DecisionAnswer | None = None
    measurement: MeasurementResult | None = None

    @property
    def verdict(self) -> str | None:
        if self.measurement is not None:
            return self.measurement.verdict
        return verdict(self.answer) if self.answer is not None else None

    @property
    def accepted(self) -> bool | None:
        result = self.verdict
        return result == "accepted" if result is not None else None

    @property
    def description(self) -> str | None:
        if self.measurement is not None:
            return "; ".join(f"{item['role']} {item['value']:g} {item['unit']}"
                             for item in self.measurement.readings)
        return self.caption.text if self.caption is not None else None

    def record(self) -> dict:
        return {"source": self.source, "verdict": self.verdict, "stage_latency_s": dict(self.latency_s),
                "prepared_image_sha256": (hashlib.sha256(self.prepared_image).hexdigest()
                                          if self.prepared_image is not None else None),
                "caption": self.caption.model_dump() if self.caption is not None else None,
                "image_answer": asdict(self.image_answer) if self.image_answer is not None else None,
                "answer": asdict(self.answer) if self.answer is not None else None,
                "measurement": asdict(self.measurement) if self.measurement is not None else None}


async def caption_image(
    query: ImageQueryTool, *, images: ImageRegistry, image: ImageInput,
    question: str, timeout_s: float, owner: str | None = None,
) -> ImageQueryResult:
    """Describe the pinned image through the existing visual tool and prompt."""

    # Re-register the pinned input immediately before inference: another frame
    # producer may have evicted the original reference during earlier inference.
    reference = images.put(image, owner=owner)
    async with asyncio.timeout(timeout_s):
        return await query.execute(ImageQueryRequest(image=reference, query=question))


async def verify_image(
    decisions: DecisionService, query: ImageQueryTool, *, images: ImageRegistry,
    image: ImageInput, condition: str | None, question: str, frame: dict, step_id: str,
    revision: int, decision_timeout_s: float, vision_timeout_s: float,
    trace: VisualCheckTrace | None = None,
    measurement: MeasurementPolicy | None = None, measurement_query: ImageQueryTool | None = None, policy=None,
) -> VisualCheckTrace:
    """Use an image decision above the configured cutoff, otherwise judge its caption.

    The cutoff applies to the selected choice, including rejected and unknown. Caption judgment
    uses the existing caption-only rule without another confidence threshold.
    Errors and cancellation propagate; they are not low-confidence answers.
    """

    policy = policy or load_policy()
    context = dict(policy=policy, condition=condition, frame=frame, step_id=step_id, revision=revision,
                   timeout_s=decision_timeout_s)
    trace = trace if trace is not None else VisualCheckTrace()
    if measurement is not None:
        trace.source = "measurement"
        if measurement_query is None:
            raise ValueError("measurement extraction client is not configured")
        with trace.stage("measurement"):
            trace.caption = await caption_image(
                measurement_query, images=images, image=image, question=question,
                timeout_s=vision_timeout_s, owner=frame.get("participant_id"),
            )
        if not trace.caption.available:
            raise ValueError("measurement extraction unavailable")
        trace.measurement = compare_measurements(trace.caption.text, measurement)
        return trace
    if condition is None:
        raise ValueError("semantic verification requires a condition")
    with trace.stage("image_decision"):
        first = await judge_image(decisions, image=image, on_prepared=trace.prepared, **context)
    trace.image_answer = first
    if first.confidence > policy.image_confidence_threshold:
        trace.answer = first
        return trace
    trace.source = "caption"
    logger.debug("Guide caption fallback step={} revision={} frame={} confidence={} threshold={}",
                 step_id, revision, frame, first.confidence, policy.image_confidence_threshold)
    with trace.stage("caption"):
        caption = await caption_image(
            query, images=images, image=image, question=question, timeout_s=vision_timeout_s,
            owner=frame.get("participant_id"),
        )
    trace.caption = caption
    if not caption.available or not caption.text.strip():
        raise ValueError("visual caption unavailable")
    with trace.stage("caption_decision"):
        final = await judge_observation(decisions, observation=caption.text, **context)
    trace.answer = final
    return trace
