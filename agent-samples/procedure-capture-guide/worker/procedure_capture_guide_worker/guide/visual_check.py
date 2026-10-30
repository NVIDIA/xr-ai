# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify a selected frame with CLEF and request a focused caption when uncertain.

Image judgment, caption generation, and caption judgment remain separate calls.
This module composes them into the sample's uniform decision rule; the backend
owns frame selection, consecutive evidence, and progress updates.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from loguru import logger
from xr_ai_models import DecisionAnswer, DecisionService, ImageInput
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryResult, ImageQueryTool

from .verification import judge_image, judge_observation

_IMAGE_CONFIDENCE_THRESHOLD = 0.80


@dataclass(slots=True)
class VisualCheckTrace:
    """Retain completed stages even when a later inference fails."""

    source: Literal["image", "caption"] = "image"
    image_answer: DecisionAnswer | None = None
    caption: ImageQueryResult | None = None
    answer: DecisionAnswer | None = None


async def caption_image(
    query: ImageQueryTool, *, images: ImageRegistry, image: ImageInput,
    question: str, timeout_s: float, owner: str | None = None,
) -> ImageQueryResult:
    """Describe the pinned image through the existing visual tool and prompt."""

    # Re-register the pinned input immediately before inference: another frame
    # producer may have evicted the original reference during image judgment.
    reference = images.put(image, owner=owner)
    async with asyncio.timeout(timeout_s):
        return await query.execute(ImageQueryRequest(image=reference, query=question))


async def verify_image(
    decisions: DecisionService, query: ImageQueryTool, *, images: ImageRegistry,
    image: ImageInput, condition: str, question: str, frame: dict, step_id: str,
    revision: int, decision_timeout_s: float, vision_timeout_s: float,
    trace: VisualCheckTrace | None = None,
) -> DecisionAnswer:
    """Use a >80% image decision, otherwise judge a caption of the same frame.

    The cutoff applies to the selected choice, including no. Caption judgment
    uses the existing caption-only rule without another confidence threshold.
    Errors and cancellation propagate; they are not low-confidence answers.
    """

    context = dict(condition=condition, frame=frame, step_id=step_id, revision=revision,
                   timeout_s=decision_timeout_s)
    trace = trace if trace is not None else VisualCheckTrace()
    first = await judge_image(decisions, image=image, **context)
    trace.image_answer = first
    if first.confidence > _IMAGE_CONFIDENCE_THRESHOLD:
        trace.answer = first
        return first
    trace.source = "caption"
    logger.debug("Guide caption fallback step={} revision={} frame={} confidence={} threshold={}",
                 step_id, revision, frame, first.confidence, _IMAGE_CONFIDENCE_THRESHOLD)
    caption = await caption_image(
        query, images=images, image=image, question=question, timeout_s=vision_timeout_s,
        owner=frame.get("participant_id"),
    )
    trace.caption = caption
    if not caption.available or not caption.text.strip():
        raise ValueError("visual caption unavailable")
    final = await judge_observation(decisions, observation=caption.text, **context)
    trace.answer = final
    return final
