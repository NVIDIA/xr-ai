# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera-backed resolution of physical color references.

The subagent LLM reliably classifies a color source and copies the user's
phrase; it does not reliably observe. This tool owns the observation: one
strict VLM query over the participant's current frame, parsed into RGB.
"""

from __future__ import annotations

import re

from loguru import logger
from pydantic import BaseModel, Field
from xr_ai_tools import Tool
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.vision import ImageQueryRequest

from ._tolerant import as_unavailable
from ._trace import current_participant_id

IMAGE_QUERY_SYSTEM_PROMPT = (
    "Reply with exactly three numbers r, g, b between 0 and 1, separated by spaces."
)

_TRIPLE = re.compile(r"^\s*(\d(?:\.\d+)?)[,\s]+(\d(?:\.\d+)?)[,\s]+(\d(?:\.\d+)?)\s*\.?\s*$")


class ResolvePhysicalColorRequest(BaseModel):
    source_words: str = Field(description="The user's words for the physical color source.")


class ResolvedColor(BaseModel):
    r: float
    g: float
    b: float


def parse_color_answer(
    answer: str, color_words: dict[str, tuple[float, float, float]]
) -> tuple[float, float, float] | None:
    match = _TRIPLE.match(answer)
    if match:
        values = tuple(float(v) for v in match.groups())
        if all(0.0 <= v <= 1.0 for v in values):
            return values
    resolved = None
    # Last color word wins: replies name the answer after any echoed context
    # ("the wall behind the red couch is white").
    for word in re.findall(r"[a-z]+", answer.lower()):
        if word in color_words:
            resolved = color_words[word]
    return resolved


def make_physical_color_tool(
    current_frame: CurrentFrameTool,
    image_query,
    color_words: dict[str, tuple[float, float, float]],
) -> Tool:
    async def resolve(req: ResolvePhysicalColorRequest) -> ResolvedColor:
        source = req.source_words[:80]
        try:
            frame = await current_frame.execute(
                CurrentFrameRequest(participant_id=current_participant_id.get())
            )
        except Exception as error:
            degraded = as_unavailable(error, "the current camera view")
            if degraded is None:
                raise
            raise degraded from error
        result = await image_query.execute(ImageQueryRequest(
            image=frame.image,
            query=f'What color is "{source}"? Answer with three numbers r, g, b, each between 0 and 1.',
        ))
        if not result.available:
            raise ValueError(f"the camera cannot currently see {source!r}: {result.text}")
        resolved = parse_color_answer(result.text, color_words)
        if resolved is None:
            raise ValueError(
                f"the camera view did not yield a color for {source!r}: {result.text[:120]}"
            )
        logger.debug("physical color {!r} -> {}", source, resolved)
        return ResolvedColor(r=resolved[0], g=resolved[1], b=resolved[2])

    return Tool(
        "resolve_physical_color",
        "Resolve a physical-world color reference against the participant's current camera view.",
        ResolvePhysicalColorRequest,
        ResolvedColor,
        resolve,
    )


__all__ = [
    "IMAGE_QUERY_SYSTEM_PROMPT",
    "ResolvePhysicalColorRequest",
    "ResolvedColor",
    "make_physical_color_tool",
    "parse_color_answer",
]
