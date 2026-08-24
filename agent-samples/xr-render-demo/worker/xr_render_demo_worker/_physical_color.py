# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera-backed resolution of physical color references.

The subagent LLM copies the user's phrase; it does not observe. This tool
owns the observation: one strict VLM query over the participant's current
frame, parsed into RGB, with an explicit not-visible answer so the VLM is
never forced to invent a color.
"""

from __future__ import annotations

import re
from typing import Protocol

from loguru import logger
from pydantic import BaseModel, Field
from xr_ai_tools import Tool
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryResult

from ._tolerant import as_unavailable
from ._trace import current_participant_id, current_reference_time_us

IMAGE_QUERY_SYSTEM_PROMPT = (
    "Reply with exactly three numbers r, g, b between 0 and 1, separated by spaces, "
    "describing the color actually visible. If the thing asked about is not visible "
    "or you cannot tell, reply exactly UNKNOWN. Never answer with a typical or "
    "assumed color."
)

_NUMBER = r"(?:\d+(?:\.\d+)?|\.\d+)"
_TRIPLE = re.compile(rf"(?<![\d.])({_NUMBER})[,;\s]+({_NUMBER})[,;\s]+({_NUMBER})(?!\d)")
_CHANNEL_TAGS = re.compile(r"\b(?:rgb|[rgb])\s*[=:]")
_WRAPPERS = re.compile(r"[*()\[\]]")

# Refusals AND hedges fail closed: an answer that qualifies its observation
# ("may be occluded; likely RGB 1 0 0", "outside the frame; the couch is
# blue") is not an observation of the requested thing.
_REFUSAL = re.compile(
    r"\b(unknown|not[_\s]?visible|cannot|can't|unable|unclear|"
    r"don'?t\s+(?:see|know)|not\s+(?:sure|certain|able)|"
    r"occluded|obscured|hidden|blocked|likely|probably|possibly|might|"
    r"may\s+be|guess\w*|assum\w*|typical\w*|usually|"
    r"outside|out\s+of\s+(?:frame|view|sight))\b",
    re.IGNORECASE,
)


class _ImageQuery(Protocol):
    async def execute(self, request: ImageQueryRequest) -> ImageQueryResult: ...


class ResolvePhysicalColorRequest(BaseModel):
    source_words: str = Field(description="The user's words for the physical color source.")


class ResolvedColor(BaseModel):
    r: float = Field(ge=0, le=1)
    g: float = Field(ge=0, le=1)
    b: float = Field(ge=0, le=1)


def parse_color_answer(
    answer: str, color_words: dict[str, tuple[float, float, float]]
) -> tuple[float, float, float] | None:
    if _REFUSAL.search(answer):
        return None
    numeric = _WRAPPERS.sub(" ", _CHANNEL_TAGS.sub(" ", answer.lower()))
    for match in _TRIPLE.finditer(numeric):
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
    image_query: _ImageQuery,
    color_words: dict[str, tuple[float, float, float]],
) -> Tool:
    # One observation per phrase per turn: repeated resolutions inside a turn
    # ("make three cones the color of my shirt") must not return three
    # different colors or defeat the creation ledger's retry dedupe.
    cache: dict[tuple[str, int, str], tuple[float, float, float]] = {}

    async def resolve(req: ResolvePhysicalColorRequest) -> ResolvedColor:
        source = req.source_words.strip()
        if len(source) > 80:
            source = source[:80].rsplit(" ", 1)[0]
            logger.debug("physical color source truncated to {!r}", source)
        key = (current_participant_id.get(), current_reference_time_us.get(), source.lower())
        if cache and next(iter(cache))[:2] != key[:2]:
            cache.clear()
        if (cached := cache.get(key)) is not None:
            return ResolvedColor(r=cached[0], g=cached[1], b=cached[2])
        try:
            frame = await current_frame.execute(
                CurrentFrameRequest(participant_id=current_participant_id.get())
            )
        except Exception as error:
            degraded = as_unavailable(error, "the current camera view")
            if degraded is None:
                raise
            logger.debug("physical color frame fetch degraded: {}", degraded)
            raise degraded from error
        try:
            result = await image_query.execute(ImageQueryRequest(
                image=frame.image,
                query=(
                    f'What color is "{source}"? Answer with three numbers r, g, b, each '
                    "between 0 and 1, or exactly UNKNOWN if you cannot see it."
                ),
            ))
        except Exception as error:
            degraded = as_unavailable(error, "the vision model")
            if degraded is None:
                raise
            logger.debug("physical color VLM query degraded: {}", degraded)
            raise degraded from error
        if not result.available:
            logger.debug("physical color {!r} unavailable: {!r}", source, result.text[:120])
            raise ValueError(f"the camera cannot currently see {source!r}: {result.text[:120]}")
        resolved = parse_color_answer(result.text, color_words)
        if resolved is None:
            logger.debug("physical color {!r} unresolved: {!r}", source, result.text[:120])
            raise ValueError(
                f"the camera view did not yield a color for {source!r}: {result.text[:120]}"
            )
        logger.debug("physical color {!r} -> {}", source, resolved)
        cache[key] = resolved
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
