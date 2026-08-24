# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera-backed resolution of physical color references.

The subagent LLM names the source; it does not observe. This tool owns the
observation: one VLM query over the participant's current frame with a
closed response grammar, full-matching ``VISIBLE r g b`` or ``UNKNOWN``.
Any other reply fails closed; no color is ever inferred from prose.
"""

from __future__ import annotations

import re
from typing import Protocol

from loguru import logger
from pydantic import BaseModel, Field
from xr_ai_tools import Tool
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryResult

from ._tolerant import reraise_unavailable
from ._trace import current_participant_id, current_trace_id

IMAGE_QUERY_SYSTEM_PROMPT = (
    'Reply with exactly "VISIBLE r g b", where r, g, b are numbers between 0 '
    "and 1 for the color you actually see, or exactly \"UNKNOWN\" when the "
    "asked-about thing is not clearly visible. Reply with nothing else; never "
    "answer with a typical or assumed color."
)

_NUMBER = r"(?:\d+(?:\.\d+)?|\.\d+)"
_VISIBLE = re.compile(
    rf"^\W*visible[\s:,]+({_NUMBER})[,\s]+({_NUMBER})[,\s]+({_NUMBER})\W*$",
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


def parse_color_answer(answer: str) -> tuple[float, float, float] | None:
    """Full-match the closed grammar; UNKNOWN, malformed, and out-of-range
    replies all yield None (no observation)."""
    match = _VISIBLE.match(answer)
    if match:
        values = tuple(float(v) for v in match.groups())
        if all(0.0 <= v <= 1.0 for v in values):
            return values
    return None


def make_physical_color_tool(
    current_frame: CurrentFrameTool,
    image_query: _ImageQuery,
) -> Tool:
    # One observation per phrase per turn: repeated resolutions inside a turn
    # ("make three cones the color of my shirt") must not return three
    # different colors or defeat the creation ledger's retry dedupe. Keyed
    # on the runtime-bound trace id; client timestamps can repeat.
    cache: dict[tuple[str, str, str], tuple[float, float, float]] = {}

    async def resolve(req: ResolvePhysicalColorRequest) -> ResolvedColor:
        source = req.source_words.strip().replace('"', "'")
        # "the color of my apron" and "my apron" name the same observation.
        source = re.sub(r"^(?:the\s+)?colou?r\s+of\s+", "", source, flags=re.IGNORECASE) or source
        if len(source) > 80:
            head = source[:80]
            clipped = head.rsplit(" ", 1)[0]
            source = clipped if len(clipped) >= 40 else head
            logger.debug("physical color source truncated to {!r}", source)
        # An empty trace id cannot distinguish turns; skip caching entirely.
        trace = current_trace_id.get()
        key = (current_participant_id.get(), trace, source.lower())
        if trace:
            if cache and next(iter(cache))[:2] != key[:2]:
                cache.clear()
            if (cached := cache.get(key)) is not None:
                return ResolvedColor(r=cached[0], g=cached[1], b=cached[2])
        try:
            frame = await current_frame.execute(
                CurrentFrameRequest(participant_id=current_participant_id.get())
            )
        except Exception as error:
            reraise_unavailable(error, "the current camera view")
        try:
            result = await image_query.execute(ImageQueryRequest(
                image=frame.image,
                query=(
                    f'What color is "{source}"? Reply with exactly "VISIBLE r g b" '
                    '(each number between 0 and 1) or exactly "UNKNOWN".'
                ),
            ))
        except Exception as error:
            reraise_unavailable(error, "the vision model")
        if not result.available:
            logger.debug("physical color {!r} unavailable: {!r}", source, result.text[:120])
            raise ValueError(f"the camera cannot currently see {source!r}: {result.text[:120]}")
        resolved = parse_color_answer(result.text)
        if resolved is None:
            logger.debug("physical color {!r} not observed: {!r}", source, result.text[:120])
            raise ValueError(
                f"the camera view did not yield an observation of {source!r}: {result.text[:120]}"
            )
        logger.debug("physical color {!r} -> {}", source, resolved)
        if trace:
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
