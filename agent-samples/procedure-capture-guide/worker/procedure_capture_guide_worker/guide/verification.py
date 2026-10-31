# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Judge whether visual evidence establishes a procedure step's condition.

The typed CLEF service judges either a selected image or a written observation.
This module keeps the decision policy separate from image questions and progress
updates. The guide backend owns frame freshness, confirmations, and transitions.
"""

from __future__ import annotations

import asyncio
import base64
import io
import math
from pathlib import Path

import nemo_relay
from loguru import logger
from PIL import Image
from xr_ai_models import DecisionAnswer, DecisionQuestion, DecisionService, ImageInput

_INSTRUCTIONS = (Path(__file__).with_name("prompts") / "guide_condition.txt").read_text(encoding="utf-8").strip()
_QUESTION = DecisionQuestion(
    instructions=_INSTRUCTIONS,
    criteria={
        "yes": "The written observation establishes the required visible condition.",
        "no": "The observation contradicts the condition, lacks a required fact, or leaves it uncertain or obscured.",
    },
)

_IMAGE_INSTRUCTIONS = (Path(__file__).with_name("prompts") / "guide_image_condition.txt").read_text(
    encoding="utf-8",
).strip()
_IMAGE_QUESTION = DecisionQuestion(
    instructions=_IMAGE_INSTRUCTIONS,
    criteria={
        "yes": "The image establishes the required visible condition.",
        "no": "The image contradicts the condition, lacks a required fact, or leaves it uncertain or obscured.",
    },
)


async def judge_image(
    decisions: DecisionService, *, condition: str, image: ImageInput,
    frame: dict, step_id: str, revision: int, timeout_s: float,
) -> DecisionAnswer:
    """Judge one selected frame without a caption or conversation history."""

    state = {"condition": condition, "image_role": "The single image is the selected current camera frame."}
    prepared = await asyncio.to_thread(_prepare_decision_image, image)
    return await _judge(
        decisions, state=state, question=_IMAGE_QUESTION, images=(prepared,),
        frame=frame, step_id=step_id, revision=revision, timeout_s=timeout_s, source="image",
    )


# Base64 expands this to ~683 KiB, leaving room for the condition and question
# within CLEF's default 1 MiB request limit. Bound pixels as well: compressed
# images can be small while exceeding the default 4096-token source-pixel budget.
_IMAGE_MAX_BYTES = 512 * 1024
_IMAGE_MAX_PIXELS = 2_000_000


def _prepare_decision_image(image: ImageInput) -> bytes:
    """Adapt camera media to CLEF without changing the selected frame identity.

    Preserve small JPEG/PNG inputs; convert WebP and bound larger inputs with
    aspect-preserving JPEG encoding. The caller retains the original for Cosmos.
    Invalid images raise rather than triggering confidence-based fallback.
    """
    if isinstance(image, Path):
        raw = image.read_bytes()
    elif isinstance(image, str):
        header, separator, payload = image.partition(",")
        if not separator or header not in {
            "data:image/jpeg;base64", "data:image/png;base64", "data:image/webp;base64",
        }:
            raise ValueError("guide decision image must be inline JPEG, PNG, or WebP")
        raw = base64.b64decode(payload, validate=True)
    elif isinstance(image, bytes):
        raw = image
    else:
        raise TypeError("guide decision image must be bytes, Path, or an inline data URL")
    with Image.open(io.BytesIO(raw), formats=["JPEG", "PNG", "WEBP"]) as source:
        source.load()
        if source.width * source.height <= _IMAGE_MAX_PIXELS:
            if source.format in {"JPEG", "PNG"} and len(raw) <= _IMAGE_MAX_BYTES:
                return raw
            # Prefer a lossless WebP conversion when it fits the same budget.
            if source.format == "WEBP":
                output = io.BytesIO()
                source.save(output, format="PNG")
                if output.tell() <= _IMAGE_MAX_BYTES:
                    return output.getvalue()
        rgb = source.convert("RGB")
        try:
            scale = min(1.0, math.sqrt(_IMAGE_MAX_PIXELS / (rgb.width * rgb.height)))
            while True:
                size = (max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale)))
                with rgb.resize(size, Image.Resampling.LANCZOS) as resized:
                    output = io.BytesIO()
                    resized.save(output, format="JPEG", quality=90)
                if output.tell() <= _IMAGE_MAX_BYTES:
                    return output.getvalue()
                scale *= min(0.9, math.sqrt(_IMAGE_MAX_BYTES / output.tell()))
        finally:
            rgb.close()


async def condition_met(
    decisions: DecisionService, *, condition: str, observation: str,
    frame: dict, step_id: str, revision: int, timeout_s: float,
) -> bool:
    """Return the caption-only decision for callers evaluating written evidence."""

    answer = await judge_observation(
        decisions, condition=condition, observation=observation,
        frame=frame, step_id=step_id, revision=revision, timeout_s=timeout_s,
    )
    return answer.choice == "yes"


async def judge_observation(
    decisions: DecisionService, *, condition: str, observation: str,
    frame: dict, step_id: str, revision: int, timeout_s: float,
) -> DecisionAnswer:
    """Judge a description as evidence, without an image or prior decision."""

    if not observation.strip():
        raise ValueError("visual observation must not be empty")
    state = {"condition": condition, "current_observation": observation}
    # Persist evidence before the request so a decision failure cannot erase it.
    logger.debug(
        "Guide visual evidence step={} revision={} frame={} condition={!r} observation={!r}",
        step_id, revision, frame, condition, observation,
    )
    return await _judge(
        decisions, state=state, question=_QUESTION, images=(),
        frame=frame, step_id=step_id, revision=revision, timeout_s=timeout_s, source="caption",
    )


async def _judge(
    decisions: DecisionService, *, state: dict, question: DecisionQuestion, images: tuple[ImageInput, ...],
    frame: dict, step_id: str, revision: int, timeout_s: float, source: str,
) -> DecisionAnswer:
    with nemo_relay.scope.scope(
        "procedure-guide.visual-condition", nemo_relay.ScopeType.Agent,
        input=state, metadata={"frame": frame, "step_id": step_id, "revision": revision, "source": source},
    ) as scope:
        # Bound the whole operation, including clients that do not enforce the
        # transport timeout. Each decision stage has its own deadline.
        async with asyncio.timeout(timeout_s):
            response = await decisions.decide(
                state, {"condition_met": question}, **({"images": images} if images else {}), timeout=timeout_s,
            )
        answer = response.answers["condition_met"]
        if answer.choice not in question.criteria:
            raise ValueError(f"CLEF returned invalid guide condition result {answer.choice!r}")
        logger.debug(
            "Guide visual decision step={} revision={} frame={} source={} choice={} confidence={}",
            step_id, revision, frame, source, answer.choice, answer.confidence,
        )
        nemo_relay.scope.event(
            "procedure-guide.visual-condition-result", handle=scope,
            data={"source": source, "choice": answer.choice, "confidence": answer.confidence},
        )
        return answer
