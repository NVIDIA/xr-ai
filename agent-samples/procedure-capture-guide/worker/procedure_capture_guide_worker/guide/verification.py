# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Judge whether visual evidence establishes a procedure step's condition.

The typed CLEF service judges either a selected image or a written observation.
This module keeps the decision policy separate from image questions and progress
updates. The guide backend owns frame freshness, confirmations, and transitions.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import nemo_relay
from loguru import logger
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
    return await _judge(
        decisions, state=state, question=_IMAGE_QUESTION, images=(image,),
        frame=frame, step_id=step_id, revision=revision, timeout_s=timeout_s, source="image",
    )


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
