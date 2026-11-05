# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Judge whether visual evidence establishes a procedure step's condition.

Cosmos supplies observations; the typed CLEF service judges the written evidence.
This module keeps the decision policy separate from image questions and progress
updates. The guide backend owns frame freshness, confirmations, and transitions.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

import nemo_relay
from loguru import logger
from xr_ai_models import DecisionAnswer, DecisionQuestion, DecisionService, ImageInput

_INSTRUCTIONS = (Path(__file__).with_name("prompts") / "guide_condition.txt").read_text(encoding="utf-8").strip()
_QUESTION = DecisionQuestion(
    instructions=_INSTRUCTIONS,
    criteria={
        "accepted": "The observation establishes every required visible fact.",
        "rejected": "Visible evidence contradicts a required condition.",
        "unknown": "A required fact is missing, ambiguous, obscured, or not observable.",
    },
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
    return verdict(answer) == "accepted"


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
        verdict(answer)
        return answer


def verdict(answer: DecisionAnswer) -> str:
    """An affirmative plurality is insufficient when two choices mean do not pass."""
    expected = {"accepted", "rejected", "unknown"}
    if answer.choice not in expected or set(answer.probabilities) != expected:
        raise ValueError("invalid three-way evidence result")
    if any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
           or not 0 <= value <= 1 for value in answer.probabilities.values()):
        raise ValueError("invalid evidence probabilities")
    if answer.choice == "accepted" and answer.probabilities["accepted"] <= (
        answer.probabilities["rejected"] + answer.probabilities["unknown"]
    ):
        return "unknown"
    return answer.choice
