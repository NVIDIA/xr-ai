# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Judge whether visual evidence establishes a procedure step's condition.

Cosmos supplies observations; the existing typed CLEF service chooses yes or no.
This module keeps the decision policy separate from image questions and progress
updates. The guide backend owns frame freshness, confirmations, and transitions.
"""

from __future__ import annotations

from pathlib import Path

import nemo_relay
from loguru import logger
from xr_ai_models import DecisionQuestion, DecisionService

_INSTRUCTIONS = (Path(__file__).with_name("prompts") / "guide_condition.txt").read_text(encoding="utf-8").strip()
_QUESTION = DecisionQuestion(
    instructions=_INSTRUCTIONS,
    criteria={
        "yes": "The written observation establishes the required visible condition.",
        "no": "The observation contradicts the condition, lacks a required fact, or leaves it uncertain or obscured.",
    },
)


async def condition_met(
    decisions: DecisionService, *, condition: str, observation: str,
    frame: dict, step_id: str, revision: int, timeout_s: float,
) -> bool:
    # No image or conversation history reaches this decision. The written
    # observation is evidence to assess, never an instruction to execute.
    state = {"condition": condition, "current_observation": observation}
    # Persist evidence before the request so a decision failure cannot erase
    # the observation needed to diagnose a missed or incorrect completion.
    logger.debug(
        "Guide visual evidence step={} revision={} frame={} condition={!r} observation={!r}",
        step_id, revision, frame, condition, observation,
    )
    with nemo_relay.scope.scope(
        "procedure-guide.visual-condition", nemo_relay.ScopeType.Agent,
        input=state, metadata={"frame": frame, "step_id": step_id, "revision": revision},
    ) as scope:
        response = await decisions.decide(state, {"condition_met": _QUESTION}, timeout=timeout_s)
        answer = response.answers["condition_met"]
        if answer.choice not in _QUESTION.criteria:
            raise ValueError(f"CLEF returned invalid guide condition result {answer.choice!r}")
        logger.debug(
            "Guide visual decision step={} revision={} frame={} choice={} confidence={}",
            step_id, revision, frame, answer.choice, answer.confidence,
        )
        nemo_relay.scope.event(
            "procedure-guide.visual-condition-result", handle=scope,
            data={"choice": answer.choice, "confidence": answer.confidence},
        )
        return answer.choice == "yes"
