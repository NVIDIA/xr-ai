# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classify guide controls and authorize their exact native state transition.

CLEF interprets user intent; it never constructs tool names or arguments.
Informational answers receive read-only tools, while the backend still owns
readiness, revision checks, timers, and the effects of each authorized control.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import asdict

from xr_ai_models import DecisionService

from .policy import load_policy

_OPERATIONS = {
    "advance": ("workflow__advance", {"skip": False}, "Leave the current step for the next, if verified."),
    "skip": ("workflow__advance", {"skip": True}, "Bypass this step and leave it without requiring verification."),
    "restart": ("workflow__restart", {}, "Discard prior progress and begin the selected guide at step one."),
    "reset": ("workflow__reset", {}, "End this guide run and make guidance inactive."),
}


async def classify_control(decisions: DecisionService, text: str, context: dict, *, timeout_s: float, policy=None):
    """Return a fixed tool binding and diagnostic decisions, never model arguments."""
    policy = policy or load_policy()
    control_question = policy.control_question.decision()
    authorization_question = policy.authorization_question.decision()
    async with asyncio.timeout(timeout_s):
        # Step labels and verification state are not intent. Including them can
        # make an incomplete step override a clear request to proceed or skip.
        response = await decisions.decide(
            {"request": text, "guide_active": context["active"]},
            {"action": control_question}, timeout=timeout_s,
        )
    answer = response.answers["action"]
    if answer.choice not in control_question.criteria:
        raise ValueError("invalid guide control classification")
    record = {"classification": asdict(answer)}
    if answer.choice in {"answer", "unsupported", "status"}:
        return None, record
    name, arguments, description = _OPERATIONS[answer.choice]
    async with asyncio.timeout(timeout_s):
        response = await decisions.decide(
            {"request": text, "proposed_action": {"description": description,
             "operation": name, "arguments": arguments, "target": "the selected procedure guide"}},
            {"authorization": authorization_question}, timeout=timeout_s,
        )
    authorization = response.answers["authorization"]
    record["authorization"] = asdict(authorization)
    if (authorization.choice not in authorization_question.criteria
            or set(authorization.probabilities) != set(authorization_question.criteria)
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or not 0 <= v <= 1
                   for v in authorization.probabilities.values())):
        raise ValueError("invalid guide authorization probabilities")
    positive = authorization.probabilities["command"]
    other = sum(v for k, v in authorization.probabilities.items() if k != "command")
    if authorization.choice != "command" or positive <= other:
        return None, record
    return (name, dict(arguments)), record
