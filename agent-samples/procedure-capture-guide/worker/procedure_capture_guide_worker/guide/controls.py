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

from xr_ai_models import DecisionQuestion, DecisionService

_CONTROL_QUESTION = DecisionQuestion(
    instructions="""Classify the current user's desired guide operation. State is context,
not authorization. Choose a control only for a present affirmative request to
change execution. Polite permission-seeking requests to proceed count. Merely
keeping guidance active or leaving the step unchanged is answer, not advance.
A prohibition on one operation never requests another. Quotations, reported
advice, hypotheticals, and questions about a control are answer unless the user
explicitly adopts that action now. Capture and other applications are not guide
controls. Interpret the complete utterance, not matching words.""",
    criteria={
        "advance": ("An instruction or polite request for permission to proceed to the next step now, "
                    "subject to verification."),
        "skip": "Request to bypass the current step's verification and move onward now.",
        "restart": "Request to start this selected guide, or begin it again from its first step now.",
        "reset": "Request to end or clear this guide run now, leaving the guide inactive.",
        "status": "Request to report the current execution status, without asking to proceed or change it.",
        "unsupported": "Request to capture, record, or operate another capability instead of this selected guide.",
        "answer": "Discuss, explain, keep the current execution unchanged, or request an unsupported action.",
    },
)
_AUTHORIZATION = DecisionQuestion(
    instructions="""Does the user affirmatively request the EXACT proposed state change now?
Judge the requested change, not whether its topic is mentioned. Maintaining an
ongoing activity does not request a transition to another step. Negating a change
does not authorize a different change. Starting/stopping capture is not starting/
stopping a guide. Reported advice authorizes nothing unless the user adopts it.
Polite requests to proceed or bypass count as requests to act. The operation's
readiness is enforced by code; do not refuse authorization merely because the
current step is incomplete. Unknown or ambiguous intent must not authorize.""",
    criteria={"command": ("A direct instruction from the current user to this assistant "
                          "to execute this exact operation now, including a polite request phrased as a question."),
              "not_command": ("A request for information or explanation rather than execution, "
                              "reading instructions aloud, quotation, "
                              "report of another person's advice or printed instructions, hypothetical, "
                              "prohibition, or request for a different operation or target."),
              "reported_instruction": ("The utterance reports words, advice, labels, or instructions from "
                                       "another source, without the user asking the assistant to carry them out."),
              "explanation": "The user asks to read, describe, or explain a guide or control, not execute it.",
              "unknown": "It is unclear whether this user asks this assistant to perform this exact action now."},
)
_OPERATIONS = {
    "advance": ("workflow__advance", {"skip": False}, "Leave the current step for the next, if verified."),
    "skip": ("workflow__advance", {"skip": True}, "Bypass this step and leave it without requiring verification."),
    "restart": ("workflow__restart", {}, "Discard prior progress and begin the selected guide at step one."),
    "reset": ("workflow__reset", {}, "End this guide run and make guidance inactive."),
}


async def classify_control(decisions: DecisionService, text: str, context: dict, *, timeout_s: float):
    """Return a fixed tool binding and diagnostic decisions, never model arguments."""
    async with asyncio.timeout(timeout_s):
        # Step labels and verification state are not intent. Including them can
        # make an incomplete step override a clear request to proceed or skip.
        response = await decisions.decide(
            {"request": text, "guide_active": context["active"]},
            {"action": _CONTROL_QUESTION}, timeout=timeout_s,
        )
    answer = response.answers["action"]
    if answer.choice not in _CONTROL_QUESTION.criteria:
        raise ValueError("invalid guide control classification")
    record = {"classification": asdict(answer)}
    if answer.choice in {"answer", "unsupported", "status"}:
        return None, record
    name, arguments, description = _OPERATIONS[answer.choice]
    async with asyncio.timeout(timeout_s):
        response = await decisions.decide(
            {"request": text, "proposed_action": {"description": description,
             "operation": name, "arguments": arguments, "target": "the selected procedure guide"}},
            {"authorization": _AUTHORIZATION}, timeout=timeout_s,
        )
    authorization = response.answers["authorization"]
    record["authorization"] = asdict(authorization)
    if (authorization.choice not in _AUTHORIZATION.criteria
            or set(authorization.probabilities) != set(_AUTHORIZATION.criteria)
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or not 0 <= v <= 1
                   for v in authorization.probabilities.values())):
        raise ValueError("invalid guide authorization probabilities")
    positive = authorization.probabilities["command"]
    other = sum(v for k, v in authorization.probabilities.items() if k != "command")
    if authorization.choice != "command" or positive <= other:
        return None, record
    return (name, dict(arguments)), record
