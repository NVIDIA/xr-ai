# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded semantic admissions; the SDK router owns routing and turn lifetime."""

from xr_ai_models import DecisionQuestion


async def _choose(decisions, question, turn, state):
    response = await decisions.decide(
        {"request": turn.query,
         "history": [{"request": item.user, "answer": item.assistant} for item in turn.history], **state},
        {"action": DecisionQuestion(question.instructions, question.criteria)},
    )
    choice = response.answers["action"].choice
    if choice not in question.criteria:
        raise ValueError("invalid control choice")
    return choice


async def _authorized(decisions, policy, turn, *, operation, description, arguments, target):
    question = policy.authorization
    response = await decisions.decide(
        {
            "request": turn.query,
            "proposed_action": {
                "description": description,
                "operation": operation,
                "arguments": arguments,
                "target": target,
            },
        },
        {"authorization": DecisionQuestion(question.instructions, question.criteria)},
    )
    answer = response.answers["authorization"]
    if answer.choice not in question.criteria:
        raise ValueError("invalid command authorization")
    # Unknown and negative are both non-authorizing outcomes. Splitting them
    # into two classes must not let a minority positive authorize a mutation.
    positive = answer.probabilities["command"]
    alternatives = sum(value for label, value in answer.probabilities.items() if label != "command")
    return answer.choice == "command" and positive > alternatives
