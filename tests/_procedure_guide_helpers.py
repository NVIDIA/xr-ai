# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provide scripted CLEF outcomes for offline procedure-runtime tests.

YES/NO markers are test inputs, not the production visual answer contract.
Model accuracy is exercised separately with descriptive observations and images.
"""

from unittest.mock import AsyncMock, Mock

from xr_ai_models import DecisionAnswer, DecisionResponse


def clef_response(choice="yes"):
    return DecisionResponse(
        answers={"condition_met": DecisionAnswer(choice, 0.9, {choice: 0.9, "no" if choice == "yes" else "yes": 0.1})},
        model="test-clef", usage={}, raw={},
    )


def stub_decisions():
    async def decide(state, _questions, **_kwargs):
        return clef_response("yes" if state["current_observation"] == "YES" else "no")

    return Mock(decide=AsyncMock(side_effect=decide), health=AsyncMock(return_value=True), close=AsyncMock())
