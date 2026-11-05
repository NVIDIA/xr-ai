# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provide scripted CLEF outcomes for offline procedure-runtime tests.

YES/NO markers are test inputs, not the production visual answer contract.
Model accuracy is exercised separately with descriptive observations and images.
"""

from unittest.mock import AsyncMock, Mock

from xr_ai_models import DecisionAnswer, DecisionResponse


def clef_response(choice="yes", confidence=0.9):
    choice = {"yes": "accepted", "no": "rejected"}.get(choice, choice)
    probabilities = {"accepted": 0.0, "rejected": 0.0, "unknown": 0.0}
    probabilities[choice] = confidence
    probabilities["rejected" if choice == "accepted" else "accepted"] = 1 - confidence
    return DecisionResponse(
        answers={"condition_met": DecisionAnswer(choice, confidence, probabilities)},
        model="test-clef", usage={}, raw={},
    )


def stub_decisions():
    async def decide(state, _questions, **kwargs):
        if "action" in _questions or "authorization" in _questions:
            key = next(iter(_questions))
            controls = {
                "start guide": "restart", "stop guide": "reset", "please continue": "advance",
                "next": "advance", "skip": "skip", "status": "status", "what is my status?": "status",
            }
            choice = controls.get(state["request"].lower(), "answer") if key == "action" else "command"
            probabilities = {label: float(label == choice) for label in _questions[key].criteria}
            return DecisionResponse(answers={key: DecisionAnswer(choice, 1.0, probabilities)},
                                    model="scripted-controls", usage={}, raw={})
        if kwargs.get("images"):
            image = kwargs["images"][0]
            # Scripted backend tests use byte markers; real-camera harness tests
            # deliberately exercise the caption fallback with encoded images.
            if isinstance(image, bytes) and image.startswith((b"\x89PNG", b"\xff\xd8")):
                return clef_response("no", 0.75)
            return clef_response("yes" if image == b"YES" else "no")
        return clef_response("yes" if state["current_observation"] == "YES" else "no")

    return Mock(decide=AsyncMock(side_effect=decide), health=AsyncMock(return_value=True), close=AsyncMock())
