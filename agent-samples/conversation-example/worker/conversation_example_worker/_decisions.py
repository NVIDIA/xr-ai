# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ask Clef one traced question and reject a choice outside its criteria."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import nemo_relay
from xr_ai_models import DecisionAnswer, DecisionQuestion, DecisionService, ImageInput


async def decide(
    decisions: DecisionService,
    scope: str,
    key: str,
    question: DecisionQuestion,
    state: Mapping[str, Any],
    *,
    participant_id: str,
    images: Sequence[ImageInput] = (),
) -> DecisionAnswer:
    """Ask ``question`` as ``key`` about ``state`` under the ``conversation-example.<scope>`` Relay scope.

    :raises ValueError: Clef returned a choice that is not one of the question's criteria.
    """

    with nemo_relay.scope.scope(
        f"conversation-example.{scope}",
        nemo_relay.ScopeType.Agent,
        input=dict(state),
        metadata={"participant_id": participant_id},
    ) as handle:
        # Text-only questions send no images argument.
        extra = {"images": images} if images else {}
        response = await decisions.decide(dict(state), {key: question}, **extra)
        answer = response.answers[key]
        nemo_relay.scope.event(
            f"conversation-example.{scope}-result",
            handle=handle,
            data={"choice": answer.choice, "confidence": answer.confidence},
        )
    if answer.choice not in question.criteria:
        raise ValueError(f"Clef returned invalid {key} choice {answer.choice!r}")
    return answer
