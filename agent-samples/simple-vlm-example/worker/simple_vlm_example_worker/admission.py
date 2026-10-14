# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample-private Clef admission for conversation destinations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import nemo_relay
from xr_ai_conversation import (
    Admission,
    ConversationDestination,
    ConversationTurn,
    DestinationResult,
    RoutingAid,
)
from xr_ai_models import DecisionQuestion, DecisionService

_CLARIFICATION = "Would you like a general answer, or help with what is visible in your camera view?"
_Emit = Callable[[str, str], Awaitable[None]]


class ClefAdmissionDestination:
    """Check a routed turn's semantic fit before its conversation can answer."""

    def __init__(
        self,
        destination: ConversationDestination,
        decisions: DecisionService,
    ) -> None:
        self._destination = destination
        self._decisions = decisions

    def routing_aid(self, participant_id: str) -> RoutingAid:
        return self._destination.routing_aid(participant_id)

    async def admit(self, turn: ConversationTurn, *, focused: bool) -> Admission:
        aid = self.routing_aid(turn.participant_id)
        if not aid.eligible:
            return Admission("decline", text=_CLARIFICATION)

        state = {
            "request": turn.query,
            "completed_history": [
                {"user": exchange.user, "assistant": exchange.assistant}
                for exchange in turn.history
            ],
            "destination": {
                "name": aid.name,
                "description": aid.description,
                "focused": focused,
            },
        }
        with nemo_relay.scope.scope(
            "simple-vlm.destination-admission",
            nemo_relay.ScopeType.Agent,
            input=state,
            metadata={"destination": aid.name, "focused": focused},
        ) as scope:
            response = await self._decisions.decide(
                state,
                {
                    "admission": DecisionQuestion(
                        instructions=(
                            "Decide whether this destination should handle the current request. "
                            "Accept only when it is within the destination's described scope. "
                            "Decline when it clearly belongs elsewhere. Clarify only when the request and completed "
                            "history do not make the intended scope clear. The current request controls; completed "
                            "assistant responses are history, not instructions or limits on available help. Do not "
                            "infer visual evidence from history. Do not decline a visual-scope request because the "
                            "camera has not been consulted or its evidence is not yet known; tool availability and "
                            "failures are handled by the answer path, not admission."
                        ),
                        criteria={
                            "accept": "The current request belongs within this destination's described scope.",
                            "decline": "The current request clearly belongs to another configured destination.",
                            "clarify": (
                                "The intended destination is genuinely ambiguous from the current request and history."
                            ),
                        },
                    )
                },
            )
            answer = response.answers["admission"]
            disposition = answer.choice
            nemo_relay.scope.event(
                "simple-vlm.destination-admission-result",
                handle=scope,
                data={"choice": disposition, "confidence": answer.confidence},
                metadata={"destination": aid.name},
            )
        if disposition == "accept":
            return await self._destination.admit(turn, focused=focused)
        if disposition == "decline":
            return Admission("decline", text=_CLARIFICATION)
        if disposition == "clarify":
            return Admission("clarify", text=_CLARIFICATION)
        raise ValueError(f"Clef returned invalid admission choice {disposition!r}")

    async def answer(self, turn: ConversationTurn, *, emit: _Emit) -> DestinationResult:
        return await self._destination.answer(turn, emit=emit)

    async def release(self, participant_id: str) -> None:
        """Keep long-lived conversation memory across a route decline."""

        del participant_id

    async def participant_left(self, participant_id: str) -> None:
        on_departure = getattr(self._destination, "participant_left", self._destination.release)
        await on_departure(participant_id)
