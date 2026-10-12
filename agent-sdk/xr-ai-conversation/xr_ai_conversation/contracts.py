# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed contracts shared by conversation destinations and their front end."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True, slots=True)
class ConversationExchange:
    """One completed user and assistant exchange."""

    user: str
    """User request in the completed exchange."""

    assistant: str
    """Completed destination answer, excluding progress acknowledgements."""


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """The current user turn and bounded, completed context for a destination."""

    participant_id: str
    """Participant whose state and focus apply to this turn."""

    turn_id: str
    """Stable identity used to correlate streamed output and cancellation."""

    query: str
    """Unmodified current user request."""

    history: tuple[ConversationExchange, ...] = ()
    """Recent completed exchanges supplied to routing and admission."""


@dataclass(frozen=True, slots=True)
class RoutingAid:
    """A destination's concise declared scope for route selection."""

    name: str
    """Stable route name used for attribution and decision labels."""

    scope: str
    """Requests this destination is equipped to handle."""

    exclusions: str = ""
    """Requests this destination must not claim."""

    eligible: bool = True
    """Whether routing may select this destination for the participant now."""

    eligibility: str = "available"
    """Current participant-specific eligibility state."""

    priority: int = 0
    """Tie-breaking preference applied after scope relevance."""


@dataclass(frozen=True, slots=True)
class Admission:
    """Destination decision made before it speaks or starts work."""

    disposition: Literal["accept", "decline", "clarify", "release"]
    """Whether the destination may begin work or should return control."""

    preamble: str = ""
    """Optional short acknowledgment spoken only after acceptance."""

    text: str = ""
    """Clarification or release message when no answer work is accepted."""


@dataclass(frozen=True, slots=True)
class DestinationResult:
    """Completion text for history and an optional foreground transition."""

    disposition: Literal["answer", "decline", "focus", "release", "clarify"] = "answer"
    """Outcome of this destination's admitted work."""

    text: str = ""
    """Completed answer to retain in history; progress acknowledgements are excluded."""


Emit = Callable[[str, str], Awaitable[None]]


class ConversationDestination(Protocol):
    """A destination that owns its state and can answer one admitted turn."""

    def routing_aid(self, participant_id: str) -> RoutingAid:
        """Describe this participant's current eligibility and route scope."""

    async def admit(self, turn: ConversationTurn, *, focused: bool) -> Admission:
        """Accept, decline, clarify, or release before producing spoken output."""

    async def answer(self, turn: ConversationTurn, *, emit: Emit) -> DestinationResult:
        """Answer an admitted turn, streaming each chunk with its source name."""

    async def release(self, participant_id: str) -> None:
        """Release destination-owned foreground resources for one participant."""


__all__ = [
    "Admission",
    "ConversationDestination",
    "ConversationExchange",
    "ConversationTurn",
    "DestinationResult",
    "Emit",
    "RoutingAid",
]
