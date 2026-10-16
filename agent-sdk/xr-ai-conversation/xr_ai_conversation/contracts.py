# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed contracts shared by conversation destinations and their front end."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class ConversationContextUpdate(BaseModel):
    """A named topic update supplied as background context to a conversation.

    The topic name identifies the source. Incremental updates are accumulated
    and may be compacted in the background; summaries replace that source's
    current background as published when it fits the configured byte budget. An
    oversized summary removes the existing section for that source; oversized
    incremental updates are dropped as whole records. When accumulated
    incremental records exceed the budget, the oldest whole records are removed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["incremental", "summary"]
    """Whether ``text`` adds source material or replaces its background."""

    text: str = Field(min_length=1)
    """At least one source-owned character; interpretation remains the publisher's responsibility."""


@dataclass(frozen=True, slots=True)
class ConversationExchange:
    """One completed user and assistant exchange."""

    user: str
    """User request in the completed exchange."""

    assistant: str
    """Completed user-visible reply, excluding progress acknowledgments."""


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """The current user turn and complete exchanges selected for this destination."""

    participant_id: str
    """Participant whose state and focus apply to this turn."""

    turn_id: str
    """Stable identity used to correlate streamed output and cancellation."""

    query: str
    """Unmodified current user request."""

    history: tuple[ConversationExchange, ...] = ()
    """Newest whole completed exchanges that fit the front-end byte budget."""


@dataclass(frozen=True, slots=True)
class RoutingAid:
    """Brief routing metadata, separate from the destination's full instructions."""

    name: str
    """Stable route name used for attribution and decision labels."""

    description: str
    """Concise description used as a route-selection hint, not an authorization rule."""

    priority: int = 0
    """Tie-breaking preference after request relevance."""

    eligible: bool = True
    """Whether this destination may be selected for this participant now."""

    eligibility: str = "available"
    """Current participant-specific eligibility state."""


@dataclass(frozen=True, slots=True)
class Admission:
    """Destination decision made before it speaks or starts work.

    ``accept`` permits work, ``decline`` returns control for another route,
    ``clarify`` ends routing with a question, and ``release`` clears focus and
    returns the same query to ordinary routing.
    """

    disposition: Literal["accept", "decline", "clarify", "release"]
    """Whether the destination may begin work or should return control."""

    preamble: str = ""
    """Optional short acknowledgment emitted only after acceptance."""

    text: str = ""
    """Optional text for clarification or release; release text precedes rerouting."""


@dataclass(frozen=True, slots=True)
class DestinationResult:
    """Completed answer plus the front end's post-answer focus transition."""

    disposition: Literal["answer", "focus", "release", "clarify"] = "answer"
    """Destinations return answer, focus, or release; the front end returns clarify before answer work.

    A post-answer ``release`` consumes this turn and clears focus. It does not
    reroute the request; use ``Admission("release")`` for that behavior.
    """

    text: str = ""
    """Completed answer to retain in history; progress acknowledgements are excluded."""


Emit = Callable[[str, str], Awaitable[None]]
"""Async callback that emits a text chunk tagged with its destination source."""


class ConversationDestination(Protocol):
    """A destination that owns its state and handles admitted or fallback turns."""

    def routing_aid(self, participant_id: str) -> RoutingAid:
        """Describe this participant's current eligibility and route scope."""

    async def admit(self, turn: ConversationTurn, *, focused: bool) -> Admission:
        """Accept, decline, clarify, or release before producing spoken output."""

    async def answer(self, turn: ConversationTurn, *, emit: Emit) -> DestinationResult:
        """Answer a turn after admission or as the configured default fallback.

        Stream chunks and return answer, focus, or release. After selected
        destinations decline through the bounded routing attempts, the front
        end may call an eligible configured default once without ``admit()``;
        destinations must support that standalone answer path.

        Destinations decline or clarify in ``admit()``, before output or work begins.
        Returning ``DestinationResult("release")`` consumes the current request
        after the answer and clears focus; it does not reroute that request.
        Destination exceptions and task cancellation propagate to the caller.
        The front end suppresses later chunks after interruption but cannot
        retract output already delivered to ``emit``.
        If routing returns ``clarify`` with usable scores for eligible
        destinations, the front end tries the highest-scoring candidate through
        normal admission. Missing or invalid eligible scores, or no eligible
        candidate, still produce a clarification. After exhausted declines,
        clarification is also used when the configured default is unavailable.
        """

    async def release(self, participant_id: str) -> None:
        """Release or clear destination-owned state and resources for one participant."""


__all__ = [
    "Admission",
    "ConversationDestination",
    "ConversationContextUpdate",
    "ConversationExchange",
    "ConversationTurn",
    "DestinationResult",
    "Emit",
    "RoutingAid",
]
