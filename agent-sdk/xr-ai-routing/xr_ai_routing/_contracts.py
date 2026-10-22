# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define routing contracts for imports within the ``xr_ai_routing`` package."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True, slots=True)
class RoutingTurnContext:
    """Read-only routing metadata captured from the current asynchronous context.

    The snapshot is stable after it is returned, including when the router
    later updates its private context for another completed exchange.
    Outside an active routed turn, the identifiers are ``None`` and the
    history metadata tuples are empty.
    """

    exchange_id: object | None
    """Identity of the current completed exchange, when one is active."""

    history_exchange_ids: tuple[object | None, ...]
    """Exchange identities aligned with the active bounded history."""

    history_timestamps: tuple[int | None, ...]
    """History timestamps aligned with the active history, in Unix microseconds."""


@dataclass(frozen=True, slots=True)
class AgentExchange:
    """One completed user request and destination-provided response context."""

    user: str
    """User request in the completed exchange."""

    assistant: str
    """Completed response context, excluding progress acknowledgments."""


@dataclass(frozen=True, slots=True)
class AgentTurn:
    """The current user turn and complete exchanges selected for this destination."""

    participant_id: str
    """Participant whose state and focus apply to this turn."""

    turn_id: str
    """Stable identity used to correlate streamed output and cancellation."""

    query: str
    """Unmodified current user request."""

    history: tuple[AgentExchange, ...] = ()
    """Newest whole completed exchanges that fit the router's byte budget."""

    timestamp_us: int | None = None
    """Originating input time in Unix microseconds, when supplied by the caller."""


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

    def __post_init__(self) -> None:
        if self.disposition not in {"accept", "decline", "clarify", "release"}:
            raise ValueError("admission disposition must be accept, decline, clarify, or release")


@dataclass(frozen=True, slots=True)
class DestinationResult:
    """Optional completed response context and the router's focus transition."""

    disposition: Literal["answer", "focus", "release", "clarify"] = "answer"
    """Destinations return answer, focus, or release; the router returns clarify before answer work.

    A post-answer ``release`` consumes this turn and clears focus. It does not
    reroute the request; use ``Admission("release")`` for that behavior.
    """

    text: str = ""
    """Optional completed response context to retain in history.

    This text is not delivered as output by the router. Destinations can omit
    it when no routing context is needed; progress acknowledgements are excluded.
    """


class RoutableAgent(Protocol):
    """An agent that owns its state and handles admitted or fallback turns.

    The router calls ``release()`` when it releases a destination during
    rerouting or clears its focus. On participant departure it calls an optional
    ``participant_left(participant_id)`` method when present, otherwise it calls
    ``release()``. Implement that optional hook when departure should clear
    durable participant state that route release retains; it is not required by
    this protocol.
    """

    def routing_aid(self, participant_id: str) -> RoutingAid:
        """Describe this participant's current eligibility and route scope."""

    async def admit(self, turn: AgentTurn, *, focused: bool) -> Admission:
        """Accept, decline, clarify, or release before producing spoken output."""

    async def answer(
        self,
        turn: AgentTurn,
        *,
        started: Callable[[], bool],
    ) -> DestinationResult:
        """Answer a turn after admission or as the configured default fallback.

        Deliver output directly and return answer, focus, or release. After selected
        destinations decline through the bounded routing attempts, the front
        end may call an eligible configured default once without ``admit()``,
        unless that default already declined the current turn;
        destinations must support that standalone answer path.

        Destinations decline or clarify in ``admit()``, before output or work begins.
        Returning ``DestinationResult("release")`` consumes the current request
        after the answer and clears focus; it does not reroute that request.
        Destination exceptions and task cancellation propagate to the caller.
        Call ``started()`` immediately before the first meaningful output or
        application action. It suppresses a pending acknowledgment, supplies
        the first-response timing sample, and returns whether the response
        leads the turn's output: ``False`` means router control text such as a
        preamble or acknowledgment already started this turn, so the response
        should follow it instead of interrupting it. Repeated calls return the
        same result, and calls after ``answer()`` returns have no effect.
        Completion and focus still use the return value, with optional response
        context in ``DestinationResult.text`` for the router's history.
        Destinations own output publication and stream finalization. A failure
        before ``started()`` is reported to the participant by the router;
        after it, the destination owns any failure output.
        Cancellation must drain their output and model work before returning.
        A routing ``clarify`` choice with usable scores selects the
        highest-scoring eligible destination through normal admission. Without
        eligible scores the router asks for clarification. After exhausted
        declines, the default is not retried if
        it already declined this turn, and
        clarification is also used when the configured default is unavailable.

        :param turn: Current request and bounded completed history.
        :param started: Response-start signal for this answer.
        """

    async def release(self, participant_id: str) -> None:
        """Handle route release; the router also uses it for departure fallback.

        This method runs when the router releases a destination during
        rerouting or clears its focus. It is also the departure fallback when
        no ``participant_left()`` hook is present. A destination can clear
        active resources here while retaining completed participant data; use
        the optional departure hook for stronger cleanup.
        """
