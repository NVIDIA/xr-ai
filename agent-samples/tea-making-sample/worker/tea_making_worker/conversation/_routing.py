# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small state-scoped admissions following conversation-example's router contract."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

import nemo_relay
from xr_ai_models import DecisionQuestion, DecisionService
from xr_ai_routing import Admission, AgentTurn, DestinationResult, RoutingAid
from xr_ai_tools import ToolSet

from ..sop._engine import GuidanceAgent
from ._policy import ConversationPolicy, _Action, _ActionQuestion, _ToolDestinationPolicy


class _Execute(Protocol):
    async def __call__(
        self, turn: AgentTurn, action: _Action, started: Callable[[], Awaitable[None]], *, workflow: bool = False
    ) -> str: ...


async def _choose(
    decisions: DecisionService, question: _ActionQuestion, turn: AgentTurn, tools: ToolSet, state: dict[str, Any]
) -> _Action:
    eligible = {
        key: action
        for key, action in question.actions.items()
        if all(
            tools.get(name) is not None
            for name in ([action.tool] if action.tool else [])
            + ([action.observation.evaluation.function] if action.observation and action.observation.evaluation else [])
        )
    }
    if not eligible:
        return _Action(criterion="Unavailable", handoff=True)
    if len(eligible) == 1:
        return next(iter(eligible.values()))
    response = await decisions.decide(
        {
            "request": turn.query,
            "history": [{"request": item.user, "answer": item.assistant} for item in turn.history],
            **state,
        },
        {
            "action": DecisionQuestion(
                question.instructions, {key: action.criterion for key, action in eligible.items()}
            )
        },
    )
    choice = response.answers["action"].choice
    if choice not in eligible:
        raise ValueError("invalid destination action")
    nemo_relay.scope.event("tea.foreground.action", data={"choice": choice, "choices": list(eligible)})
    return eligible[choice]


async def _authorized(decisions, policy, action, turn, target):
    if not action.authorize:
        return True
    question = policy.authorization
    response = await decisions.decide(
        {
            "request": turn.query,
            "proposed_action": {
                "description": action.criterion,
                "operation": action.tool,
                "arguments": action.arguments,
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


@dataclass(frozen=True)
class _Pending:
    turn_id: str
    revision: int | None
    action: _Action
    session: object | None = None


class _WorkflowDestination:
    def __init__(
        self,
        guidance: GuidanceAgent,
        decisions: DecisionService,
        policy: ConversationPolicy,
        tools: Callable[[str], ToolSet],
        execute: _Execute,
        stale: Callable[[AgentTurn, Callable[[], Awaitable[None]]], Awaitable[None]],
    ) -> None:
        self._guidance, self._decisions, self._policy = guidance, decisions, policy
        self._tools, self._execute, self._stale = tools, execute, stale
        self._pending: dict[str, _Pending] = {}

    def routing_aid(self, participant_id: str) -> RoutingAid:
        session = self._guidance.store.find(participant_id)
        active = session is not None and session.active
        return RoutingAid(
            "workflow",
            self._policy.workflow_description,
            eligibility="active procedure" if active else "idle; explicit start required",
        )

    async def admit(self, turn: AgentTurn, *, focused: bool) -> Admission:
        self._pending.pop(turn.participant_id, None)
        session = self._guidance.store.get(turn.participant_id)
        revision = session.revision
        state: dict[str, Any] = {
            "active": session.active,
            "workflow": self._guidance.workflow.name,
            "destination": {"name": "workflow", "focused": focused},
        }
        tools = self._tools(turn.participant_id)
        if session.active and session.step_id is not None:
            step = self._guidance.workflow.step(session.step_id)
            state["step"] = {
                "title": step.title,
                "complete": step.is_complete(session.state),
            }
            state["known"] = self._guidance.workflow.project(step, session.state)
            action = await _choose(self._decisions, self._policy.active_question, turn, tools, state)
        else:
            action = await _choose(self._decisions, self._policy.idle_question, turn, tools, state)
        if action.handoff:
            return Admission("decline")
        if not await _authorized(self._decisions, self._policy, action, turn, self._guidance.workflow.name):
            return Admission("decline")
        if session.revision != revision and action.tool is not None:
            return Admission("accept")  # answer emits the configured stale-state message, without execution.
        self._pending[turn.participant_id] = _Pending(turn.turn_id, revision, action, session)
        return Admission("accept")

    async def answer(self, turn: AgentTurn, *, started: Callable[[], Awaitable[None]]) -> DestinationResult:
        pending = self._pending.pop(turn.participant_id, None)
        session = self._guidance.store.find(turn.participant_id)
        if (
            pending is None
            or pending.turn_id != turn.turn_id
            or session is None
            or session is not pending.session
            or (pending.action.tool is not None and session.revision != pending.revision)
        ):
            await self._stale(turn, started)
            text = ""
        else:
            text = await self._execute(turn, pending.action, started, workflow=True)
        # State owns procedure position; router focus owns only turn admission.
        session = self._guidance.store.find(turn.participant_id)
        return DestinationResult("focus" if session is not None and session.active else "release", text=text)

    async def release(self, participant_id: str) -> None:
        self._pending.pop(participant_id, None)


class _ToolDestination:
    def __init__(
        self,
        name: str,
        policy: _ToolDestinationPolicy,
        decisions: DecisionService,
        tools: Callable[[str], ToolSet],
        execute: _Execute,
        foreground_policy: ConversationPolicy,
    ) -> None:
        self._foreground_policy = foreground_policy
        self._name, self._policy, self._decisions = name, policy, decisions
        self._tools, self._execute = tools, execute
        self._pending: dict[str, _Pending] = {}

    def routing_aid(self, participant_id: str) -> RoutingAid:
        return RoutingAid(self._name, self._policy.description)

    async def admit(self, turn: AgentTurn, *, focused: bool) -> Admission:
        del focused
        self._pending.pop(turn.participant_id, None)
        action = await _choose(self._decisions, self._policy.question, turn, self._tools(turn.participant_id), {})
        if action.handoff or not await _authorized(
            self._decisions, self._foreground_policy, action, turn, self._policy.description
        ):
            return Admission("decline")
        self._pending[turn.participant_id] = _Pending(turn.turn_id, None, action)
        return Admission("accept")

    async def answer(self, turn: AgentTurn, *, started: Callable[[], Awaitable[None]]) -> DestinationResult:
        pending = self._pending.pop(turn.participant_id, None)
        if pending is None or pending.turn_id != turn.turn_id:
            return DestinationResult()
        text = await self._execute(turn, pending.action, started)
        return DestinationResult(text=text)

    async def release(self, participant_id: str) -> None:
        self._pending.pop(participant_id, None)


class _ConversationDestination:
    def __init__(self, policy: ConversationPolicy, execute: _Execute) -> None:
        self._policy, self._execute = policy, execute

    def routing_aid(self, participant_id: str) -> RoutingAid:
        return RoutingAid("conversation", self._policy.conversation_description)

    async def admit(self, turn: AgentTurn, *, focused: bool) -> Admission:
        return Admission("accept")

    async def answer(self, turn: AgentTurn, *, started: Callable[[], Awaitable[None]]) -> DestinationResult:
        text = await self._execute(turn, _Action(criterion="Generate an answer", generate=True), started)
        return DestinationResult(text=text)

    async def release(self, participant_id: str) -> None:
        pass


def _make_router(guidance, decisions, policy, workflow_tools, application_tools, execute, stale, output):
    """Reuse SDK focus, decline/reroute, cancellation, and participant cleanup."""
    from xr_ai_routing import AgentRouter

    destinations = {
        "workflow": _WorkflowDestination(guidance, decisions, policy, workflow_tools, execute, stale),
        "conversation": _ConversationDestination(policy, execute),
    }
    destinations.update(
        {
            name: _ToolDestination(name, item, decisions, application_tools, execute, policy)
            for name, item in policy.destinations.items()
        }
    )
    return AgentRouter(destinations, decisions, output=output, default_destination="conversation")
