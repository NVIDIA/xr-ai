# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interpret one tea interaction, apply native effects, and reuse SDK Conversation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from xr_ai_conversation import Conversation
from xr_ai_models import DecisionInputTooLargeError, DecisionQuestion, DecisionService
from xr_ai_routing import Admission, AgentTurn, DestinationResult, RoutingAid

from ..components.sop_run_recording import _attempt
from .tea_sop_agent import SOPAgent, _InteractionProposal


class _TeaInteraction:
    """One application entry; the runner owns work and Conversation owns dialogue."""

    def __init__(self, runner: SOPAgent, conversation: Conversation, decisions: DecisionService):
        self._runner = runner
        self._conversation = conversation
        self._decisions = decisions

    def routing_aid(self, _participant_id: str) -> RoutingAid:
        """Describe this whole application; no second router selects speech acts."""
        return RoutingAid("tea", self._runner.config.conversation_description)

    async def admit(self, _turn: AgentTurn, *, focused: bool) -> Admission:
        """Accept the application turn; semantic interpretation precedes any effect."""
        del focused
        return Admission("accept")

    async def answer(self, turn: AgentTurn, *, started: Callable[[], Awaitable[None]]) -> DestinationResult:
        """Record the real semantic turn, native effects, and cancellation boundary."""
        async with _attempt(
            turn.participant_id, "interaction", request_id=turn.turn_id,
            snapshot=lambda: self._runner.context_snapshot(turn.participant_id),
        ):
            return await self._answer(turn, started=started)

    async def _answer(self, turn: AgentTurn, *, started: Callable[[], Awaitable[None]]) -> DestinationResult:
        """Interpret user meaning; SOP speaks native effects and Conversation answers QA."""
        snapshot = self._runner._interaction_snapshot(turn.participant_id)
        if snapshot is None:
            return DestinationResult()
        config = self._runner.config
        referent = config.authorization_question
        questions = {"operation": config.control_question}
        if "instrument_unit" in snapshot.state and config.instrument_unit_question is not None:
            questions["instrument_unit"] = config.instrument_unit_question
        # Semantic action interpretation needs the latest exchange for anaphora,
        # not a second copy of all dialogue. Conversation still receives the
        # original turn/history unchanged and manages its own full context.
        previous = turn.history[-1] if turn.history else None
        state = {"sop": snapshot.state, "previous_exchange": (
            {"user": previous.user, "assistant": previous.assistant} if previous is not None else None
        )}
        operation_state = {"status": snapshot.state["status"], "current_step": snapshot.state["current_step"]}
        if "instrument_unit" in snapshot.state:
            operation_state["instrument_unit"] = snapshot.state["instrument_unit"]
        try:
            result = await self._decisions.decide(
                {"sop": operation_state, "previous_exchange": state["previous_exchange"], "request": turn.query}, questions
            )
            # A backend may finish after swallowing cancellation. Never move its
            # result into a fresh task or recreate departed participant context.
            if asyncio.current_task().cancelling():
                raise asyncio.CancelledError
            current = self._runner._interaction_snapshot(turn.participant_id)
            if current is None or current.session is not snapshot.session:
                return DestinationResult()
            operation = result.answers["operation"]
            requested_target = None
            if operation.choice != "none" and _InteractionProposal.supported(operation):
                # The proposed operation is not authority. Resolve its target only
                # if this exact action survives the user's final intended meaning.
                allowed_targets = {"assistant_speech", "physical_activity", "other", "not_requested", "unclear"}
                if operation.choice in {"start", "stop", "restart"}:
                    allowed_targets.add("workflow")
                else:
                    allowed_targets.update(snapshot.state["steps"])
                target_result = await self._decisions.decide(
                    {**state, "candidate_operation": {
                        "label": operation.choice, "meaning": config.control_question.criteria[operation.choice],
                    }, "request": turn.query},
                    {"referent": DecisionQuestion(
                        referent.instructions,
                        {label: criterion.format_map({"operation_criterion": config.control_question.criteria[operation.choice]})
                         for label, criterion in referent.criteria.items() if label in allowed_targets},
                    )},
                )
                if asyncio.current_task().cancelling():
                    raise asyncio.CancelledError
                requested_target = target_result.answers["referent"]
            proposal = _InteractionProposal(operation, requested_target, result.answers.get("instrument_unit"))
        except DecisionInputTooLargeError:
            # The backend's authoritative limit does not authorize an action.
            # Preserve the original user request/history for normal Conversation.
            proposal = None
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError
        current = self._runner._interaction_snapshot(turn.participant_id)
        if current is None or current.session is not snapshot.session:
            return DestinationResult()

        response_claimed = False

        async def started_once() -> None:
            nonlocal response_claimed
            if not response_claimed:
                await started()
                response_claimed = True

        native_reply = await self._runner._apply_interaction(snapshot, turn, proposal, started=started_once)
        if native_reply is not None:
            return await self._runner._reply_to_control(turn, native_reply, started=started_once)
        return await self._conversation.answer(turn, started=started_once)

    async def release(self, _participant_id: str) -> None:
        """Releasing a turn does not stop the runner or close borrowed Conversation."""
