# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-scoped routing across conversation and application destinations."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Mapping
from functools import partial

import nemo_relay
from xr_ai_models import DecisionQuestion, DecisionService

from .contracts import (
    ConversationDestination,
    ConversationExchange,
    ConversationTurn,
    DestinationResult,
    Emit,
)


class ConversationFrontEnd:
    """Route a turn, check admission before speech, and retain explicit focus.

    A focused destination receives the participant's next turn first. It can
    answer, release focus, or decline and return control to ordinary routing.
    Route choices use the typed decision service; a declined route is excluded
    from one bounded reroute. Destinations own their model and application state.
    """

    def __init__(
        self,
        destinations: Mapping[str, ConversationDestination],
        decisions: DecisionService,
        *,
        default_destination: str = "conversation",
    ) -> None:
        if not destinations or default_destination not in destinations:
            raise ValueError("default destination must be registered")
        self._destinations = dict(destinations)
        self._decisions = decisions
        self._default_destination = default_destination
        self._focus: dict[str, str] = {}
        self._tasks: dict[str, asyncio.Task[DestinationResult]] = {}
        self._generation: dict[str, object] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._history: dict[str, deque[ConversationExchange]] = {}
        self._destinations_used: dict[str, set[str]] = {}

    async def handle(self, turn: ConversationTurn, *, emit: Emit) -> DestinationResult:
        """Route and answer one turn, emitting source-tagged text chunks."""

        participant_id = turn.participant_id
        lock = self._locks.setdefault(participant_id, asyncio.Lock())
        async with lock:
            await self.interrupt(participant_id)
            history = tuple(self._history.get(participant_id, ())) or turn.history
            bounded = tuple(ConversationExchange(item.user[:240], item.assistant[:240]) for item in history[-4:])
            current_turn = ConversationTurn(
                participant_id=turn.participant_id,
                turn_id=turn.turn_id,
                query=turn.query,
                history=bounded,
            )
            generation = object()
            self._generation[participant_id] = generation

            async def emit_if_current(source: str, text: str) -> None:
                if self._generation.get(participant_id) is generation:
                    await emit(source, text)

            task = asyncio.create_task(
                self._handle(current_turn, emit=emit_if_current, generation=generation),
                name=f"conversation-front-end:{participant_id}",
                context=nemo_relay.fork_asyncio_context(),
            )
            self._tasks[participant_id] = task
        try:
            return await task
        finally:
            if self._tasks.get(turn.participant_id) is task:
                self._tasks.pop(turn.participant_id, None)
            if self._generation.get(turn.participant_id) is generation:
                self._generation.pop(turn.participant_id, None)

    async def interrupt(self, participant_id: str | None = None) -> None:
        """Cancel active response work while retaining completed history and focus."""

        if participant_id is None:
            tasks = tuple(self._tasks.values())
            self._tasks.clear()
            self._generation.clear()
        else:
            task = self._tasks.pop(participant_id, None)
            self._generation.pop(participant_id, None)
            tasks = () if task is None else (task,)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self) -> None:
        """Cancel active work and release all destination-owned focus."""

        await self.interrupt()
        participants = self._destinations_used.keys() | self._focus.keys()
        for participant_id in tuple(participants):
            await self.participant_left(participant_id)

    async def _handle(self, turn: ConversationTurn, *, emit: Emit, generation: object) -> DestinationResult:

        focused = self._focus.get(turn.participant_id)
        candidates = [focused] if focused in self._destinations else list(self._destinations)
        if focused is not None:
            aid = self._destinations[focused].routing_aid(turn.participant_id)
            if not aid.eligible:
                await self._destinations[focused].release(turn.participant_id)
                if self._generation.get(turn.participant_id) is not generation:
                    raise asyncio.CancelledError()
                self._focus.pop(turn.participant_id, None)
                self._forget_destination(turn.participant_id, focused)
                candidates = [name for name in self._destinations if name != focused]
                focused = None
        for attempt in range(2):
            name = candidates[0] if focused and attempt == 0 else await self._select(turn, candidates)
            if self._generation.get(turn.participant_id) is not generation:
                raise asyncio.CancelledError()
            if name is None:
                text = "Which part would you like help with?"
                await emit(self._default_destination, text)
                return DestinationResult("clarify", text)
            destination = self._destinations[name]
            self._destinations_used.setdefault(turn.participant_id, set()).add(name)
            admission = await destination.admit(turn, focused=(name == focused))
            if self._generation.get(turn.participant_id) is not generation:
                raise asyncio.CancelledError()
            if admission.disposition == "release":
                await destination.release(turn.participant_id)
                if self._generation.get(turn.participant_id) is not generation:
                    raise asyncio.CancelledError()
                self._focus.pop(turn.participant_id, None)
                self._forget_destination(turn.participant_id, name)
                if admission.text:
                    await emit(name, admission.text)
                    self._remember(turn, admission.text)
                focused = None
                candidates = [item for item in self._destinations if item != name]
                if not candidates:
                    return DestinationResult("clarify", admission.text)
                continue
            if admission.disposition == "clarify":
                if admission.text:
                    await emit(name, admission.text)
                return DestinationResult("clarify", admission.text)
            if admission.disposition == "decline":
                await destination.release(turn.participant_id)
                if self._generation.get(turn.participant_id) is not generation:
                    raise asyncio.CancelledError()
                self._forget_destination(turn.participant_id, name)
                if name == focused:
                    self._focus.pop(turn.participant_id, None)
                    focused = None
                candidates = [item for item in candidates if item != name]
                if not candidates or attempt == 1:
                    text = admission.text or "Which part would you like help with?"
                    await emit(self._default_destination, text)
                    return DestinationResult("clarify", text)
                continue

            if admission.preamble:
                await emit(name, admission.preamble)
            result = await destination.answer(
                turn,
                emit=partial(_emit_from, emit, name),
            )
            if self._generation.get(turn.participant_id) is not generation:
                raise asyncio.CancelledError()
            if result.disposition not in {"answer", "focus", "release"}:
                raise ValueError("destinations must decline or clarify in admit(), before answer()")
            if result.disposition == "focus":
                self._focus[turn.participant_id] = name
            elif result.disposition == "release":
                await destination.release(turn.participant_id)
                if self._generation.get(turn.participant_id) is not generation:
                    raise asyncio.CancelledError()
                if self._focus.get(turn.participant_id) == name:
                    self._focus.pop(turn.participant_id, None)
                self._forget_destination(turn.participant_id, name)
            elif result.disposition == "answer" and self._focus.get(turn.participant_id) == name:
                self._focus.pop(turn.participant_id, None)
            if result.disposition in {"answer", "focus", "release"} and result.text.strip():
                self._remember(turn, result.text)
            return result
        raise AssertionError("bounded routing loop exhausted")

    async def participant_left(self, participant_id: str) -> None:
        """Release destination-owned state and history for one departed participant."""

        await self.interrupt(participant_id)
        used = self._destinations_used.pop(participant_id, set())
        focused = self._focus.pop(participant_id, None)
        if focused is not None:
            used.add(focused)
        self._history.pop(participant_id, None)
        self._locks.pop(participant_id, None)
        for name in used:
            await self._destinations[name].release(participant_id)

    def _forget_destination(self, participant_id: str, destination: str) -> None:
        used = self._destinations_used.get(participant_id)
        if used is not None:
            used.discard(destination)
            if not used:
                self._destinations_used.pop(participant_id, None)

    def _remember(self, turn: ConversationTurn, answer: str) -> None:
        history = self._history.setdefault(turn.participant_id, deque(maxlen=4))
        history.append(ConversationExchange(turn.query[:240], answer[:240]))

    async def _select(self, turn: ConversationTurn, candidates: list[str]) -> str | None:
        if not candidates:
            return None
        if len(candidates) == 1:
            name = candidates[0]
            aid = self._destinations[name].routing_aid(turn.participant_id)
            if aid.name != name:
                raise ValueError(f"routing aid name {aid.name!r} does not match destination {name!r}")
            return name if aid.eligible else None
        aids = []
        for name in candidates:
            aid = self._destinations[name].routing_aid(turn.participant_id)
            if aid.name != name:
                raise ValueError(f"routing aid name {aid.name!r} does not match destination {name!r}")
            aids.append(aid)
        criteria = {
            aid.name: (
                f"{aid.name} has routing description={aid.description!r}; "
                f"eligible={aid.eligible}; eligibility={aid.eligibility}; priority={aid.priority}."
            )
            for aid in aids
        }
        criteria["clarify"] = "The request cannot safely be assigned to one available destination."
        state = {
            "participant_id": turn.participant_id,
            "request": turn.query,
            "completed_history": [
                {"user": item.user[:240], "assistant": item.assistant[:240]} for item in turn.history[-4:]
            ],
            "destinations": [
                {
                    "name": aid.name,
                    "description": aid.description,
                    "eligible": aid.eligible,
                    "eligibility": aid.eligibility,
                    "priority": aid.priority,
                }
                for aid in aids
            ],
        }
        response = await self._decisions.decide(
            state,
            {
                "destination": DecisionQuestion(
                    instructions=(
                        "Choose the best eligible destination using its description and routing hints. "
                        "Descriptions are hints for routing, not access controls or guarantees that a "
                        "destination can enforce its application policy. Use priority only "
                        "to break ties after relevance; never select an ineligible destination. Choose "
                        "clarify when no eligible destination is a suitable match."
                    ),
                    criteria=criteria,
                )
            },
        )
        selected = response.answers["destination"].choice
        eligible = {aid.name for aid in aids if aid.eligible}
        return selected if selected in eligible else None


async def _emit_from(emit: Emit, destination: str, _source: str, text: str) -> None:
    await emit(destination, text)


__all__ = ["ConversationFrontEnd"]
