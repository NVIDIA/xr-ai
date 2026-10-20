# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise routing lifecycle and model boundaries with controlled interleavings."""

import asyncio

import pytest
from xr_ai_models import DecisionAnswer, DecisionResponse
from xr_ai_routing import Admission, AgentExchange, AgentRouter, AgentTurn, DestinationResult, RoutingAid


class _Destination:
    def __init__(self, name="conversation"):
        self.name = name
        self.eligible = True
        self.answers = []
        self.histories = []
        self.releases = []

    def routing_aid(self, _participant):
        return RoutingAid(self.name, self.name, eligible=self.eligible)

    async def admit(self, _turn, *, focused):
        return Admission("accept")

    async def answer(self, turn, *, emit):
        self.answers.append(turn.query)
        self.histories.append(turn.history)
        await emit(self.name, "answer")
        return DestinationResult("answer", "answer")

    async def release(self, participant):
        self.releases.append(participant)


async def _emit(_source, _text):
    pass


async def _schedule_checkpoint():
    checkpoint = asyncio.get_running_loop().create_future()
    asyncio.get_running_loop().call_soon(checkpoint.set_result, None)
    await checkpoint


@pytest.mark.parametrize("operation", ["replacement", "departure", "stop"])
async def test_interruption_keeps_cleanup_owned_across_concurrent_operations(operation):
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    finish_cleanup = asyncio.Event()
    cleanup_done = asyncio.Event()
    overlap = []

    class Destination(_Destination):
        async def answer(self, turn, *, emit):
            if turn.query == "old":
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleaning.set()
                    await finish_cleanup.wait()
                    cleanup_done.set()
            if not cleanup_done.is_set():
                overlap.append("answer")
            return await super().answer(turn, emit=emit)

        async def release(self, participant):
            if not cleanup_done.is_set():
                overlap.append("release")
            await super().release(participant)

    destination = Destination()
    router = AgentRouter({"conversation": destination}, None)
    old = asyncio.create_task(router.handle(AgentTurn("alice", "old", "old"), emit=_emit))
    await entered.wait()
    interrupt = asyncio.create_task(router.interrupt("alice"))
    await cleaning.wait()
    if operation == "replacement":
        followup = asyncio.create_task(router.handle(AgentTurn("alice", "new", "new"), emit=_emit))
    elif operation == "departure":
        followup = asyncio.create_task(router.participant_left("alice"))
    else:
        followup = asyncio.create_task(router.stop())
    try:
        await _schedule_checkpoint()
        await _schedule_checkpoint()
        premature = followup.done()
        still_owned = bool(router._tasks)
    finally:
        finish_cleanup.set()
        outcomes = await asyncio.gather(old, interrupt, followup, return_exceptions=True)
    assert not premature
    assert still_owned
    assert overlap == []
    assert isinstance(outcomes[0], asyncio.CancelledError)
    assert outcomes[1] is None
    if operation == "replacement":
        assert outcomes[2] == DestinationResult("answer", "answer")
        assert destination.answers == ["new"]
        assert destination.releases == []
    else:
        assert outcomes[2] is None
        assert destination.answers == []
        assert destination.releases == ["alice"]


async def test_repeated_interrupt_cancellation_does_not_cancel_child_cleanup_again():
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    finish_cleanup = asyncio.Event()
    cleanup_done = asyncio.Event()

    class Destination(_Destination):
        async def answer(self, turn, *, emit):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await finish_cleanup.wait()
                cleanup_done.set()

    router = AgentRouter({"conversation": Destination()}, None)
    old = asyncio.create_task(router.handle(AgentTurn("alice", "old", "old"), emit=_emit))
    await entered.wait()
    interrupt = asyncio.create_task(router.interrupt("alice"))
    await cleaning.wait()
    try:
        interrupt.cancel()
        await _schedule_checkpoint()
        interrupt.cancel()
        await _schedule_checkpoint()
        pending = not interrupt.done()
    finally:
        finish_cleanup.set()
        await asyncio.gather(old, interrupt, return_exceptions=True)
    assert pending
    assert cleanup_done.is_set()
    assert interrupt.cancelled()
    assert not router._tasks


async def test_cancelled_handle_keeps_its_response_owned_until_cleanup_finishes():
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    finish_cleanup = asyncio.Event()
    cleanup_done = asyncio.Event()

    class Destination(_Destination):
        async def answer(self, turn, *, emit):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await finish_cleanup.wait()
                cleanup_done.set()

    router = AgentRouter({"conversation": Destination()}, None)
    handle = asyncio.create_task(router.handle(AgentTurn("alice", "old", "old"), emit=_emit))
    await entered.wait()
    handle.cancel()
    await cleaning.wait()
    try:
        handle.cancel()
        await _schedule_checkpoint()
        pending = not handle.done()
        owned = bool(router._tasks)
    finally:
        finish_cleanup.set()
        await asyncio.gather(handle, return_exceptions=True)
    assert pending and owned
    assert cleanup_done.is_set()
    assert handle.cancelled()


async def test_interrupted_preamble_cannot_start_destination_answer():
    entered = asyncio.Event()
    finish_preamble = asyncio.Event()

    class Destination(_Destination):
        async def admit(self, _turn, *, focused):
            return Admission("accept", preamble="Starting")

    async def emit(_source, _text):
        entered.set()
        try:
            await finish_preamble.wait()
        except asyncio.CancelledError:
            await finish_preamble.wait()

    destination = Destination()
    router = AgentRouter({"conversation": destination}, None)
    handle = asyncio.create_task(router.handle(AgentTurn("alice", "turn", "query"), emit=emit))
    await entered.wait()
    interrupt = asyncio.create_task(router.interrupt("alice"))
    await _schedule_checkpoint()
    finish_preamble.set()
    await asyncio.gather(handle, interrupt, return_exceptions=True)
    assert destination.answers == []


async def test_eligibility_is_rechecked_after_model_inference():
    entered = asyncio.Event()
    finish = asyncio.Event()

    class Decisions:
        async def decide(self, _state, _questions):
            entered.set()
            await finish.wait()
            return DecisionResponse(
                {"destination": DecisionAnswer(
                    "specialist", 0.8, {"specialist": 0.8, "conversation": 0.1, "clarify": 0.1},
                )},
                "fake", {}, {},
            )

    specialist = _Destination("specialist")
    default = _Destination()
    router = AgentRouter({"conversation": default, "specialist": specialist}, Decisions())
    handle = asyncio.create_task(router.handle(AgentTurn("alice", "turn", "query"), emit=_emit))
    await entered.wait()
    specialist.eligible = False
    finish.set()
    assert (await handle).disposition == "clarify"
    assert specialist.answers == default.answers == []


async def test_failed_ineligible_focus_release_does_not_wedge_later_turns():
    class Decisions:
        async def decide(self, _state, _questions):
            choice = "specialist" if specialist.eligible else "conversation"
            return DecisionResponse(
                {"destination": DecisionAnswer(
                    choice, 1.0, {name: float(name == choice) for name in _questions["destination"].criteria},
                )},
                "fake", {}, {},
            )

    class Specialist(_Destination):
        async def answer(self, _turn, *, emit):
            return DestinationResult("focus", "focused")

        async def release(self, participant):
            raise RuntimeError("release failed")

    specialist = Specialist("specialist")
    default = _Destination()
    router = AgentRouter({"conversation": default, "specialist": specialist}, Decisions())
    # Establish focus through actual admission and answer, without private state mutation.
    default.eligible = False
    await router.handle(AgentTurn("alice", "one", "focus"), emit=_emit)
    specialist.eligible = False
    default.eligible = True
    with pytest.raises(RuntimeError, match="release failed"):
        await router.handle(AgentTurn("alice", "two", "release"), emit=_emit)
    assert (await router.handle(AgentTurn("alice", "three", "recover"), emit=_emit)).disposition == "answer"
    assert "specialist" in router._destinations_used["alice"]


async def test_stop_clears_history_only_participants_and_allows_fresh_reuse():
    destination = _Destination()
    destination.eligible = False
    router = AgentRouter({"conversation": destination}, None)
    await router.handle(AgentTurn("alice", "old", "unavailable"), emit=_emit)
    await router.stop()
    assert not router._history
    assert not router._history_times
    assert not router._history_exchange_ids
    assert not router._locks
    destination.eligible = True
    await router.handle(AgentTurn("alice", "new", "available"), emit=_emit)
    assert destination.histories == [()]


async def test_retained_history_is_bounded_and_an_empty_history_is_not_reseeded():
    destination = _Destination()
    router = AgentRouter({"conversation": destination}, None, context_budget_bytes=1)
    for index in range(3):
        await router.handle(
            AgentTurn("alice", str(index), "query", (AgentExchange("seed", "seed"),)), emit=_emit,
        )
        assert router._history["alice"] == []
        assert router._history_times["alice"] == []
        assert router._history_exchange_ids["alice"] == []
    assert destination.histories == [(), (), ()]


def test_admission_rejects_unknown_disposition_at_construction():
    with pytest.raises(ValueError, match="admission disposition"):
        Admission("reject")


@pytest.mark.parametrize("operation", ["departure", "stop"])
async def test_closed_admission_is_rechecked_after_replacement_drains_old_response(operation):
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    finish = asyncio.Event()

    class Destination(_Destination):
        async def answer(self, turn, *, emit):
            if turn.query == "old":
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleaning.set()
                    await finish.wait()
            return await super().answer(turn, emit=emit)

    destination = Destination()
    router = AgentRouter({"conversation": destination}, None)
    old = asyncio.create_task(router.handle(AgentTurn("alice", "old", "old"), emit=_emit))
    await entered.wait()
    replacement = asyncio.create_task(router.handle(AgentTurn("alice", "new", "new"), emit=_emit))
    await cleaning.wait()
    if operation == "departure":
        closing = asyncio.create_task(router.participant_left("alice"))
    else:
        closing = asyncio.create_task(router.stop())
    await _schedule_checkpoint()
    finish.set()
    outcomes = await asyncio.gather(old, replacement, closing, return_exceptions=True)
    assert isinstance(outcomes[1], RuntimeError)
    assert destination.answers == []
    assert not router._tasks
