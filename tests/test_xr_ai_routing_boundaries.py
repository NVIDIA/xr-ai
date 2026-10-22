# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise routing lifecycle and model boundaries with controlled interleavings."""

import asyncio

import pytest
from _routing_test_output import emit_destination as _emit_destination
from _routing_test_output import route as _route
from xr_ai_models import ChatResponse, DecisionAnswer, DecisionResponse
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

    async def answer(self, turn):
        emit = _emit_destination
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
        async def answer(self, turn):
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
            return await super().answer(turn)

        async def release(self, participant):
            if not cleanup_done.is_set():
                overlap.append("release")
            await super().release(participant)

    destination = Destination()
    router = AgentRouter({"conversation": destination}, None)
    old = asyncio.create_task(_route(router, AgentTurn("alice", "old", "old"), emit=_emit))
    await entered.wait()
    interrupt = asyncio.create_task(router.interrupt("alice"))
    await cleaning.wait()
    if operation == "replacement":
        followup = asyncio.create_task(_route(router, AgentTurn("alice", "new", "new"), emit=_emit))
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
        async def answer(self, turn):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await finish_cleanup.wait()
                cleanup_done.set()

    router = AgentRouter({"conversation": Destination()}, None)
    old = asyncio.create_task(_route(router, AgentTurn("alice", "old", "old"), emit=_emit))
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
        async def answer(self, turn):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await finish_cleanup.wait()
                cleanup_done.set()

    router = AgentRouter({"conversation": Destination()}, None)
    handle = asyncio.create_task(_route(router, AgentTurn("alice", "old", "old"), emit=_emit))
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
    handle = asyncio.create_task(_route(router, AgentTurn("alice", "turn", "query"), emit=emit))
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
    handle = asyncio.create_task(_route(router, AgentTurn("alice", "turn", "query"), emit=_emit))
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
        async def answer(self, _turn):
            return DestinationResult("focus", "focused")

        async def release(self, participant):
            raise RuntimeError("release failed")

    specialist = Specialist("specialist")
    default = _Destination()
    router = AgentRouter({"conversation": default, "specialist": specialist}, Decisions())
    # Establish focus through actual admission and answer, without private state mutation.
    default.eligible = False
    await _route(router, AgentTurn("alice", "one", "focus"), emit=_emit)
    specialist.eligible = False
    default.eligible = True
    with pytest.raises(RuntimeError, match="release failed"):
        await _route(router, AgentTurn("alice", "two", "release"), emit=_emit)
    assert (await _route(router, AgentTurn("alice", "three", "recover"), emit=_emit)).disposition == "answer"
    assert "specialist" in router._destinations_used["alice"]


async def test_stop_clears_history_only_participants_and_allows_fresh_reuse():
    destination = _Destination()
    destination.eligible = False
    router = AgentRouter({"conversation": destination}, None)
    await _route(router, AgentTurn("alice", "old", "unavailable"), emit=_emit)
    await router.stop()
    assert not router._history
    assert not router._history_times
    assert not router._history_exchange_ids
    assert not router._locks
    destination.eligible = True
    await _route(router, AgentTurn("alice", "new", "available"), emit=_emit)
    assert destination.histories == [()]


async def test_retained_history_is_bounded_and_an_empty_history_is_not_reseeded():
    destination = _Destination()
    router = AgentRouter({"conversation": destination}, None, context_budget_bytes=1)
    for index in range(3):
        await _route(router,
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
        async def answer(self, turn):
            if turn.query == "old":
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleaning.set()
                    await finish.wait()
            return await super().answer(turn)

    destination = Destination()
    router = AgentRouter({"conversation": destination}, None)
    old = asyncio.create_task(_route(router, AgentTurn("alice", "old", "old"), emit=_emit))
    await entered.wait()
    replacement = asyncio.create_task(_route(router, AgentTurn("alice", "new", "new"), emit=_emit))
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


@pytest.mark.parametrize("operation", ["departure", "stop"])
async def test_canceled_cleanup_waiter_owns_the_complete_transition(operation):
    entered = asyncio.Event()
    response_cleaning = asyncio.Event()
    finish_response = asyncio.Event()
    destination_cleaning = asyncio.Event()
    finish_destination = asyncio.Event()

    class Destination(_Destination):
        async def answer(self, turn):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                response_cleaning.set()
                await finish_response.wait()

        async def release(self, participant):
            destination_cleaning.set()
            await finish_destination.wait()
            await super().release(participant)

    destination = Destination()
    router = AgentRouter({"conversation": destination}, None)
    response = asyncio.create_task(_route(router, AgentTurn("alice", "old", "old"), emit=_emit))
    await entered.wait()
    cleanup = asyncio.create_task(router.stop() if operation == "stop" else router.participant_left("alice"))
    await response_cleaning.wait()
    try:
        cleanup.cancel()
        await _schedule_checkpoint()
        cleanup.cancel()
        await _schedule_checkpoint()
        finish_response.set()
        await asyncio.wait_for(destination_cleaning.wait(), timeout=1)
        assert not cleanup.done()
        with pytest.raises(RuntimeError):
            await _route(router, AgentTurn("alice", "new", "new"), emit=_emit)
    finally:
        finish_response.set()
        finish_destination.set()
        outcomes = await asyncio.gather(response, cleanup, return_exceptions=True)
    assert isinstance(outcomes[1], asyncio.CancelledError)
    assert destination.releases == ["alice"]
    assert not router._history
    assert not router._tasks
    assert not router._locks


@pytest.mark.parametrize("operation", ["departure", "stop"])
async def test_concurrent_cleanup_waiters_join_one_destination_release(operation):
    cleaning = asyncio.Event()
    finish = asyncio.Event()

    class Destination(_Destination):
        async def release(self, participant):
            self.releases.append(participant)
            cleaning.set()
            await finish.wait()

    destination = Destination()
    router = AgentRouter({"conversation": destination}, None)
    await _route(router, AgentTurn("alice", "old", "old"), emit=_emit)
    first = asyncio.create_task(router.stop() if operation == "stop" else router.participant_left("alice"))
    await cleaning.wait()
    second = asyncio.create_task(router.stop() if operation == "stop" else router.participant_left("alice"))
    try:
        await _schedule_checkpoint()
        first.cancel()
        await _schedule_checkpoint()
        assert not first.done() and not second.done()
    finally:
        finish.set()
        outcomes = await asyncio.gather(first, second, return_exceptions=True)
    assert isinstance(outcomes[0], asyncio.CancelledError)
    assert outcomes[1] is None
    assert destination.releases == ["alice"]


@pytest.mark.parametrize("operation", ["departure", "stop"])
async def test_failed_cleanup_retains_admission_and_retries_only_owned_failures(operation):
    class Destination(_Destination):
        fail = True

        async def release(self, participant):
            self.releases.append(participant)
            if participant == "alice" and self.fail:
                raise RuntimeError("cleanup failed")

    destination = Destination()
    router = AgentRouter({"conversation": destination}, None)
    await _route(router, AgentTurn("alice", "old", "old"), emit=_emit)
    await _route(router, AgentTurn("bob", "old", "old"), emit=_emit)
    with pytest.raises(ExceptionGroup, match="failed to release"):
        await (router.stop() if operation == "stop" else router.participant_left("alice"))
    with pytest.raises(RuntimeError):
        await _route(router, AgentTurn("alice", "blocked", "blocked"), emit=_emit)
    if operation == "departure":
        assert (await _route(router, AgentTurn("bob", "new", "new"), emit=_emit)).disposition == "answer"
    else:
        with pytest.raises(RuntimeError, match="router is stopping"):
            await _route(router, AgentTurn("bob", "blocked", "blocked"), emit=_emit)
    destination.fail = False
    await (router.stop() if operation == "stop" else router.participant_left("alice"))
    assert destination.releases.count("alice") == 2
    if operation == "stop":
        assert destination.releases.count("bob") == 1
    assert (await _route(router, AgentTurn("alice", "reopened", "reopened"), emit=_emit)).disposition == "answer"


async def test_canceled_shutdown_preserves_a_cleanup_failure_as_its_cause():
    cleaning = asyncio.Event()
    finish = asyncio.Event()

    class Destination(_Destination):
        async def release(self, participant):
            cleaning.set()
            await finish.wait()
            raise RuntimeError("cleanup failed")

    router = AgentRouter({"conversation": Destination()}, None)
    await _route(router, AgentTurn("alice", "old", "old"), emit=_emit)
    shutdown = asyncio.create_task(router.stop())
    await cleaning.wait()
    shutdown.cancel()
    await _schedule_checkpoint()
    finish.set()
    with pytest.raises(asyncio.CancelledError) as cancellation:
        await shutdown
    assert isinstance(cancellation.value.__cause__, ExceptionGroup)
    with pytest.raises(RuntimeError, match="router is stopping"):
        await _route(router, AgentTurn("alice", "blocked", "blocked"), emit=_emit)


@pytest.mark.parametrize("operation", ["interrupt", "departure", "stop", "handle"])
@pytest.mark.parametrize("target", ["alice", "bob"])
async def test_destination_cannot_deadlock_by_awaiting_router_operations(operation, target):
    class Destination(_Destination):
        async def answer(self, turn):
            emit = _emit_destination
            if operation == "interrupt":
                await router.interrupt(target)
            elif operation == "departure":
                await router.participant_left(target)
            elif operation == "stop":
                await router.stop()
            else:
                await _route(router, AgentTurn(target, "nested", "nested"), emit=emit)

    router = AgentRouter({"conversation": Destination()}, None)
    with pytest.raises(RuntimeError, match="cannot await"):
        await asyncio.wait_for(_route(router, AgentTurn("alice", "turn", "query"), emit=_emit), timeout=1)


@pytest.mark.parametrize("child_callback", [False, True])
async def test_acknowledgment_callback_inherits_router_ownership(child_callback, caplog):
    finish_answer = asyncio.Event()
    callback_done = asyncio.Event()

    class Destination(_Destination):
        async def answer(self, turn):
            await finish_answer.wait()
            return await super().answer(turn)

    class Decisions:
        async def decide(self, _state, _questions):
            return DecisionResponse(
                {"first_answer_timing": DecisionAnswer(
                    "slow", 1.0, {"slow": 1.0, "within_delay": 0.0, "uncertain": 0.0},
                )},
                "fake", {}, {},
            )

    class Acknowledgment:
        async def chat(self, _messages, **_kwargs):
            return ChatResponse("Let me think through your question.", None, None, "stop", {})

    async def emit(_source, text):
        if text.startswith("Let me think"):
            try:
                if child_callback:
                    await asyncio.create_task(router.stop())
                else:
                    await router.stop()
            finally:
                callback_done.set()

    destination = Destination()
    router = AgentRouter(
        {"conversation": destination}, Decisions(),
        acknowledgment_llm=Acknowledgment(), acknowledgment_delay_secs=10,
    )
    response = asyncio.create_task(_route(router, AgentTurn("alice", "turn", "query"), emit=emit))
    try:
        await asyncio.wait_for(callback_done.wait(), timeout=1)
    finally:
        finish_answer.set()
    assert (await asyncio.wait_for(response, timeout=1)).disposition == "answer"
    failures = [record for record in caplog.records if "optional acknowledgment task failed" in record.message]
    assert len(failures) == 1
    assert isinstance(failures[0].exc_info[1], RuntimeError)
    assert "router-owned work cannot await" in str(failures[0].exc_info[1])
    assert destination.releases == []
    await router.stop()
    assert destination.releases == ["alice"]
