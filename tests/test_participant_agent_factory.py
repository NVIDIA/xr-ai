# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check scoped resource ownership through real routing and runtime lifecycles."""

from __future__ import annotations

import asyncio

import pytest
from xr_ai_interaction import ParticipantJoined, ParticipantLeft, ResponseOutput, UserQuery
from xr_ai_models import DecisionAnswer, DecisionResponse
from xr_ai_routing import Admission, AgentRouter, AgentTurn, DestinationResult, ParticipantAgentFactory, RoutingAid
from xr_ai_runtime import AgentRuntime, Topic

QUERIES = Topic("participant-factory.query", UserQuery)
OUTPUTS = Topic("participant-factory.output", ResponseOutput)
JOINED = Topic("participant-factory.joined", ParticipantJoined)
LEFT = Topic("participant-factory.left", ParticipantLeft)


class _Destination:
    """Own one background task independently of the routed answer lifetime."""

    def __init__(self, participant_id, *, name="conversation", models=None):
        self.participant_id = participant_id
        self.name = name
        self.models = models
        self.closed = 0
        self.released = 0
        self.answer_entered = asyncio.Event()
        self.answer_finished = asyncio.Event()
        self.answer_release = None
        self.background = None

    def routing_aid(self, participant_id):
        assert participant_id == self.participant_id
        return RoutingAid(self.name, "A participant-scoped test capability")

    async def admit(self, turn, *, focused):
        assert turn.participant_id == self.participant_id
        return Admission("accept")

    async def answer(self, turn, *, started):
        self.answer_entered.set()
        try:
            if self.answer_release is not None:
                await self.answer_release.wait()
            await started()
            return DestinationResult("focus", turn.query)
        finally:
            self.answer_finished.set()

    async def release(self, participant_id):
        assert participant_id == self.participant_id
        self.released += 1

    async def participant_left(self, participant_id):
        raise AssertionError("scoped departure must use its one explicit disposer")

    async def close(self):
        self.closed += 1
        if self.background is not None:
            self.background.cancel()
            await asyncio.gather(self.background, return_exceptions=True)


async def _output(_turn, _name, _text):
    """The destinations return context only; no transport is needed for these tests."""


def _router(destinations, *, initialize=None, cleanup=None, decisions=None):
    return AgentRouter(
        destinations,
        decisions,
        output=_output,
        query_topic=QUERIES,
        output_topic=OUTPUTS,
        participant_joined_topic=JOINED,
        participant_left_topic=LEFT,
        initialize_participant=initialize,
        cleanup_participant=cleanup,
    )


async def _turn(router, participant="alice", text="hello"):
    return await router.handle(AgentTurn(participant, text, text))


async def _wait(event):
    await asyncio.wait_for(event.wait(), timeout=2)


@pytest.mark.asyncio
async def test_empty_room_stop_does_not_allocate_participant_agents():
    async def create(_participant):
        raise AssertionError("stopping an empty room must not initialize destinations")

    router = _router({"conversation": ParticipantAgentFactory(create, _Destination.close)})
    await router.stop()
    await router.stop()


@pytest.mark.asyncio
async def test_participants_share_clients_but_not_work_or_lifetime():
    models = object()
    created = []

    async def create(participant):
        destination = _Destination(participant, models=models)
        destination.background = asyncio.create_task(asyncio.Event().wait())
        created.append(destination)
        return destination

    router = _router({"conversation": ParticipantAgentFactory(create, _Destination.close)})
    await asyncio.gather(_turn(router), _turn(router, "bob"))
    alice, bob = sorted(created, key=lambda item: item.participant_id)
    assert alice is not bob and alice.models is bob.models is models
    await router.participant_left("alice")
    assert alice.closed == 1 and alice.background.done()
    assert bob.closed == 0 and not bob.background.done()
    await _turn(router, "bob", "continue")
    assert len(created) == 2
    await router.stop()
    assert bob.closed == 1 and bob.background.done()


@pytest.mark.asyncio
async def test_join_initializes_all_routes_once_and_closes_unselected_routes():
    created = []
    closed = []

    def factory(name):
        async def create(participant):
            destination = _Destination(participant, name=name)
            created.append(destination)
            return destination

        async def close(destination):
            closed.append(destination.name)
            await destination.close()

        return ParticipantAgentFactory(create, close)

    router = _router({"conversation": factory("conversation"), "workflow": factory("workflow")})
    runtime = AgentRuntime()
    runtime.register("router", router.runtime_subscriber())
    async with runtime:
        await asyncio.gather(*(runtime.publish(JOINED, ParticipantJoined(), participant_id="alice") for _ in range(2)))
        assert len(created) == 2
        assert all(not destination.answer_entered.is_set() for destination in created)
        await runtime.publish(LEFT, ParticipantLeft(), participant_id="alice")
        assert closed == ["workflow", "conversation"]
        await router.stop()


def test_reusing_factory_descriptor_for_two_route_names_is_rejected():
    async def create(participant):
        return _Destination(participant)

    factory = ParticipantAgentFactory(create, _Destination.close)
    with pytest.raises(ValueError, match="multiple route names"):
        _router({"conversation": factory, "other": factory})


@pytest.mark.asyncio
async def test_factory_cannot_borrow_another_participants_instance():
    destination = _Destination("alice")

    async def create(_participant):
        return destination

    router = _router({"conversation": ParticipantAgentFactory(create, _Destination.close)})
    await _turn(router)
    with pytest.raises(ValueError, match="dedicated instances"):
        await _turn(router, "bob")
    assert destination.closed == 0
    await _turn(router, text="still valid")
    await router.stop()
    assert destination.closed == 1


@pytest.mark.asyncio
async def test_factory_cannot_claim_an_existing_shared_destination():
    destination = _Destination("alice")

    async def create(_participant):
        return destination

    router = _router(
        {
            "conversation": destination,
            "workflow": ParticipantAgentFactory(create, _Destination.close),
        }
    )
    with pytest.raises(ValueError, match="dedicated instances"):
        await _turn(router)
    assert destination.closed == 0
    await router.stop()


@pytest.mark.asyncio
async def test_turn_interruption_preserves_initialized_background_work():
    created = []

    async def create(participant):
        destination = _Destination(participant)
        destination.background = asyncio.create_task(asyncio.Event().wait())
        destination.answer_release = asyncio.Event()
        created.append(destination)
        return destination

    router = _router({"conversation": ParticipantAgentFactory(create, _Destination.close)})
    answer = asyncio.create_task(_turn(router))
    while not created:
        await asyncio.sleep(0)
    destination = created[0]
    await _wait(destination.answer_entered)
    await router.interrupt("alice")
    assert isinstance((await asyncio.gather(answer, return_exceptions=True))[0], asyncio.CancelledError)
    assert destination.answer_finished.is_set()
    assert destination.closed == 0 and not destination.background.done()
    destination.answer_release.set()
    await _turn(router, text="second")
    assert len(created) == 1
    await router.stop()
    assert destination.closed == 1 and destination.background.done()


@pytest.mark.asyncio
async def test_failed_scope_close_defers_frame_cleanup_and_blocks_rejoin_until_retry():
    created = []
    application_cleanup = []
    fail_close = True

    async def create(participant):
        destination = _Destination(participant)
        destination.background = asyncio.create_task(asyncio.Event().wait())
        created.append(destination)
        return destination

    async def close(destination):
        if fail_close:
            raise ValueError("background still owns frames")
        await destination.close()

    async def cleanup(participant):
        assert all(destination.background.done() for destination in created)
        application_cleanup.append(participant)

    router = _router({"conversation": ParticipantAgentFactory(create, close)}, cleanup=cleanup)
    runtime = AgentRuntime()
    runtime.register("router", router.runtime_subscriber())
    async with runtime:
        await runtime.publish(JOINED, ParticipantJoined(), participant_id="alice")
        with pytest.raises(ExceptionGroup):
            await router.participant_left("alice")
        assert not created[0].background.done()
        assert application_cleanup == []
        with pytest.raises(RuntimeError, match="leaving"):
            await _turn(router)
        with pytest.raises(ExceptionGroup):
            await runtime.publish(JOINED, ParticipantJoined(), participant_id="alice")
        assert len(created) == 1 and not application_cleanup
        fail_close = False
        await runtime.publish(JOINED, ParticipantJoined(), participant_id="alice")
        assert created[0].background.done() and created[0].closed == 1
        assert application_cleanup == ["alice"] and len(created) == 2
        await router.stop()


@pytest.mark.asyncio
async def test_failed_batch_rolls_back_then_retries_pending_application_cleanup_before_allocating():
    created = []
    cleanup_attempts = 0

    async def create(participant):
        destination = _Destination(participant)
        created.append(destination)
        return destination

    async def fail_create(_participant):
        raise ValueError("second factory failed")

    async def cleanup(_participant):
        nonlocal cleanup_attempts
        cleanup_attempts += 1
        if cleanup_attempts <= 2:
            raise ValueError("application cleanup failed")

    router = _router(
        {
            "conversation": ParticipantAgentFactory(create, _Destination.close),
            "other": ParticipantAgentFactory(fail_create, _Destination.close),
        },
        cleanup=cleanup,
    )
    with pytest.raises(ValueError, match="application cleanup failed"):
        await _turn(router)
    assert len(created) == 1 and created[0].closed == 1
    with pytest.raises(ValueError, match="application cleanup failed"):
        await _turn(router, text="retry")
    assert len(created) == 1 and created[0].closed == 1
    await router.stop()
    assert cleanup_attempts == 3
    assert not router._participant_agents


@pytest.mark.asyncio
async def test_failed_application_initializer_remains_cleanup_owned_before_any_factory_returns():
    allocated = []
    cleanup_attempts = 0

    async def initialize(participant):
        allocated.append(participant)
        raise ValueError("application initializer failed")

    async def create(_participant):
        raise AssertionError("factory must not run after application initialization failed")

    async def cleanup(_participant):
        nonlocal cleanup_attempts
        cleanup_attempts += 1
        if cleanup_attempts == 1:
            raise ValueError("cleanup not yet possible")
        allocated.clear()

    router = _router(
        {"conversation": ParticipantAgentFactory(create, _Destination.close)}, initialize=initialize, cleanup=cleanup
    )
    with pytest.raises(ValueError, match="cleanup not yet possible"):
        await _turn(router)
    assert allocated == ["alice"]
    await router.stop()
    assert allocated == [] and cleanup_attempts == 2


@pytest.mark.asyncio
async def test_departure_drains_late_factory_result_before_rejoining_same_participant():
    creation_entered = asyncio.Event()
    creation_cancelled = asyncio.Event()
    allow_creation = asyncio.Event()
    created = []
    cleanup = []

    async def create(participant):
        if not created:
            creation_entered.set()
            try:
                await allow_creation.wait()
            except asyncio.CancelledError:
                creation_cancelled.set()
                await allow_creation.wait()
        destination = _Destination(participant)
        created.append(destination)
        return destination

    async def cleanup_application(participant):
        assert all(destination.closed == 1 for destination in created)
        cleanup.append(participant)

    router = _router({"conversation": ParticipantAgentFactory(create, _Destination.close)}, cleanup=cleanup_application)
    runtime = AgentRuntime()
    runtime.register("router", router.runtime_subscriber())
    async with runtime:
        initial_join = asyncio.create_task(runtime.publish(JOINED, ParticipantJoined(), participant_id="alice"))
        await _wait(creation_entered)
        departure = asyncio.create_task(router.participant_left("alice"))
        await _wait(creation_cancelled)
        replacement_join = asyncio.create_task(runtime.publish(JOINED, ParticipantJoined(), participant_id="alice"))
        assert not departure.done() and not replacement_join.done() and not cleanup
        allow_creation.set()
        await asyncio.wait_for(asyncio.gather(initial_join, departure, replacement_join), timeout=2)
        assert len(created) == 2 and created[0].closed == 1 and created[1].closed == 0
        assert cleanup == ["alice"]
        await _turn(router, text="fresh")
        await router.stop()
        assert created[1].closed == 1


@pytest.mark.asyncio
async def test_other_scope_closes_when_one_fails_and_success_is_not_repeated():
    close_attempts = []
    fail_workflow = True
    cleanup_calls = []

    def factory(name):
        async def create(participant):
            return _Destination(participant, name=name)

        async def close(destination):
            close_attempts.append(destination.name)
            if destination.name == "workflow" and fail_workflow:
                raise ValueError("workflow close failed")
            await destination.close()

        return ParticipantAgentFactory(create, close)

    async def cleanup(participant):
        cleanup_calls.append(participant)

    router = _router({"conversation": factory("conversation"), "workflow": factory("workflow")}, cleanup=cleanup)
    runtime = AgentRuntime()
    runtime.register("router", router.runtime_subscriber())
    async with runtime:
        await runtime.publish(JOINED, ParticipantJoined(), participant_id="alice")
        with pytest.raises(ExceptionGroup):
            await router.participant_left("alice")
        assert close_attempts == ["workflow", "conversation"] and cleanup_calls == []
        fail_workflow = False
        await router.participant_left("alice")
        assert close_attempts == ["workflow", "conversation", "workflow"]
        assert cleanup_calls == ["alice"]
        await router.stop()


@pytest.mark.asyncio
async def test_cancelled_initializer_and_repeatedly_cancelled_departure_keep_application_cleanup_owned():
    allocated = []
    initializer_entered = asyncio.Event()
    cleanup_entered = asyncio.Event()
    allow_cleanup = asyncio.Event()

    async def initialize(participant):
        allocated.append(participant)
        initializer_entered.set()
        await asyncio.Event().wait()

    async def create(_participant):
        raise AssertionError("cancelled initialization must not advance to a factory")

    async def cleanup(_participant):
        cleanup_entered.set()
        await allow_cleanup.wait()
        allocated.clear()

    router = _router(
        {"conversation": ParticipantAgentFactory(create, _Destination.close)}, initialize=initialize, cleanup=cleanup
    )
    runtime = AgentRuntime()
    runtime.register("router", router.runtime_subscriber())
    async with runtime:
        join = asyncio.create_task(runtime.publish(JOINED, ParticipantJoined(), participant_id="alice"))
        await _wait(initializer_entered)
        departure = asyncio.create_task(router.participant_left("alice"))
        await _wait(cleanup_entered)
        departure.cancel()
        await asyncio.sleep(0)
        departure.cancel()
        assert not departure.done() and allocated == ["alice"]
        with pytest.raises(RuntimeError, match="leaving"):
            await _turn(router)
        allow_cleanup.set()
        outcomes = await asyncio.wait_for(asyncio.gather(join, departure, return_exceptions=True), timeout=2)
        assert isinstance(outcomes[1], asyncio.CancelledError)
        assert allocated == []
        await router.stop()


@pytest.mark.asyncio
async def test_scoped_and_room_shared_routes_keep_independent_routing_and_departure():
    choices = iter(("conversation", "workflow"))
    history = []
    shared_departures = []
    created = []

    class RoomAgent:
        def routing_aid(self, _participant):
            return RoutingAid("conversation", "Shared conversation")

        async def admit(self, _turn, *, focused):
            return Admission("accept")

        async def answer(self, turn, *, started):
            history.append(turn)
            await started()
            return DestinationResult("answer", "room answer")

        async def release(self, participant):
            shared_departures.append(participant)

    class Decisions:
        async def decide(self, _state, questions, **_kwargs):
            choice = next(choices)
            return DecisionResponse({"destination": DecisionAnswer(choice, 1.0, {choice: 1.0})}, "fake", {}, {})

    async def create(participant):
        destination = _Destination(participant, name="workflow")
        created.append(destination)
        return destination

    router = _router(
        {"conversation": RoomAgent(), "workflow": ParticipantAgentFactory(create, _Destination.close)},
        decisions=Decisions(),
    )
    await _turn(router, text="general question")
    await _turn(router, text="start workflow")
    assert len(history) == 1 and created[0].answer_entered.is_set()
    assert router._focus == {"alice": "workflow"}
    await router.participant_left("alice")
    assert created[0].closed == 1 and shared_departures == ["alice"]
    await router.stop()


@pytest.mark.asyncio
async def test_cancelled_disposer_is_reported_without_abandoning_other_routes_or_participants():
    closed = []
    cleaned = []
    created = []
    fail_alice = True

    def factory(name):
        async def create(participant):
            destination = _Destination(participant, name=name)
            destination.background = asyncio.create_task(asyncio.Event().wait())
            created.append(destination)
            return destination

        async def close(destination):
            if fail_alice and destination.participant_id == "alice" and destination.name == "workflow":
                raise asyncio.CancelledError("cancelled child did not establish cleanup")
            closed.append((destination.participant_id, destination.name))
            await destination.close()

        return ParticipantAgentFactory(create, close)

    async def cleanup(participant):
        cleaned.append(participant)

    router = _router({"conversation": factory("conversation"), "workflow": factory("workflow")}, cleanup=cleanup)
    runtime = AgentRuntime()
    runtime.register("router", router.runtime_subscriber())
    async with runtime:
        await runtime.publish(JOINED, ParticipantJoined(), participant_id="alice")
        await runtime.publish(JOINED, ParticipantJoined(), participant_id="bob")
        with pytest.raises(ExceptionGroup) as raised:
            await router.stop()
        failures = raised.value.exceptions[0].exceptions[0].exceptions
        assert any(isinstance(error.__cause__, asyncio.CancelledError) for error in failures)
        assert set(closed) == {("alice", "conversation"), ("bob", "conversation"), ("bob", "workflow")}
        assert cleaned == ["bob"]
        failed_owner = next(item for item in created if item.participant_id == "alice" and item.name == "workflow")
        assert router._participant_agents["alice"].instances["workflow"] is failed_owner
        assert not failed_owner.background.done()
        assert all(item.background.done() for item in created if item is not failed_owner)
        fail_alice = False
        await router.stop()
        assert closed.count(("alice", "conversation")) == 1
        assert closed.count(("alice", "workflow")) == 1
        assert cleaned == ["bob", "alice"]
        assert failed_owner.closed == 1 and failed_owner.background.done()
