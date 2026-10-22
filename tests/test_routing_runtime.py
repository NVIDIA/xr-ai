# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise AgentRouter through actual typed runtime delivery and direct output."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from contextlib import asynccontextmanager

import pytest
from xr_ai_models import DecisionAnswer, DecisionResponse
from xr_ai_routing import (
    Admission,
    AgentExchange,
    AgentRouter,
    AgentTurn,
    DestinationResult,
    RoutingAid,
    current_turn_context,
)
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, Topic, subscribe
from xr_ai_voice import UserQuery, VoiceInterrupted, VoiceOutput, VoiceParticipantJoined, VoiceParticipantLeft

QUERIES = Topic("routing-test.queries", UserQuery)
OUTPUT = Topic("routing-test.output", VoiceOutput, telemetry="none")
JOINED = Topic("routing-test.joined", VoiceParticipantJoined)
LEFT = Topic("routing-test.left", VoiceParticipantLeft)
INTERRUPTED = Topic("routing-test.interrupted", VoiceInterrupted)


class _Output(Agent):
    def __init__(self):
        super().__init__()
        self.items = []
        self.changed = asyncio.Event()

    @subscribe(OUTPUT)
    async def receive(self, value: VoiceOutput, ctx: RuntimeContext):
        self.items.append((value, ctx.metadata))
        self.changed.set()

    async def wait(self, count):
        while len(self.items) < count:
            self.changed.clear()
            await self.changed.wait()


async def _settled(router):
    """Wait until the router owns no turn, including its done-callback bookkeeping."""

    while router._tasks:
        await asyncio.wait(tuple(router._tasks.values()))
        await asyncio.sleep(0)


class _Destination:
    def __init__(self, runtime, name="conversation"):
        self.runtime = runtime
        self.name = name
        self.description = f"Explicit {name} capability"
        self.turns = []
        self.contexts = []
        self.aids = []
        self.admissions = deque()
        self.results = deque()
        self.departures = []
        self.started = asyncio.Event()
        self.finished = asyncio.Event()
        self.block = False
        self.cleanup_entered = asyncio.Event()
        self.cleanup_release = None
        self.release_failures = 0

    def routing_aid(self, participant_id):
        self.aids.append((participant_id, self.description))
        return RoutingAid(self.name, self.description)

    async def admit(self, turn, *, focused):
        return self.admissions.popleft() if self.admissions else Admission("accept")

    async def answer(self, turn, *, started):
        self.turns.append(turn)
        self.contexts.append(current_turn_context())
        self.started.set()
        try:
            if self.block:
                await asyncio.Event().wait()
            await started()
            await self.runtime.publish(
                OUTPUT,
                VoiceOutput(
                    text=f"{self.name}: {turn.query}", response_id=turn.turn_id, timestamp_us=turn.timestamp_us
                ),
                participant_id=turn.participant_id,
                source=self.name,
            )
            return self.results.popleft() if self.results else DestinationResult(text=f"context:{turn.query}")
        finally:
            self.finished.set()

    async def release(self, participant_id):
        self.departures.append(participant_id)
        self.cleanup_entered.set()
        if self.cleanup_release is not None:
            await self.cleanup_release.wait()
        if self.release_failures:
            self.release_failures -= 1
            raise ValueError("release failed")


class _Decisions:
    def __init__(self, *choices):
        self.choices = deque(choices)
        self.calls = []

    async def decide(self, state, questions, **_kwargs):
        self.calls.append((state, questions))
        key = next(iter(questions))
        choice = self.choices.popleft() if key == "destination" else "slow"
        return DecisionResponse(
            {key: DecisionAnswer(choice, 1.0, {choice: 1.0})},
            "fake",
            {},
            {},
        )


@asynccontextmanager
async def _app(destinations=None, decisions=None, **kwargs):
    runtime = AgentRuntime()
    output = runtime.register("output", _Output())
    if destinations is None:
        destinations = {"conversation": _Destination(runtime)}
    elif callable(destinations):
        destinations = destinations(runtime)
    router = AgentRouter(
        destinations,
        decisions,
        output=_control,
        query_topic=QUERIES,
        output_topic=OUTPUT,
        participant_joined_topic=JOINED,
        participant_left_topic=LEFT,
        interrupted_topic=INTERRUPTED,
        **kwargs,
    )
    runtime.register("router", router.runtime_subscriber())
    async with runtime:
        try:
            yield runtime, router, destinations, output
        finally:
            await router.stop()


async def _query(runtime, text, participant="alice"):
    await runtime.publish(QUERIES, UserQuery(text=text, timestamp_us=1234), participant_id=participant)


async def _control(_turn, _source, _text):
    return None


@pytest.mark.parametrize("started_first", [False, True])
async def test_runtime_turn_failure_publishes_at_most_one_failure_message(started_first):
    class Failing(_Destination):
        async def answer(self, turn, *, started):
            if started_first:
                await started()
            raise RuntimeError("destination failed")

    async with _app(lambda runtime: {"conversation": Failing(runtime)}) as (runtime, router, _, output):
        await _query(runtime, "request")
        await _settled(router)
        texts = [value.text for value, _ in output.items]
        assert texts == ([] if started_first else ["I couldn't complete that request. Please try again."])


async def test_direct_and_published_turns_each_run_in_one_owned_execution_task():
    entered, release = asyncio.Event(), asyncio.Event()
    answer_tasks = []

    class Decisions(_Decisions):
        async def decide(self, *args, **kwargs):
            if not self.calls:
                entered.set()
                await release.wait()
            return await super().decide(*args, **kwargs)

    class Destination(_Destination):
        async def answer(self, turn, *, started):
            answer_tasks.append(asyncio.current_task())
            return await super().answer(turn, started=started)

    async with _app(
        lambda runtime: {name: Destination(runtime, name) for name in ("conversation", "specialist")},
        Decisions("conversation", "conversation"),
    ) as (runtime, router, _destinations, _output):
        await asyncio.wait_for(_query(runtime, "published"), 1)
        await entered.wait()
        owned = router._tasks["alice"]
        assert not owned.done()
        release.set()
        await _settled(router)
        await router.handle(AgentTurn("alice", "direct", "direct"))
        assert answer_tasks[0] is owned
        assert all(task.get_name() == "routing-turn:alice" for task in answer_tasks)


@pytest.mark.parametrize("entry", ["direct", "published"])
async def test_query_queued_on_retired_lock_cannot_enter_rejoined_participant_lifecycle(entry):
    queued, resume = asyncio.Event(), asyncio.Event()

    class DelayedLock(asyncio.Lock):
        delay_next = True

        async def __aenter__(self):
            if self.delay_next:
                self.delay_next = False
                queued.set()
                await resume.wait()
            return await super().__aenter__()

    async with _app() as (runtime, router, destinations, _output):
        await _query(runtime, "first")
        await _settled(router)
        retired = router._locks["alice"] = DelayedLock()
        stale = asyncio.create_task(
            router.handle(AgentTurn("alice", "stale", "stale")) if entry == "direct" else _query(runtime, "stale")
        )
        await queued.wait()
        await runtime.publish(LEFT, VoiceParticipantLeft(), participant_id="alice")
        await runtime.publish(JOINED, VoiceParticipantJoined(), participant_id="alice")
        assert router._locks["alice"] is not retired
        resume.set()
        if entry == "direct":
            with pytest.raises(RuntimeError, match="lifecycle changed"):
                await stale
        else:
            await stale
        await _query(runtime, "fresh")
        await _settled(router)
        assert [turn.query for turn in destinations["conversation"].turns] == ["first", "fresh"]
        assert destinations["conversation"].turns[-1].history == ()


async def test_interrupted_route_failure_cannot_emit_runtime_fallback():
    entered = asyncio.Event()

    class Decisions(_Decisions):
        async def decide(self, *_args, **_kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise ValueError("selection failed after cancellation") from None

    async with _app(
        lambda runtime: {name: _Destination(runtime, name) for name in ("conversation", "specialist")},
        Decisions(),
    ) as (runtime, router, destinations, output):
        await _query(runtime, "old")
        await entered.wait()
        await router.interrupt("alice")
        assert output.items == []
        assert not any(destination.turns for destination in destinations.values())
        assert router._history["alice"] == []


async def test_typed_topics_direct_attribution_history_and_dynamic_routing_aids():
    decisions = _Decisions("specialist", "conversation")
    async with _app(
        lambda runtime: {name: _Destination(runtime, name) for name in ("conversation", "specialist")},
        decisions,
    ) as (runtime, router, destinations, output):
        specialist = destinations["specialist"]
        await _query(runtime, "check the camera")
        await _settled(router)
        specialist.description = "Explicit updated continuation context; camera task completed"
        await _query(runtime, "what did you find?")
        await _settled(router)
        conversation = destinations["conversation"]
        assert conversation.turns[0].history == (AgentExchange("check the camera", "context:check the camera"),)
        assert conversation.contexts[0].history_exchange_ids == (specialist.contexts[0].exchange_id,)
        assert conversation.contexts[0].history_timestamps[0] is not None
        assert "updated continuation context" in decisions.calls[1][1]["destination"].criteria["specialist"]
        assert [meta.source for _, meta in output.items] == ["specialist", "conversation"]
        assert all(
            value.response_id == turn.turn_id
            for (value, _), turn in zip(
                output.items,
                [specialist.turns[0], conversation.turns[0]],
                strict=True,
            )
        )
        assert conversation.turns[0].timestamp_us == 1234
        assert router._history["alice"][-1].assistant == "context:what did you find?"


async def test_explicit_focus_admission_release_and_finite_router_output():
    decisions = _Decisions("specialist", "conversation")
    async with _app(
        lambda runtime: {name: _Destination(runtime, name) for name in ("conversation", "specialist")},
        decisions,
    ) as (runtime, router, destinations, output):
        specialist = destinations["specialist"]
        specialist.results.append(DestinationResult("focus", "specialist context"))
        await _query(runtime, "start")
        await _settled(router)
        specialist.admissions.append(Admission("release", text="Task complete."))
        await _query(runtime, "tell me more")
        await _settled(router)
        assert len(decisions.calls) == 1
        control, metadata = output.items[1]
        assert metadata.source == "router"
        assert control.text == "Task complete." and control.final and control.response_id is None
        assert destinations["conversation"].turns[0].history[-1].assistant == "Task complete."
        assert specialist.departures == ["alice"]


async def test_supersession_interrupt_leave_rejoin_and_stop_drain_owned_work():
    async with _app() as (runtime, router, destinations, output):
        destination = destinations["conversation"]
        destination.block = True
        await _query(runtime, "old")
        await destination.started.wait()
        await runtime.publish(INTERRUPTED, VoiceInterrupted(), participant_id="alice")
        assert destination.finished.is_set() and output.items == []
        destination.block = False
        await _query(runtime, "new")
        await _settled(router)
        assert destination.turns[-1].history == ()
        await runtime.publish(LEFT, VoiceParticipantLeft(), participant_id="alice")
        await _query(runtime, "late")
        assert len(destination.turns) == 2
        with pytest.raises(RuntimeError, match="participant has left"):
            await router.handle(AgentTurn("alice", "late", "late"))
        await runtime.publish(JOINED, VoiceParticipantJoined(), participant_id="alice")
        await _query(runtime, "rejoined")
        await _settled(router)
        assert destination.turns[-1].history == ()
        await router.stop()
        await _query(runtime, "fresh lifecycle")
        await _settled(router)
        assert destination.turns[-1].history == ()


async def test_alice_cleanup_does_not_block_bob_query_or_interrupt():
    async with _app() as (runtime, router, destinations, _output):
        destination = destinations["conversation"]
        await _query(runtime, "alice request")
        await _settled(router)
        destination.cleanup_release = asyncio.Event()
        left = asyncio.create_task(runtime.publish(LEFT, VoiceParticipantLeft(), participant_id="alice"))
        await destination.cleanup_entered.wait()
        destination.started.clear()
        destination.finished.clear()
        destination.block = True
        await asyncio.wait_for(_query(runtime, "bob request", "bob"), 1)
        await destination.started.wait()
        await asyncio.wait_for(runtime.publish(INTERRUPTED, VoiceInterrupted(), participant_id="bob"), 1)
        assert destination.finished.is_set() and not left.done()
        destination.cleanup_release.set()
        await left


def test_runtime_topic_models_and_incomplete_configuration_rejected():
    destination = _Destination(None)
    with pytest.raises(ValueError, match="direct routing requires output"):
        AgentRouter({"conversation": destination}, None)
    with pytest.raises(ValueError, match="output_topic"):
        AgentRouter({"conversation": destination}, None, query_topic=QUERIES)
    with pytest.raises(ValueError, match="runtime options"):
        AgentRouter({"conversation": destination}, None, output_topic=OUTPUT)
    with pytest.raises(TypeError, match="UserQuery"):
        AgentRouter(
            {"conversation": destination}, None, query_topic=Topic("wrong-query", VoiceOutput), output_topic=OUTPUT
        )
    with pytest.raises(TypeError, match="VoiceOutput"):
        AgentRouter(
            {"conversation": destination}, None, query_topic=QUERIES, output_topic=Topic("wrong-output", UserQuery)
        )
    with pytest.raises(ValueError, match="distinct"):
        AgentRouter(
            {"conversation": destination}, None, query_topic=QUERIES, output_topic=Topic(QUERIES.name, VoiceOutput)
        )
    with pytest.raises(ValueError, match="participant_left_topic requires participant_joined_topic"):
        AgentRouter(
            {"conversation": destination}, None, query_topic=QUERIES, output_topic=OUTPUT, participant_left_topic=LEFT
        )
    with pytest.raises(ValueError, match="runtime_subscriber"):
        AgentRouter({"conversation": destination}, None, output=_control).runtime_subscriber()


async def test_runtime_control_uses_query_context_without_direct_output_callback():
    received = []

    class Input(Agent):
        @subscribe(QUERIES)
        async def receive(self, _value: UserQuery, ctx: RuntimeContext):
            received.append(ctx.metadata)

    runtime = AgentRuntime()
    runtime.register("input", Input())
    output = runtime.register("output", _Output())
    destination = _Destination(runtime)
    destination.admissions.append(Admission("clarify", text="Which view?"))
    router = AgentRouter({"conversation": destination}, None, query_topic=QUERIES, output_topic=OUTPUT)
    runtime.register("router", router.runtime_subscriber())
    async with runtime:
        try:
            with pytest.raises(ValueError, match="direct handle.*requires output"):
                await router.handle(AgentTurn("alice", "direct", "request"))
            assert router._tasks == {} and destination.turns == []
            await _query(runtime, "request")
            await _settled(router)
            assert len(output.items) == 1
            control, metadata = output.items[0]
            assert control.text == "Which view?" and control.final and control.response_id is None
            assert control.timestamp_us == 1234
            assert metadata.source == "router" and metadata.participant_id == "alice"
            assert metadata.parent_message_id == received[0].message_id
            assert metadata.correlation_id == received[0].correlation_id
            assert destination.turns == []
        finally:
            await router.stop()


@pytest.mark.parametrize("operation", ["departure", "stop"])
async def test_runtime_admission_closure_is_rechecked_after_direct_response_drain(operation):
    old_started, cleanup_entered, cleanup_release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    replacement_started = asyncio.Event()

    class Destination(_Destination):
        async def answer(self, turn, *, started):
            if turn.query == "old":
                old_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleanup_entered.set()
                    await cleanup_release.wait()
            replacement_started.set()
            return await super().answer(turn, started=started)

    async with _app(lambda runtime: {"conversation": Destination(runtime)}) as (_runtime, router, *_):
        old = asyncio.create_task(router.handle(AgentTurn("alice", "old", "old")))
        await old_started.wait()
        replacement = asyncio.create_task(router.handle(AgentTurn("alice", "replacement", "replacement")))
        await cleanup_entered.wait()
        closing = asyncio.create_task(router.stop() if operation == "stop" else router.participant_left("alice"))
        checkpoint = asyncio.Event()
        asyncio.get_running_loop().call_soon(checkpoint.set)
        await checkpoint.wait()
        try:
            with pytest.raises(RuntimeError):
                await router.handle(AgentTurn("alice", "late", "late"))
            cleanup_release.set()
            with pytest.raises(RuntimeError):
                await asyncio.wait_for(replacement, 1)
            assert not replacement_started.is_set()
        finally:
            cleanup_release.set()
            outcomes = await asyncio.gather(old, replacement, closing, return_exceptions=True)
        assert isinstance(outcomes[0], asyncio.CancelledError)
        assert outcomes[-1] is None


@pytest.mark.parametrize(
    "topic,event",
    [
        (QUERIES, UserQuery(text="recursive", timestamp_us=1)),
        (JOINED, VoiceParticipantJoined()),
        (LEFT, VoiceParticipantLeft()),
        (INTERRUPTED, VoiceInterrupted()),
    ],
)
async def test_router_owned_work_cannot_reenter_through_runtime_publication(topic, event):
    errors = []

    class Reentrant(_Destination):
        async def answer(self, turn, *, started):
            try:
                await self.runtime.publish(topic, event, participant_id=turn.participant_id)
            except BaseExceptionGroup as error:
                errors.append(error)
            return DestinationResult()

    async with _app(lambda runtime: {"conversation": Reentrant(runtime)}) as (runtime, router, *_):
        await _query(runtime, "request")
        await _settled(router)
        assert errors and errors[0].subgroup(RuntimeError) is not None


@pytest.mark.parametrize(
    "topic,event",
    [
        (JOINED, VoiceParticipantJoined()),
        (LEFT, VoiceParticipantLeft()),
        (QUERIES, UserQuery(text="late during cleanup", timestamp_us=1)),
    ],
)
async def test_events_delivered_during_stop_wait_for_it_and_start_no_turn(topic, event):
    async with _app() as (runtime, router, destinations, _output):
        destination = destinations["conversation"]
        await _query(runtime, "first")
        await _settled(router)
        destination.cleanup_release = asyncio.Event()
        stopping = asyncio.create_task(router.stop())
        await destination.cleanup_entered.wait()
        stale = asyncio.create_task(runtime.publish(topic, event, participant_id="alice"))
        checkpoint = asyncio.Event()
        asyncio.get_running_loop().call_soon(checkpoint.set)
        await checkpoint.wait()
        destination.cleanup_release.set()
        await asyncio.gather(stopping, stale)
        await _settled(router)
        assert len(destination.turns) == 1


async def test_rejoin_retries_failed_departure_and_readmits_participant():
    async with _app() as (runtime, router, destinations, _output):
        destination = destinations["conversation"]
        await _query(runtime, "first")
        await _settled(router)
        destination.release_failures = 1
        with pytest.raises(BaseExceptionGroup):
            await runtime.publish(LEFT, VoiceParticipantLeft(), participant_id="alice")
        await runtime.publish(JOINED, VoiceParticipantJoined(), participant_id="alice")
        await _query(runtime, "after rejoin")
        await _settled(router)
        assert [turn.query for turn in destination.turns] == ["first", "after rejoin"]
        assert destination.departures == ["alice", "alice"]


async def test_runtime_logs_failed_turns_and_dropped_queries(caplog):
    class Failing(_Destination):
        async def answer(self, turn, *, started):
            raise RuntimeError("destination failed")

    async with _app(lambda runtime: {"conversation": Failing(runtime)}) as (runtime, router, *_):
        with caplog.at_level(logging.WARNING, logger="xr_ai_routing._router"):
            await _query(runtime, "fails")
            await _settled(router)
            await runtime.publish(LEFT, VoiceParticipantLeft(), participant_id="alice")
            await _query(runtime, "dropped")
        messages = [record.getMessage() for record in caplog.records]
        assert any("runtime routing failed" in message for message in messages)
        assert any("runtime query dropped" in message and "participant has left" in message for message in messages)
