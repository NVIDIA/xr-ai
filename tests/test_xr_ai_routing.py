# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from collections import deque
from contextlib import contextmanager
from dataclasses import FrozenInstanceError

import pytest
import xr_ai_routing as routing_module
import xr_ai_routing._router as router_module
from _routing_test_output import destination_output as _destination_output
from _routing_test_output import emit_control as _emit_control
from _routing_test_output import route as _route
from xr_ai_models import ChatResponse, DecisionAnswer, DecisionResponse
from xr_ai_routing import (
    Admission,
    AgentExchange,
    AgentRouter,
    AgentTurn,
    DestinationResult,
    RoutableAgent,
    RoutingAid,
    RoutingTurnContext,
    current_turn_context,
)
from xr_ai_routing._history import (
    _active_exchange_id,
    _active_history_exchange_ids,
    _active_history_timestamps,
)
from xr_ai_routing._router import (
    _bounded_history_with_metadata,
    _is_valid_acknowledgment,
)


def test_routable_agent_is_the_only_destination_protocol_export() -> None:
    assert RoutableAgent.__name__ == "RoutableAgent"
    assert routing_module.__all__ == [
        "Admission",
        "AgentExchange",
        "AgentRouter",
        "AgentTurn",
        "DestinationResult",
        "RoutableAgent",
        "RoutingAid",
        "RoutingTurnContext",
        "current_turn_context",
    ]


@pytest.mark.asyncio
async def test_direct_response_signal_cancels_blocked_ack_without_emitting_text(monkeypatch):
    now = 0.0
    monkeypatch.setattr(router_module, "_monotonic", lambda: now)
    responded = asyncio.Event()
    respond = asyncio.Event()
    finish = asyncio.Event()
    notifications = []
    llm = _AcknowledgmentLLM(blocked=True)

    class Destination(_AckDestination):
        async def answer(self, _turn, *, started):
            await llm.started.wait()
            await respond.wait()
            notifications.extend((await started(), await started()))
            responded.set()
            await finish.wait()
            return DestinationResult("focus", "The application updated the scene.")

    destination = Destination()
    router = AgentRouter(
        {"destination": destination},
        _Prediction("slow", {"slow": 0.9, "within_delay": 0.05, "uncertain": 0.05}), output=_emit_control,
        default_destination="destination",
        acknowledgment_llm=llm,
        acknowledgment_delay_secs=10.0,
    )
    output = []
    task = asyncio.create_task(_ack_output(router, "Update the scene.", output))
    await asyncio.wait_for(llm.started.wait(), timeout=1)
    now = 2.0
    respond.set()
    await asyncio.wait_for(responded.wait(), timeout=1)
    await asyncio.wait_for(llm.cancelled.wait(), timeout=1)
    assert not task.done()
    assert output == []
    assert notifications == [True, True]
    now = 50.0
    finish.set()
    result = await asyncio.wait_for(task, timeout=1)
    assert result.disposition == "focus"
    assert router._focus == {"alice": "destination"}
    assert router._history["alice"] == [AgentExchange("Update the scene.", result.text)]
    assert router._first_answer_samples["alice"]["destination"] == [("Update the scene.", 2.0)]


@pytest.mark.parametrize("ending", ["completion", "failure", "interrupt", "replacement", "departure", "stop"])
@pytest.mark.asyncio
async def test_response_signal_from_old_answer_cannot_affect_later_turn(ending):
    signals = {}
    entered = {name: asyncio.Event() for name in ("old", "new")}
    finish = {name: asyncio.Event() for name in ("old", "new")}

    class Destination(_AckDestination):
        async def answer(self, turn, *, started):
            signals[turn.query] = started
            entered[turn.query].set()
            try:
                await finish[turn.query].wait()
            except asyncio.CancelledError:
                await started()
                raise
            if ending == "failure" and turn.query == "old":
                raise ValueError("destination failed")
            return DestinationResult()

    router = AgentRouter({"destination": Destination()}, None, output=_emit_control, default_destination="destination")
    output = []
    old = asyncio.create_task(_ack_output(router, "old", output))
    await asyncio.wait_for(entered["old"].wait(), timeout=1)
    if ending in {"completion", "failure"}:
        finish["old"].set()
        old_result = (await asyncio.gather(old, return_exceptions=True))[0]
        if ending == "failure":
            assert isinstance(old_result, ValueError)
        else:
            assert old_result == DestinationResult()
    elif ending != "replacement":
        operation = {
            "interrupt": lambda: router.interrupt("alice"),
            "departure": lambda: router.participant_left("alice"),
            "stop": router.stop,
        }[ending]
        await operation()
    new = asyncio.create_task(_ack_output(router, "new", output))
    await asyncio.wait_for(entered["new"].wait(), timeout=1)
    if ending not in {"completion", "failure"}:
        assert isinstance((await asyncio.gather(old, return_exceptions=True))[0], asyncio.CancelledError)
    assert await signals["old"]() is False
    assert await signals["new"]() is True
    finish["new"].set()
    await asyncio.wait_for(new, timeout=1)
    assert await signals["new"]() is False
    assert output == ([("destination", router_module._FAILURE_TEXT)] if ending == "failure" else [])
    assert router._history["alice"] == []
    assert router._focus == {}
    assert [query for query, _duration in router._first_answer_samples["alice"]["destination"]] == ["new"]


@pytest.mark.asyncio
async def test_direct_destination_output_keeps_its_source_and_explicit_progress():
    notifications = []

    class Destination(_AckDestination):
        async def answer(self, _turn, *, started):
            emit = _destination_output(started)
            await emit("wrong-source", " ")
            notifications.append(await started())
            await emit("wrong-source", "answer")
            notifications.append(await started())
            return DestinationResult("answer", "routing context")

    router = AgentRouter({"destination": Destination()}, None, output=_emit_control, default_destination="destination")
    output = []
    await _ack_output(router, "request", output)
    assert notifications == [True, True]
    assert output == [("wrong-source", " "), ("wrong-source", "answer")]
    assert router._history["alice"] == [AgentExchange("request", "routing context")]
    assert len(router._first_answer_samples["alice"]["destination"]) == 1


class _Prediction:
    def __init__(self, choice="uncertain", probabilities=None, *, error=None, blocked=False, cancel_error=None) -> None:
        self.choice = choice
        self.probabilities = probabilities or {"slow": 0.0, "within_delay": 0.0, "uncertain": 1.0}
        self.error = error
        self.blocked = blocked
        self.cancel_error = cancel_error
        self.calls = []
        self.called = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.finished = asyncio.Event()
        self.task: asyncio.Task | None = None
        self.cancel_blocked = False
        self.release_cancel = asyncio.Event()
        self.cancel_count = 0

    async def decide(self, state, questions, **_kwargs):
        self.task = asyncio.current_task()
        self.calls.append((state, questions))
        self.called.set()
        try:
            if self.blocked:
                await asyncio.Event().wait()
            if self.error is not None:
                raise self.error
            return DecisionResponse(
                answers={
                    "first_answer_timing": DecisionAnswer(
                        self.choice,
                        self.probabilities.get(self.choice, 0.0),
                        self.probabilities,
                    )
                },
                model="fake",
                usage={},
                raw={},
            )
        except asyncio.CancelledError as error:
            self.cancel_count += 1
            self.cancelled.set()
            if self.cancel_blocked:
                await self.release_cancel.wait()
            if self.cancel_error is not None:
                raise self.cancel_error from error
            raise
        finally:
            self.finished.set()


def test_current_turn_context_returns_immutable_snapshot_of_private_metadata():
    assert current_turn_context() == RoutingTurnContext(None, (), ())

    current_id = object()
    history_ids = (object(), None)
    timestamps = (1_800_000_000_000_000, None)
    current_token = _active_exchange_id.set(current_id)
    ids_token = _active_history_exchange_ids.set(history_ids)
    timestamps_token = _active_history_timestamps.set(timestamps)
    try:
        snapshot = current_turn_context()
        assert snapshot == RoutingTurnContext(current_id, history_ids, timestamps)
        _active_exchange_id.set(object())
        _active_history_exchange_ids.set(())
        _active_history_timestamps.set(())
        assert snapshot == RoutingTurnContext(current_id, history_ids, timestamps)
        with pytest.raises(FrozenInstanceError):
            snapshot.exchange_id = None  # type: ignore[misc]
    finally:
        _active_history_timestamps.reset(timestamps_token)
        _active_history_exchange_ids.reset(ids_token)
        _active_exchange_id.reset(current_token)


class _AcknowledgmentLLM:
    def __init__(self, content="Let me think through that.", *, blocked=False, error=None) -> None:
        self.content = content
        self.blocked = blocked
        self.error = error
        self.calls = []
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()

    async def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        self.started.set()
        try:
            if self.blocked:
                await self.release.wait()
            if self.error is not None:
                raise self.error
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return ChatResponse(self.content, None, None, "stop", {})


class _AckDestination:
    def __init__(self, *, preamble="", blocked=False, answer="Here is the answer.") -> None:
        self.preamble = preamble
        self.blocked = blocked
        self.answer_text = answer
        self.answer_started = asyncio.Event()
        self.answer_emit_started = asyncio.Event()
        self.release_answer = asyncio.Event()
        self.calls = []

    def routing_aid(self, _participant_id):
        return RoutingAid("destination", "Answers requests about the selected topic.")

    async def admit(self, _turn, *, focused):
        return Admission("accept", preamble=self.preamble)

    async def answer(self, turn, *, started):
        emit = _destination_output(started)
        self.calls.append(turn.query)
        self.answer_started.set()
        if self.blocked:
            await self.release_answer.wait()
        if self.answer_text:
            self.answer_emit_started.set()
            await emit("destination", self.answer_text)
        return DestinationResult("answer", self.answer_text)

    async def release(self, _participant_id):
        pass


async def _ack_output(front_end, query, output, *, participant="alice"):
    async def emit(source, text):
        output.append((source, text))

    return await _route(front_end,
        AgentTurn(participant, f"turn-{participant}-{query}", query),
        emit=emit,
    )


class _Decisions:
    def __init__(self, *choices: str | DecisionAnswer) -> None:
        self.choices = deque(choices)
        self.calls = []

    async def decide(self, state, questions, *, timeout=None):
        self.calls.append((state, questions, timeout))
        choice = self.choices.popleft()
        answer = choice if isinstance(choice, DecisionAnswer) else DecisionAnswer(
            choice, 1.0, {label: float(label == choice) for label in questions["destination"].criteria}
        )
        return DecisionResponse(
            answers={"destination": answer},
            model="fake",
            usage={},
            raw={},
        )

    async def health(self):
        return True

    async def close(self):
        pass


class _Destination:
    def __init__(
        self,
        name,
        *,
        admissions=(),
        results=(),
        eligibility="available",
        description=None,
        eligible=True,
    ) -> None:
        self.name = name
        self.admissions = deque(admissions)
        self.results = deque(results)
        self.eligibility = eligibility
        self.description = description or f"Requests for {name}"
        self.eligible = eligible
        self.calls = []
        self.releases = []
        self.answer_turns = []

    def routing_aid(self, _participant_id):
        return RoutingAid(
            self.name,
            self.description,
            eligible=self.eligible,
            eligibility=self.eligibility,
        )

    async def admit(self, turn, *, focused):
        self.calls.append(("admit", turn.query, focused))
        return self.admissions.popleft() if self.admissions else Admission("accept")

    async def answer(self, turn, *, started):
        emit = _destination_output(started)
        self.calls.append(("answer", turn.query))
        self.answer_turns.append(turn)
        result = self.results.popleft() if self.results else DestinationResult()
        if result.text:
            await emit(self.name, result.text)
        return result

    async def release(self, participant_id):
        self.releases.append(participant_id)


async def _run(front_end, *, participant="alice", query="turn", history=()):
    output = []

    async def emit(source, text):
        output.append((source, text))

    result = await _route(front_end,
        AgentTurn(participant, f"turn-{query}", query, history),
        emit=emit,
    )
    return result, output


@pytest.mark.asyncio
async def test_general_conversation_is_a_normal_route_between_stateless_app_turns():
    decisions = _Decisions("xr", "conversation", "xr")
    conversation = _Destination("conversation")
    xr = _Destination("xr")
    front_end = AgentRouter({"conversation": conversation, "xr": xr}, decisions, output=_emit_control)

    await _run(front_end, query="Add a cube.")
    await _run(front_end, query="What is a cube?")
    await _run(front_end, query="Delete the cube.")

    assert [call[0] for call in xr.calls] == ["admit", "answer", "admit", "answer"]
    assert [call[0] for call in conversation.calls] == ["admit", "answer"]
    assert len(decisions.calls) == 3


@pytest.mark.asyncio
async def test_explicit_focus_routes_cross_domain_turns_until_release_per_participant():
    decisions = _Decisions("tea", "conversation", "xr")
    conversation = _Destination("conversation")
    tea = _Destination(
        "tea",
        results=(
            DestinationResult("focus", "Guide started."),
            DestinationResult("focus"),
            DestinationResult("release", "Guide closed."),
        ),
    )
    xr = _Destination("xr")
    front_end = AgentRouter({"conversation": conversation, "tea": tea, "xr": xr}, decisions, output=_emit_control)

    await _run(front_end, query="Start the guide.")
    await _run(front_end, query="Add a cube.")
    await _run(front_end, query="Leave the guide.")
    await _run(front_end, query="What is a cube?")
    await _run(front_end, participant="bob", query="Add a cube.")

    assert [call[:2] for call in tea.calls if call[0] == "answer"] == [
        ("answer", "Start the guide."),
        ("answer", "Add a cube."),
        ("answer", "Leave the guide."),
    ]
    assert any(call[2] is True for call in tea.calls if call[0] == "admit")
    assert xr.calls == [
        ("admit", "Add a cube.", False),
        ("answer", "Add a cube."),
    ]
    assert conversation.calls[-2][1] == "What is a cube?"
    assert len(decisions.calls) == 3


@pytest.mark.asyncio
async def test_focused_destination_answer_retains_focus_until_explicit_release():
    decisions = _Decisions("tea", "conversation")
    conversation = _Destination("conversation")
    tea = _Destination(
        "tea",
        results=(
            DestinationResult("focus"),
            DestinationResult("answer", "Still guiding."),
            DestinationResult("release", "Guide closed."),
        ),
    )
    front_end = AgentRouter({"conversation": conversation, "tea": tea}, decisions, output=_emit_control)

    await _run(front_end, query="Start the guide.")
    await _run(front_end, query="What does this step mean?")
    await _run(front_end, query="Leave the guide.")
    await _run(front_end, query="What did we discuss?")

    assert tea.releases == ["alice"]
    assert ("answer", "What does this step mean?") in tea.calls
    assert ("answer", "Leave the guide.") in tea.calls
    assert conversation.calls[-2][1] == "What did we discuss?"
    assert len(decisions.calls) == 2


@pytest.mark.asyncio
async def test_declined_route_reroutes_once_and_preamble_precedes_attributed_result():
    decisions = _Decisions("tea", "xr")
    conversation = _Destination("conversation")
    tea = _Destination("tea", admissions=(Admission("decline"),))
    xr = _Destination(
        "xr",
        admissions=(Admission("accept", preamble="I will add it."),),
        results=(DestinationResult("answer", "Added."),),
    )
    front_end = AgentRouter({"conversation": conversation, "tea": tea, "xr": xr}, decisions, output=_emit_control)

    result, output = await _run(front_end, query="Add a blue cube.")

    assert result.disposition == "answer"
    assert output == [("xr", "I will add it."), ("xr", "Added.")]
    assert len(decisions.calls) == 2
    assert "tea" not in decisions.calls[-1][0]["destinations"]


@pytest.mark.asyncio
async def test_destination_cannot_decline_or_clarify_after_answer_work_starts():
    destination = _Destination("conversation", results=(DestinationResult("clarify", "too late"),))
    front_end = AgentRouter({"conversation": destination}, _Decisions(), output=_emit_control)

    with pytest.raises(ValueError, match="before answer"):
        await _run(front_end, query="Tell me about this.")


@pytest.mark.asyncio
async def test_participant_departure_releases_only_that_participants_focus():
    decisions = _Decisions("tea", "tea", "conversation")
    tea = _Destination(
        "tea",
        results=(DestinationResult("focus"), DestinationResult("focus")),
    )
    front_end = AgentRouter({"conversation": _Destination("conversation"), "tea": tea}, decisions, output=_emit_control)

    await _run(front_end, participant="alice", query="Start guide.")
    await _run(front_end, participant="bob", query="Start guide.")
    await front_end.participant_left("alice")
    await _run(front_end, participant="bob", query="Continue guide.")
    await _run(front_end, participant="alice", query="What is a cube?")

    assert tea.releases == ["alice"]
    assert ("admit", "Continue guide.", True) in tea.calls
    assert ("admit", "What is a cube?", True) not in tea.calls
    assert len(decisions.calls) == 3


@pytest.mark.asyncio
async def test_departure_and_stop_release_destinations_used_without_focus():
    decisions = _Decisions("conversation", "conversation")
    conversation = _Destination("conversation")
    front_end = AgentRouter({"conversation": conversation}, decisions, output=_emit_control)

    await _run(front_end, participant="alice", query="What is a cube?")
    await front_end.participant_left("alice")
    await _run(front_end, participant="bob", query="What is a sphere?")
    await front_end.stop()

    assert conversation.releases == ["alice", "bob"]


@pytest.mark.asyncio
async def test_departure_attempts_all_releases_and_retains_failures_for_retry():
    class _FailOnce(_Destination):
        def __init__(self, name):
            super().__init__(name)
            self.fail = True

        async def release(self, participant_id):
            self.releases.append(participant_id)
            if self.fail:
                raise RuntimeError(f"{self.name} release failed")

    decisions = _Decisions("tea", "xr")
    tea = _FailOnce("tea")
    xr = _FailOnce("xr")
    front_end = AgentRouter(
        {"conversation": _Destination("conversation"), "tea": tea, "xr": xr}, decisions, output=_emit_control,
    )
    await _run(front_end, query="Start the guide.")
    await _run(front_end, query="Add a cube.")

    with pytest.raises(ExceptionGroup, match="failed to release destinations") as error:
        await front_end.participant_left("alice")
    assert len(error.value.exceptions) == 2
    assert tea.releases == ["alice"]
    assert xr.releases == ["alice"]

    tea.fail = False
    xr.fail = False
    await front_end.participant_left("alice")
    assert tea.releases == ["alice", "alice"]
    assert xr.releases == ["alice", "alice"]


@pytest.mark.asyncio
async def test_departure_blocks_new_turns_until_destination_cleanup_completes():
    class _SlowDeparture(_Destination):
        def __init__(self, name):
            super().__init__(name)
            self.departure_started = asyncio.Event()
            self.finish_departure = asyncio.Event()

        async def participant_left(self, participant_id):
            self.releases.append(participant_id)
            self.departure_started.set()
            await self.finish_departure.wait()

    destination = _SlowDeparture("conversation")
    front_end = AgentRouter({"conversation": destination}, _Decisions(), output=_emit_control)
    await _run(front_end, query="before cleanup")
    departure = asyncio.create_task(front_end.participant_left("alice"))
    await destination.departure_started.wait()

    with pytest.raises(RuntimeError, match="participant is leaving"):
        await _run(front_end, query="during cleanup")

    destination.finish_departure.set()
    await departure
    await _run(front_end, query="after cleanup")
    await front_end.participant_left("alice")

    assert destination.releases == ["alice", "alice"]


@pytest.mark.asyncio
async def test_release_result_keeps_destination_for_optional_departure_hook():
    class _DurableDestination(_Destination):
        def __init__(self):
            super().__init__("app", results=(DestinationResult("release", "Done."),))
            self.departures = []

        async def participant_left(self, participant_id):
            self.departures.append(participant_id)

    destination = _DurableDestination()
    front_end = AgentRouter(
        {"conversation": _Destination("conversation"), "app": destination},
        _Decisions("app"), output=_emit_control,
    )

    await _run(front_end)
    await front_end.participant_left("alice")

    assert destination.releases == ["alice"]
    assert destination.departures == ["alice"]


@pytest.mark.asyncio
async def test_seeded_history_is_retained_with_unknown_timestamps():
    seed = (AgentExchange("Earlier question", "Earlier answer"),)
    destination = _Destination("conversation")
    front_end = AgentRouter({"conversation": destination}, _Decisions(), output=_emit_control)

    await _run(front_end, query="First turn", history=seed)
    await _run(front_end, query="Second turn")

    assert destination.answer_turns[0].history == seed
    assert destination.answer_turns[1].history[0] == seed[0]
    assert front_end._history_times["alice"][0] is None


@pytest.mark.asyncio
async def test_focused_decline_returns_to_ordinary_route_selection():
    tea = _Destination("tea", admissions=(Admission("accept"), Admission("decline")))
    tea.results.append(DestinationResult("focus"))
    visual = _Destination("visual", results=(DestinationResult("answer", "A blue cup."),))
    front_end = AgentRouter(
        {"conversation": _Destination("conversation"), "tea": tea, "visual": visual},
        _Decisions("tea", "visual"), output=_emit_control,
    )

    await _run(front_end, query="Start the guide")
    result, output = await _run(front_end, query="What color is the cup?")

    assert result == DestinationResult("answer", "A blue cup.")
    assert output == [("visual", "A blue cup.")]
    assert visual.calls == [("admit", "What color is the cup?", False), ("answer", "What color is the cup?")]


@pytest.mark.asyncio
async def test_repeated_admission_release_returns_existing_no_route_clarification():
    destinations = {
        "conversation": _Destination("conversation"),
        "first": _Destination("first", admissions=(Admission("release"),)),
        "second": _Destination("second", admissions=(Admission("release"),)),
        "third": _Destination("third"),
    }
    front_end = AgentRouter(
        destinations,
        _Decisions("first", "second"), output=_emit_control,
    )

    result, output = await _run(front_end)

    assert result == DestinationResult("clarify", "Which part would you like help with?")
    assert output == [("conversation", "Which part would you like help with?")]


@pytest.mark.asyncio
async def test_stop_attempts_release_for_every_participant_after_failure():
    class _AlwaysFail(_Destination):
        async def release(self, participant_id):
            self.releases.append(participant_id)
            raise RuntimeError(f"{participant_id} release failed")

    decisions = _Decisions("conversation", "conversation")
    conversation = _AlwaysFail("conversation")
    front_end = AgentRouter({"conversation": conversation}, decisions, output=_emit_control)
    await _run(front_end, participant="alice", query="A")
    await _run(front_end, participant="bob", query="B")

    with pytest.raises(ExceptionGroup, match="failed to release conversation destinations") as error:
        await front_end.stop()

    assert len(error.value.exceptions) == 2
    assert set(conversation.releases) == {"alice", "bob"}


@pytest.mark.asyncio
async def test_destination_error_after_preamble_reports_failure_and_propagates():
    class _BrokenAnswer(_Destination):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fail_once = True
            self.answer_histories = []

        async def answer(self, turn, *, started):
            emit = _destination_output(started)
            self.answer_histories.append(turn.history)
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("destination failed")
            await emit(self.name, "Recovered.")
            return DestinationResult("answer", "Recovered.")

    destination = _BrokenAnswer("xr", admissions=(Admission("accept", preamble="I will try."),))
    front_end = AgentRouter({"xr": destination}, _Decisions(), output=_emit_control, default_destination="xr")
    output = []

    async def emit(source, text):
        output.append((source, text))

    with pytest.raises(RuntimeError, match="destination failed"):
        await _route(front_end, AgentTurn("alice", "turn-1", "Do it."), emit=emit)
    failure = ("xr", router_module._FAILURE_TEXT)
    assert output == [("xr", "I will try."), failure]

    result = await _route(front_end, AgentTurn("alice", "turn-2", "Try again."), emit=emit)
    assert result.text == "Recovered."
    assert output == [("xr", "I will try."), failure, ("xr", "Recovered.")]
    assert destination.answer_histories == [(), ()]


@pytest.mark.asyncio
async def test_unresolved_or_invalid_route_clarifies_without_answering_default():
    for choice in ("clarify", "not-registered"):
        decisions = _Decisions(choice)
        conversation = _Destination("conversation", results=(DestinationResult("answer", "I can help."),))
        xr = _Destination("xr")
        front_end = AgentRouter({"conversation": conversation, "xr": xr}, decisions, output=_emit_control)

        result, output = await _run(
            front_end,
            query=(
                "Can you tell me how I can rearrange the tasks behind me so "
                "that I can climb on them and remove the lights"
            ),
        )

        assert result == DestinationResult("clarify", "Which part would you like help with?")
        assert output == [("conversation", "Which part would you like help with?")]
        assert conversation.calls == []
        assert xr.calls == []


@pytest.mark.asyncio
async def test_route_scope_records_choice_probabilities_and_eligible_candidates(monkeypatch):
    class _RouteDecisions(_Decisions):
        async def decide(self, state, questions, *, timeout=None):
            self.calls.append((state, questions, timeout))
            return DecisionResponse(
                answers={
                    "destination": DecisionAnswer(
                        "clarify",
                        0.61,
                        {"clarify": 0.61, "conversation": 0.27, "observation": 0.12},
                    )
                },
                model="fake",
                usage={},
                raw={},
            )

    events = []
    monkeypatch.setattr(
        "xr_ai_routing._router.nemo_relay.scope.event",
        lambda name, **kwargs: events.append((name, kwargs["data"])),
    )
    decisions = _RouteDecisions()
    conversation = _Destination("conversation", results=(DestinationResult("answer", "General answer."),))
    observation = _Destination("observation", description="Questions about the current camera view")
    front_end = AgentRouter({"conversation": conversation, "observation": observation}, decisions, output=_emit_control)

    result, output = await _run(front_end, query="How can I move the objects behind me?")

    assert result == DestinationResult("answer", "General answer.")
    assert output == [("conversation", "General answer.")]
    assert conversation.calls == [
        ("admit", "How can I move the objects behind me?", False),
        ("answer", "How can I move the objects behind me?"),
    ]
    assert observation.calls == []
    assert events == [
        (
            "conversation.destination-route-choice",
            {
                "router_choice": "clarify",
                "router_confidence": 0.61,
                "router_probabilities": {"clarify": 0.61, "conversation": 0.27, "observation": 0.12},
                "selected_destination": "conversation",
                "selected_destination_probability": 0.27,
                "selection_reason": "highest_scored_eligible_after_clarify",
                "eligible_destinations": ["conversation", "observation"],
            },
        )
    ]


@pytest.mark.asyncio
async def test_router_uses_explicit_eligible_visual_destination():
    decisions = _Decisions(
        DecisionAnswer(
            "environment",
            0.55,
            {"environment": 0.55, "clarify": 0.05, "observation": 0.25, "conversation": 0.15},
        )
    )
    conversation = _Destination("conversation", results=(DestinationResult("answer", "General."),))
    observation = _Destination("observation", results=(DestinationResult("answer", "Observed."),))
    environment = _Destination("environment", results=(DestinationResult("answer", "A grounded answer."),))
    front_end = AgentRouter(
        {"conversation": conversation, "observation": observation, "environment": environment},
        decisions, output=_emit_control,
    )

    result, output = await _run(
        front_end,
        query="Can we do the next step for making pasta?",
    )

    assert result == DestinationResult("answer", "A grounded answer.")
    assert output == [("environment", "A grounded answer.")]
    assert environment.calls == [
        ("admit", "Can we do the next step for making pasta?", False),
        ("answer", "Can we do the next step for making pasta?"),
    ]
    assert conversation.calls == []
    assert observation.calls == []


@pytest.mark.asyncio
async def test_router_uses_explicit_observation_for_current_camera_question():
    decisions = _Decisions(
        DecisionAnswer(
            "observation",
            0.75,
            {"conversation": 0.1, "observation": 0.75, "environment": 0.1, "clarify": 0.05},
        )
    )
    conversation = _Destination("conversation", results=(DestinationResult("answer", "General."),))
    observation = _Destination("observation", results=(DestinationResult("answer", "A blue cup."),))
    environment = _Destination("environment", results=(DestinationResult("answer", "A grounded answer."),))
    front_end = AgentRouter(
        {"conversation": conversation, "observation": observation, "environment": environment},
        decisions, output=_emit_control,
    )

    query = "What color is the cup in the current camera view?"
    result, output = await _run(front_end, query=query)

    assert result == DestinationResult("answer", "A blue cup.")
    assert output == [("observation", "A blue cup.")]
    assert observation.calls == [("admit", query, False), ("answer", query)]
    assert conversation.calls == []
    assert environment.calls == []


@pytest.mark.asyncio
async def test_uncertain_router_tie_uses_destination_registration_order():
    decisions = _Decisions(
        DecisionAnswer(
            "clarify",
            0.1,
            {"observation": 0.45, "clarify": 0.1, "conversation": 0.45},
        )
    )
    conversation = _Destination("conversation", results=(DestinationResult("answer", "General."),))
    observation = _Destination("observation", results=(DestinationResult("answer", "Observed."),))
    front_end = AgentRouter({"conversation": conversation, "observation": observation}, decisions, output=_emit_control)

    result, output = await _run(front_end, query="What is in front of me?")

    assert result == DestinationResult("answer", "General.")
    assert output == [("conversation", "General.")]
    assert conversation.calls == [
        ("admit", "What is in front of me?", False),
        ("answer", "What is in front of me?"),
    ]
    assert observation.calls == []


@pytest.mark.parametrize("choice", ["clarify", "conversation"])
async def test_uncertain_fallback_uses_only_eligible_routes_without_overriding_explicit_choice(choice):
    probabilities = {"unavailable": 0.5, "observation": 0.3, "conversation": 0.15, "clarify": 0.05}
    decisions = _Decisions(DecisionAnswer(choice, probabilities[choice], probabilities))
    conversation = _Destination("conversation", results=(DestinationResult("answer", "General."),))
    observation = _Destination("observation", results=(DestinationResult("answer", "Observed."),))
    unavailable = _Destination("unavailable", eligible=False)
    router = AgentRouter(
        {"conversation": conversation, "observation": observation, "unavailable": unavailable},
        decisions, output=_emit_control,
    )

    result, output = await _run(router, query="What is in the current view?")

    expected = "observation" if choice == "clarify" else "conversation"
    assert result == DestinationResult("answer", "Observed." if choice == "clarify" else "General.")
    assert output == [(expected, result.text)]
    assert unavailable.calls == []


@pytest.mark.asyncio
async def test_uncertain_router_with_no_eligible_scores_still_clarifies():
    decisions = _Decisions(DecisionAnswer("clarify", 0.2, {"clarify": 0.2, "conversation": 0.4, "specialist": 0.4}))
    conversation = _Destination("conversation", eligible=False)
    specialist = _Destination("specialist", eligible=False)
    front_end = AgentRouter({"conversation": conversation, "specialist": specialist}, decisions, output=_emit_control)

    result, output = await _run(front_end, query="Help me.")

    assert result == DestinationResult("clarify", "Which part would you like help with?")
    assert output == [("conversation", "Which part would you like help with?")]
    assert conversation.calls == []
    assert specialist.calls == []


@pytest.mark.asyncio
async def test_uncertain_router_with_zero_probability_mass_still_clarifies():
    decisions = _Decisions(DecisionAnswer("clarify", 1.0, {"clarify": 1.0, "conversation": 0.0, "observation": 0.0}))
    conversation = _Destination("conversation")
    observation = _Destination("observation")
    front_end = AgentRouter({"conversation": conversation, "observation": observation}, decisions, output=_emit_control)

    result, output = await _run(front_end, query="What is in the current view?")

    assert result == DestinationResult("clarify", "Which part would you like help with?")
    assert output == [("conversation", "Which part would you like help with?")]
    assert conversation.calls == []
    assert observation.calls == []


@pytest.mark.asyncio
async def test_route_prompt_exposes_unavailable_destination_eligibility():
    decisions = _Decisions("xr")
    conversation = _Destination("conversation", results=(DestinationResult("answer", "I can help generally."),))
    xr = _Destination("xr", eligibility="unavailable", eligible=False)
    front_end = AgentRouter({"conversation": conversation, "xr": xr}, decisions, output=_emit_control)

    result, output = await _run(front_end, query="Change the virtual scene.")

    assert result == DestinationResult("clarify", "Which part would you like help with?")
    assert output == [("conversation", "Which part would you like help with?")]
    routed_destinations = decisions.calls[0][0]["destinations"]
    assert next(item for item in routed_destinations if item["name"] == "xr")["eligibility"] == "unavailable"
    assert next(item for item in routed_destinations if item["name"] == "xr")["eligible"] is False
    assert xr.calls == []
    assert conversation.calls == []


@pytest.mark.asyncio
async def test_single_ineligible_destination_is_not_admitted():
    destination = _Destination("conversation", eligibility="not configured", eligible=False)
    front_end = AgentRouter({"conversation": destination}, _Decisions(), output=_emit_control)

    result, output = await _run(front_end, query="What is visible?")

    assert result.disposition == "clarify"
    assert output == [("conversation", "Which part would you like help with?")]
    assert destination.calls == []


@pytest.mark.asyncio
async def test_exhausted_declines_answer_once_with_default_and_shared_history():
    decisions = _Decisions(
        DecisionAnswer(
            "specialist",
            0.8,
            {"conversation": 0.1, "specialist": 0.8, "visual": 0.05, "clarify": 0.05},
        ),
        DecisionAnswer(
            "visual",
            0.85,
            {"conversation": 0.1, "visual": 0.85, "clarify": 0.05},
        ),
    )
    history = (AgentExchange("Earlier request", "Earlier response"),)
    default = _Destination(
        "conversation",
        admissions=(Admission("decline", text="must not be spoken"),),
        results=(DestinationResult("answer", "I can still help."),),
    )
    specialist = _Destination("specialist", admissions=(Admission("decline"),))
    visual = _Destination("visual", admissions=(Admission("decline"),))
    front_end = AgentRouter(
        {"conversation": default, "specialist": specialist, "visual": visual},
        decisions, output=_emit_control,
        default_destination="conversation",
    )

    result, output = await _run(front_end, query="Continue that.", history=history)

    assert result == DestinationResult("answer", "I can still help.")
    assert output == [("conversation", "I can still help.")]
    assert [call[0] for call in specialist.calls] == ["admit"]
    assert [call[0] for call in visual.calls] == ["admit"]
    assert [item["name"] for item in decisions.calls[0][0]["destinations"]] == [
        "conversation",
        "specialist",
        "visual",
    ]
    assert [item["name"] for item in decisions.calls[1][0]["destinations"]] == [
        "conversation",
        "visual",
    ]
    assert default.calls == [("answer", "Continue that.")]
    assert default.answer_turns[0].history == history
    assert len(decisions.calls) == 2


@pytest.mark.asyncio
async def test_ineligible_route_clarifies_instead_of_answering_default():
    decisions = _Decisions("specialist")
    default = _Destination("conversation", results=(DestinationResult("answer", "General help."),))
    specialist = _Destination("specialist", eligible=False)
    front_end = AgentRouter(
        {"conversation": default, "specialist": specialist},
        decisions, output=_emit_control,
        default_destination="conversation",
    )

    result, output = await _run(front_end, query="Help me.")

    assert result == DestinationResult("clarify", "Which part would you like help with?")
    assert output == [("conversation", "Which part would you like help with?")]
    assert default.calls == []
    assert specialist.calls == []


@pytest.mark.asyncio
async def test_already_declined_default_is_not_forced_to_answer(monkeypatch):
    default = _Destination(
        "conversation",
        admissions=(Admission("decline"),),
        results=(DestinationResult("answer", "General help."),),
    )
    specialist = _Destination("specialist", admissions=(Admission("decline"),))
    front_end = AgentRouter(
        {"conversation": default, "specialist": specialist},
        _Decisions("specialist", "conversation"), output=_emit_control,
        default_destination="conversation",
    )
    reasons = []
    monkeypatch.setattr(
        front_end,
        "_trace_default_fallback",
        lambda _turn, reason, *, eligible: reasons.append((reason, eligible)),
    )

    result, _output = await _run(front_end, query="Help me.")

    assert result.disposition == "clarify"
    assert not default.answer_turns
    assert reasons == [("exhausted_declines", True)]


@pytest.mark.asyncio
async def test_exhausted_declines_clarify_when_default_is_unavailable():
    specialist = _Destination("specialist", admissions=(Admission("decline"),))
    default = _Destination("conversation", eligible=False)
    front_end = AgentRouter(
        {"conversation": default, "specialist": specialist},
        _Decisions("specialist"), output=_emit_control,
        default_destination="conversation",
    )

    result, output = await _run(front_end, query="Help me.")

    assert result.disposition == "clarify"
    assert output == [("conversation", "Which part would you like help with?")]
    assert specialist.releases == ["alice"]
    assert default.calls == []


@pytest.mark.asyncio
async def test_destination_admission_clarification_does_not_trigger_default_answer():
    default = _Destination("conversation", results=(DestinationResult("answer", "Should not run."),))
    specialist = _Destination("specialist", admissions=(Admission("clarify", text="Which one?"),))
    front_end = AgentRouter(
        {"conversation": default, "specialist": specialist},
        _Decisions("specialist"), output=_emit_control,
        default_destination="conversation",
    )

    result, output = await _run(front_end, query="Please handle that.")

    assert result == DestinationResult("clarify", "Which one?")
    assert output == [("specialist", "Which one?")]
    assert specialist.calls == [("admit", "Please handle that.", False)]
    assert default.calls == []


@pytest.mark.asyncio
async def test_domain_specific_visual_question_routes_by_declared_scope():
    decisions = _Decisions("museum_guide")
    conversation = _Destination("conversation", description="General conversation and general knowledge")
    museum = _Destination(
        "museum_guide",
        description="Visual questions about museum exhibits and artwork",
    )
    front_end = AgentRouter(
        {"conversation": conversation, "museum_guide": museum},
        decisions, output=_emit_control,
    )

    result, _output = await _run(front_end, query="What is written on this exhibit label?")

    declared = {item["name"]: item["description"] for item in decisions.calls[0][0]["destinations"]}
    assert declared["museum_guide"] == "Visual questions about museum exhibits and artwork"
    assert museum.calls == [
        ("admit", "What is written on this exhibit label?", False),
        ("answer", "What is written on this exhibit label?"),
    ]
    assert conversation.calls == []
    assert result.disposition == "answer"


@pytest.mark.asyncio
async def test_route_decision_receives_bounded_completed_context():
    decisions = _Decisions("xr")
    conversation = _Destination("conversation")
    xr = _Destination("xr")
    front_end = AgentRouter({"conversation": conversation, "xr": xr}, decisions, output=_emit_control)
    history = tuple(AgentExchange(f"request {index}", f"reply {index}") for index in range(5))

    await _run(front_end, query="Add the next object.", history=history)

    routed_history = decisions.calls[0][0]["completed_history"]
    assert routed_history == [
        {"user": f"request {index}", "assistant": f"reply {index}", "recorded_around_utc": "unknown"}
        for index in range(5)
    ]
    assert xr.calls[0] == ("admit", "Add the next object.", False)


@pytest.mark.asyncio
async def test_replacement_turn_cancels_old_answer_before_it_can_emit():
    class _WaitingDestination(_Destination):
        def __init__(self):
            super().__init__("xr")
            self.started = asyncio.Event()

        async def answer(self, turn, *, started):
            emit = _destination_output(started)
            self.calls.append(("answer", turn.query))
            if turn.query == "Slow request":
                self.started.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    # Simulate a delegate that finishes a callback after cancellation.
                    pass
                await emit(self.name, "stale result")
                return DestinationResult("focus", "stale result")
            await emit(self.name, "fresh result")
            return DestinationResult("answer", "fresh result")

    decisions = _Decisions("xr", "xr")
    xr = _WaitingDestination()
    front_end = AgentRouter({"conversation": _Destination("conversation"), "xr": xr}, decisions, output=_emit_control)
    output = []

    async def emit(source, text):
        output.append((source, text))

    task = asyncio.create_task(
        _route(front_end,
            AgentTurn("alice", "turn-1", "Slow request"),
            emit=emit,
        )
    )
    await xr.started.wait()
    result = await _route(front_end,
        AgentTurn("alice", "turn-2", "Replacement request"),
        emit=emit,
    )
    assert result.text == "fresh result"
    with pytest.raises(asyncio.CancelledError):
        await task
    assert output == [("xr", "fresh result")]
    assert front_end._focus == {}  # noqa: SLF001
    assert [item.assistant for item in front_end._history["alice"]] == ["fresh result"]  # noqa: SLF001


@pytest.mark.asyncio
async def test_front_end_attaches_completion_recency_to_its_owned_history():
    decisions = _Decisions("conversation", "conversation")
    conversation = _Destination(
        "conversation",
        results=(DestinationResult("answer", "first answer"), DestinationResult("answer", "second answer")),
    )
    front_end = AgentRouter(
        {"conversation": conversation, "observation": _Destination("observation")}, decisions, output=_emit_control
    )

    await _run(front_end, query="first request")
    await _run(front_end, query="follow-up")

    history = decisions.calls[1][0]["completed_history"]
    assert history[0]["user"] == "first request"
    assert history[0]["recorded_around_utc"] != "unknown"


@pytest.mark.asyncio
async def test_completed_exchange_ids_remain_aligned_when_history_is_trimmed():
    destination = _Destination(
        "conversation",
        results=(DestinationResult("answer", "first answer"), DestinationResult("answer", "second answer")),
    )
    front_end = AgentRouter({"conversation": destination}, _Decisions(), output=_emit_control)

    await _run(front_end, query="first request")
    await _run(front_end, query="second request")

    history = tuple(front_end._history["alice"])
    history_times = tuple(front_end._history_times["alice"])
    history_ids = tuple(front_end._history_exchange_ids["alice"])
    assert history_ids[0] is not history_ids[1]
    assert len(history_ids) == len(history_times) == len(history) == 2

    newest = history[-1]
    newest_timestamp = history_times[-1]
    item = {
        "user": newest.user,
        "assistant": newest.assistant,
        "recorded_around_utc": router_module._format_minute(newest_timestamp),
    }
    budget = len(json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    bounded, bounded_times, bounded_ids = _bounded_history_with_metadata(
        history, history_times, history_ids, budget, "alice"
    )

    assert bounded == (newest,)
    assert bounded_times == (newest_timestamp,)
    assert bounded_ids[0] is history_ids[-1]


@pytest.mark.asyncio
async def test_admission_release_reroute_receives_history_with_separate_exchange_ids():
    observed = {}
    controls = []

    async def output_control(turn, source, text):
        controls.append((turn, source, text))
        await _emit_control(turn, source, text)

    class _ReroutedDestination(_Destination):
        async def answer(self, turn, *, started):
            observed["history"] = turn.history
            observed["timestamps"] = _active_history_timestamps.get()
            observed["history_ids"] = _active_history_exchange_ids.get()
            observed["active_id"] = _active_exchange_id.get()
            return await super().answer(turn, started=started)

    releasing = _Destination(
        "scene",
        admissions=(Admission("release", text="Leaving scene mode."),),
    )
    conversation = _ReroutedDestination(
        "conversation",
        admissions=(Admission("accept", preamble="I will answer."),),
        results=(DestinationResult("answer", "The answer."),),
    )
    front_end = AgentRouter(
        {"scene": releasing, "conversation": conversation},
        _Decisions("scene"), output=output_control,
        default_destination="conversation",
    )
    output = []

    async def emit(source, text):
        output.append((source, text))

    result = await _route(front_end,
        AgentTurn("alice", "turn-1", "Continue with the request.", timestamp_us=13),
        emit=emit,
    )

    assert result.text == "The answer."
    assert output == [
        ("scene", "Leaving scene mode."),
        ("conversation", "I will answer."),
        ("conversation", "The answer."),
    ]
    assert controls[0][0].history == ()
    assert controls[1][0].history == (AgentExchange("Continue with the request.", "Leaving scene mode."),)
    assert [(turn.participant_id, turn.turn_id, turn.timestamp_us) for turn, _, _ in controls] == [
        ("alice", "turn-1", 13), ("alice", "turn-1", 13),
    ]
    assert observed["history"] == (AgentExchange("Continue with the request.", "Leaving scene mode."),)
    assert len(observed["timestamps"]) == len(observed["history_ids"]) == 1
    assert observed["timestamps"][0] is not None
    assert observed["history_ids"][0] is front_end._history_exchange_ids["alice"][0]
    assert observed["active_id"] is not observed["history_ids"][0]
    assert front_end._history_exchange_ids["alice"][0] is not front_end._history_exchange_ids["alice"][1]


def test_routing_history_byte_budget_counts_timestamp_and_utf8_json_fields():
    history = (AgentExchange("質問", "答え"),)
    timestamp = 1_790_856_000_000_000
    token = _active_history_timestamps.set((timestamp,))
    try:
        rendered = '{"user":"質問","assistant":"答え","recorded_around_utc":"2026-10-01 12:00 UTC"}'
        exact_bytes = len(rendered.encode("utf-8"))
        retained, retained_times, _ = _bounded_history_with_metadata(
            history, (timestamp,), (None,), exact_bytes, "alice"
        )
        dropped, dropped_times, _ = _bounded_history_with_metadata(
            history, (timestamp,), (None,), exact_bytes - 1, "alice"
        )
    finally:
        _active_history_timestamps.reset(token)

    assert retained == history
    assert retained_times == (timestamp,)
    assert dropped == ()
    assert dropped_times == ()


@pytest.mark.asyncio
async def test_acknowledgment_is_disabled_by_default_and_fast_answer_cancels_prediction():
    destination = _AckDestination()
    decisions = _Prediction()
    llm = _AcknowledgmentLLM()
    front_end = AgentRouter(
        {"destination": destination}, decisions, output=_emit_control, default_destination="destination",
    )  # type: ignore[arg-type]
    output = []

    await _ack_output(front_end, "Quick request.", output)

    assert output == [("destination", "Here is the answer.")]
    assert not decisions.calls
    assert not llm.calls

    enabled = AgentRouter(
        {"destination": destination},
        decisions, output=_emit_control,  # type: ignore[arg-type]
        default_destination="destination",
        acknowledgment_llm=llm,  # type: ignore[arg-type]
        acknowledgment_delay_secs=20.0,
    )
    await _ack_output(enabled, "Another quick request.", output)
    assert output[-1] == ("destination", "Here is the answer.")
    assert not decisions.calls
    assert not llm.calls


@pytest.mark.asyncio
async def test_confident_slow_prediction_acknowledges_then_records_only_real_answer():
    destination = _AckDestination(blocked=True)
    decisions = _Prediction(
        "slow",
        {"slow": 0.8, "within_delay": 0.1, "uncertain": 0.1},
    )
    llm = _AcknowledgmentLLM()
    front_end = AgentRouter(
        {"destination": destination},
        decisions, output=_emit_control,  # type: ignore[arg-type]
        default_destination="destination",
        acknowledgment_llm=llm,  # type: ignore[arg-type]
        acknowledgment_delay_secs=10.0,
    )
    output = []
    emitted = asyncio.Event()

    async def emit(source, text):
        output.append((source, text))
        emitted.set()

    task = asyncio.create_task(_route(front_end, AgentTurn("alice", "turn-1", "Explain the request."), emit=emit))
    await asyncio.wait_for(destination.answer_started.wait(), timeout=1)
    await asyncio.wait_for(decisions.called.wait(), timeout=1)
    await asyncio.wait_for(llm.started.wait(), timeout=1)
    await asyncio.wait_for(emitted.wait(), timeout=1)
    destination.release_answer.set()
    result = await asyncio.wait_for(task, timeout=1)

    assert result.disposition == "answer"
    assert output == [
        ("destination", "Let me think through that. "),
        ("destination", "Here is the answer."),
    ]
    assert llm.calls[0][1] == {"max_tokens": 64, "enable_thinking": False}
    acknowledgment_messages = llm.calls[0][0]
    assert acknowledgment_messages[0].role == "system"
    assert "Let me think through [short neutral restatement" in acknowledgment_messages[0].content
    assert "Do not include findings, advice, recommendations" in acknowledgment_messages[0].content
    assert "Current user request: Explain the request." in acknowledgment_messages[1].content
    assert acknowledgment_messages[1].content.endswith("Explain the request.")
    assert "prior_first_answer_examples" in decisions.calls[0][0]
    assert front_end._history["alice"][-1].assistant == "Here is the answer."
    assert front_end._history["alice"][-1].assistant != output[0][1]
    assert front_end._first_answer_samples["alice"]["destination"][0][0] == "Explain the request."


@pytest.mark.asyncio
async def test_committed_acknowledgment_is_delivered_before_direct_answer_output():
    destination = _AckDestination(blocked=True)
    front_end = AgentRouter(
        {"destination": destination},
        _Prediction(), output=_emit_control,  # type: ignore[arg-type]
        default_destination="destination",
        acknowledgment_llm=_AcknowledgmentLLM(),  # type: ignore[arg-type]
        acknowledgment_delay_secs=0.001,
    )
    output = []
    ack_entered, ack_release = asyncio.Event(), asyncio.Event()

    async def emit(source, text):
        if text.startswith("Let me think"):
            ack_entered.set()
            await ack_release.wait()
        output.append((source, text))

    task = asyncio.create_task(
        _route(front_end, AgentTurn("alice", "turn-1", "An unfamiliar request."), emit=emit)
    )
    await asyncio.wait_for(ack_entered.wait(), timeout=1)
    destination.release_answer.set()
    await asyncio.wait_for(destination.answer_emit_started.wait(), timeout=1)
    for _ in range(5):
        await asyncio.sleep(0)
    assert output == []
    ack_release.set()
    await asyncio.wait_for(task, timeout=1)
    assert output == [("destination", "Let me think through that. "), ("destination", "Here is the answer.")]


@pytest.mark.asyncio
async def test_uncertain_prediction_uses_delay_fallback_and_admission_preamble_suppresses_ack():
    destination = _AckDestination(blocked=True)
    decisions = _Prediction()
    llm = _AcknowledgmentLLM()
    front_end = AgentRouter(
        {"destination": destination},
        decisions, output=_emit_control,  # type: ignore[arg-type]
        default_destination="destination",
        acknowledgment_llm=llm,  # type: ignore[arg-type]
        acknowledgment_delay_secs=0.001,
    )
    output = []
    emitted = asyncio.Event()

    async def emit(source, text):
        output.append((source, text))
        emitted.set()

    task = asyncio.create_task(
        _route(front_end, AgentTurn("alice", "turn-1", "An unfamiliar request."), emit=emit)
    )
    await asyncio.wait_for(destination.answer_started.wait(), timeout=1)
    await asyncio.wait_for(emitted.wait(), timeout=1)
    assert output == [("destination", "Let me think through that. ")]
    destination.release_answer.set()
    await asyncio.wait_for(task, timeout=1)

    preamble_destination = _AckDestination(preamble="I will check.")
    preamble_front_end = AgentRouter(
        {"destination": preamble_destination},
        decisions, output=_emit_control,  # type: ignore[arg-type]
        default_destination="destination",
        acknowledgment_llm=llm,  # type: ignore[arg-type]
        acknowledgment_delay_secs=0.001,
    )
    previous_calls = len(decisions.calls)
    previous_llm_calls = len(llm.calls)
    await _ack_output(preamble_front_end, "Request with preamble.", output)
    assert len(decisions.calls) == previous_calls
    assert len(llm.calls) == previous_llm_calls
    assert output[-2:] == [
        ("destination", "I will check."),
        ("destination", "Here is the answer."),
    ]


@pytest.mark.asyncio
async def test_real_answer_cancels_pending_ack_generation_without_waiting():
    destination = _AckDestination(blocked=True)
    decisions = _Prediction("slow", {"slow": 0.9, "within_delay": 0.05, "uncertain": 0.05})
    llm = _AcknowledgmentLLM(blocked=True)
    front_end = AgentRouter(
        {"destination": destination},
        decisions, output=_emit_control,  # type: ignore[arg-type]
        default_destination="destination",
        acknowledgment_llm=llm,  # type: ignore[arg-type]
        acknowledgment_delay_secs=10.0,
    )
    output = []
    task = asyncio.create_task(_ack_output(front_end, "Wait on model output.", output))
    await asyncio.wait_for(llm.started.wait(), timeout=1)
    destination.release_answer.set()
    await asyncio.wait_for(task, timeout=1)

    assert output == [("destination", "Here is the answer.")]
    assert llm.cancelled.is_set()


@pytest.mark.asyncio
async def test_answer_does_not_reuse_interrupt_slot_after_ack_was_delivered():
    destination = _AckDestination(blocked=True)
    decisions = _Prediction("slow", {"slow": 0.9, "within_delay": 0.05, "uncertain": 0.05})
    llm = _AcknowledgmentLLM()
    front_end = AgentRouter(
        {"destination": destination},
        decisions, output=_emit_control,  # type: ignore[arg-type]
        default_destination="destination",
        acknowledgment_llm=llm,  # type: ignore[arg-type]
        acknowledgment_delay_secs=10.0,
    )
    output = []
    first = True
    acknowledgment_emit_started = asyncio.Event()
    acknowledgment_emit_release = asyncio.Event()

    async def emit(source, text):
        nonlocal first
        interrupt = first
        first = False
        output.append((source, text, interrupt))
        if text.startswith("Let me think through"):
            acknowledgment_emit_started.set()
            await acknowledgment_emit_release.wait()

    task = asyncio.create_task(
        _route(front_end, AgentTurn("alice", "turn-1", "Wait on model output."), emit=emit)
    )
    await asyncio.wait_for(acknowledgment_emit_started.wait(), timeout=1)
    destination.release_answer.set()
    await asyncio.wait_for(destination.answer_emit_started.wait(), timeout=1)
    acknowledgment_emit_release.set()
    await asyncio.wait_for(task, timeout=1)

    assert output == [
        ("destination", "Let me think through that. ", True),
        ("destination", "Here is the answer.", False),
    ]
    assert llm.calls


@pytest.mark.asyncio
async def test_answer_timing_excludes_committed_acknowledgment_output(monkeypatch):
    destination = _AckDestination(blocked=True)
    decisions = _Prediction("slow", {"slow": 0.9, "within_delay": 0.05, "uncertain": 0.05})
    llm = _AcknowledgmentLLM()
    front_end = AgentRouter(
        {"destination": destination},
        decisions, output=_emit_control,  # type: ignore[arg-type]
        default_destination="destination",
        acknowledgment_llm=llm,  # type: ignore[arg-type]
        acknowledgment_delay_secs=10.0,
    )
    now = 0.0
    monkeypatch.setattr(router_module, "_monotonic", lambda: now)
    acknowledgment_emit_started = asyncio.Event()
    acknowledgment_emit_release = asyncio.Event()

    async def emit(_source, text):
        if text.startswith("Let me think through"):
            acknowledgment_emit_started.set()
            await acknowledgment_emit_release.wait()

    task = asyncio.create_task(
        _route(front_end, AgentTurn("alice", "turn-1", "Wait on model output."), emit=emit)
    )
    await asyncio.wait_for(acknowledgment_emit_started.wait(), timeout=1)
    now = 1.0
    destination.release_answer.set()
    await asyncio.wait_for(destination.answer_emit_started.wait(), timeout=1)
    await asyncio.sleep(0)
    now = 50.0
    acknowledgment_emit_release.set()
    await asyncio.wait_for(task, timeout=1)

    assert front_end._first_answer_samples["alice"]["destination"] == [("Wait on model output.", 1.0)]


@pytest.mark.parametrize(
    ("text", "valid"),
    [
        ("Let me think through creating a clear walkway.", True),
        ("One practical way is to create a clear central pathway.", False),
        ("I can help you rearrange the room to create a clear walkway.", False),
        ("Let me think through creating a clear walkway. Move the chairs.", False),
    ],
)
def test_acknowledgment_gate_rejects_answer_content(text, valid):
    assert _is_valid_acknowledgment(text) is valid


@pytest.mark.parametrize(
    "generated",
    [
        "Let me think through tell me the next one and then summarize all of them in one paragraph.",
        "Let me think through removing the ceiling light fixture and placing it on the floor safely.",
        "Let me think through moving the desk behind you to a different spot.",
        "Let me think through how safe it is to climb on the desk behind you to remove the lights.",
    ],
)
@pytest.mark.asyncio
async def test_invalid_acknowledgment_uses_neutral_fallback_without_recording_it(generated):
    destination = _AckDestination(blocked=True)
    decisions = _Prediction()
    llm = _AcknowledgmentLLM(content=generated)
    front_end = AgentRouter(
        {"destination": destination},
        decisions, output=_emit_control,  # type: ignore[arg-type]
        default_destination="destination",
        acknowledgment_llm=llm,  # type: ignore[arg-type]
        acknowledgment_delay_secs=0.001,
    )
    output = []
    fallback_emitted = asyncio.Event()

    async def emit(source, text):
        output.append((source, text))
        if text == "Let me think that through. ":
            fallback_emitted.set()

    task = asyncio.create_task(
        _route(front_end, AgentTurn("alice", "turn-1", "Continue with the next request."), emit=emit)
    )
    await asyncio.wait_for(destination.answer_started.wait(), timeout=1)
    await asyncio.wait_for(fallback_emitted.wait(), timeout=1)
    destination.release_answer.set()
    result = await asyncio.wait_for(task, timeout=1)

    assert result.disposition == "answer"
    assert output == [
        ("destination", "Let me think that through. "),
        ("destination", "Here is the answer."),
    ]
    assert front_end._history["alice"][-1].assistant == "Here is the answer."


@pytest.mark.parametrize("disposition", ["focus", "release"])
@pytest.mark.asyncio
async def test_completed_focus_and_release_answers_train_first_answer_timing(disposition):
    class Destination(_AckDestination):
        async def answer(self, _turn, *, started):
            emit = _destination_output(started)
            await emit("destination", self.answer_text)
            return DestinationResult(disposition, self.answer_text)

    destination = Destination()
    front_end = AgentRouter(
        {"destination": destination},
        _Decisions(), output=_emit_control,
        default_destination="destination",
    )

    async def emit(_source, _text):
        pass

    result = await _route(front_end,
        AgentTurn("alice", "turn-1", "Wait on model output."),
        emit=emit,
    )

    assert result.disposition == disposition
    assert front_end._first_answer_samples["alice"]["destination"][0][0] == "Wait on model output."


@pytest.mark.asyncio
async def test_unexpected_prediction_failure_is_logged_without_failing_answer(caplog):
    destination = _AckDestination(blocked=True)
    decisions = _Prediction(error=RuntimeError("prediction bug"))
    llm = _AcknowledgmentLLM()
    front_end = AgentRouter(
        {"destination": destination},
        decisions, output=_emit_control,  # type: ignore[arg-type]
        default_destination="destination",
        acknowledgment_llm=llm,  # type: ignore[arg-type]
        acknowledgment_delay_secs=0.001,
    )
    output = []
    task = asyncio.create_task(_ack_output(front_end, "Quick answer while prediction runs.", output))
    await asyncio.wait_for(destination.answer_started.wait(), timeout=1)
    await asyncio.wait_for(decisions.called.wait(), timeout=1)
    destination.release_answer.set()
    await asyncio.wait_for(task, timeout=1)

    assert output == [("destination", "Here is the answer.")]
    assert not llm.calls
    failure = next(record for record in caplog.records if "optional acknowledgment task failed" in record.message)
    assert failure.exc_info is not None
    assert isinstance(failure.exc_info[1], RuntimeError)
    assert "prediction bug" in str(failure.exc_info[1])


@pytest.mark.asyncio
async def test_prediction_hang_does_not_delay_answer_and_starts_ack_after_delay():
    destination = _AckDestination(blocked=True)
    decisions = _Prediction(blocked=True)
    llm = _AcknowledgmentLLM(error=RuntimeError("acknowledgment bug"))
    front_end = AgentRouter(
        {"destination": destination},
        decisions, output=_emit_control,  # type: ignore[arg-type]
        default_destination="destination",
        acknowledgment_llm=llm,  # type: ignore[arg-type]
        acknowledgment_delay_secs=0.001,
    )
    output = []
    task = asyncio.create_task(_ack_output(front_end, "Quick answer while prediction runs.", output))
    await asyncio.wait_for(destination.answer_started.wait(), timeout=1)
    await asyncio.wait_for(decisions.called.wait(), timeout=1)
    await asyncio.wait_for(llm.started.wait(), timeout=1)
    destination.release_answer.set()
    await asyncio.wait_for(task, timeout=1)

    assert output == [("destination", "Here is the answer.")]
    assert decisions.cancelled.is_set()
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_cancelled_acknowledgment_reports_prediction_cleanup_failure_once(caplog, monkeypatch):
    @contextmanager
    def scope(*_args, **_kwargs):
        yield None

    monkeypatch.setattr(router_module.nemo_relay.scope, "scope", scope)
    destination = _AckDestination(blocked=True)
    decisions = _Prediction(blocked=True, cancel_error=RuntimeError("prediction cleanup failed"))
    llm = _AcknowledgmentLLM(blocked=True)
    front_end = AgentRouter(
        {"destination": destination},
        decisions, output=_emit_control,  # type: ignore[arg-type]
        default_destination="destination",
        acknowledgment_llm=llm,  # type: ignore[arg-type]
        acknowledgment_delay_secs=0.001,
    )
    output = []
    task = asyncio.create_task(_ack_output(front_end, "Answer while prediction is cancelled.", output))
    await asyncio.wait_for(destination.answer_started.wait(), timeout=1)
    await asyncio.wait_for(decisions.called.wait(), timeout=1)
    await asyncio.wait_for(llm.started.wait(), timeout=1)
    await asyncio.wait_for(decisions.finished.wait(), timeout=1)
    destination.release_answer.set()
    await asyncio.wait_for(task, timeout=1)

    assert output[-1] == ("destination", "Here is the answer.")
    assert decisions.cancelled.is_set()
    assert decisions.task is not None and decisions.task.done()
    assert llm.cancelled.is_set()
    failures = [record for record in caplog.records if "optional acknowledgment task failed" in record.message]
    assert len(failures) == 1
    assert failures[0].exc_info is not None
    assert isinstance(failures[0].exc_info[1], RuntimeError)
    assert "prediction cleanup failed" in str(failures[0].exc_info[1])


@pytest.mark.asyncio
async def test_repeated_acknowledgment_cancellation_drains_and_reports_child_failure_once(caplog, monkeypatch):
    @contextmanager
    def scope(*_args, **_kwargs):
        yield None

    monkeypatch.setattr(router_module.nemo_relay.scope, "scope", scope)
    destination = _AckDestination(blocked=True)
    decisions = _Prediction(blocked=True, cancel_error=RuntimeError("prediction cleanup failed"))
    decisions.cancel_blocked = True
    llm = _AcknowledgmentLLM(blocked=True)
    front_end = AgentRouter(
        {"destination": destination},
        decisions, output=_emit_control,  # type: ignore[arg-type]
        default_destination="destination",
        acknowledgment_llm=llm,  # type: ignore[arg-type]
        acknowledgment_delay_secs=0.001,
    )
    output = []
    drain_started = asyncio.Event()
    original_shield = asyncio.shield

    def observe_drain(future):
        task = asyncio.current_task()
        if task is not None and task.get_name().startswith("conversation-acknowledgment:"):
            drain_started.set()
        return original_shield(future)

    monkeypatch.setattr(router_module.asyncio, "shield", observe_drain)
    owner_task = asyncio.create_task(_ack_output(front_end, "Answer while cleanup is canceled.", output))
    await asyncio.wait_for(destination.answer_started.wait(), timeout=1)
    await asyncio.wait_for(decisions.called.wait(), timeout=1)
    await asyncio.wait_for(llm.started.wait(), timeout=1)
    acknowledgment_task = next(
        task
        for task in asyncio.all_tasks()
        if task.get_name().startswith("conversation-acknowledgment:")
    )
    acknowledgment_done = asyncio.Event()
    acknowledgment_task.add_done_callback(lambda _task: acknowledgment_done.set())

    acknowledgment_task.cancel()
    await asyncio.wait_for(decisions.cancelled.wait(), timeout=1)
    await asyncio.wait_for(drain_started.wait(), timeout=1)
    await asyncio.sleep(0)
    acknowledgment_task.cancel()
    acknowledgment_task.cancel()
    assert decisions.cancel_count == 1
    decisions.release_cancel.set()
    await asyncio.wait_for(acknowledgment_done.wait(), timeout=1)
    destination.release_answer.set()

    result = await asyncio.wait_for(owner_task, timeout=1)
    with pytest.raises(asyncio.CancelledError):
        await acknowledgment_task

    assert result.disposition == "answer"
    assert output[-1] == ("destination", "Here is the answer.")
    assert decisions.task is not None and decisions.task.done()
    failures = [record for record in caplog.records if "optional acknowledgment task failed" in record.message]
    assert len(failures) == 1
    assert failures[0].exc_info is not None
    assert isinstance(failures[0].exc_info[1], RuntimeError)
    assert "prediction cleanup failed" in str(failures[0].exc_info[1])


@pytest.mark.asyncio
async def test_turn_cancellation_cancels_ack_and_departure_drops_timing_examples():
    destination = _AckDestination(blocked=True)
    decisions = _Prediction("slow", {"slow": 0.9, "within_delay": 0.05, "uncertain": 0.05})
    llm = _AcknowledgmentLLM(blocked=True)
    front_end = AgentRouter(
        {"destination": destination},
        decisions, output=_emit_control,  # type: ignore[arg-type]
        default_destination="destination",
        acknowledgment_llm=llm,  # type: ignore[arg-type]
        acknowledgment_delay_secs=10.0,
    )
    output = []
    task = asyncio.create_task(_ack_output(front_end, "Cancel this turn.", output))
    await asyncio.wait_for(llm.started.wait(), timeout=1)
    await front_end.interrupt("alice")
    with pytest.raises(asyncio.CancelledError):
        await task

    assert llm.cancelled.is_set()
    assert output == []
    front_end._first_answer_samples["alice"] = {"destination": [("old request", 0.1)]}
    await front_end.participant_left("alice")
    assert "alice" not in front_end._first_answer_samples
