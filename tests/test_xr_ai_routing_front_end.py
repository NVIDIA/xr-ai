# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from collections import deque

import pytest
import xr_ai_routing as routing_module
import xr_ai_routing.front_end as front_end_module
from xr_ai_models import ChatResponse, DecisionAnswer, DecisionResponse
from xr_ai_routing import (
    Admission,
    ConversationExchange,
    ConversationFrontEnd,
    ConversationTurn,
    DestinationResult,
    RoutableAgent,
    RoutingAid,
)
from xr_ai_routing._history import (
    _active_exchange_id,
    _active_history_exchange_ids,
    _active_history_timestamps,
)
from xr_ai_routing.front_end import (
    _bounded_history_with_metadata,
    _bounded_history_with_times,
    _is_valid_acknowledgment,
)


def test_routable_agent_replaces_destination_protocol_export() -> None:
    assert RoutableAgent.__name__ == "RoutableAgent"
    assert routing_module.__all__ == [
        "Admission",
        "Conversation",
        "ConversationContextUpdate",
        "ConversationExchange",
        "ConversationFrontEnd",
        "ConversationTurn",
        "DestinationResult",
        "RoutableAgent",
        "RoutingAid",
    ]
    assert not hasattr(routing_module, "ConversationDestination")


class _Prediction:
    def __init__(self, choice="uncertain", probabilities=None, *, error=None, blocked=False) -> None:
        self.choice = choice
        self.probabilities = probabilities or {"slow": 0.0, "within_delay": 0.0, "uncertain": 1.0}
        self.error = error
        self.blocked = blocked
        self.calls = []
        self.called = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def decide(self, state, questions, **_kwargs):
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
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


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

    async def answer(self, turn, *, emit):
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

    return await front_end.handle(
        ConversationTurn(participant, f"turn-{participant}-{query}", query),
        emit=emit,
    )


class _Decisions:
    async def decide(self, *_args, **_kwargs):
        raise AssertionError("one registered destination does not need route inference")


class _Decisions:
    def __init__(self, *choices: str | DecisionAnswer) -> None:
        self.choices = deque(choices)
        self.calls = []

    async def decide(self, state, questions, *, timeout=None):
        self.calls.append((state, questions, timeout))
        choice = self.choices.popleft()
        answer = choice if isinstance(choice, DecisionAnswer) else DecisionAnswer(choice, 1.0, {choice: 1.0})
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

    async def answer(self, turn, *, emit):
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

    result = await front_end.handle(
        ConversationTurn(participant, f"turn-{query}", query, history),
        emit=emit,
    )
    return result, output


@pytest.mark.asyncio
async def test_general_conversation_is_a_normal_route_between_stateless_app_turns():
    decisions = _Decisions("xr", "conversation", "xr")
    conversation = _Destination("conversation")
    xr = _Destination("xr")
    front_end = ConversationFrontEnd({"conversation": conversation, "xr": xr}, decisions)

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
    front_end = ConversationFrontEnd({"conversation": conversation, "tea": tea, "xr": xr}, decisions)

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
    front_end = ConversationFrontEnd({"conversation": conversation, "tea": tea}, decisions)

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
    front_end = ConversationFrontEnd({"conversation": conversation, "tea": tea, "xr": xr}, decisions)

    result, output = await _run(front_end, query="Add a blue cube.")

    assert result.disposition == "answer"
    assert output == [("xr", "I will add it."), ("xr", "Added.")]
    assert len(decisions.calls) == 2
    assert "tea" not in decisions.calls[-1][0]["destinations"]


@pytest.mark.asyncio
async def test_destination_cannot_decline_or_clarify_after_answer_work_starts():
    destination = _Destination("conversation", results=(DestinationResult("clarify", "too late"),))
    front_end = ConversationFrontEnd({"conversation": destination}, _Decisions())

    with pytest.raises(ValueError, match="before answer"):
        await _run(front_end, query="Tell me about this.")


@pytest.mark.asyncio
async def test_participant_departure_releases_only_that_participants_focus():
    decisions = _Decisions("tea", "tea", "conversation")
    tea = _Destination(
        "tea",
        results=(DestinationResult("focus"), DestinationResult("focus")),
    )
    front_end = ConversationFrontEnd({"conversation": _Destination("conversation"), "tea": tea}, decisions)

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
    front_end = ConversationFrontEnd({"conversation": conversation}, decisions)

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
    front_end = ConversationFrontEnd({"conversation": _Destination("conversation"), "tea": tea, "xr": xr}, decisions)
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
    front_end = ConversationFrontEnd({"conversation": destination}, _Decisions())
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
    front_end = ConversationFrontEnd(
        {"conversation": _Destination("conversation"), "app": destination},
        _Decisions("app"),
    )

    await _run(front_end)
    await front_end.participant_left("alice")

    assert destination.releases == ["alice"]
    assert destination.departures == ["alice"]


@pytest.mark.asyncio
async def test_seeded_history_is_retained_with_unknown_timestamps():
    seed = (ConversationExchange("Earlier question", "Earlier answer"),)
    destination = _Destination("conversation")
    front_end = ConversationFrontEnd({"conversation": destination}, _Decisions())

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
    front_end = ConversationFrontEnd(
        {"conversation": _Destination("conversation"), "tea": tea, "visual": visual},
        _Decisions("tea", "visual"),
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
    front_end = ConversationFrontEnd(
        destinations,
        _Decisions("first", "second"),
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
    front_end = ConversationFrontEnd({"conversation": conversation}, decisions)
    await _run(front_end, participant="alice", query="A")
    await _run(front_end, participant="bob", query="B")

    with pytest.raises(ExceptionGroup, match="failed to release conversation destinations") as error:
        await front_end.stop()

    assert len(error.value.exceptions) == 2
    assert set(conversation.releases) == {"alice", "bob"}


@pytest.mark.asyncio
async def test_destination_error_after_preamble_propagates_to_turn_caller():
    class _BrokenAnswer(_Destination):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fail_once = True
            self.answer_histories = []

        async def answer(self, turn, *, emit):
            self.answer_histories.append(turn.history)
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("destination failed")
            await emit(self.name, "Recovered.")
            return DestinationResult("answer", "Recovered.")

    destination = _BrokenAnswer("xr", admissions=(Admission("accept", preamble="I will try."),))
    front_end = ConversationFrontEnd({"xr": destination}, _Decisions(), default_destination="xr")
    output = []

    async def emit(source, text):
        output.append((source, text))

    with pytest.raises(RuntimeError, match="destination failed"):
        await front_end.handle(ConversationTurn("alice", "turn-1", "Do it."), emit=emit)
    assert output == [("xr", "I will try.")]

    result = await front_end.handle(ConversationTurn("alice", "turn-2", "Try again."), emit=emit)
    assert result.text == "Recovered."
    assert output == [("xr", "I will try."), ("xr", "Recovered.")]
    assert destination.answer_histories == [(), ()]


@pytest.mark.asyncio
async def test_unresolved_or_invalid_route_clarifies_without_answering_default():
    for choice in ("clarify", "not-registered"):
        decisions = _Decisions(choice)
        conversation = _Destination("conversation", results=(DestinationResult("answer", "I can help."),))
        xr = _Destination("xr")
        front_end = ConversationFrontEnd({"conversation": conversation, "xr": xr}, decisions)

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
        "xr_ai_routing.front_end.nemo_relay.scope.event",
        lambda name, **kwargs: events.append((name, kwargs["data"])),
    )
    decisions = _RouteDecisions()
    conversation = _Destination("conversation", results=(DestinationResult("answer", "General answer."),))
    observation = _Destination("observation", description="Questions about the current camera view")
    front_end = ConversationFrontEnd({"conversation": conversation, "observation": observation}, decisions)

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
                "eligible_scores_available": True,
            },
        )
    ]


@pytest.mark.asyncio
async def test_uncertain_router_uses_highest_probability_eligible_visual_destination():
    decisions = _Decisions(
        DecisionAnswer(
            "clarify",
            0.05,
            {"environment": 0.55, "clarify": 0.05, "observation": 0.25, "conversation": 0.15},
        )
    )
    conversation = _Destination("conversation", results=(DestinationResult("answer", "General."),))
    observation = _Destination("observation", results=(DestinationResult("answer", "Observed."),))
    environment = _Destination("environment", results=(DestinationResult("answer", "A grounded answer."),))
    front_end = ConversationFrontEnd(
        {"conversation": conversation, "observation": observation, "environment": environment},
        decisions,
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
async def test_uncertain_router_uses_observation_for_current_camera_question():
    decisions = _Decisions(
        DecisionAnswer(
            "clarify",
            0.05,
            {"conversation": 0.1, "observation": 0.75, "environment": 0.1, "clarify": 0.05},
        )
    )
    conversation = _Destination("conversation", results=(DestinationResult("answer", "General."),))
    observation = _Destination("observation", results=(DestinationResult("answer", "A blue cup."),))
    environment = _Destination("environment", results=(DestinationResult("answer", "A grounded answer."),))
    front_end = ConversationFrontEnd(
        {"conversation": conversation, "observation": observation, "environment": environment},
        decisions,
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
    front_end = ConversationFrontEnd({"conversation": conversation, "observation": observation}, decisions)

    result, output = await _run(front_end, query="What is in front of me?")

    assert result == DestinationResult("answer", "General.")
    assert output == [("conversation", "General.")]
    assert conversation.calls == [
        ("admit", "What is in front of me?", False),
        ("answer", "What is in front of me?"),
    ]
    assert observation.calls == []


@pytest.mark.asyncio
async def test_uncertain_router_with_no_eligible_scores_still_clarifies():
    decisions = _Decisions(DecisionAnswer("clarify", 0.2, {"clarify": 0.2, "conversation": 0.4, "specialist": 0.4}))
    conversation = _Destination("conversation", eligible=False)
    specialist = _Destination("specialist", eligible=False)
    front_end = ConversationFrontEnd({"conversation": conversation, "specialist": specialist}, decisions)

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
    front_end = ConversationFrontEnd({"conversation": conversation, "observation": observation}, decisions)

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
    front_end = ConversationFrontEnd({"conversation": conversation, "xr": xr}, decisions)

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
    front_end = ConversationFrontEnd({"conversation": destination}, _Decisions())

    result, output = await _run(front_end, query="What is visible?")

    assert result.disposition == "clarify"
    assert output == [("conversation", "Which part would you like help with?")]
    assert destination.calls == []


@pytest.mark.asyncio
async def test_exhausted_declines_answer_once_with_default_and_shared_history():
    decisions = _Decisions(
        DecisionAnswer(
            "clarify",
            0.05,
            {"conversation": 0.1, "specialist": 0.8, "visual": 0.05, "clarify": 0.05},
        ),
        DecisionAnswer(
            "clarify",
            0.05,
            {"conversation": 0.1, "visual": 0.85, "clarify": 0.05},
        ),
    )
    history = (ConversationExchange("Earlier request", "Earlier response"),)
    default = _Destination(
        "conversation",
        admissions=(Admission("decline", text="must not be spoken"),),
        results=(DestinationResult("answer", "I can still help."),),
    )
    specialist = _Destination("specialist", admissions=(Admission("decline"),))
    visual = _Destination("visual", admissions=(Admission("decline"),))
    front_end = ConversationFrontEnd(
        {"conversation": default, "specialist": specialist, "visual": visual},
        decisions,
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
    front_end = ConversationFrontEnd(
        {"conversation": default, "specialist": specialist},
        decisions,
        default_destination="conversation",
    )

    result, output = await _run(front_end, query="Help me.")

    assert result == DestinationResult("clarify", "Which part would you like help with?")
    assert output == [("conversation", "Which part would you like help with?")]
    assert default.calls == []
    assert specialist.calls == []


@pytest.mark.asyncio
async def test_default_fallback_records_route_reason(monkeypatch):
    default = _Destination(
        "conversation",
        admissions=(Admission("decline"),),
        results=(DestinationResult("answer", "General help."),),
    )
    specialist = _Destination("specialist", admissions=(Admission("decline"),))
    front_end = ConversationFrontEnd(
        {"conversation": default, "specialist": specialist},
        _Decisions("specialist", "conversation"),
        default_destination="conversation",
    )
    reasons = []
    monkeypatch.setattr(
        front_end,
        "_trace_default_fallback",
        lambda _turn, reason, *, eligible: reasons.append((reason, eligible)),
    )

    result, _output = await _run(front_end, query="Help me.")

    assert result.disposition == "answer"
    assert reasons == [("exhausted_declines", True)]


@pytest.mark.asyncio
async def test_exhausted_declines_clarify_when_default_is_unavailable():
    specialist = _Destination("specialist", admissions=(Admission("decline"),))
    default = _Destination("conversation", eligible=False)
    front_end = ConversationFrontEnd(
        {"conversation": default, "specialist": specialist},
        _Decisions("specialist"),
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
    front_end = ConversationFrontEnd(
        {"conversation": default, "specialist": specialist},
        _Decisions("specialist"),
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
    front_end = ConversationFrontEnd(
        {"conversation": conversation, "museum_guide": museum},
        decisions,
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
    front_end = ConversationFrontEnd({"conversation": conversation, "xr": xr}, decisions)
    history = tuple(ConversationExchange(f"request {index}", f"reply {index}") for index in range(5))

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

        async def answer(self, turn, *, emit):
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
    front_end = ConversationFrontEnd({"conversation": _Destination("conversation"), "xr": xr}, decisions)
    output = []

    async def emit(source, text):
        output.append((source, text))

    task = asyncio.create_task(
        front_end.handle(
            ConversationTurn("alice", "turn-1", "Slow request"),
            emit=emit,
        )
    )
    await xr.started.wait()
    result = await front_end.handle(
        ConversationTurn("alice", "turn-2", "Replacement request"),
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
    front_end = ConversationFrontEnd(
        {"conversation": conversation, "observation": _Destination("observation")}, decisions
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
    front_end = ConversationFrontEnd({"conversation": destination}, _Decisions())

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
        "recorded_around_utc": front_end_module._format_minute(newest_timestamp),
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

    class _ReroutedDestination(_Destination):
        async def answer(self, turn, *, emit):
            observed["history"] = turn.history
            observed["timestamps"] = _active_history_timestamps.get()
            observed["history_ids"] = _active_history_exchange_ids.get()
            observed["active_id"] = _active_exchange_id.get()
            return await super().answer(turn, emit=emit)

    releasing = _Destination(
        "scene",
        admissions=(Admission("release", text="Leaving scene mode."),),
    )
    conversation = _ReroutedDestination(
        "conversation",
        results=(DestinationResult("answer", "The answer."),),
    )
    front_end = ConversationFrontEnd(
        {"scene": releasing, "conversation": conversation},
        _Decisions("scene"),
        default_destination="conversation",
    )
    output = []

    async def emit(source, text):
        output.append((source, text))

    result = await front_end.handle(
        ConversationTurn("alice", "turn-1", "Continue with the request."),
        emit=emit,
    )

    assert result.text == "The answer."
    assert output == [
        ("scene", "Leaving scene mode."),
        ("conversation", "The answer."),
    ]
    assert observed["history"] == (ConversationExchange("Continue with the request.", "Leaving scene mode."),)
    assert len(observed["timestamps"]) == len(observed["history_ids"]) == 1
    assert observed["timestamps"][0] is not None
    assert observed["history_ids"][0] is front_end._history_exchange_ids["alice"][0]
    assert observed["active_id"] is not observed["history_ids"][0]
    assert front_end._history_exchange_ids["alice"][0] is not front_end._history_exchange_ids["alice"][1]


def test_routing_history_byte_budget_counts_timestamp_and_utf8_json_fields():
    history = (ConversationExchange("質問", "答え"),)
    timestamp = 1_790_856_000_000_000
    token = _active_history_timestamps.set((timestamp,))
    try:
        rendered = '{"user":"質問","assistant":"答え","recorded_around_utc":"2026-10-01 12:00 UTC"}'
        exact_bytes = len(rendered.encode("utf-8"))
        retained, retained_times = _bounded_history_with_times(history, (timestamp,), exact_bytes, "alice")
        dropped, dropped_times = _bounded_history_with_times(history, (timestamp,), exact_bytes - 1, "alice")
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
    front_end = ConversationFrontEnd({"destination": destination}, decisions, default_destination="destination")  # type: ignore[arg-type]
    output = []

    await _ack_output(front_end, "Quick request.", output)

    assert output == [("destination", "Here is the answer.")]
    assert not decisions.calls
    assert not llm.calls

    enabled = ConversationFrontEnd(
        {"destination": destination},
        decisions,  # type: ignore[arg-type]
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
    front_end = ConversationFrontEnd(
        {"destination": destination},
        decisions,  # type: ignore[arg-type]
        default_destination="destination",
        acknowledgment_llm=llm,  # type: ignore[arg-type]
        acknowledgment_delay_secs=10.0,
    )
    output = []
    emitted = asyncio.Event()

    async def emit(source, text):
        output.append((source, text))
        emitted.set()

    task = asyncio.create_task(front_end.handle(ConversationTurn("alice", "turn-1", "Explain the request."), emit=emit))
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
async def test_uncertain_prediction_uses_delay_fallback_and_admission_preamble_suppresses_ack():
    destination = _AckDestination(blocked=True)
    decisions = _Prediction()
    llm = _AcknowledgmentLLM()
    front_end = ConversationFrontEnd(
        {"destination": destination},
        decisions,  # type: ignore[arg-type]
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
        front_end.handle(ConversationTurn("alice", "turn-1", "An unfamiliar request."), emit=emit)
    )
    await asyncio.wait_for(destination.answer_started.wait(), timeout=1)
    await asyncio.wait_for(emitted.wait(), timeout=1)
    assert output == [("destination", "Let me think through that. ")]
    destination.release_answer.set()
    await asyncio.wait_for(task, timeout=1)

    preamble_destination = _AckDestination(preamble="I will check.")
    preamble_front_end = ConversationFrontEnd(
        {"destination": preamble_destination},
        decisions,  # type: ignore[arg-type]
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
    front_end = ConversationFrontEnd(
        {"destination": destination},
        decisions,  # type: ignore[arg-type]
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
    front_end = ConversationFrontEnd(
        {"destination": destination},
        decisions,  # type: ignore[arg-type]
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
        front_end.handle(ConversationTurn("alice", "turn-1", "Wait on model output."), emit=emit)
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
    front_end = ConversationFrontEnd(
        {"destination": destination},
        decisions,  # type: ignore[arg-type]
        default_destination="destination",
        acknowledgment_llm=llm,  # type: ignore[arg-type]
        acknowledgment_delay_secs=10.0,
    )
    now = 0.0
    monkeypatch.setattr(front_end_module, "_monotonic", lambda: now)
    acknowledgment_emit_started = asyncio.Event()
    acknowledgment_emit_release = asyncio.Event()

    async def emit(_source, text):
        if text.startswith("Let me think through"):
            acknowledgment_emit_started.set()
            await acknowledgment_emit_release.wait()

    task = asyncio.create_task(
        front_end.handle(ConversationTurn("alice", "turn-1", "Wait on model output."), emit=emit)
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
    front_end = ConversationFrontEnd(
        {"destination": destination},
        decisions,  # type: ignore[arg-type]
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
        front_end.handle(ConversationTurn("alice", "turn-1", "Continue with the next request."), emit=emit)
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
        async def answer(self, _turn, *, emit):
            await emit("destination", self.answer_text)
            return DestinationResult(disposition, self.answer_text)

    destination = Destination()
    front_end = ConversationFrontEnd(
        {"destination": destination},
        _Decisions(),
        default_destination="destination",
    )

    async def emit(_source, _text):
        pass

    result = await front_end.handle(
        ConversationTurn("alice", "turn-1", "Wait on model output."),
        emit=emit,
    )

    assert result.disposition == disposition
    assert front_end._first_answer_samples["alice"]["destination"][0][0] == "Wait on model output."


@pytest.mark.asyncio
async def test_prediction_failure_or_hang_does_not_delay_real_answer():
    for decisions in (_Prediction(error=RuntimeError("prediction failed")), _Prediction(blocked=True)):
        destination = _AckDestination(blocked=True)
        llm = _AcknowledgmentLLM(error=RuntimeError("acknowledgment failed"))
        front_end = ConversationFrontEnd(
            {"destination": destination},
            decisions,  # type: ignore[arg-type]
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
        if decisions.blocked:
            assert decisions.cancelled.is_set()
        assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_turn_cancellation_cancels_ack_and_departure_drops_timing_examples():
    destination = _AckDestination(blocked=True)
    decisions = _Prediction("slow", {"slow": 0.9, "within_delay": 0.05, "uncertain": 0.05})
    llm = _AcknowledgmentLLM(blocked=True)
    front_end = ConversationFrontEnd(
        {"destination": destination},
        decisions,  # type: ignore[arg-type]
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
