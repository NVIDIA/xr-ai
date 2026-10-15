# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import sys
from collections import deque
from pathlib import Path

import pytest
from xr_ai_models import DecisionAnswer, DecisionResponse

_SAMPLE_DIR = Path(__file__).resolve().parents[1] / "agent-samples" / "simple-vlm-example"
sys.path.insert(0, str(_SAMPLE_DIR / "worker"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-sdk"))

from xr_ai_conversation import (  # noqa: E402
    Admission,
    ConversationExchange,
    ConversationFrontEnd,
    ConversationTurn,
    DestinationResult,
    RoutingAid,
)
from xr_ai_conversation._context import _active_history_timestamps  # noqa: E402
from xr_ai_conversation.front_end import _bounded_history_with_times  # noqa: E402


class _Decisions:
    def __init__(self, *choices: str) -> None:
        self.choices = deque(choices)
        self.calls = []

    async def decide(self, state, questions, *, timeout=None):
        self.calls.append((state, questions, timeout))
        choice = self.choices.popleft()
        return DecisionResponse(
            answers={"destination": DecisionAnswer(choice, 1.0, {choice: 1.0})},
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
        "xr_ai_conversation.front_end.nemo_relay.scope.event",
        lambda name, **kwargs: events.append((name, kwargs["data"])),
    )
    decisions = _RouteDecisions()
    conversation = _Destination("conversation")
    observation = _Destination("observation", description="Questions about the current camera view")
    front_end = ConversationFrontEnd({"conversation": conversation, "observation": observation}, decisions)

    result, output = await _run(front_end, query="How can I move the objects behind me?")

    assert result.disposition == "clarify"
    assert output == [("conversation", "Which part would you like help with?")]
    assert events == [
        (
            "conversation.destination-route-choice",
            {
                "choice": "clarify",
                "confidence": 0.61,
                "probabilities": {"clarify": 0.61, "conversation": 0.27, "observation": 0.12},
                "selected_destination": None,
                "eligible_destinations": ["conversation", "observation"],
            },
        )
    ]


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
    decisions = _Decisions("specialist", "visual")
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
