# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decision-evaluator fixture loading and reporting contracts."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest
from _conversation_output import CapturingAgentRouter as AgentRouter
from _conversation_output import capture_emit
from _conversation_output import handle as _handle
from xr_ai_models import DecisionAnswer, DecisionResponse
from xr_ai_routing import Admission, AgentTurn, DestinationResult, RoutableAgent, RoutingAid

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "agent-samples/conversation-example/worker"))
_RUNNER = _ROOT / "agent-samples/conversation-example/eval/run_decision_eval.py"
_SPEC = importlib.util.spec_from_file_location("conversation_example_decision_eval", _RUNNER)
assert _SPEC is not None and _SPEC.loader is not None
evaluator = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = evaluator
_SPEC.loader.exec_module(evaluator)

from test_conversation_example_flows import completed_history, score_turn  # noqa: E402


class _FixtureDecisionService:
    """Deterministic plumbing fake; its choices are not model-evaluation results."""

    def __init__(self, recorder, flows, cases, *, router_choices=None):
        self.recorder = recorder
        self.router_choices = router_choices or {}
        self.fixtures = {f"{flow['name']}/{turn['name']}": turn for flow in flows for turn in flow["turns"]}
        self.fixtures.update({case["name"]: case for case in cases})

    async def decide(self, _state, questions, *, images=(), timeout=None):
        del images, timeout
        fixture = self.fixtures[self.recorder.case_name]
        answers = {}
        for key, question in questions.items():
            criteria = question.criteria
            if key == "destination":
                choices = fixture.get("acceptable_router_choices") or (
                    ["clarify"]
                    if fixture.get("route") == "clarify"
                    else fixture.get("acceptable_destinations", [fixture.get("destination")])
                )
                choice = next((item for item in choices if item in criteria), None)
                choice = self.router_choices.get(self.recorder.case_name, choice)
            else:
                choice = (fixture.get("destination_actions") or {}).get("notes" if key == "note_action" else "alerts")
            if choice not in criteria:
                choice = next(iter(criteria))
            if key == "destination" and choice == "clarify":
                probabilities = {item: (0.6 if item == "clarify" else 0.4 / (len(criteria) - 1)) for item in criteria}
                answers[key] = DecisionAnswer(choice, probabilities[choice], probabilities)
            else:
                probabilities = {item: float(item == choice) for item in criteria}
                answers[key] = DecisionAnswer(choice, 1.0, probabilities)
        return DecisionResponse(answers, "deterministic-test-double", {}, {})


@pytest.mark.parametrize("ambiguous_choice", ["clarify", "conversation"])
def test_decision_evaluator_preflights_and_runs_every_fixture_without_answer_calls(ambiguous_choice):
    flows, cases = evaluator.load_fixtures()
    plan = evaluator.preflight(flows, cases)
    assert plan["total_turns"] == 56
    assert plan["planned_decision_calls"] <= plan["max_decision_calls"]

    recorder = evaluator._DecisionRecorder(None, max_calls=plan["max_decision_calls"])
    recorder.service = _FixtureDecisionService(
        recorder,
        flows,
        cases,
        router_choices={
            "clarify_visual_referent_then_inspect/ambiguous_visual_reference": ambiguous_choice,
        },
    )

    import asyncio

    turns, summary = asyncio.run(evaluator.run_decisions(flows, cases, recorder, context_budget=12_000))

    assert len(turns) == plan["total_turns"]
    assert all(turn["answer"] is None and turn["completed"] is False for turn in turns)
    assert all(turn["answer_status"] == "not_scored_decision_only" for turn in turns)
    assert all(turn["evidence"] is None for turn in turns)
    assert all(turn["evidence_status"] == "not_scored_no_tool_calls" for turn in turns)
    ambiguous = next(
        turn
        for turn in turns
        if turn["fixture"].endswith("clarify_visual_referent_then_inspect/ambiguous_visual_reference")
    )
    assert ambiguous["raw_router_choice"] == ambiguous_choice
    assert ambiguous["effective_destination"] == "conversation"
    assert ambiguous["route_outcome"] == "selected"
    assert ambiguous["attempts"][0]["admission"] == "accept"
    assert ambiguous["route_outcome_match"] is True
    note_capture = next(turn for turn in turns if turn["fixture"].endswith("save_accessibility_note"))
    assert note_capture["published_context_update_match"] is True
    assert note_capture["published_context_updates"][-1]["source"] == "conversation-example.foreground-notes"
    assert note_capture["published_context_updates"][-1]["mode"] == "summary"
    assert note_capture["focus_before"] == "notes"
    assert note_capture["focus_before_match"] is True
    note_recall = next(turn for turn in turns if turn["fixture"].endswith("focused_notes_read_back_after_handoff"))
    assert note_recall["focus_before"] == "notes"
    assert note_recall["focus_before_match"] is True
    assert summary["answer_scoring"] == "not_run"
    assert summary["evidence_scoring"] == "not_run"
    assert len(recorder.calls) <= plan["max_decision_calls"]
    assert all("acceptable_destinations" not in str(call["request_state"]) for call in recorder.calls)
    assert all(turn["history_source"] in {"authored_teacher_forced", "empty_fixture_history"} for turn in turns)
    single_case_participants = [
        turn["participant_id"] for turn in turns if turn["fixture"] in {case["name"] for case in cases}
    ]
    assert len(single_case_participants) == len(set(single_case_participants))

    continuation_call = next(
        call for call in recorder.calls if call["case"] == "general_planning_continuation/continue_general_plan"
    )
    previous = continuation_call["request_state"]["completed_history"]
    assert previous == [
        {
            "user": "I'm planning a small community garden. What should I consider first?",
            "assistant": "Start by checking how much sunlight the garden gets and where water is available.",
            "recorded_around_utc": previous[0]["recorded_around_utc"],
        }
    ]
    all_requests = json.dumps([call["request_state"] for call in recorder.calls])
    assert "Authored routing-only fixture context" not in all_requests
    assert "Leave enough room for paths between the beds" not in all_requests
    assert "Completed conversation exchange (context only)" not in json.dumps(continuation_call["request_state"])
    assert "Start by checking how much sunlight" in json.dumps(continuation_call["request_state"])
    assert "18 percent of 250 is 45" not in json.dumps(continuation_call["request_state"])

    incomplete = {"answer": None, "completed": False}
    history = completed_history([], [{"query": "test"}], [incomplete])
    assert history == []
    assert (
        score_turn(
            {"answer_expectation": "nonempty"},
            {"answer": None, "completed": False},
        )["answer"]
        == "not_scored"
    )


@pytest.mark.parametrize("decision_only", [False, True])
@pytest.mark.parametrize(
    "scenario", ["ranked", "only_lower_eligible", "no_eligible", "zero_eligible_probabilities", "admit_clarify"]
)
async def test_uncertain_routing_probes_eligible_destination_before_clarifying(decision_only, scenario):
    class Destination(RoutableAgent):
        def __init__(self, name, eligible):
            self.name = name
            self.eligible = eligible
            self.admissions = 0
            self.answers = 0

        def routing_aid(self, _participant_id):
            return RoutingAid(self.name, self.name, eligible=self.eligible)

        async def admit(self, _turn, *, focused):
            self.admissions += 1
            if scenario == "admit_clarify":
                return Admission("clarify", text="Which item do you mean?")
            return Admission("accept")

        async def answer(self, _turn, *, started):
            self.answers += 1
            await capture_emit(self.name, "Accepted the request.")
            return DestinationResult("answer", "Accepted the request.")

        async def release(self, _participant_id):
            pass

        async def participant_left(self, _participant_id):
            pass

    class Decisions:
        async def decide(self, _state, _questions, **_kwargs):
            probabilities = (
                {"clarify": 1.0, "conversation": 0.0, "observation": 0.0}
                if scenario == "zero_eligible_probabilities"
                else {"clarify": 0.61, "conversation": 0.27, "observation": 0.12}
            )
            return DecisionResponse(
                {"destination": DecisionAnswer("clarify", probabilities["clarify"], probabilities)}, "fake", {}, {}
            )

    recorder = evaluator._DecisionRecorder(Decisions(), max_calls=1)
    destinations = {
        "conversation": Destination("conversation", scenario not in {"only_lower_eligible", "no_eligible"}),
        "observation": Destination("observation", scenario != "no_eligible"),
    }
    selected = (
        None
        if scenario in {"no_eligible", "zero_eligible_probabilities"}
        else "observation"
        if scenario == "only_lower_eligible"
        else "conversation"
    )
    expected_outcome = "clarify" if selected is None or scenario == "admit_clarify" else "selected"
    router = AgentRouter(destinations, recorder)
    emitted = []

    async def emit(source, text):
        emitted.append((source, text))

    try:
        if decision_only:
            result = await evaluator._evaluate_turn(
                "uncertain-request",
                {"query": "Can you check that one?", "acceptable_destinations": [selected] if selected else []},
                "alice",
                (),
                [],
                router,
                destinations,
                recorder,
                defaultdict(list),
                None,
                {"alice": []},
            )
            assert result["effective_destination"] == selected
            assert result["route_outcome"] == expected_outcome
            if scenario == "admit_clarify":
                assert result["attempts"][-1]["admission"] == "clarify"
        else:
            result = await _handle(
                router, AgentTurn("alice", "uncertain-request", "Can you check that one?"), emit=emit
            )
            assert result.disposition == ("clarify" if expected_outcome == "clarify" else "answer")
            if scenario == "admit_clarify":
                assert result.text == "Which item do you mean?"
                assert emitted == [("conversation", "Which item do you mean?")]
        assert {name: destination.admissions for name, destination in destinations.items()} == {
            name: int(name == selected) for name in destinations
        }
        assert sum(destination.answers for destination in destinations.values()) == (
            1 if not decision_only and expected_outcome == "selected" else 0
        )
    finally:
        await router.stop()


@pytest.mark.parametrize("decision_only", [False, True])
async def test_declined_default_never_becomes_the_answer_fallback(decision_only):
    class DecliningDestination(RoutableAgent):
        def __init__(self, name):
            self.name = name
            self.admissions = 0
            self.answers = 0

        def routing_aid(self, _participant_id):
            return RoutingAid(self.name, self.name)

        async def admit(self, _turn, *, focused):
            self.admissions += 1
            return Admission("decline")

        async def answer(self, _turn, *, started):
            self.answers += 1
            return DestinationResult("answer", "Must not answer a declined request.")

        async def release(self, _participant_id):
            pass

        async def participant_left(self, _participant_id):
            pass

    class Decisions:
        def __init__(self):
            self.choices = iter(("conversation", "observation"))

        async def decide(self, _state, questions, **_kwargs):
            choice = next(self.choices)
            probabilities = {name: float(name == choice) for name in questions["destination"].criteria}
            return DecisionResponse({"destination": DecisionAnswer(choice, 1.0, probabilities)}, "fake", {}, {})

    recorder = evaluator._DecisionRecorder(Decisions(), max_calls=2)
    destinations = {name: DecliningDestination(name) for name in ("conversation", "observation")}
    router = AgentRouter(destinations, recorder)

    async def emit(_source, _text):
        pass

    try:
        if decision_only:
            result = await evaluator._evaluate_turn(
                "two-declines",
                {"query": "Check the request.", "acceptable_destinations": []},
                "alice",
                (),
                [],
                router,
                destinations,
                recorder,
                defaultdict(list),
                None,
                {"alice": []},
            )
            assert result["effective_destination"] is None
            assert result["route_outcome"] == "clarify"
            assert [attempt["admission"] for attempt in result["attempts"]] == ["decline", "decline"]
        else:
            result = await _handle(router, AgentTurn("alice", "two-declines", "Check the request."), emit=emit)
            assert result.disposition == "clarify"
        assert all(destination.admissions == 1 for destination in destinations.values())
        assert all(destination.answers == 0 for destination in destinations.values())
    finally:
        await router.stop()


def test_preflight_rejects_participant_reuse_across_independent_flows():
    from copy import deepcopy

    flows, cases = evaluator.load_fixtures()
    first, second = deepcopy(flows[:2])
    first["turns"][0]["participant_id"] = "shared-eval-participant"
    second["turns"][0]["participant_id"] = "shared-eval-participant"
    with pytest.raises(ValueError, match="shared by independent flows"):
        evaluator.preflight([first, second], cases)


def test_resume_preserves_prior_calls_and_runs_only_unattempted_flows():
    import asyncio

    flows, _cases = evaluator.load_fixtures()
    first, second = flows[:2]
    prior = {
        "case": f"{first['name']}/{first['turns'][0]['name']}",
        "participant_id": f"eval:{first['name']}",
        "response": {"destination": {"choice": "conversation"}},
    }
    recorder = evaluator._DecisionRecorder(None, max_calls=100, prior_calls=[prior])
    recorder.service = _FixtureDecisionService(recorder, [first, second], [])
    turns, _summary = asyncio.run(evaluator.run_decisions([first, second], [], recorder, context_budget=12_000))

    prior_flow = [turn for turn in turns if turn["fixture"].startswith(f"{first['name']}/")]
    assert prior_flow[0]["run_status"] == "not_scored_prior_attempt_preserved"
    assert all(turn["run_status"] == "not_scored_flow_state_not_replayed" for turn in prior_flow[1:])
    assert prior_flow[0]["answer_status"] == "not_scored_prior_attempt_preserved"
    assert prior_flow[1]["answer"] is None and prior_flow[1]["completed"] is False
    assert _summary["turns_with_prior_decision_calls_preserved"] == 1
    assert _summary["turns_not_run_because_prior_flow_state_was_not_replayed"] == len(prior_flow) - 1
    assert any(call is prior for call in recorder.calls)
    assert any(call["case"].startswith(f"{second['name']}/") for call in recorder.calls)


def test_resume_accepts_harness_only_change_but_rejects_changed_decision_inputs(tmp_path):
    evaluator_path = str(Path(evaluator.__file__).resolve().relative_to(evaluator._ROOT))
    history_source = "agent-sdk/xr-ai-routing/xr_ai_routing/_history.py"
    hashes = {
        evaluator_path: "old-harness",
        "yaml/models.json": "same-config",
        history_source: "same-history-source",
    }
    log = tmp_path / "prior.jsonl"
    log.write_text(
        json.dumps({"type": "decision", "case": "already/attempted"})
        + "\n"
        + json.dumps({"type": "summary", "preflight": {"sha256": hashes}})
        + "\n",
        encoding="utf-8",
    )
    calls, prior_hash = evaluator._load_resume_calls(log, {**hashes, evaluator_path: "new-harness"})
    assert len(calls) == 1
    assert prior_hash == "old-harness"

    changed_source = dict(hashes, **{history_source: "changed-history-source"})
    with pytest.raises(ValueError, match="hashes do not match"):
        evaluator._load_resume_calls(log, changed_source)


def test_preflight_fingerprints_shared_conversation_history_source():
    flows, cases = evaluator.load_fixtures()
    source = "agent-sdk/xr-ai-routing/xr_ai_routing/_history.py"

    assert source in evaluator.preflight(flows, cases)["sha256"]


def test_preflight_fingerprints_general_conversation_prompt():
    flows, cases = evaluator.load_fixtures()
    source = "agent-samples/conversation-example/worker/conversation_example_worker/prompts/conversation.txt"

    assert source in evaluator.preflight(flows, cases)["sha256"]
