# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from xr_ai_models import DecisionAnswer, DecisionResponse

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "agent-samples/simple-vlm-example/worker"))
_RUNNER = _ROOT / "agent-samples/simple-vlm-example/eval/run_decision_eval.py"
_SPEC = importlib.util.spec_from_file_location("simple_vlm_decision_eval", _RUNNER)
assert _SPEC is not None and _SPEC.loader is not None
evaluator = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = evaluator
_SPEC.loader.exec_module(evaluator)

from test_simple_vlm_conversation_flows import completed_history, score_turn  # noqa: E402


class _FixtureDecisionService:
    """Deterministic plumbing fake; its choices are not model-evaluation results."""

    def __init__(self, recorder, flows, cases):
        self.recorder = recorder
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
            else:
                choice = (fixture.get("destination_actions") or {}).get("notes" if key == "note_action" else "monitor")
            if choice not in criteria:
                choice = next(iter(criteria))
            if key == "destination" and choice == "clarify":
                probabilities = {item: (0.6 if item == "clarify" else 0.4 / (len(criteria) - 1)) for item in criteria}
                answers[key] = DecisionAnswer(choice, probabilities[choice], probabilities)
            else:
                probabilities = {item: float(item == choice) for item in criteria}
                answers[key] = DecisionAnswer(choice, 1.0, probabilities)
        return DecisionResponse(answers, "deterministic-test-double", {}, {})


def test_decision_evaluator_preflights_and_runs_every_fixture_without_answer_calls():
    flows, cases = evaluator.load_fixtures()
    plan = evaluator.preflight(flows, cases)
    assert plan["total_turns"] == 64
    assert plan["planned_decision_calls"] <= plan["max_decision_calls"]

    recorder = evaluator._DecisionRecorder(None, max_calls=plan["max_decision_calls"])
    recorder.service = _FixtureDecisionService(recorder, flows, cases)

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
    assert ambiguous["raw_router_choice"] == "clarify"
    assert ambiguous["effective_destination"] in {"conversation", "observation"}
    assert ambiguous["route_outcome_match"] is True
    note_capture = next(turn for turn in turns if turn["fixture"].endswith("alice_dictates_ar_vr_note_verbatim"))
    assert note_capture["published_context_update_match"] is True
    assert note_capture["published_context_updates"][-1]["source"] == "simple-vlm.foreground-notes"
    assert note_capture["published_context_updates"][-1]["mode"] == "incremental"
    assert note_capture["focus_before"] == "notes"
    assert note_capture["focus_before_match"] is True
    note_recall = next(
        turn for turn in turns if turn["fixture"].endswith("alice_general_conversation_uses_saved_note_context")
    )
    assert note_recall["focus_before"] is None
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
    assert "Completed conversation exchange (context only)" in json.dumps(continuation_call["request_state"])
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
    hashes = {
        evaluator_path: "old-harness",
        "yaml/models.json": "same-config",
        "prompts/system.txt": "same-prompt",
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

    changed_source = dict(hashes, **{"prompts/system.txt": "changed-prompt"})
    with pytest.raises(ValueError, match="hashes do not match"):
        evaluator._load_resume_calls(log, changed_source)


def test_missing_monitor_record_skips_dependent_flow_but_runs_independent_flows():
    import asyncio
    from copy import deepcopy

    flows, cases = evaluator.load_fixtures()
    monitor_flow = deepcopy(
        next(
            flow for flow in flows if flow["name"] == "active_monitor_routes_task_coverage_separately_from_current_view"
        )
    )
    monitor_flow["turns"][0]["acceptable_destinations"] = ["notes"]
    monitor_flow["turns"][0]["admission_expectations"] = {"notes": "accept"}
    monitor_flow["turns"][0]["destination_actions"] = {"notes": "other"}
    independent_flow = next(flow for flow in flows if flow["name"] == "notes_handoff_releases_to_general_conversation")
    selected_cases = []

    recorder = evaluator._DecisionRecorder(None, max_calls=100)
    recorder.service = _FixtureDecisionService(recorder, [monitor_flow, independent_flow], selected_cases)
    turns, _summary = asyncio.run(
        evaluator.run_decisions([monitor_flow, independent_flow], selected_cases, recorder, context_budget=12_000)
    )

    missing = next(turn for turn in turns if turn["fixture"].endswith("active_task_question_uses_monitor_context"))
    assert missing["run_status"] == "not_scored_missing_precondition"
    dependent = next(turn for turn in turns if turn["fixture"].endswith("monitor_status_is_a_control"))
    assert dependent["run_status"] == "not_scored_upstream_precondition_missing"
    assert any("notes_handoff_releases_to_general_conversation/start_notes" == call["case"] for call in recorder.calls)
