# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural and per-layer checks for the multi-turn conversation eval corpus."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

_CORPUS = (
    Path(__file__).resolve().parents[1] / "agent-samples" / "simple-vlm-example" / "eval" / "conversation-flows.yaml"
)


def _contains_any(text: str, expected: list[str]) -> bool:
    lowered = text.casefold()
    return any(term.casefold() in lowered for term in expected)


def _terms(value: Any) -> list[str]:
    return list(map(str, value if isinstance(value, list) else [value]))


def score_turn(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    """Score observed routing, admission, evidence, and answer layers independently.

    Missing observations stay ``not_scored`` so partial traces do not masquerade
    as failures or passes. The caller supplies actual runtime outputs; this helper
    never manufactures history or decisions from later turns in the corpus.
    """
    scores: dict[str, Any] = {}

    destinations = expected.get("acceptable_destinations")
    router_choices = expected.get("acceptable_router_choices", destinations)
    if router_choices is None or "router_choice" not in actual:
        scores["router_choice"] = "not_scored"
    else:
        scores["router_choice"] = actual["router_choice"] in router_choices

    if destinations is None or "effective_destination" not in actual:
        scores["effective_destination"] = "not_scored"
    elif not destinations:
        scores["effective_destination"] = actual["effective_destination"] is None
    else:
        effective_destination = actual["effective_destination"]
        scores["effective_destination"] = (
            "not_scored" if effective_destination is None else effective_destination in destinations
        )

    route_outcome = expected.get("route_outcome")
    if route_outcome is None or "route_outcome" not in actual:
        scores["route_outcome"] = "not_scored"
    else:
        scores["route_outcome"] = actual["route_outcome"] == route_outcome

    router_invoked = expected.get("router_invoked")
    if router_invoked is not None:
        scores["router_invoked"] = (
            actual["router_invoked"] == router_invoked if "router_invoked" in actual else "not_scored"
        )

    disposition = expected.get("destination_disposition")
    if disposition is not None:
        scores["destination_disposition"] = (
            actual["destination_disposition"] == disposition if "destination_disposition" in actual else "not_scored"
        )

    completion = expected.get("completion_expectation")
    if completion is not None:
        if "completed" not in actual:
            scores["completion"] = "not_scored"
        else:
            scores["completion"] = actual["completed"] is (completion == "completed")

    expected_updates = expected.get("context_update_expectations")
    if expected_updates is not None:
        if "context_updates" not in actual:
            scores["context_updates"] = {item["source"]: "not_scored" for item in expected_updates}
        else:
            observed_updates = {item["source"]: item["mode"] for item in actual["context_updates"]}
            scores["context_updates"] = {
                item["source"]: observed_updates.get(item["source"]) == item["mode"] for item in expected_updates
            }

    forbidden_sources = expected.get("context_updates_must_not_include")
    if forbidden_sources is not None:
        if "context_updates" not in actual:
            scores["context_updates_excluded"] = "not_scored"
        else:
            observed_sources = {str(item["source"]) for item in actual["context_updates"]}
            scores["context_updates_excluded"] = not (set(forbidden_sources) & observed_sources)

    update_text_contains = expected.get("context_update_text_contains")
    if update_text_contains is not None:
        if "context_updates" not in actual:
            scores["context_update_text"] = "not_scored"
        else:
            expected_text = str(update_text_contains)
            update_text = "\n".join(str(item.get("text", "")) for item in actual["context_updates"])
            scores["context_update_text"] = expected_text in update_text

    expected_actions = expected.get("destination_actions")
    if expected_actions is not None:
        actual_actions = actual.get("destination_actions")
        if actual_actions is None:
            scores["destination_actions"] = "not_scored"
        else:
            scores["destination_actions"] = {
                name: actual_actions.get(name) == action for name, action in expected_actions.items()
            }

    context_sources = expected.get("context_sources_before")
    if context_sources is not None:
        if "context_sources_before" not in actual:
            scores["context_sources_before"] = "not_scored"
        else:
            observed_sources = set(actual["context_sources_before"])
            scores["context_sources_before"] = set(context_sources) <= observed_sources

    context_record_requirement = expected.get("context_record_required_before")
    context_record_ready = True
    if context_record_requirement is not None:
        context_before = actual.get("context_before")
        source = str(context_record_requirement["source"])
        marker = str(context_record_requirement["contains"])
        if context_before is None:
            context_record_ready = False
            scores["context_record_precondition"] = "not_scored"
        else:
            source_text = str(context_before.get(source, ""))
            context_record_ready = bool(source_text.strip()) and marker in source_text
            scores["context_record_precondition"] = context_record_ready

    focus_after = expected.get("focus_after", ...)
    if focus_after is not ...:
        scores["focus_after"] = actual["focus_after"] == focus_after if "focus_after" in actual else "not_scored"
    focus_before = expected.get("focus_before", ...)
    if focus_before is not ...:
        scores["focus_before"] = actual["focus_before"] == focus_before if "focus_before" in actual else "not_scored"

    expected_admissions = expected.get("admission_expectations") or {}
    actual_admissions = actual.get("admissions") or {}
    scores["admission"] = {
        candidate: (actual_admissions[candidate] == outcome if candidate in actual_admissions else "not_scored")
        for candidate, outcome in expected_admissions.items()
    }

    evidence_expectation = expected.get("evidence_expectation")
    if evidence_expectation is None:
        scores["evidence"] = "not_scored"
    else:
        evidence = actual.get("evidence")
        if not isinstance(evidence, dict) or "tool_names" not in evidence:
            scores["evidence"] = "not_scored"
        else:
            tool_names = set(map(str, evidence["tool_names"]))
            visual_tools = {"current_view", "inspect_current_view", "ask_images", "image_query"}
            observed_visual = bool(tool_names & visual_tools)
            evidence_text = evidence.get("text")
            has_result = evidence_text is not None and bool(str(evidence_text).strip())
            if evidence_expectation in {"none", "completed_history_only"}:
                scores["evidence"] = not observed_visual
            elif evidence_expectation == "current_view":
                scores["evidence"] = (
                    "not_scored"
                    if "current_view" in tool_names and not has_result
                    else "current_view" in tool_names and has_result
                )
            elif evidence_expectation == "current_view_then_reason":
                if observed_visual and not has_result:
                    scores["evidence"] = "not_scored"
                else:
                    scores["evidence"] = "inspect_current_view" in tool_names and observed_visual and has_result
            elif evidence_expectation == "completed_history_or_current_view":
                if observed_visual and not has_result:
                    scores["evidence"] = "not_scored"
                else:
                    scores["evidence"] = not observed_visual or has_result
            else:
                scores["evidence"] = "not_scored"

    answer = actual.get("answer")
    answer_terms = expected.get("answer_any_of")
    should_clarify = expected.get("answer_should_clarify")
    clarification_terms = expected.get("clarification_any_of")
    if not context_record_ready:
        scores["answer"] = "not_scored"
    elif answer is None:
        scores["answer"] = "not_scored"
    elif answer_terms is not None:
        scores["answer"] = _contains_any(str(answer), _terms(answer_terms))
    elif should_clarify is True:
        terms = _terms(clarification_terms) if clarification_terms else []
        scores["answer"] = bool(terms) and _contains_any(str(answer), terms)
    elif should_clarify is False:
        terms = _terms(clarification_terms) if clarification_terms else []
        scores["answer"] = not _contains_any(str(answer), terms)
    elif expected.get("answer_expectation") == "nonempty":
        scores["answer"] = bool(str(answer).strip())
    else:
        scores["answer"] = "not_scored"

    return scores


def completed_history(
    initial_history: list[dict[str, str]],
    turns: list[dict[str, Any]],
    actual_turns: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Return prior completed exchanges, in order, without reading future turns."""
    history = list(initial_history)
    for turn, actual in zip(turns, actual_turns, strict=False):
        answer = actual.get("answer")
        if actual.get("completed", answer is not None) and answer is not None:
            history.append({"user": turn["query"], "assistant": str(answer)})
    return history


def completed_flow_state_before_turns(
    flow: dict[str, Any],
    actual_turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build participant state without applying unfinished or future turn output."""
    default_participant = str(flow.get("participant_id", "participant"))
    initial_history = list(flow.get("initial_history", []))
    histories: dict[str, list[dict[str, str]]] = {default_participant: initial_history}
    contexts: dict[str, dict[str, str]] = {}
    focused: dict[str, str] = {}
    states: list[dict[str, Any]] = []

    for turn, actual in zip(flow["turns"], actual_turns, strict=False):
        participant_id = str(turn.get("participant_id", default_participant))
        history = histories.setdefault(participant_id, [])
        participant_context = contexts.setdefault(participant_id, {})
        focus_before = focused.get(participant_id)
        states.append(
            {
                "participant_id": participant_id,
                "history": list(history),
                "context": dict(participant_context),
                "focused_destination": focus_before,
            }
        )
        if not actual.get("completed", actual.get("answer") is not None):
            continue

        answer = actual.get("answer")
        if answer is not None:
            history.append({"user": turn["query"], "assistant": str(answer)})
        for update in actual.get("context_updates", []):
            source = str(update["source"])
            text = str(update["text"])
            if update["mode"] == "summary":
                participant_context[source] = text
            elif update["mode"] == "incremental":
                previous = participant_context.get(source)
                participant_context[source] = f"{previous}\n{text}" if previous else text
            else:
                raise ValueError(f"unsupported context update mode: {update['mode']!r}")

        disposition = actual.get("destination_disposition")
        destination = actual.get("effective_destination")
        if disposition == "focus" and destination is not None:
            focused[participant_id] = str(destination)
        elif disposition == "release" and focused.get(participant_id) == destination:
            focused.pop(participant_id, None)

    return states


def test_multi_turn_eval_corpus_is_ordered_and_layer_scored() -> None:
    flows = yaml.safe_load(_CORPUS.read_text(encoding="utf-8"))
    assert isinstance(flows, list)
    assert flows

    flow_names: set[str] = set()
    for flow in flows:
        assert flow["name"] not in flow_names
        flow_names.add(flow["name"])
        assert flow["turns"]
        turn_names: set[str] = set()
        for turn in flow["turns"]:
            assert turn["name"] not in turn_names
            turn_names.add(turn["name"])
            assert turn["query"].strip()
            destinations = turn.get("acceptable_destinations")
            assert destinations is not None
            assert set(destinations) <= {"conversation", "observation", "monitor", "notes"}
            if turn.get("route_outcome") == "clarify":
                assert turn.get("acceptable_router_choices") == ["clarify"]
                assert destinations == []
            else:
                assert destinations
            assert turn.get("evidence_expectation") in {
                "none",
                "current_view",
                "current_view_then_reason",
                "completed_history_only",
                "completed_history_or_current_view",
            }
            assert (
                turn.get("answer_any_of")
                or turn.get("answer_should_clarify") is not None
                or turn.get("answer_expectation") == "nonempty"
                or turn.get("completion_expectation") == "cancelled"
            )
            if "destination_disposition" in turn:
                assert turn["destination_disposition"] in {"answer", "focus", "release"}
            if "router_invoked" in turn:
                assert isinstance(turn["router_invoked"], bool)
            if "context_update_expectations" in turn:
                assert all(
                    update["source"].startswith("simple-vlm.") and update["mode"] in {"incremental", "summary"}
                    for update in turn["context_update_expectations"]
                )
            if "context_record_required_before" in turn:
                requirement = turn["context_record_required_before"]
                assert requirement["source"].startswith("simple-vlm.")
                assert requirement["contains"].strip()

    by_name = {flow["name"]: flow for flow in flows}
    assert {
        "general_ar_vr_followup_topic_shift_and_shared_recall",
        "direct_visual_handoff_to_general_conversation",
        "background_monitor_and_focused_notes_are_participant_scoped",
        "notes_handoff_releases_to_general_conversation",
        "poisoned_camera_unavailable_history_does_not_block_fresh_view",
        "monitor_retarget_is_a_routed_control_before_old_scope_view",
    } <= set(by_name)
    assert all(
        "expert" not in destination
        for flow in flows
        for turn in flow["turns"]
        for destination in turn.get("acceptable_destinations", [])
    )
    assert any(
        "AR/VR" in turn["query"] and turn.get("acceptable_destinations") == ["conversation"]
        for flow in flows
        for turn in flow["turns"]
    )
    assert any(
        turn.get("evidence_expectation") == "current_view"
        and "acceptable_destinations" in turn
        and "observation" in turn["acceptable_destinations"]
        for flow in flows
        for turn in flow["turns"]
    )
    coverage = by_name["active_monitor_routes_task_coverage_separately_from_current_view"]
    turns_by_name = {turn["name"]: turn for turn in coverage["turns"]}
    assert turns_by_name["active_task_question_uses_monitor_context"]["query"] == "Where is my pen?"
    assert turns_by_name["active_task_question_uses_monitor_context"]["acceptable_destinations"] == ["conversation"]
    assert (
        turns_by_name["active_task_question_uses_monitor_context"]["evidence_expectation"]
        == "completed_history_only"
    )
    assert turns_by_name["active_task_question_uses_monitor_context"]["context_record_required_before"] == {
        "source": "simple-vlm.monitor-window",
        "contains": "(frame source time)",
    }
    assert turns_by_name["explicit_fresh_camera_request_overrides_monitor_coverage"]["acceptable_destinations"] == [
        "observation"
    ]
    assert turns_by_name["stopped_monitor_history_remains_available_to_conversation"]["acceptable_destinations"] == [
        "conversation"
    ]
    monitor = by_name["background_monitor_and_focused_notes_are_participant_scoped"]
    assert [turn["destination_disposition"] for turn in monitor["turns"] if "destination_disposition" in turn] == [
        "answer",
        "answer",
        "answer",
        "focus",
        "focus",
        "focus",
        "focus",
        "release",
        "answer",
        "answer",
        "answer",
        "answer",
        "answer",
    ]
    assert [turn.get("router_invoked") for turn in monitor["turns"]] == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
    ]
    assert all(turn.get("focus_after") != "monitor" for turn in monitor["turns"])
    assert [turn.get("focus_after") for turn in monitor["turns"]][3:8] == [
        "notes",
        "notes",
        "notes",
        "notes",
        None,
    ]
    assert any(turn.get("participant_id") == "bob" for turn in monitor["turns"])


def test_direct_observation_is_available_to_later_general_turn_without_future_output() -> None:
    flows = yaml.safe_load(_CORPUS.read_text(encoding="utf-8"))
    flow = next(item for item in flows if item["name"] == "direct_visual_handoff_to_general_conversation")
    actual = [
        {"completed": True, "answer": "A display is beside a doorway."},
        {"completed": False, "answer": None},
    ]

    before = completed_flow_state_before_turns(flow, actual)

    assert before[1]["history"] == [{"user": flow["turns"][0]["query"], "assistant": "A display is beside a doorway."}]


def test_poisoned_camera_unavailable_history_still_requires_fresh_observation() -> None:
    flows = yaml.safe_load(_CORPUS.read_text(encoding="utf-8"))
    flow = next(
        item for item in flows if item["name"] == "poisoned_camera_unavailable_history_does_not_block_fresh_view"
    )
    assert flow["initial_history"] == [
        {
            "user": "Can you read what is in front of me?",
            "assistant": "I cannot access a camera in this chat.",
        }
    ]
    turn = flow["turns"][0]
    assert turn["acceptable_destinations"] == ["observation"]
    assert turn["evidence_expectation"] == "current_view"
    assert turn["query"] not in flow["initial_history"][0]["user"]


def test_monitor_retarget_control_precedes_an_old_scope_current_view_request() -> None:
    flows = yaml.safe_load(_CORPUS.read_text(encoding="utf-8"))
    flow = next(item for item in flows if item["name"] == "monitor_retarget_is_a_routed_control_before_old_scope_view")
    start, retarget, old_scope_view = flow["turns"]
    assert start["acceptable_destinations"] == ["monitor"]
    assert start["destination_actions"] == {"monitor": "start"}
    assert retarget["acceptable_destinations"] == ["monitor"]
    assert retarget["destination_actions"] == {"monitor": "retarget"}
    assert "entrance" in retarget["query"] and "pen" in retarget["query"]
    assert old_scope_view["acceptable_destinations"] == ["observation"]
    assert old_scope_view["evidence_expectation"] == "current_view"
    assert "current view" in old_scope_view["query"]


@pytest.mark.parametrize(
    ("context_before", "expected_precondition", "expected_answer"),
    [
        (None, "not_scored", "not_scored"),
        (
            {"simple-vlm.monitor-window": "Background visual monitoring status: active."},
            False,
            "not_scored",
        ),
        (
            {
                "simple-vlm.monitor-window": (
                    "Background visual monitoring status: active.\n"
                    "- 2026-10-16 07:00 UTC (frame source time): Pen beside the notebook."
                )
            },
            True,
            True,
        ),
    ],
)
def test_monitor_lookup_answer_requires_completed_timestamped_observation(
    context_before, expected_precondition, expected_answer
) -> None:
    flows = yaml.safe_load(_CORPUS.read_text(encoding="utf-8"))
    flow = next(
        flow
        for flow in flows
        if flow["name"] == "active_monitor_routes_task_coverage_separately_from_current_view"
    )
    expected = flow["turns"][1]
    scores = score_turn(
        expected,
        {
            "effective_destination": "conversation",
            "route_outcome": "select",
            "admissions": {"conversation": "accept"},
            "evidence": {"tool_names": []},
            "answer": "The pen is beside the notebook.",
            "context_before": context_before,
        },
    )

    assert scores["context_record_precondition"] is expected_precondition
    assert scores["answer"] == expected_answer


def test_background_monitor_and_notes_focus_are_independent_and_participant_scoped() -> None:
    flow = {
        "name": "monitor_background_with_notes_focus",
        "participants": ["alice", "bob"],
        "turns": [
            {"participant_id": "alice", "query": "Start background monitoring."},
            {"participant_id": "bob", "query": "A separate question."},
            {"participant_id": "alice", "query": "A general question while monitoring."},
            {"participant_id": "alice", "query": "Take some notes."},
            {"participant_id": "alice", "query": "Record this AR/VR note."},
            {"participant_id": "alice", "query": "Read back my notes."},
            {"participant_id": "alice", "query": "I'm finished taking notes."},
            {"participant_id": "alice", "query": "An AR/VR knowledge question after focus."},
            {"participant_id": "alice", "query": "Stop background monitoring."},
            {"participant_id": "alice", "query": "A regular question after stop."},
        ],
    }
    actual = [
        {
            "completed": True,
            "answer": "Monitoring started.",
            "effective_destination": "monitor",
            "destination_disposition": "answer",
            "context_updates": [{"source": "simple-vlm.monitor-window", "mode": "summary", "text": "active"}],
        },
        {"completed": True, "answer": "Bob's independent answer.", "effective_destination": "conversation"},
        {
            "completed": True,
            "answer": "General answer.",
            "effective_destination": "conversation",
        },
        {
            "completed": True,
            "answer": "Note taking started.",
            "effective_destination": "notes",
            "destination_disposition": "focus",
            "destination_actions": {"notes": "start"},
        },
        {
            "completed": True,
            "answer": "Note recorded.",
            "effective_destination": "notes",
            "destination_disposition": "focus",
            "destination_actions": {"notes": "capture"},
            "context_updates": [
                {"source": "simple-vlm.foreground-notes", "mode": "incremental", "text": "Record this AR/VR note."}
            ],
        },
        {
            "completed": True,
            "answer": "Your notes are ...",
            "effective_destination": "notes",
            "destination_disposition": "focus",
            "destination_actions": {"notes": "readback"},
        },
        {
            "completed": True,
            "answer": "Note taking finished.",
            "effective_destination": "notes",
            "destination_disposition": "release",
            "destination_actions": {"notes": "finish"},
        },
        {
            "completed": True,
            "answer": "General answer about AR/VR.",
            "effective_destination": "conversation",
            "context_updates": [{"source": "simple-vlm.monitor-window", "mode": "summary", "text": "latest window"}],
        },
        {
            "completed": True,
            "answer": "Monitoring stopped.",
            "effective_destination": "monitor",
            "destination_disposition": "answer",
            "context_updates": [{"source": "simple-vlm.monitor-window", "mode": "summary", "text": "stopped window"}],
        },
        {"completed": True, "answer": "Regular answer.", "effective_destination": "conversation"},
    ]

    before = completed_flow_state_before_turns(flow, actual)

    assert before[1]["focused_destination"] is None
    assert before[1]["context"] == {}
    assert before[2]["focused_destination"] is None
    assert before[2]["context"] == {"simple-vlm.monitor-window": "active"}
    assert before[3]["focused_destination"] is None
    assert before[4]["focused_destination"] == "notes"
    assert before[5]["focused_destination"] == "notes"
    assert before[6]["focused_destination"] == "notes"
    assert before[7]["focused_destination"] is None
    assert before[7]["context"] == {
        "simple-vlm.monitor-window": "active",
        "simple-vlm.foreground-notes": "Record this AR/VR note.",
    }
    assert before[8]["focused_destination"] is None
    assert before[8]["context"] == {
        "simple-vlm.monitor-window": "latest window",
        "simple-vlm.foreground-notes": "Record this AR/VR note.",
    }
    assert before[9]["focused_destination"] is None
    assert before[9]["context"] == {
        "simple-vlm.monitor-window": "stopped window",
        "simple-vlm.foreground-notes": "Record this AR/VR note.",
    }


def test_flow_history_uses_only_completed_turns() -> None:
    flow = {
        "participant_id": "alice",
        "turns": [
            {"query": "Ask about AR/VR accessibility."},
            {"query": "Continue with that recommendation."},
            {"query": "Recall the earlier suggestion."},
        ],
    }
    actual = [
        {
            "completed": False,
            "answer": None,
        },
        {
            "completed": True,
            "answer": "The conversation completed a recommendation.",
            "destination_disposition": "answer",
            "effective_destination": "conversation",
        },
        {"completed": True, "answer": "The completed suggestion is available."},
    ]

    before = completed_flow_state_before_turns(flow, actual)

    assert before[1]["history"] == []
    assert before[2]["history"] == [
        {
            "user": "Continue with that recommendation.",
            "assistant": "The conversation completed a recommendation.",
        }
    ]

    flow = {
        "initial_history": [{"user": "Earlier", "assistant": "Seed answer."}],
        "turns": [
            {"query": "First question"},
            {"query": "Follow up"},
            {"query": "Never use future turns"},
        ],
    }
    history = completed_history(
        flow["initial_history"],
        flow["turns"],
        [{"completed": False, "answer": "Unfinished partial answer."}],
    )
    assert history == [{"user": "Earlier", "assistant": "Seed answer."}]

    # Traces without a completion marker retain the helper's historical
    # convention that a present answer is a completed exchange.
    history = completed_history(
        flow["initial_history"],
        flow["turns"],
        [{"answer": "Observed first answer."}],
    )
    assert history[-1] == {"user": "First question", "assistant": "Observed first answer."}


@pytest.mark.parametrize(
    ("actual", "expected", "expected_scores"),
    [
        (
            {
                "router_choice": "observation",
                "effective_destination": "observation",
                "route_outcome": "select",
                "admissions": {"observation": "accept"},
                "evidence": {"text": "The display is visible.", "tool_names": ["current_view", "image_query"]},
                "answer": "A whiteboard is visible beside the table.",
            },
            {
                "acceptable_destinations": ["observation"],
                "acceptable_router_choices": ["observation"],
                "route_outcome": "select",
                "admission_expectations": {"observation": "accept"},
                "evidence_expectation": "current_view",
                "answer_any_of": ["whiteboard", "table"],
            },
            {
                "router_choice": True,
                "effective_destination": True,
                "route_outcome": True,
                "admission": {"observation": True},
                "evidence": True,
                "answer": True,
            },
        ),
        (
            {
                "router_choice": "clarify",
                "effective_destination": None,
                "route_outcome": "clarify",
            },
            {
                "acceptable_destinations": [],
                "acceptable_router_choices": ["clarify"],
                "route_outcome": "clarify",
            },
            {
                "router_choice": True,
                "effective_destination": True,
                "route_outcome": True,
                "admission": {},
                "evidence": "not_scored",
                "answer": "not_scored",
            },
        ),
        (
            {
                "router_choice": "observation",
                "effective_destination": "observation",
                "route_outcome": "select",
                "admissions": {},
                "evidence": None,
                "answer": None,
            },
            {
                "acceptable_destinations": ["conversation"],
                "acceptable_router_choices": ["conversation"],
                "route_outcome": "select",
                "admission_expectations": {"conversation": "accept"},
                "evidence_expectation": "none",
                "answer_any_of": ["general"],
            },
            {
                "router_choice": False,
                "effective_destination": False,
                "route_outcome": True,
                "admission": {"conversation": "not_scored"},
                "evidence": "not_scored",
                "answer": "not_scored",
            },
        ),
    ],
)
def test_turn_scoring_keeps_pipeline_layers_separate(actual, expected, expected_scores) -> None:
    assert score_turn(expected, actual) == expected_scores


def test_note_action_and_context_content_are_scored_separately_from_answer() -> None:
    expected = {
        "destination_actions": {"notes": "capture"},
        "context_update_expectations": [{"source": "simple-vlm.foreground-notes", "mode": "incremental"}],
        "context_update_text_contains": "AR/VR labels stay readable",
        "context_updates_must_not_include": ["simple-vlm.monitor-window"],
        "answer_expectation": "nonempty",
    }
    actual = {
        "destination_actions": {"notes": "capture"},
        "context_updates": [
            {
                "source": "simple-vlm.foreground-notes",
                "mode": "incremental",
                "text": "AR/VR labels stay readable at arm's length.",
            }
        ],
        "answer": "I recorded that note.",
    }

    assert score_turn(expected, actual) == {
        "router_choice": "not_scored",
        "effective_destination": "not_scored",
        "route_outcome": "not_scored",
        "destination_actions": {"notes": True},
        "context_updates": {"simple-vlm.foreground-notes": True},
        "context_update_text": True,
        "context_updates_excluded": True,
        "admission": {},
        "evidence": "not_scored",
        "answer": True,
    }


def test_retargeted_monitor_old_visual_scope_case_is_held_out() -> None:
    flows = yaml.safe_load(_CORPUS.read_text(encoding="utf-8"))
    flow = next(flow for flow in flows if flow["name"] == "retargeted_monitor_does_not_cover_old_visual_scope")
    assert flow["image"] == "closed_sign"
    history = flow["initial_history"]
    assert any("entrance sign" in item["user"] for item in history)
    assert any("ARRIVALS" in item["assistant"] for item in history)
    assert any("entrance door" in item["user"] for item in history)
    assert any("blue" in item["assistant"] for item in history)
    assert any(
        "Retarget monitoring from entrance arrivals to my desk pen" in item["user"]
        for item in history
    )
    assert any("desk is beside the window" in item["assistant"] for item in history)
    turn = flow["turns"][0]
    assert "sign" in turn["query"]
    assert "current view" in turn["query"]
    assert turn["acceptable_destinations"] == ["observation"]
    assert turn["evidence_expectation"] == "current_view"
