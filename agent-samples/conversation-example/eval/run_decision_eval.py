# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one bounded, teacher-forced Clef routing/control pass over sample fixtures.

This evaluator does not call answer, VLM, voice, or compaction services; alert
compilation returns a fixed fixture watch. Prior turns use explicitly authored
fixture completions; they are inputs, never scored as generated answers.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[3]
_SAMPLE = _ROOT / "agent-samples/conversation-example"
sys.path.insert(0, str(_SAMPLE / "worker"))

from conversation_example_worker._alert_agent import VisualAlertAgent  # noqa: E402
from conversation_example_worker._app import _CONVERSATION_SCOPE  # noqa: E402
from conversation_example_worker._note_taking import NoteTakingAgent  # noqa: E402
from conversation_example_worker._visual_agents import VisualObservationAgent  # noqa: E402
from xr_ai_conversation import (  # noqa: E402
    ContextManager,
    Conversation,
)
from xr_ai_models import ChatResponse, DecisionResponse, load_models_config, make_decision  # noqa: E402
from xr_ai_routing import (  # noqa: E402
    AgentExchange,
    AgentRouter,
    AgentTurn,
    RoutingAid,
)
from xr_ai_tools.current_frame import ImageFrame  # noqa: E402
from xr_ai_voice import VOICE_OUTPUT_TOPIC  # noqa: E402

_FLOW_FILE = _SAMPLE / "eval/conversation-flows.yaml"
_CASE_FILE = _SAMPLE / "eval/conversation-cases.yaml"
_MODELS_FILE = _SAMPLE / "yaml/models.json"
_MAX_CALLS = 100
_TRACE_SOURCES = (
    _SAMPLE / "worker/conversation_example_worker/_app.py",
    _SAMPLE / "worker/conversation_example_worker/_note_taking.py",
    _SAMPLE / "worker/conversation_example_worker/_alert_agent.py",
    _SAMPLE / "worker/conversation_example_worker/_visual_agents.py",
    _SAMPLE / "worker/conversation_example_worker/prompts/conversation.txt",
    _SAMPLE / "worker/conversation_example_worker/prompts/note_control.txt",
    _SAMPLE / "worker/conversation_example_worker/prompts/alert_control.txt",
    _SAMPLE / "worker/conversation_example_worker/prompts/alert_condition.txt",
    _SAMPLE / "worker/conversation_example_worker/prompts/alert_description.txt",
    _ROOT / "agent-sdk/xr-ai-routing/xr_ai_routing/_router.py",
    _ROOT / "agent-sdk/xr-ai-routing/xr_ai_routing/_history.py",
)
_ALERT_SOURCE = "conversation-example.alert-capability"
_NOTES_SOURCE = "conversation-example.foreground-notes"
_OBSERVATION_SOURCE = "conversation-example.observation-capability"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_resume_calls(path: Path, expected_hashes: dict[str, str]) -> tuple[list[dict[str, Any]], str]:
    calls = []
    summary = None
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("type") == "decision":
            calls.append({key: value for key, value in item.items() if key != "type"})
        elif item.get("type") == "summary":
            summary = item
    if summary is None:
        raise ValueError("resume source must contain a completed summary record")
    previous_hashes = summary.get("preflight", {}).get("sha256")
    if not isinstance(previous_hashes, dict):
        raise ValueError("resume source summary has no fixture/config/source hashes")
    evaluator_path = str(Path(__file__).relative_to(_ROOT))
    previous_sources = {key: value for key, value in previous_hashes.items() if key != evaluator_path}
    expected_sources = {key: value for key, value in expected_hashes.items() if key != evaluator_path}
    if previous_sources != expected_sources:
        raise ValueError("resume source fixture/config/source hashes do not match the current evaluation snapshot")
    return calls, previous_hashes[evaluator_path]


class _DecisionRecorder:
    def __init__(self, service: Any, *, max_calls: int, prior_calls: list[dict[str, Any]] | None = None) -> None:
        self.service = service
        self.max_calls = max_calls
        self.calls: list[dict[str, Any]] = list(prior_calls or [])
        self.case_name = ""
        self.participant_id = ""

    async def decide(self, state: Any, questions: Any, *, images=(), timeout=None) -> DecisionResponse:
        if len(self.calls) >= self.max_calls:
            raise RuntimeError(f"decision evaluation reached its {_MAX_CALLS}-call safety cap")
        started = time.monotonic()
        trace = {
            "case": self.case_name,
            "participant_id": self.participant_id,
            "request_state": _jsonable(state),
            "questions": _jsonable(questions),
        }
        try:
            response = await self.service.decide(state, questions, images=images, timeout=timeout)
        except Exception as error:
            # Record the failed model call before the evaluator's owner logs and propagates it.
            self.calls.append(
                {
                    **trace,
                    "error": f"{type(error).__name__}: {error}",
                    "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
                }
            )
            raise
        self.calls.append(
            {
                **trace,
                "response": {
                    name: {
                        "choice": item.choice,
                        "confidence": item.confidence,
                        "probabilities": dict(item.probabilities),
                    }
                    for name, item in response.answers.items()
                },
                "model": response.model,
                "usage": _jsonable(response.usage),
                "raw": _jsonable(response.raw),
                "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
            }
        )
        return response


class _NoAnswerModel:
    async def stream(self, *_args, **_kwargs):
        raise AssertionError("decision-only evaluation must not generate answers")


class _FixtureAlertCompiler:
    """Teacher-force alert compilation: routing fixtures score control decisions, not compiled watches."""

    async def chat(self, *_args, **_kwargs) -> ChatResponse:
        watch = {"status": "ready", "condition": "the fixture condition is visible", "relevance": "the fixture subject"}
        return ChatResponse(content=json.dumps(watch), reasoning=None, tool_calls=None, finish_reason="stop", raw={})


class _Frames:
    def __init__(self) -> None:
        self.block = asyncio.Event()

    async def execute(self, _request: Any) -> ImageFrame:
        # Routing-only evaluation starts jobs but never supplies camera observations.
        await self.block.wait()
        raise AssertionError("decision-only evaluation must not acquire an image")


class _NoVoiceRuntime:
    async def publish(self, _topic, _message, **_metadata) -> None:
        """Discard sample control replies; this evaluator never runs voice services."""


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def load_fixtures() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    flows = yaml.safe_load(_FLOW_FILE.read_text(encoding="utf-8"))
    cases = yaml.safe_load(_CASE_FILE.read_text(encoding="utf-8"))
    if not isinstance(flows, list) or not isinstance(cases, list):
        raise ValueError("conversation fixtures must be YAML lists")
    return flows, cases


def preflight(flows: list[dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    flow_turns = sum(len(flow["turns"]) for flow in flows)
    names: set[str] = set()
    participant_owners: dict[str, str] = {}
    routes = controls = release_reroutes = 0
    for flow in flows:
        participants = {str(flow.get("participant_id", f"eval:{flow['name']}"))}
        participants.update(str(turn["participant_id"]) for turn in flow["turns"] if "participant_id" in turn)
        for participant in participants:
            owner = participant_owners.setdefault(participant, flow["name"])
            if owner != flow["name"]:
                raise ValueError(
                    f"participant {participant!r} is shared by independent flows {owner!r} and {flow['name']!r}"
                )
        for turn in flow["turns"]:
            name = f"{flow['name']}/{turn['name']}"
            if name in names:
                raise ValueError(f"duplicate fixture turn {name!r}")
            names.add(name)
            if not turn.get("query", "").strip() or not turn.get("acceptable_destinations"):
                raise ValueError(f"fixture {name!r} lacks a query or route label")
            if not turn.get("authored_completion", "").strip():
                raise ValueError(f"fixture {name!r} lacks explicit authored history completion metadata")
            routes += bool(turn.get("router_invoked", True))
            controls += len(turn.get("destination_actions") or {})
            release_reroutes += "release" in (turn.get("admission_expectations") or {}).values()
    for case in cases:
        name = str(case["name"])
        if name in names:
            raise ValueError(f"duplicate fixture turn {name!r}")
        names.add(name)
        if not case.get("query", "").strip() or not case.get("destination") or not case.get("route"):
            raise ValueError(f"single-turn fixture {name!r} lacks query/destination")
        routes += 1
    planned = routes + controls + release_reroutes
    if planned > _MAX_CALLS:
        raise ValueError(f"planned decision count {planned} exceeds the {_MAX_CALLS}-call cap")
    return {
        "flow_count": len(flows),
        "flow_turns": flow_turns,
        "single_turn_cases": len(cases),
        "total_turns": flow_turns + len(cases),
        "route_calls_planned": routes,
        "native_action_calls_planned_max": controls,
        "release_reroutes_expected": release_reroutes,
        "planned_decision_calls": planned,
        "max_decision_calls": _MAX_CALLS,
        "call_count_note": (
            "The plan includes labeled controls and expected reroutes; unexpected declines can add bounded FE reroutes."
        ),
        "history_policy": "authored teacher-forced history only; never generated or answer-scored",
        "sha256": {
            str(path.relative_to(_ROOT)): _sha256(path)
            for path in (_FLOW_FILE, _CASE_FILE, _MODELS_FILE, Path(__file__), *_TRACE_SOURCES)
        },
    }


def _destinations(recorder: _DecisionRecorder, context_budget: int):
    frames = _Frames()
    output_runtime = _NoVoiceRuntime()
    context_sources = (_ALERT_SOURCE, _NOTES_SOURCE, _OBSERVATION_SOURCE)
    section_budget = context_budget // (len(context_sources) + 1)
    if section_budget <= 0:
        raise ValueError("context budget must provide a positive budget for dialogue and each context source")
    topic_budgets = {name: section_budget for name in context_sources}
    context_updates: dict[str, list[dict[str, str]]] = defaultdict(list)

    async def publish_alerts(participant: str, update: Any) -> None:
        context_updates[participant].append({"source": _ALERT_SOURCE, "mode": "summary", "text": update.text})

    async def publish_notes(participant: str, update: Any) -> None:
        context_updates[participant].append({"source": _NOTES_SOURCE, "mode": "summary", "text": update.text})

    manager = ContextManager(
        _NoAnswerModel(),
        summary_context_topics=context_sources,
        dialogue_budget_bytes=section_budget,
        topic_budget_bytes=topic_budgets,
        compaction_input_budget_bytes=context_budget,
    )
    conversation = Conversation(
        _NoAnswerModel(),
        context=manager,
        routing=RoutingAid("conversation", _CONVERSATION_SCOPE),
        runtime=output_runtime,
        output_topic=VOICE_OUTPUT_TOPIC,
    )
    notes_conversation = Conversation(
        _NoAnswerModel(),
        context=ContextManager(
            _NoAnswerModel(),
            dialogue_budget_bytes=context_budget,
            compaction_input_budget_bytes=context_budget,
        ),
        runtime=output_runtime,
        output_topic=VOICE_OUTPUT_TOPIC,
    )
    alerts = VisualAlertAgent(
        frames,
        None,
        recorder,
        publish_alerts,
        vlm=None,
        llm=_FixtureAlertCompiler(),
        context=manager,
        runtime=output_runtime,
        output_topic=VOICE_OUTPUT_TOPIC,
    )
    destinations = {
        "conversation": conversation,
        "observation": VisualObservationAgent(None, None, runtime=output_runtime, output_topic=VOICE_OUTPUT_TOPIC),
        "alerts": alerts,
        "notes": NoteTakingAgent(
            notes_conversation,
            recorder,
            context_budget_bytes=context_budget,
            publish_context=publish_notes,
            runtime=output_runtime,
            output_topic=VOICE_OUTPUT_TOPIC,
        ),
    }

    async def discard_router_output(_turn: AgentTurn, _source: str, _text: str) -> None:
        # The decision evaluator calls selection/admission directly and does not
        # exercise user-facing router control output.
        pass

    return (
        AgentRouter(
            destinations,
            recorder,
            output=discard_router_output,
            context_budget_bytes=context_budget,
        ),
        destinations,
        alerts,
        notes_conversation,
        frames,
        context_updates,
    )


async def _started() -> bool:
    """Response-start signal for destination answers called without a router."""

    return True


async def run_decisions(flows, cases, recorder, *, context_budget: int):
    router, destinations, alerts, notes_conversation, frames, context_updates = _destinations(recorder, context_budget)
    turns_out: list[dict[str, Any]] = []
    metrics: dict[str, list[bool]] = defaultdict(list)
    attempted_cases = {call.get("case") for call in recorder.calls}
    attempted_flows = {case.split("/", 1)[0] for case in attempted_cases if isinstance(case, str) and "/" in case}
    try:
        for flow in flows:
            if flow["name"] in attempted_flows:
                for fixture in flow["turns"]:
                    case_name = f"{flow['name']}/{fixture['name']}"
                    reason = (
                        "not_scored_prior_attempt_preserved"
                        if case_name in attempted_cases
                        else "not_scored_flow_state_not_replayed"
                    )
                    turns_out.append(
                        _not_run_turn(
                            case_name,
                            fixture,
                            str(fixture.get("participant_id", flow.get("participant_id", f"eval:{flow['name']}"))),
                            reason,
                        )
                    )
                continue
            initial = [
                {"user": item["user"], "assistant": item["assistant"], "provenance": "authored_initial_history"}
                for item in flow.get("initial_history", [])
            ]
            default_pid = str(flow.get("participant_id", f"eval:{flow['name']}"))
            prior: dict[str, list[dict[str, str]]] = {default_pid: initial}
            for fixture in flow["turns"]:
                pid = str(fixture.get("participant_id", default_pid))
                history_rows = prior.setdefault(pid, [])
                history = tuple(AgentExchange(item["user"], item["assistant"]) for item in history_rows)
                item = await _evaluate_turn(
                    f"{flow['name']}/{fixture['name']}",
                    fixture,
                    pid,
                    history,
                    history_rows,
                    router,
                    destinations,
                    recorder,
                    metrics,
                    frames,
                    context_updates,
                )
                turns_out.append(item)
                if item.get("run_status") == "not_scored_missing_precondition":
                    for dependent in flow["turns"][flow["turns"].index(fixture) + 1 :]:
                        turns_out.append(
                            _not_run_turn(
                                f"{flow['name']}/{dependent['name']}",
                                dependent,
                                str(dependent.get("participant_id", default_pid)),
                                "not_scored_upstream_precondition_missing",
                            )
                        )
                    break
                prior[pid].append(
                    {
                        "user": fixture["query"],
                        "assistant": fixture["authored_completion"],
                        "provenance": "authored_teacher_forced_completion",
                    }
                )
        for fixture in cases:
            if fixture["name"] in attempted_cases:
                turns_out.append(
                    _not_run_turn(
                        str(fixture["name"]),
                        fixture,
                        f"decision-eval:{fixture['name']}",
                        "not_scored_prior_attempt_preserved",
                    )
                )
                continue
            history_rows = [
                {"user": row["user"], "assistant": row["assistant"], "provenance": "authored_single_case_history"}
                for row in fixture.get("history", [])
            ]
            history = tuple(AgentExchange(row["user"], row["assistant"]) for row in history_rows)
            turns_out.append(
                await _evaluate_turn(
                    str(fixture["name"]),
                    fixture,
                    f"decision-eval:{fixture['name']}",
                    history,
                    history_rows,
                    router,
                    destinations,
                    recorder,
                    metrics,
                    frames,
                    context_updates,
                )
            )
    finally:
        await alerts.stop()
        await router.stop()
        await notes_conversation.close()
    statuses = [item.get("run_status", "unknown") for item in turns_out]
    status_counts = {status: statuses.count(status) for status in sorted(set(statuses))}
    return turns_out, {
        "scores": {key: {"scored": len(values), "passed": sum(values)} for key, values in metrics.items()},
        "answer_scoring": "not_run",
        "evidence_scoring": "not_run",
        "decision_calls_total_including_resume_source": len(recorder.calls),
        "turn_status_counts": status_counts,
        "turns_with_prior_decision_calls_preserved": status_counts.get("not_scored_prior_attempt_preserved", 0),
        "turns_not_run_after_missing_precondition": status_counts.get("not_scored_missing_precondition", 0)
        + status_counts.get("not_scored_upstream_precondition_missing", 0),
        "turns_not_run_because_prior_flow_state_was_not_replayed": status_counts.get(
            "not_scored_flow_state_not_replayed", 0
        ),
    }


def _not_run_turn(name: str, fixture: dict[str, Any], participant: str, reason: str) -> dict[str, Any]:
    return {
        "fixture": name,
        "participant_id": participant,
        "query": fixture["query"],
        "run_status": reason,
        "answer": None,
        "completed": False,
        "answer_status": reason,
        "evidence": None,
        "evidence_status": reason,
        "disposition": "not_run",
    }


async def _evaluate_turn(
    name,
    fixture,
    participant,
    history,
    history_rows,
    router,
    destinations,
    recorder,
    metrics,
    frames,
    context_updates,
):
    turn = AgentTurn(participant, name, fixture["query"], history)
    updates_before = len(context_updates[participant])
    context_before = {}
    for update in context_updates[participant]:
        context_before[update["source"]] = update["text"]
    expected_focus_before = fixture.get("focus_before", ...)
    actual_focus_before = router._focus.get(participant)
    if expected_focus_before is not ... and actual_focus_before != expected_focus_before:
        metrics["focus_before"].append(False)
    elif expected_focus_before is not ...:
        metrics["focus_before"].append(True)
    candidates = [router._focus[participant]] if participant in router._focus else list(destinations)
    attempts: list[dict[str, Any]] = []
    final_destination = None
    exhausted_declines = False
    final_disposition = None
    first_router_choice = None
    for attempt in range(2):
        recorder.case_name = name
        recorder.participant_id = participant
        candidate_list = (
            candidates
            if attempt == 0
            else [key for key in destinations if key not in {entry["candidate"] for entry in attempts}]
        )
        if not candidate_list:
            break
        calls_before = len(recorder.calls)
        selected = await router._select(turn, candidate_list)
        decision = recorder.calls[-1] if len(recorder.calls) > calls_before else None
        decision_answer = (decision or {}).get("response", {}).get("destination", {})
        raw_choice = decision_answer.get("choice")
        if first_router_choice is None and decision is not None:
            first_router_choice = raw_choice
        entry = {
            "candidate": selected,
            "router_choice": raw_choice,
            "effective_selection": selected,
            "selection_call_index": len(recorder.calls) - 1 if decision is not None else None,
        }
        attempts.append(entry)
        if selected is None:
            break
        final_destination = selected
        router._destinations_used.setdefault(participant, set()).add(selected)
        admission_call_count = len(recorder.calls)
        admission = await destinations[selected].admit(turn, focused=selected == router._focus.get(participant))
        entry["admission"] = admission.disposition
        if len(recorder.calls) > admission_call_count:
            entry["admission_decision_call_index"] = admission_call_count
            native_key = "alert_action" if selected == "alerts" else "note_action" if selected == "notes" else None
            if native_key is not None:
                native_response = recorder.calls[admission_call_count]["response"].get(native_key, {})
                if native_response:
                    entry["native_decision"] = {"type": native_key, **native_response}
                    entry["action"] = native_response.get("choice")
        if admission.disposition == "decline":
            final_destination = None
            final_disposition = "decline"
            exhausted_declines = attempt == 1
            continue
        if admission.disposition == "release":
            await destinations[selected].release(participant)
            router._focus.pop(participant, None)
            if selected == "notes":
                router._forget_destination(participant, selected)
                destinations[selected]._pending.pop(participant, None)
            final_destination = None
            final_disposition = "release"
            continue
        if admission.disposition == "clarify":
            final_disposition = "clarify"
            break
        final_disposition = "accept"
        if selected == "alerts":
            pending = destinations[selected]._pending.get(participant)
            action = pending[1] if pending and pending[0] == turn.turn_id else None
            if action:
                action_call_index = len(recorder.calls) - 1
                await destinations[selected].answer(turn, started=_started)
                entry["action"] = action
                entry["action_decision_call_index"] = action_call_index
        elif selected == "notes":
            pending = destinations[selected]._pending.get(participant)
            action = pending[1] if pending and pending[0] == turn.turn_id else None
            if action in {"start", "capture", "readback", "finish"}:
                action_call_index = len(recorder.calls) - 1
                result = await destinations[selected].answer(turn, started=_started)
                entry["action"] = action
                entry["action_decision_call_index"] = action_call_index
                if result.disposition == "focus":
                    router._focus[participant] = selected
                elif result.disposition == "release":
                    await destinations[selected].release(participant)
                    router._focus.pop(participant, None)
                    router._forget_destination(participant, selected)
            elif action == "other":
                destinations[selected]._pending.pop(participant, None)
        break

    expected = set(
        fixture.get(
            "acceptable_destinations",
            ["conversation", "observation"] if fixture.get("route") == "clarify" else [fixture.get("destination")],
        )
    )
    if exhausted_declines:
        default = router._destinations[router._default_destination]
        aid = default.routing_aid(participant)
        declined = {entry["candidate"] for entry in attempts if entry.get("admission") == "decline"}
        if aid.eligible and router._default_destination not in declined:
            final_destination = router._default_destination
            attempts.append(
                {
                    "candidate": final_destination,
                    "router_choice": None,
                    "effective_selection": final_destination,
                    "admission": "not_checked_default_fallback",
                    "selection_reason": "exhausted_declines",
                }
            )
            final_disposition = "default_fallback"
    raw_choice = first_router_choice
    effective = final_destination
    router_was_invoked = any(item.get("selection_call_index") is not None for item in attempts)
    raw_default = ["clarify"] if fixture.get("route") == "clarify" else [fixture.get("destination")]
    expected_raw_choices = set(fixture.get("acceptable_router_choices", raw_default))
    result = {
        "fixture": name,
        "participant_id": participant,
        "query": turn.query,
        "run_status": "completed_decisions",
        "history_source": "authored_teacher_forced" if history else "empty_fixture_history",
        "history": [dict(item) for item in history_rows],
        "published_context_before": context_before,
        "published_context_sources_before": sorted(context_before),
        "published_context_updates": context_updates[participant][updates_before:],
        "focus_before": actual_focus_before,
        "router_invoked": router_was_invoked,
        "raw_router_choice": raw_choice,
        "effective_destination": effective,
        "attempts": attempts,
        "acceptable_destinations": sorted(item for item in expected if item),
        "raw_route_match": raw_choice in expected_raw_choices if router_was_invoked else "not_scored_focus_bypass",
        "route_outcome": "clarify" if final_disposition == "clarify" or effective is None else "selected",
        "answer": None,
        "completed": False,
        "answer_status": "not_scored_decision_only",
        "evidence": None,
        "evidence_status": "not_scored_no_tool_calls",
        "disposition": final_disposition or "not_scored_answer_path_not_run",
    }
    if router_was_invoked:
        metrics["raw_route"].append(raw_choice in expected_raw_choices)
    route_expectation = fixture.get("route_outcome")
    if route_expectation is not None:
        actual_outcome = "select" if result["route_outcome"] == "selected" else result["route_outcome"]
        result["route_outcome_match"] = actual_outcome == route_expectation
        metrics["route_outcome"].append(result["route_outcome_match"])
    if effective is not None:
        result["effective_destination_match"] = effective in expected
        metrics["effective_destination"].append(effective in expected)
    action_expectations = fixture.get("destination_actions") or {}
    admission_expectations = fixture.get("admission_expectations") or {}
    for candidate, expected_admission in admission_expectations.items():
        actual_admission = next(
            (attempt.get("admission") for attempt in attempts if attempt.get("candidate") == candidate),
            None,
        )
        if actual_admission == "not_checked_default_fallback":
            actual_admission = None
        result.setdefault("admissions", {})[candidate] = {
            "expected": expected_admission,
            "actual": actual_admission if actual_admission is not None else "not_scored_not_invoked",
            "match": actual_admission == expected_admission if actual_admission is not None else None,
        }
        if actual_admission is not None:
            metrics["native_admission"].append(actual_admission == expected_admission)
    for destination, expected_action in action_expectations.items():
        actual_action = next(
            (attempt.get("action") for attempt in attempts if attempt["candidate"] == destination), None
        )
        result.setdefault("native_action", {})[destination] = {
            "expected": expected_action,
            "actual": actual_action if actual_action is not None else "not_scored_not_selected",
            "match": actual_action == expected_action if actual_action is not None else None,
        }
        if actual_action is not None:
            metrics[f"{destination}_action"].append(actual_action == expected_action)
    has_update_expectations = fixture.get("context_update_expectations") is not None
    has_update_exclusions = fixture.get("context_updates_must_not_include") is not None
    if has_update_expectations or has_update_exclusions:
        observed_updates = result["published_context_updates"]
        expected_updates = fixture.get("context_update_expectations", [])
        forbidden_sources = fixture.get("context_updates_must_not_include", [])
        result["published_context_update_match"] = all(
            any(
                update["source"] == expected_update["source"] and update["mode"] == expected_update["mode"]
                for update in observed_updates
            )
            for expected_update in expected_updates
        ) and not any(update["source"] in forbidden_sources for update in observed_updates)
        metrics["published_context_updates"].append(result["published_context_update_match"])
    if "focus_after" in fixture:
        actual_focus = router._focus.get(participant)
        result["focus_after"] = actual_focus
        result["focus_match"] = actual_focus == fixture["focus_after"]
        metrics["focus_transition"].append(result["focus_match"])
    if expected_focus_before is not ...:
        result["focus_before_match"] = actual_focus_before == expected_focus_before
    return result


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    flows, cases = load_fixtures()
    plan = preflight(flows, cases)
    prior_calls = []
    prior_evaluator_hash = None
    if args.resume_from is not None:
        prior_calls, prior_evaluator_hash = _load_resume_calls(args.resume_from, plan["sha256"])
        plan["resume_source_evaluator_sha256"] = prior_evaluator_hash
        plan["resume_current_evaluator_sha256"] = plan["sha256"][str(Path(__file__).relative_to(_ROOT))]
        plan["resume_source_match_note"] = (
            "fixture, model config, and production source hashes match; evaluator hashes may differ for resume handling"
        )
    if len(prior_calls) + plan["planned_decision_calls"] > _MAX_CALLS:
        attempted_flows = {call["case"].split("/", 1)[0] for call in prior_calls if "/" in call.get("case", "")}
        attempted_cases = {call.get("case") for call in prior_calls}
        remaining_flows = [flow for flow in flows if flow["name"] not in attempted_flows]
        remaining_cases = [case for case in cases if case["name"] not in attempted_cases]
        plan = preflight(remaining_flows, remaining_cases)
    if len(prior_calls) + plan["planned_decision_calls"] > _MAX_CALLS:
        raise ValueError(
            f"resume would exceed the {_MAX_CALLS}-call total cap with {len(prior_calls)} preserved decisions"
        )
    plan["prior_decision_calls_preserved"] = len(prior_calls)
    plan["decision_calls_total_planned"] = len(prior_calls) + plan["planned_decision_calls"]
    print(json.dumps(plan, indent=2))
    if args.preflight_only:
        return
    models = load_models_config(_MODELS_FILE)
    recorder = _DecisionRecorder(make_decision(models, "router"), max_calls=_MAX_CALLS, prior_calls=prior_calls)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    summary = {}
    failure = None
    try:
        outputs, summary = await run_decisions(flows, cases, recorder, context_budget=12_000)
        summary = {**summary, "run_status": "completed", "preflight": plan}
    # Preserve a failure summary even when cancellation or process shutdown interrupts evaluation.
    except BaseException as error:  # noqa: BLE001 -- write failure output, then re-raise the same failure.
        failure = error
        summary = {
            "run_status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "decision_calls": len(recorder.calls),
            "answer_scoring": "not_run",
            "evidence_scoring": "not_run",
            "preflight": plan,
        }
    finally:
        with args.output.open("w", encoding="utf-8") as stream:
            for call in recorder.calls:
                stream.write(json.dumps({"type": "decision", **call}, ensure_ascii=False) + "\n")
            for item in outputs:
                stream.write(json.dumps({"type": "turn", **item}, ensure_ascii=False) + "\n")
            stream.write(json.dumps({"type": "summary", **summary}, ensure_ascii=False) + "\n")
    if failure is not None:
        raise failure
    print(json.dumps({**summary, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
