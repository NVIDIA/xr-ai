# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fast schema, coverage, and small-model prompt-budget checks."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

_SAMPLE = Path(__file__).parents[1]
sys.path.insert(0, str(_SAMPLE / "worker"))

from tea_making_worker.agents.prompts import HUMAN, ROUTER, STEP, VOICE  # noqa: E402
from tea_making_worker.agents.registry import _state_contract  # noqa: E402
from tea_making_worker.runtime.render import render_message  # noqa: E402
from tea_making_worker.spec import load_workflow  # noqa: E402


def run() -> None:
    workflow = load_workflow(_SAMPLE / "yaml" / "workflow.yaml")
    cases = yaml.safe_load((_SAMPLE / "eval" / "cases.yaml").read_text(encoding="utf-8"))
    assert len(ROUTER) <= 300
    assert len(f"{STEP}\n{HUMAN}") <= 350
    assert len(f"{VOICE}\n{HUMAN}") <= 300
    assert "natural spoken language" in HUMAN
    assert "temperature" not in HUMAN.lower()
    assert "Rewrite tool/state" in HUMAN
    assert "already_complete is status, not state" in STEP
    assert "Completion requires" not in STEP
    assert "briefly message real non-completing changes" in STEP
    assert "Empty on no change/completion" in STEP
    assert "if unavailable, say so" in VOICE
    assert "Never route these to ask_step" in ROUTER
    style = cases["spoken_style_guard"]
    assert all(fragment in style["compact_input"] for fragment in style["forbidden_fragments"])
    assert all(fragment not in style["expected_output"] for fragment in style["forbidden_fragments"])
    assert not {"tea", "temperature"} & set(style["compact_input"].lower().split())
    assert {case["expected_tool"] for case in cases["routes"]} == {
        f"workflow__{name}" for name in ("start", "advance", "reset", "status", "ask_step")
    }
    assert set(cases["steps"]) == set(workflow.steps)
    assert cases["rag_fallback"]["expected_tools"] == ["rag_lookup", "workflow__commit"]
    assert cases["rag_fallback"]["expected_top_k"] == 2
    assert cases["rag_fallback"]["expected_tea_ready"] is True
    assert cases["rag_fallback"]["expected_atomic_updates"] == [
        "tea_name",
        "target_temperature_c",
        "steep_duration_s",
        "guidance_source",
        "tea_ready",
    ]
    assert cases["completed_step"] == {
        "step": "fill_water",
        "observation": "The kettle interior is visible with a clear water surface and level inside.",
        "state": {"water_filled": True},
        "expected_tools": ["workflow__commit"],
        "expected_updates": {},
        "expected_transition": None,
    }
    assert cases["false_positive_guard"]["expected_accepted"] is False
    identify = workflow.step("identify")
    assert identify.agent.tools == ("rag_lookup",)
    assert "rag_lookup" in identify.voice.tools
    assert identify.evidence is not None
    guard = cases["identification_guard"]
    readable = bool(re.fullmatch(identify.evidence.pattern, guard["readable_ocr"]))
    partial = bool(re.fullmatch(identify.evidence.pattern, guard["partial_ocr"]))
    unreadable = bool(re.fullmatch(identify.evidence.pattern, guard["unreadable_caption"]))
    empty = bool(re.fullmatch(identify.evidence.pattern, guard["empty_caption"]))
    none = bool(re.fullmatch(identify.evidence.pattern, guard["none_caption"]))
    assert readable is guard["expected_readable"]
    assert partial is guard["expected_partial"]
    assert unreadable is guard["expected_unreadable"]
    assert empty is guard["expected_empty"]
    assert none is guard["expected_empty"]
    assert guard["generic_rag_allowed"] is False
    assert guard["unreadable_expected_updates"] == {}
    assert guard["unreadable_expected_message"] == ""
    atomic = cases["identification_atomic_guard"]
    assert atomic["forbidden_draft_updates"] == [
        "tea_name",
        "target_temperature_c",
        "steep_duration_s",
        "guidance_source",
        "tea_ready",
    ]
    assert atomic["expected_unready_state"] == {}
    assert atomic["current_observation_wins"] is True
    question = str(identify.trigger.arguments["question"])
    assert "front label brand and tea/blend name" in question
    assert "Ignore slogans, claims, bag count, weight" in question
    identify_prompt = identify.agent.prompt.lower()
    assert "identify tea only from the current caption" in identify_prompt
    assert "never state or rag" in identify_prompt
    assert "never write tea_ready false" in identify_prompt
    assert "commit all with tea_ready true" in identify_prompt
    assert cases["irrelevant_retrieval_guard"]["forbidden_updates"] == [
        "target_temperature_c",
        "steep_duration_s",
        "tea_ready",
    ]
    mismatch = cases["identity_retrieval_mismatch"]
    assert mismatch["expected_tools"] == ["rag_lookup", "workflow__commit"]
    assert mismatch["expected_updates"] == {}
    assert mismatch["expected_tea_ready"] is False
    identity = cases["tea_identity_question"]
    assert identity["step"] == "identify"
    assert identity["expected_tool"] == "current_view"
    assert identity["forbidden_tool"] == "rag_lookup"
    heat = workflow.step("heat_water")
    assert heat.evidence is not None
    temperature = cases["temperature_unit_guard"]
    assert temperature["step"] == "heat_water"
    assert temperature["expected_updates"] == {"heating_started": True}
    assert temperature["expected_water_ready"] is False
    terse = cases["terse_temperature_caption"]
    assert re.fullmatch(heat.evidence.pattern, terse["observation"])
    assert terse["expected_evidence"] is True
    assert terse["expected_updates"] == {"heating_started": True}
    assert terse["expected_water_ready"] is False
    missing_unit = cases["temperature_missing_unit_guard"]
    assert missing_unit["expected_updates"] == {}
    assert missing_unit["expected_water_ready"] is False
    state_notice = cases["state_update_notice"]
    assert state_notice["expected_updates"] == {"heating_started": True}
    assert state_notice["expected_message_intent"] == "heating started"
    assert state_notice["expected_complete"] is False
    routine_notice = cases["routine_notice_guard"]
    assert routine_notice["expected_updates"] == {}
    assert routine_notice["expected_messages"] == ["", "", ""]
    fill = workflow.step("fill_water")
    assert "closed vessel" in fill.agent.prompt
    assert "user report is not" in fill.voice.prompt
    assert "sets heating_started true" in heat.agent.prompt
    assert "Never store readings/conversions" in heat.agent.prompt
    assert "current heating, reading, or readiness" in heat.voice.prompt
    context = cases["observation_context"]
    assert context["keys"] == ["observation", "already_complete", "state"]
    assert context["prior_status_is_state_value"] is False
    assert context["contract_includes"] == ["field_descriptions", "completion_condition"]
    assert context["writable_fields_are_goals"] is False
    rag_config = yaml.safe_load((_SAMPLE / "yaml" / "rag_service.yaml").read_text(encoding="utf-8"))
    retrieval = cases["retrieval_context"]
    assert rag_config["chunk_size"] == retrieval["chunk_size"]
    assert rag_config["overlap"] == retrieval["overlap"]
    assert retrieval["query_suffix"] in identify.agent.prompt
    steeping = workflow.step("start_steeping")
    assert steeping.evidence is not None
    assert not re.fullmatch(
        steeping.evidence.pattern,
        cases["steeping_negative_guard"]["observation"],
    )
    assert cases["steeping_negative_guard"]["expected_accepted"] is False
    assert "do not call clock__now" in steeping.agent.prompt
    assert "not visual confirmation" in steeping.voice.prompt
    timer = workflow.step("steep_timer")
    assert "While false, commit empty" in timer.agent.prompt
    assert "for every timer or completion question" in timer.voice.prompt
    assert cases["timer_running_guard"]["expected_updates"] == {}
    assert cases["timer_running_guard"]["expected_steeping_complete"] is False
    voice_tools = cases["voice_tool_guards"]
    assert {case["expected_tool"] for case in voice_tools.values()} == {
        "current_view",
        "clock__timer",
    }
    spoken = cases["spoken_values"]
    assert render_message("{{ value | temperature_c }}", {"value": 100}) == spoken["temperature_c"]
    assert render_message("{{ value | duration }}", {"value": 240}) == spoken["duration_240_s"]
    assert render_message("{{ value | duration }}", {"value": 65}) == spoken["duration_65_s"]
    for step in workflow.steps.values():
        question = str(step.trigger.arguments.get("question", ""))
        assert len(question) <= 240, f"{step.id} VLM question is too large"
        assert len(step.agent.prompt) <= 420, f"{step.id} observation prompt is too large"
        assert len(step.voice.prompt) <= 300, f"{step.id} voice prompt is too large"
        assert len(_state_contract(workflow, step)) <= 500, f"{step.id} state contract is too large"
    print(f"validated {len(workflow.steps)} steps and {len(cases['routes'])} routes")


if __name__ == "__main__":
    run()
