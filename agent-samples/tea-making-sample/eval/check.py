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

from tea_making_worker.agents.prompts import (  # noqa: E402
    HUMAN,
    ROOT,
    STEP,
    TEA,
    VOICE,
)
from tea_making_worker.agents.registry import (  # noqa: E402
    _ROOT_TOOLS,
    _TEA_MANAGEMENT_TOOLS,
    _state_contract,
)
from tea_making_worker.runtime.render import render_message  # noqa: E402
from tea_making_worker.spec import load_workflow  # noqa: E402


def run() -> None:
    workflow = load_workflow(_SAMPLE / "yaml" / "workflow.yaml")
    cases = yaml.safe_load((_SAMPLE / "eval" / "cases.yaml").read_text(encoding="utf-8"))
    assert len(f"{ROOT}\n{HUMAN}") <= 450
    assert len(f"{TEA}\n{VOICE}\n{HUMAN}") <= 550
    assert len(f"{STEP}\n{HUMAN}") <= 350
    assert "natural spoken language" in HUMAN
    assert "temperature" not in HUMAN.lower()
    assert "Rewrite tool/state" in HUMAN
    assert "already_complete is status, not state" in STEP
    assert "Completion requires" not in STEP
    assert "message only with a real non-completing state change" in STEP
    assert "Empty on no change or completion" in STEP
    assert "if unavailable, say so" in VOICE
    assert "Explicit request to begin tea guidance: workflow__start" in ROOT
    assert "Next/continue/advance" in TEA
    assert "Questions using these words are not commands" in TEA
    assert tuple(map(str, _ROOT_TOOLS)) == ("workflow__start", "current_view", "rag_lookup")
    assert tuple(map(str, _TEA_MANAGEMENT_TOOLS)) == (
        "workflow__advance",
        "workflow__reset",
        "workflow__restart",
        "workflow__status",
    )
    style = cases["spoken_style_guard"]
    assert all(fragment in style["compact_input"] for fragment in style["forbidden_fragments"])
    assert all(fragment not in style["expected_output"] for fragment in style["forbidden_fragments"])
    assert not {"tea", "temperature"} & set(style["compact_input"].lower().split())
    lifecycle_actions = {
        f"workflow__{name}" for name in ("start", "advance", "reset", "restart", "status")
    }
    assert {case["expected_tool"] for case in cases["routes"]} == {"answer"} | lifecycle_actions
    matrix = cases["route_state_matrix"]
    assert set(matrix["active_steps"]) == set(workflow.steps)
    active_actions = {
        f"workflow__{name}" for name in ("advance", "reset", "restart", "status")
    }
    assert {case["expected_tool"] for case in matrix["active_cases"]} == {"answer"} | active_actions
    assert {case["expected_tool"] for case in matrix["idle_cases"]} == {
        "workflow__start",
        "answer",
    }
    advance_cases = [
        case for case in matrix["active_cases"] if case["expected_tool"] == "workflow__advance"
    ]
    assert {case["expected_advanced"] for case in advance_cases} == {False, True}
    general = cases["general_queries"]
    assert general["tea_knowledge"]["expected_tools"] == ["rag_lookup"]
    assert general["visible_scene"]["expected_tools"] == ["current_view"]
    assert general["visible_tea"]["expected_order"] == ["current_view", "rag_lookup"]
    assert general["safety"] == {"state_updates": False, "lifecycle_tools": False}
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
    assert "water_ready" not in workflow.state_fields
    temperature = cases["temperature_unit_guard"]
    assert temperature["step"] == "heat_water"
    assert temperature["forbidden_tool"] == "temperature__verify"
    assert temperature["expected_updates"] == {"heating_started": True}
    assert temperature["expected_complete"] is True
    terse = cases["terse_temperature_caption"]
    assert re.fullmatch(heat.evidence.pattern, terse["observation"])
    assert terse["expected_evidence"] is True
    assert terse["expected_updates"] == {"heating_started": True}
    assert terse["expected_complete"] is True
    split_captions = cases["split_temperature_caption_guard"]
    assert all(re.fullmatch(heat.evidence.pattern, item) for item in split_captions["accepted"])
    assert not any(re.fullmatch(heat.evidence.pattern, item) for item in split_captions["rejected"])
    missing_unit = cases["temperature_missing_unit_guard"]
    assert missing_unit["forbidden_tool"] == "temperature__verify"
    assert missing_unit["expected_updates"] == {}
    assert missing_unit["expected_complete"] is False
    absent = cases["temperature_absent_guard"]
    assert absent["observation"] == "Not visible."
    assert absent["forbidden_tool"] == "temperature__verify"
    assert absent["expected_updates"] == {}
    detection = cases["heating_detection_guard"]
    assert all(re.fullmatch(heat.evidence.pattern, item) for item in detection["accepted"])
    assert not any(re.fullmatch(heat.evidence.pattern, item) for item in detection["rejected"])
    assert detection["expected_updates"] == {"heating_started": True}
    assert detection["forbidden_tool"] == "temperature__verify"
    voice_check = cases["temperature_voice_check"]
    assert voice_check["expected_tools"] == ["current_view", "temperature__verify"]
    assert voice_check["expected_verify_request"] == {"reading": 70, "unit": "celsius"}
    assert voice_check["expected_ready"] is False
    assert voice_check["expected_state_updates"] == {}
    readout = cases["temperature_readout"]
    assert readout["expected_tool"] == "current_view"
    assert readout["forbidden_tool"] == "temperature__verify"
    assert readout["expected_answer"] == "70 degrees Celsius"
    completion_notice = cases["heating_completion_notice"]
    assert completion_notice["expected_updates"] == {"heating_started": True}
    assert completion_notice["expected_message"] == heat.complete_message
    assert completion_notice["expected_complete"] is True
    routine_notice = cases["routine_notice_guard"]
    assert routine_notice["expected_updates"] == {}
    assert routine_notice["attempted_messages"]
    assert routine_notice["expected_spoken_messages"] == []
    fill = workflow.step("fill_water")
    assert "closed vessel" in fill.agent.prompt
    assert "user report is not" in fill.voice.prompt
    assert heat.agent.tools == ()
    assert "Set heating_started true only when input is false" in heat.agent.prompt
    assert "Celsius or Fahrenheit" in str(heat.trigger.arguments["question"])
    assert "Celsius or Fahrenheit" in heat.agent.prompt
    assert "Do not compare temperatures" in heat.agent.prompt
    assert heat.voice.tools == ("current_view", "temperature__verify")
    assert "For a reading" in heat.voice.prompt
    assert "repeat the visible number and unit; do not verify" in heat.voice.prompt
    assert "For hot-enough checks" in heat.voice.prompt
    assert "C/F" not in heat.voice.prompt
    assert "reading is unavailable" in heat.voice.prompt
    assert heat.complete_when == {"heating_started": True}
    assert "next" not in heat.complete_message.lower()
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
    no_water = cases["steeping_no_water_guard"]
    assert not re.fullmatch(steeping.evidence.pattern, no_water["observation"])
    assert no_water["expected_accepted"] is False
    assert no_water["expected_updates"] == {}
    assert no_water["forbidden_tool"] == "clock__now"
    assert "two separately visible facts" in str(steeping.trigger.arguments["question"])
    assert "both visible water" in steeping.agent.prompt
    assert "do not call clock__now" in steeping.agent.prompt
    assert "Require both visible water" in steeping.voice.prompt
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
        foreground = f"{TEA}\n{VOICE}\n{workflow.foreground_prompt}\n{step.voice.prompt}\n{HUMAN}"
        assert len(foreground) <= 850, f"{step.id} foreground prompt is too large"
        assert len(_state_contract(workflow, step)) <= 500, f"{step.id} state contract is too large"
    route_count = (
        len(cases["routes"])
        + len(matrix["idle_cases"])
        + len(matrix["active_steps"]) * len(matrix["active_cases"])
    )
    print(f"validated {len(workflow.steps)} steps and {route_count} routes")


if __name__ == "__main__":
    run()
