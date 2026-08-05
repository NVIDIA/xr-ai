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

from tea_making_worker.agents.prompts import ROUTER, STEP, VOICE  # noqa: E402
from tea_making_worker.runtime.render import render_message  # noqa: E402
from tea_making_worker.spec import load_workflow  # noqa: E402


def run() -> None:
    workflow = load_workflow(_SAMPLE / "yaml" / "workflow.yaml")
    cases = yaml.safe_load((_SAMPLE / "eval" / "cases.yaml").read_text(encoding="utf-8"))
    assert len(ROUTER) <= 300
    assert len(STEP) <= 350
    assert len(VOICE) <= 300
    assert "natural spoken language" in VOICE
    assert "temperature" not in VOICE.lower()
    assert "natural spoken language" in STEP
    assert {case["expected_tool"] for case in cases["routes"]} == {
        f"workflow__{name}" for name in ("start", "advance", "reset", "status", "ask_step")
    }
    assert set(cases["steps"]) == set(workflow.steps)
    assert cases["rag_fallback"]["expected_tools"] == ["rag_lookup", "workflow__commit"]
    assert cases["rag_fallback"]["expected_top_k"] == 2
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
    unreadable = bool(re.fullmatch(identify.evidence.pattern, guard["unreadable_caption"]))
    empty = bool(re.fullmatch(identify.evidence.pattern, guard["empty_caption"]))
    none = bool(re.fullmatch(identify.evidence.pattern, guard["none_caption"]))
    assert readable is guard["expected_readable"]
    assert unreadable is guard["expected_unreadable"]
    assert empty is guard["expected_empty"]
    assert none is guard["expected_empty"]
    assert guard["generic_rag_allowed"] is False
    steeping = workflow.step("start_steeping")
    assert steeping.evidence is not None
    assert not re.fullmatch(
        steeping.evidence.pattern,
        cases["steeping_negative_guard"]["observation"],
    )
    assert cases["steeping_negative_guard"]["expected_accepted"] is False
    spoken = cases["spoken_values"]
    assert render_message("{{ value | temperature_c }}", {"value": 100}) == spoken["temperature_c"]
    assert render_message("{{ value | duration }}", {"value": 240}) == spoken["duration_240_s"]
    assert render_message("{{ value | duration }}", {"value": 65}) == spoken["duration_65_s"]
    for step in workflow.steps.values():
        question = str(step.trigger.arguments.get("question", ""))
        assert len(question) <= 240, f"{step.id} VLM question is too large"
        assert len(step.agent.prompt) <= 420, f"{step.id} observation prompt is too large"
        assert len(step.voice.prompt) <= 300, f"{step.id} voice prompt is too large"
    print(f"validated {len(workflow.steps)} steps and {len(cases['routes'])} routes")


if __name__ == "__main__":
    run()
