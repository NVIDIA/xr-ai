# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository-level contracts for the tea-making guidance sample."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / "agent-samples" / "tea-making-sample"
sys.path.insert(0, str(_SAMPLE / "worker"))

from tea_making_worker.agents.prompts import ROUTER, STEP, VOICE  # noqa: E402
from tea_making_worker.functions.vision import CurrentViewRequest  # noqa: E402
from tea_making_worker.runtime.render import render_message  # noqa: E402
from tea_making_worker.runtime.state import SessionStore  # noqa: E402
from tea_making_worker.spec import load_workflow  # noqa: E402

_MAIN_SPEC = importlib.util.spec_from_file_location("tea_making_sample_main", _SAMPLE / "main.py")
assert _MAIN_SPEC is not None and _MAIN_SPEC.loader is not None
sample_main = importlib.util.module_from_spec(_MAIN_SPEC)
_MAIN_SPEC.loader.exec_module(sample_main)


def _workflow():
    return load_workflow(_SAMPLE / "yaml" / "workflow.yaml")


def test_workflow_is_uniform_sparse_and_prompt_bounded() -> None:
    workflow = _workflow()
    raw = yaml.safe_load((_SAMPLE / "yaml" / "workflow.yaml").read_text(encoding="utf-8"))
    assert list(workflow.steps) == [
        "identify",
        "fill_water",
        "heat_water",
        "start_steeping",
        "steep_timer",
    ]
    assert len(ROUTER) <= 300
    assert len(STEP) <= 350
    assert len(VOICE) <= 300
    assert "natural spoken language" in VOICE
    assert "temperature" not in VOICE.lower()
    assert "natural spoken language" in STEP
    for step in workflow.steps.values():
        assert step.trigger.function
        assert step.complete_when
        assert len(str(step.trigger.arguments.get("question", ""))) <= 240
        assert len(step.agent.prompt) <= 420
        assert len(step.voice.prompt) <= 300
        assert set((*step.reads, *step.writes)) <= workflow.state_fields.keys()
        assert "workflow__advance" not in (*step.agent.tools, *step.voice.tools)
        question = str(step.trigger.arguments.get("question", ""))
        assert "return only" not in question.lower()
        assert "return exactly" not in question.lower()
    assert all("auto_advance" not in step for step in raw["steps"])
    assert set(CurrentViewRequest.model_fields) == {"question"}
    for step in tuple(workflow.steps.values())[:4]:
        assert "participant_id" not in step.trigger.arguments


def test_state_commit_waits_for_explicit_advance() -> None:
    workflow = _workflow()
    store = SessionStore(workflow)
    session = store.get("participant")
    assert store.start(session) == workflow.step("identify").enter_message
    for index in range(2):
        store.observe(
            session,
            "A tea package label reads Oolong, 88 C, steep 4 minutes.",
            f"identify-{index}",
        )

    before = dict(session.state)
    rejected = store.commit(session, {"water_filled": True}, "")
    assert not rejected.accepted
    assert session.state == before

    accepted = store.commit(
        session,
        {
            "tea_name": "oolong",
            "target_temperature_c": 88,
            "steep_duration_s": 240,
            "guidance_source": "package",
            "tea_ready": True,
        },
        "",
    )
    assert accepted.accepted and accepted.complete
    assert session.step_id == "identify"
    assert store.step_complete(session)
    assert store.advance(session, skip=False) == workflow.step("fill_water").enter_message
    assert session.step_id == "fill_water"
    assert workflow.project(workflow.step("fill_water"), session.state) == {"water_filled": False}


def test_model_profiles_offer_split_reuse_and_managed_omni() -> None:
    split = json.loads((_SAMPLE / "yaml" / "models.split.json").read_text(encoding="utf-8"))["models"]
    omni = json.loads((_SAMPLE / "yaml" / "models.omni.json").read_text(encoding="utf-8"))["models"]
    worker = yaml.safe_load((_SAMPLE / "yaml" / "tea_making_worker.yaml").read_text(encoding="utf-8"))

    assert worker["models_config"] == "models.split.json"
    assert split["agent_llm"]["adapter"]["preset"] == "nemotron3_nano"
    assert split["vlm"]["adapter"]["preset"] == "cosmos_vlm"
    assert {split[role]["deployment"]["ownership"] for role in ("agent_llm", "vlm", "stt", "embedding")} == {"reused"}
    assert omni["agent_llm"]["adapter"]["preset"] == "nemotron_omni"
    assert omni["agent_llm"]["deployment"]["service"] == omni["vlm"]["deployment"]["service"] == "omni"


def test_identification_keeps_native_rag_fallback() -> None:
    identify = _workflow().step("identify")
    rag = yaml.safe_load((_SAMPLE / "yaml" / "rag_service.yaml").read_text(encoding="utf-8"))

    assert identify.agent.tools == ("rag_lookup",)
    assert "rag_lookup" in identify.voice.tools
    assert rag["documents_dir"] == "../rag-documents"
    assert rag["embedding_role"] == "embedding"
    assert (_SAMPLE / "rag-documents" / "tea-brewing.md").is_file()


def test_visual_evidence_uses_plain_captions_and_rejects_absence() -> None:
    workflow = _workflow()
    cases = yaml.safe_load((_SAMPLE / "eval" / "cases.yaml").read_text(encoding="utf-8"))
    for step_id in ("identify", "fill_water", "heat_water", "start_steeping"):
        step = workflow.step(step_id)
        assert step.evidence is not None
        assert re.fullmatch(step.evidence.pattern, cases["steps"][step_id]["observation"])
    fill = workflow.step("fill_water")
    assert fill.evidence is not None
    assert not re.fullmatch(fill.evidence.pattern, cases["false_positive_guard"]["observation"])
    identify = workflow.step("identify")
    assert identify.evidence is not None
    assert identify.evidence.consecutive == 2
    assert re.fullmatch(identify.evidence.pattern, "NUMI ORGANIC BLACK TEA BREAKFAST BLEND")
    assert not re.fullmatch(identify.evidence.pattern, "Too dark to discern tea package text.")
    assert not re.fullmatch(identify.evidence.pattern, "There are no visible texts in this frame.")
    assert not re.fullmatch(identify.evidence.pattern, "none")


def test_user_facing_values_use_natural_units() -> None:
    assert render_message("{{ value | temperature_c }}", {"value": 100}) == "100 degrees Celsius"
    assert render_message("{{ value | duration }}", {"value": 240}) == "4 minutes"
    assert render_message("{{ value | duration }}", {"value": 65}) == "1 minute and 5 seconds"


def test_split_launcher_reuses_model_servers(monkeypatch) -> None:
    monkeypatch.setattr(sample_main, "detect_gpu_config", lambda: "96G_blackwell")
    processes = sample_main._build_processes()
    modes = {process.name: process.launch_mode for process in processes}
    assert modes == {
        "agent-llm": "reuse",
        "vlm": "reuse",
        "stt": "reuse",
        "embedding": "reuse",
        "tts": "own",
        "rag": "own",
        "hub": "own",
        "worker": "own",
    }


def test_eval_cases_cover_every_route_and_step() -> None:
    workflow = _workflow()
    cases = yaml.safe_load((_SAMPLE / "eval" / "cases.yaml").read_text(encoding="utf-8"))
    assert set(cases["steps"]) == set(workflow.steps)
    assert {case["expected_tool"] for case in cases["routes"]} == {
        f"workflow__{name}" for name in ("start", "advance", "reset", "status", "ask_step")
    }
    assert cases["rag_fallback"]["expected_tools"] == ["rag_lookup", "workflow__commit"]
    assert cases["rag_fallback"]["expected_top_k"] == 2
    assert cases["completed_step"]["expected_updates"] == {}
    assert cases["completed_step"]["expected_transition"] is None
    assert cases["false_positive_guard"]["expected_accepted"] is False
    assert cases["steeping_negative_guard"]["expected_accepted"] is False
