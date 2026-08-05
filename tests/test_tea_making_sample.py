# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository-level contracts for the tea-making guidance sample."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
import sys
import tempfile
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / "agent-samples" / "tea-making-sample"
sys.path.insert(0, str(_SAMPLE / "worker"))

from tea_making_worker.agents.prompts import HUMAN, ROUTER, STEP, VOICE  # noqa: E402
from tea_making_worker.agents.registry import _state_contract  # noqa: E402
from tea_making_worker.config import load_config  # noqa: E402
from tea_making_worker.functions.vision import CurrentViewRequest  # noqa: E402
from tea_making_worker.functions.workflow import CommitRequest  # noqa: E402
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
    assert len(f"{STEP}\n{HUMAN}") <= 350
    assert len(f"{VOICE}\n{HUMAN}") <= 300
    assert "natural spoken language" in HUMAN
    assert "temperature" not in HUMAN.lower()
    assert "Rewrite tool/state" in HUMAN
    assert "already_complete is status, not state" in STEP
    assert "briefly message real non-completing changes" in STEP
    assert "Empty on no change/completion" in STEP
    assert "Never route these to ask_step" in ROUTER
    for step, source in zip(workflow.steps.values(), raw["steps"], strict=True):
        assert step.trigger.function
        assert step.complete_when
        assert step.state_on_skip
        assert "state_on_skip" in source
        assert "skip_state" not in source
        assert len(str(step.trigger.arguments.get("question", ""))) <= 240
        assert len(step.agent.prompt) <= 420
        assert len(step.voice.prompt) <= 300
        assert len(_state_contract(workflow, step)) <= 500
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
    store.observe(
        session,
        "A tea package label reads Oolong, 88 C, steep 4 minutes.",
        "identify",
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


def test_model_profiles_always_use_omni_for_agents() -> None:
    cosmos = json.loads((_SAMPLE / "yaml" / "models.cosmos.json").read_text(encoding="utf-8"))["models"]
    omni = json.loads((_SAMPLE / "yaml" / "models.omni.json").read_text(encoding="utf-8"))["models"]

    for models in (cosmos, omni):
        assert models["agent_llm"]["adapter"]["preset"] == "nemotron_omni"
        assert models["agent_llm"]["adapter"]["default_extras"] == {
            "chat_template_kwargs": {"enable_thinking": False},
        }
        reused = {
            models[role]["deployment"]["ownership"]
            for role in ("agent_llm", "vlm", "stt", "embedding")
        }
        assert reused == {"reused"}
    assert cosmos["agent_llm"]["deployment"]["service"] == "omni"
    assert cosmos["vlm"]["adapter"]["preset"] == "cosmos_vlm"
    assert cosmos["vlm"]["deployment"]["service"] == "vlm"
    assert omni["agent_llm"]["deployment"]["service"] == omni["vlm"]["deployment"]["service"] == "omni"
    assert omni["vlm"]["adapter"]["default_extras"] == {
        "max_tokens": 128,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_identification_keeps_native_rag_fallback() -> None:
    identify = _workflow().step("identify")
    rag = yaml.safe_load((_SAMPLE / "yaml" / "rag_service.yaml").read_text(encoding="utf-8"))

    assert identify.agent.tools == ("rag_lookup",)
    assert "rag_lookup" in identify.voice.tools
    observation_prompt = identify.agent.prompt.lower()
    assert "identify tea only from the current caption" in observation_prompt
    assert "never state or rag" in observation_prompt
    assert "commit all with tea_ready true" in observation_prompt
    assert "never write tea_ready false" in observation_prompt
    question = str(identify.trigger.arguments["question"])
    assert "front label brand and tea/blend name" in question
    assert "Ignore slogans, claims, bag count, weight" in question
    assert "never RAG" in identify.voice.prompt
    assert "after the exact name is known" in identify.voice.prompt
    assert rag["documents_dir"] == "../rag-documents"
    assert rag["embedding_role"] == "embedding"
    assert rag["chunk_size"] == 700
    assert rag["overlap"] == 100
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
    assert identify.evidence.consecutive == 1
    assert re.fullmatch(identify.evidence.pattern, "NUMI ORGANIC BLACK TEA BREAKFAST BLEND")
    assert not re.fullmatch(identify.evidence.pattern, "Too dark to discern tea package text.")
    assert not re.fullmatch(identify.evidence.pattern, "There are no visible texts in this frame.")
    assert not re.fullmatch(identify.evidence.pattern, "none")


def test_heat_policy_converts_units_before_comparing() -> None:
    step = _workflow().step("heat_water")
    prompt = step.agent.prompt

    assert step.agent.tools == ("temperature__verify",)
    assert "call temperature__verify" in prompt
    assert "exact number/unit and state target" in prompt
    assert "never calculate" in prompt
    assert "always call workflow__commit" in prompt
    assert "When ready is true, include water_ready=true" in prompt
    assert "When ready is false, leave water_ready out completely" in prompt
    assert "heating_started=true only if input state is false" in prompt
    assert "never return JSON/text" in prompt
    assert step.evidence is not None
    assert re.fullmatch(step.evidence.pattern, "164F")
    message_description = CommitRequest.model_fields["message"].description
    assert message_description is not None
    assert "real non-completing state change" in message_description


def test_user_facing_values_use_natural_units() -> None:
    assert render_message("{{ value | temperature_c }}", {"value": 100}) == "100 degrees Celsius"
    assert render_message("{{ value | duration }}", {"value": 240}) == "4 minutes"
    assert render_message("{{ value | duration }}", {"value": 65}) == "1 minute and 5 seconds"


def test_launcher_requires_explicit_model_and_voice_modes() -> None:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        sample_main.run([])
    help_text = output.getvalue()
    assert "--model-mode {omni,cosmos}" in help_text
    assert "--voice-mode {wake-word,always-on}" in help_text

    args = sample_main._parse_args(["--model-mode", "omni", "--voice-mode", "always-on"])
    assert args is not None
    assert (args.model_mode, args.voice_mode) == ("omni", "always-on")


def test_launch_modes_align_worker_rag_voice_and_processes() -> None:
    expected_services = {
        "omni": {"omni", "stt", "embedding", "tts", "rag", "hub", "worker"},
        "cosmos": {"omni", "vlm", "stt", "embedding", "tts", "rag", "hub", "worker"},
    }
    original_detect = sample_main.detect_gpu_config
    sample_main.detect_gpu_config = lambda: "96G_blackwell"
    try:
        with tempfile.TemporaryDirectory() as directory:
            runtime_dir = Path(directory)
            for model_mode, model_file in sample_main._MODEL_CONFIGS.items():
                for voice_mode, voice_file in sample_main._VOICE_CONFIGS.items():
                    worker_path, rag_path = sample_main._materialize_configs(
                        runtime_dir,
                        model_mode,
                        voice_mode,
                    )
                    worker = load_config(worker_path)
                    rag = yaml.safe_load(rag_path.read_text(encoding="utf-8"))
                    assert worker.models_config == (_SAMPLE / "yaml" / model_file).resolve()
                    assert worker.voice_gate_config == (_SAMPLE / "yaml" / voice_file).resolve()
                    gate = yaml.safe_load(worker.voice_gate_config.read_text(encoding="utf-8"))
                    if voice_mode == "wake-word":
                        assert gate["magic_phrases"] == ["agent", "hey agent"]
                        assert gate["followup_grace_s"] == 5.0
                    else:
                        assert gate["magic_phrases"] == []
                    assert Path(rag["models_config"]) == worker.models_config
                    assert Path(rag["documents_dir"]) == (_SAMPLE / "rag-documents").resolve()

                processes = sample_main._build_processes(worker_path, rag_path)
                assert {process.name for process in processes} == expected_services[model_mode]
                assert all(
                    process.launch_mode == ("own" if process.name in {"tts", "rag", "hub", "worker"} else "reuse")
                    for process in processes
                )
    finally:
        sample_main.detect_gpu_config = original_detect


def test_eval_cases_cover_every_route_and_step() -> None:
    workflow = _workflow()
    cases = yaml.safe_load((_SAMPLE / "eval" / "cases.yaml").read_text(encoding="utf-8"))
    assert set(cases["steps"]) == set(workflow.steps)
    assert {case["expected_tool"] for case in cases["routes"]} == {
        f"workflow__{name}" for name in ("start", "advance", "reset", "status", "ask_step")
    }
    assert cases["rag_fallback"]["expected_tools"] == ["rag_lookup", "workflow__commit"]
    assert cases["rag_fallback"]["expected_top_k"] == 2
    assert cases["rag_fallback"]["expected_tea_ready"] is True
    assert cases["irrelevant_retrieval_guard"]["forbidden_updates"] == [
        "target_temperature_c",
        "steep_duration_s",
        "tea_ready",
    ]
    assert cases["identity_retrieval_mismatch"]["expected_updates"] == {}
    assert cases["identity_retrieval_mismatch"]["expected_tea_ready"] is False
    assert cases["identification_guard"]["unreadable_expected_updates"] == {}
    assert cases["identification_atomic_guard"]["expected_unready_state"] == {}
    assert cases["identification_atomic_guard"]["current_observation_wins"] is True
    assert cases["tea_identity_question"]["expected_tool"] == "current_view"
    assert cases["tea_identity_question"]["forbidden_tool"] == "rag_lookup"
    assert cases["temperature_unit_guard"]["expected_updates"] == {"heating_started": True}
    assert cases["temperature_unit_guard"]["expected_tool"] == "temperature__verify"
    assert cases["temperature_unit_guard"]["expected_water_ready"] is False
    assert cases["temperature_missing_unit_guard"]["expected_water_ready"] is False
    assert cases["temperature_repeated_below_target_guard"]["expected_updates"] == [{}, {}]
    assert cases["temperature_repeated_below_target_guard"]["expected_tool"] == "temperature__verify"
    assert cases["temperature_repeated_below_target_guard"]["expected_water_ready"] is False
    assert cases["temperature_above_target_ready"]["expected_updates"] == {"water_ready": True}
    assert cases["temperature_above_target_ready"]["expected_tool"] == "temperature__verify"
    assert cases["temperature_above_target_ready"]["expected_water_ready"] is True
    assert cases["state_update_notice"]["expected_updates"] == {"heating_started": True}
    assert cases["state_update_notice"]["expected_message_intent"] == "heating started"
    assert cases["state_update_notice"]["expected_complete"] is False
    assert cases["routine_notice_guard"]["expected_messages"] == ["", "", ""]
    assert cases["observation_context"]["keys"] == ["observation", "already_complete", "state"]
    assert cases["observation_context"]["prior_status_is_state_value"] is False
    assert cases["observation_context"]["contract_includes"] == [
        "field_descriptions",
        "completion_condition",
    ]
    assert cases["completed_step"]["expected_updates"] == {}
    assert cases["completed_step"]["expected_transition"] is None
    assert cases["false_positive_guard"]["expected_accepted"] is False
    assert cases["steeping_negative_guard"]["expected_accepted"] is False
    assert cases["timer_running_guard"]["expected_steeping_complete"] is False
    assert cases["voice_tool_guards"]["timer_question"]["expected_tool"] == "clock__timer"
