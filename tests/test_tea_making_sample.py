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
sys.path.insert(0, str(_SAMPLE))
sys.path.insert(0, str(_SAMPLE / "worker"))

from tea_making_viewer.config import Source  # noqa: E402
from tea_making_viewer.config import load_config as load_viewer_config  # noqa: E402
from tea_making_viewer.store import EventStore, JsonlWatcher  # noqa: E402
from tea_making_worker.agents.prompts import (  # noqa: E402
    HUMAN,
    STEP,
    TEA,
    VOICE,
)
from tea_making_worker.agents.registry import (  # noqa: E402
    _TEA_MANAGEMENT_TOOLS,
    _state_contract,
)
from tea_making_worker.applications.compose import root_function_specs  # noqa: E402
from tea_making_worker.applications.manager.spec import load_application_catalog  # noqa: E402
from tea_making_worker.applications.manager.types import InvocationEffect  # noqa: E402
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
    catalog = load_application_catalog(_SAMPLE / "yaml" / "applications.yaml")
    root_functions = root_function_specs(catalog)
    root_prompt = (
        f"{catalog.root_prompt}\nRoutes: {'; '.join(function.catalog_entry() for function in root_functions)}\n{HUMAN}"
    )
    raw = yaml.safe_load((_SAMPLE / "yaml" / "workflow.yaml").read_text(encoding="utf-8"))
    assert list(workflow.steps) == [
        "identify",
        "fill_water",
        "heat_water",
        "start_steeping",
        "steep_timer",
    ]
    assert len(root_prompt) <= 950
    assert len(f"{TEA}\n{VOICE}\n{HUMAN}") <= 550
    assert len(f"{STEP}\n{HUMAN}") <= 350
    assert "Natural spoken prose only" in HUMAN
    assert "temperature" not in HUMAN.lower()
    assert "No Markdown, lists, code syntax, formatting marks" in HUMAN
    assert "internal names" in HUMAN
    assert "already_complete is status, not state" in STEP
    assert "message only with a real non-completing state change" in STEP
    assert "Empty on no change or completion" in STEP
    assert "Next/continue/advance" in TEA
    assert "Query background only when needed" in VOICE
    assert "Questions using these words are not commands" in TEA
    assert {function.name for function in root_functions} == {
        "current_view",
        "rag_lookup",
        "application_context__query",
        "workflow__start",
        "application_manager__status",
        "change_watch__start",
        "change_watch__stop",
        "change_watch__status",
        "transcript__start",
        "transcript__stop",
        "transcript__status",
        "video_log__start",
        "video_log__stop",
        "video_log__status",
    }
    assert next(function for function in root_functions if function.name == "workflow__start").effect == (
        InvocationEffect.FOREGROUND
    )
    assert all(
        function.effect == InvocationEffect.BACKGROUND
        for function in root_functions
        if function.name.startswith(("change_watch__", "transcript__", "video_log__"))
    )
    assert tuple(map(str, _TEA_MANAGEMENT_TOOLS)) == (
        "workflow__advance",
        "workflow__reset",
        "workflow__restart",
        "workflow__status",
    )
    for step, source in zip(workflow.steps.values(), raw["steps"], strict=True):
        assert step.trigger.function
        assert step.complete_when
        assert step.state_on_skip
        assert "state_on_skip" in source
        assert "skip_state" not in source
        assert len(str(step.trigger.arguments.get("question", ""))) <= 240
        assert len(step.agent.prompt) <= 420
        assert len(step.voice.prompt) <= 300
        foreground = f"{TEA}\n{VOICE}\n{workflow.foreground_prompt}\n{step.voice.prompt}\n{HUMAN}"
        assert len(foreground) <= 850
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
        reused = {models[role]["deployment"]["ownership"] for role in ("agent_llm", "vlm", "stt", "embedding")}
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


def test_magpie_profile_preserves_native_voice_quality() -> None:
    config = yaml.safe_load((_SAMPLE / "yaml" / "magpie_tts_server.yaml").read_text(encoding="utf-8"))

    assert config["speed"] == 1.0


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


def test_heat_policy_tracks_start_without_readiness() -> None:
    step = _workflow().step("heat_water")
    prompt = step.agent.prompt

    assert step.agent.tools == ()
    assert "Set heating_started true only when input is false" in prompt
    assert "Celsius or Fahrenheit" in prompt
    assert "Do not compare temperatures or track whether the water is ready" in prompt
    assert step.complete_when == {"heating_started": True}
    assert step.evidence is not None
    assert re.fullmatch(step.evidence.pattern, "164F")
    message_description = CommitRequest.model_fields["message"].description
    assert message_description is not None
    assert "real non-completing state change" in message_description


def test_user_facing_values_use_natural_units() -> None:
    assert render_message("{{ value | temperature_c }}", {"value": 100}) == "100 degrees Celsius"
    assert render_message("{{ value | duration }}", {"value": 240}) == "4 minutes"
    assert render_message("{{ value | duration }}", {"value": 65}) == "1 minute and 5 seconds"


def test_launcher_requires_explicit_launch_modes() -> None:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        sample_main.run([])
    help_text = output.getvalue()
    assert "--model-mode {omni,cosmos}" in help_text
    assert "--voice-mode {wake-word,always-on}" in help_text
    assert "--tts-mode {piper,magpie}" in help_text

    args = sample_main._parse_args(["--model-mode", "omni", "--voice-mode", "always-on", "--tts-mode", "magpie"])
    assert args is not None
    assert (args.model_mode, args.voice_mode, args.tts_mode) == ("omni", "always-on", "magpie")


def test_launch_modes_align_worker_rag_voice_and_processes() -> None:
    expected_services = {
        "omni": {"omni", "stt", "embedding", "tts", "rag", "hub", "activity-viewer", "worker"},
        "cosmos": {
            "omni",
            "vlm",
            "stt",
            "embedding",
            "tts",
            "rag",
            "hub",
            "activity-viewer",
            "worker",
        },
    }
    original_detect = sample_main.detect_gpu_config
    sample_main.detect_gpu_config = lambda: "96G_blackwell"
    try:
        with tempfile.TemporaryDirectory() as directory:
            runtime_dir = Path(directory)
            for model_mode, model_file in sample_main._MODEL_CONFIGS.items():
                for voice_mode, voice_file in sample_main._VOICE_CONFIGS.items():
                    for tts_mode, (tts_preset, tts_url) in sample_main._TTS_CONFIGS.items():
                        worker_path, rag_path = sample_main._materialize_configs(
                            runtime_dir,
                            model_mode,
                            voice_mode,
                            tts_mode,
                        )
                        worker = load_config(worker_path)
                        rag = yaml.safe_load(rag_path.read_text(encoding="utf-8"))
                        models = json.loads(worker.models_config.read_text(encoding="utf-8"))
                        expected_models = json.loads((_SAMPLE / "yaml" / model_file).read_text(encoding="utf-8"))
                        expected_models["models"]["tts"]["adapter"] = {"preset": tts_preset}
                        expected_models["models"]["tts"]["endpoint"]["base_url"] = tts_url
                        assert models == expected_models
                        assert worker.applications_config == (_SAMPLE / "yaml" / "applications.yaml").resolve()
                        assert worker.voice_gate_config == (_SAMPLE / "yaml" / voice_file).resolve()
                        assert worker.silence_duration == 1.2
                        gate = yaml.safe_load(worker.voice_gate_config.read_text(encoding="utf-8"))
                        if voice_mode == "wake-word":
                            assert gate["magic_phrases"] == ["agent", "hey agent"]
                            assert gate["followup_grace_s"] == 5.0
                        else:
                            assert gate["magic_phrases"] == []
                        assert Path(rag["models_config"]) == worker.models_config
                        assert Path(rag["documents_dir"]) == (_SAMPLE / "rag-documents").resolve()

                        processes = sample_main._build_processes(worker_path, rag_path, tts_mode)
                        assert {process.name for process in processes} == expected_services[model_mode]
                        assert all(
                            process.launch_mode
                            == (
                                "own" if process.name in {"tts", "rag", "hub", "activity-viewer", "worker"} else "reuse"
                            )
                            for process in processes
                        )
                        tts_process = next(process for process in processes if process.name == "tts")
                        assert tts_process.project == f"../../ai-services/tts/{tts_mode}"
                        assert tts_process.command == f"{tts_mode}_tts_server"
                        assert tts_process.config == f"yaml/{tts_mode}_tts_server.yaml"
                        viewer = next(process for process in processes if process.name == "activity-viewer")
                        assert viewer.project == "."
                        assert viewer.command == "tea_making_activity_viewer"
                        assert viewer.config == "yaml/activity_viewer.json"
    finally:
        sample_main.detect_gpu_config = original_detect


def test_activity_viewer_tails_only_complete_new_jsonl_records() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source_dir = root / "transcripts"
        source_dir.mkdir()
        path = source_dir / "speaker.jsonl"
        path.write_text('{"type":"utterance","text":"old"}\n', encoding="utf-8")
        store = EventStore()
        watcher = JsonlWatcher(
            (
                Source(
                    id="transcript",
                    title="Transcript recorder",
                    location=source_dir,
                    pattern="*.jsonl",
                    format="jsonl",
                ),
            ),
            store,
        )
        watcher.baseline()

        with path.open("a", encoding="utf-8") as stream:
            stream.write('{"type":"utterance","timestamp":"2026-08-07T10:00:00Z","text":"new"}\n')
            stream.write('{"type":"summary","text":"partial"')

        assert watcher.scan() == 1
        assert [event["record"]["text"] for event in store.after(0)] == ["new"]

        with path.open("a", encoding="utf-8") as stream:
            stream.write("}\n")

        assert watcher.scan() == 1
        events = store.after(0)
        assert [event["id"] for event in events] == [1, 2]
        assert [event["record"]["text"] for event in events] == ["new", "partial"]


def test_activity_viewer_extracts_structured_agent_events_from_worker_log() -> None:
    with tempfile.TemporaryDirectory() as directory:
        worker_log = Path(directory) / "worker.log"
        store = EventStore()
        source = Source(
            id="agent",
            title="Agent activity",
            location=worker_log,
            pattern=None,
            format="event_log",
            include_prefixes=("agent.", "step."),
            exclude_events=frozenset({"step.commit_noop"}),
        )
        watcher = JsonlWatcher((source,), store)
        watcher.baseline()
        worker_log.write_text(
            "2026-08-07 18:22:09.100 DEBUG module ordinary log line\n"
            '2026-08-07 18:22:09.200 INFO module event {"event":"agent.observe.request","step":"identify"}\n'
            '2026-08-07 18:22:09.300 INFO module event {"event":"step.commit_noop"}\n',
            encoding="utf-8",
        )

        assert watcher.scan() == 1
        record = store.after(0)[0]["record"]
        assert record == {
            "event": "agent.observe.request",
            "step": "identify",
            "timestamp": "2026-08-07T18:22:09.200",
        }


def test_activity_viewer_sources_have_independent_cursors_for_one_log() -> None:
    with tempfile.TemporaryDirectory() as directory:
        worker_log = Path(directory) / "worker.log"
        store = EventStore()
        sources = (
            Source(
                id="agent",
                title="Agent activity",
                location=worker_log,
                pattern=None,
                format="event_log",
                include_prefixes=("agent.",),
            ),
            Source(
                id="change_watch",
                title="Visual change watcher",
                location=worker_log,
                pattern=None,
                format="event_log",
                include_prefixes=("change_watch.",),
            ),
        )
        watcher = JsonlWatcher(sources, store)
        watcher.baseline()
        worker_log.write_text(
            '2026-08-07 18:22:09.100 INFO module event {"event":"agent.observe.request"}\n'
            '2026-08-07 18:22:09.200 INFO module event {"event":"change_watch.caption","caption":"A mug moved."}\n',
            encoding="utf-8",
        )

        assert watcher.scan() == 2
        assert [event["source_id"] for event in store.after(0)] == ["agent", "change_watch"]


def test_activity_viewer_config_and_static_page_are_sample_local() -> None:
    run_log_dir = Path("/tmp/tea-making-viewer-test-logs")
    config = load_viewer_config(
        _SAMPLE / "yaml" / "activity_viewer.json",
        run_log_dir=run_log_dir,
    )
    assert config.host == "0.0.0.0"
    assert config.port == 8092
    assert {source.id for source in config.sources} == {
        "transcript",
        "video_log",
        "agent",
        "change_watch",
    }
    assert all(_SAMPLE in source.location.parents for source in config.sources if source.id != "agent")
    log_sources = [source for source in config.sources if source.format == "event_log"]
    assert {source.id for source in log_sources} == {"agent"}
    assert all(source.location == run_log_dir / "worker.log" for source in log_sources)
    change_watch = next(source for source in config.sources if source.id == "change_watch")
    assert change_watch.format == "jsonl"
    assert change_watch.location == (_SAMPLE / "artifacts" / "change-watch").resolve()
    page = (_SAMPLE / "tea_making_viewer" / "static" / "index.html").read_text(encoding="utf-8")
    assert "Tea Demo Activity" in page
    assert "Transcript" in page
    assert "Video activity" in page
    assert "Agent activity" in page
    assert "Visual change watcher" in page
    assert page.count('class="pane ') == 4


def test_eval_cases_cover_every_route_and_step() -> None:
    workflow = _workflow()
    cases = yaml.safe_load((_SAMPLE / "eval" / "cases.yaml").read_text(encoding="utf-8"))
    assert set(cases["steps"]) == set(workflow.steps)
    background_actions = {
        f"{app}__{operation}"
        for app in ("change_watch", "transcript", "video_log")
        for operation in ("start", "stop", "status")
    }
    routed_actions = (
        {f"workflow__{name}" for name in ("start", "advance", "reset", "restart", "status")}
        | {"application_manager__status", "application_context__query"}
        | background_actions
    )
    assert {case["expected_tool"] for case in cases["routes"]} == {"answer"} | routed_actions
    matrix = cases["route_state_matrix"]
    assert set(matrix["active_steps"]) == set(workflow.steps)
    active_actions = {f"workflow__{name}" for name in ("advance", "reset", "restart", "status")} | {
        "application_context__query"
    }
    assert {case["expected_tool"] for case in matrix["active_cases"]} == {"answer"} | active_actions
    assert {case["expected_tool"] for case in matrix["idle_cases"]} == {
        "workflow__start",
        "change_watch__start",
        "transcript__start",
        "video_log__start",
        "application_manager__status",
        "application_context__query",
        "answer",
    }
    assert {
        case["expected_advanced"] for case in matrix["active_cases"] if case["expected_tool"] == "workflow__advance"
    } == {False, True}
    assert cases["general_queries"]["tea_knowledge"]["expected_tools"] == ["rag_lookup"]
    assert cases["general_queries"]["visible_scene"]["expected_tools"] == ["current_view"]
    assert cases["general_queries"]["visible_tea"]["expected_order"] == [
        "current_view",
        "rag_lookup",
    ]
    assert cases["general_queries"]["background_context"]["expected_tools"] == ["application_context__query"]
    assert cases["general_queries"]["safety"] == {
        "state_updates": False,
        "lifecycle_tools": False,
    }
    background = cases["background_applications"]
    assert background["change_watch"]["important"]["expected_notice"] is True
    assert background["change_watch"]["insignificant"]["expected_notice"] is False
    assert all(case["instruction"] for case in background["change_watch"].values())
    assert background["transcript"]["expected_output"] is True
    assert background["transcript"]["max_sentences"] == 2
    assert background["video_log"]["expected_delta_terms"] == ["person", "parcel"]
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
    assert cases["temperature_unit_guard"]["forbidden_tool"] == "temperature__verify"
    assert cases["temperature_missing_unit_guard"]["forbidden_tool"] == "temperature__verify"
    assert cases["temperature_missing_unit_guard"]["expected_updates"] == {}
    assert cases["temperature_absent_guard"]["forbidden_tool"] == "temperature__verify"
    assert cases["temperature_absent_guard"]["expected_updates"] == {}
    assert cases["heating_detection_guard"]["expected_updates"] == {"heating_started": True}
    assert cases["heating_detection_guard"]["forbidden_tool"] == "temperature__verify"
    assert cases["temperature_voice_check"]["expected_tools"] == [
        "current_view",
        "temperature__verify",
    ]
    assert cases["temperature_voice_check"]["expected_ready"] is False
    assert cases["temperature_readout"]["expected_tool"] == "current_view"
    assert cases["temperature_readout"]["forbidden_tool"] == "temperature__verify"
    assert cases["heating_completion_notice"]["expected_updates"] == {"heating_started": True}
    assert cases["routine_notice_guard"]["attempted_messages"]
    assert cases["routine_notice_guard"]["expected_spoken_messages"] == []
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
    capability = cases["capability_answer_guard"]
    assert capability["expected_style"] == "short plain spoken sentences"
    assert {"*", "`", "__", "workflow"} <= set(capability["forbidden_fragments"])
