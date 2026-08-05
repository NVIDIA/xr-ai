# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the YAML-driven tea-making sample."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import yaml
from xr_ai_models import ToolCall, ToolDef
from xr_ai_voicegate import VoiceGate, load_voice_gate_config

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE_DIR = _REPO_ROOT / "agent-samples" / "tea-making-sample"
_WORKER_DIR = _SAMPLE_DIR / "worker"
_OMNI_DIR = _REPO_ROOT / "ai-services" / "llm" / "nemotron_omni"
sys.path.insert(0, str(_WORKER_DIR))
sys.path.insert(0, str(_OMNI_DIR))

from nemotron_omni_llm_server import __main__ as omni_server_module  # noqa: E402
from tea_making_worker import agent as agent_module  # noqa: E402
from tea_making_worker import guide as guide_module  # noqa: E402
from tea_making_worker import tools as tools_module  # noqa: E402
from tea_making_worker.agent import (  # noqa: E402
    NavigationIntent,
    StepAgentResult,
    WorkflowAgent,
)
from tea_making_worker.guide import WorkflowGuide  # noqa: E402
from tea_making_worker.step_mechanism import (  # noqa: E402
    StepEvent,
    StepIteration,
    StepMechanisms,
)
from tea_making_worker.workflow import (  # noqa: E402
    WorkflowDefinition,
    WorkflowSession,
    render_template,
    speech_text,
)

_MAIN_SPEC = importlib.util.spec_from_file_location(
    "tea_making_sample_main",
    _SAMPLE_DIR / "main.py",
)
assert _MAIN_SPEC is not None and _MAIN_SPEC.loader is not None
sample_main_module = importlib.util.module_from_spec(_MAIN_SPEC)
_MAIN_SPEC.loader.exec_module(sample_main_module)


def _workflow() -> WorkflowDefinition:
    return WorkflowDefinition.load(_SAMPLE_DIR / "yaml" / "workflow.yaml")


def test_shipped_workflow_uses_omni_native_rag_and_one_step_mechanism() -> None:
    workflow = _workflow()
    models = yaml.safe_load((_SAMPLE_DIR / "yaml" / "models.yaml").read_text())
    worker = yaml.safe_load((_SAMPLE_DIR / "yaml" / "tea_making_worker.yaml").read_text())
    final_step = workflow.step_by_id(4)

    assert models["agent_llm"]["kind"] == "preset:nemotron_omni"
    assert models["embedding"]["kind"] == "preset:nemotron_embedding"
    assert worker["rag_endpoint"] == "tcp://127.0.0.1:8340"
    assert worker["vlm_timeout_s"] == 15.0

    for profile in ("96G_blackwell", "dual_48G_ada"):
        profile_dir = _SAMPLE_DIR / "yaml" / profile
        embedding = yaml.safe_load((profile_dir / "embedding_server.yaml").read_text())
        omni = yaml.safe_load((profile_dir / "nemotron_omni_llm_server.yaml").read_text())
        stt = yaml.safe_load((profile_dir / "stt_server.yaml").read_text())

        assert omni["vllm_image"] == "vllm/vllm-openai:v0.20.0"
        assert omni["extra_pip"] == []
        assert omni["max_num_seqs"] <= 8
        assert omni["max_model_len"] <= 32768
        assert omni["tensor_parallel_size"] == 1
        assert stt["cuda_visible_devices"] in {"0", "1"}
        assert embedding["cuda_visible_devices"] in {"0", "1"}

    blackwell_dir = _SAMPLE_DIR / "yaml" / "96G_blackwell"
    blackwell_omni = yaml.safe_load((blackwell_dir / "nemotron_omni_llm_server.yaml").read_text())
    blackwell_embedding = yaml.safe_load((blackwell_dir / "embedding_server.yaml").read_text())
    assert blackwell_omni["cuda_visible_devices"] == "0"
    assert blackwell_omni["gpu_memory_utilization"] <= 0.80
    assert "moe_backend" not in blackwell_omni
    assert blackwell_embedding["cuda_visible_devices"] == "0"
    assert blackwell_embedding["gpu_memory_utilization"] <= 0.05
    assert blackwell_omni["gpu_memory_utilization"] + blackwell_embedding["gpu_memory_utilization"] < 0.86

    ada_dir = _SAMPLE_DIR / "yaml" / "dual_48G_ada"
    ada_omni = yaml.safe_load((ada_dir / "nemotron_omni_llm_server.yaml").read_text())
    ada_embedding = yaml.safe_load((ada_dir / "embedding_server.yaml").read_text())
    ada_stt = yaml.safe_load((ada_dir / "stt_server.yaml").read_text())
    assert ada_omni["cuda_visible_devices"] == "0"
    assert ada_omni["gpu_memory_utilization"] <= 0.85
    assert ada_embedding["cuda_visible_devices"] == "1"
    assert ada_embedding["gpu_memory_utilization"] <= 0.10
    assert ada_stt["cuda_visible_devices"] == "1"

    assert not (_SAMPLE_DIR / "yaml" / "rag.yaml").exists()
    assert not (_WORKER_DIR / "tea_making_worker" / "rag.py").exists()
    assert {step.mechanism for step in workflow.steps if not step.is_idle} == {
        "caption_agent"
    }
    assert final_step.vlm_prompt
    assert final_step.agent_tools == ("get_current_time", "get_timer_status")
    assert final_step.vlm_stop_when == {
        "field": "steeping_started_at_us",
        "gt": 0,
    }
    assert final_step.suppress_reminders_when == final_step.vlm_stop_when
    assert workflow.next_step(final_step.id) is None
    assert workflow.step_by_id(2).writable_fields == {"water_filled"}
    step_three = workflow.step_by_id(3)
    assert step_three.agent_tools == ()
    assert "water_temperature_current" in {field.name for field in step_three.context_fields}
    assert "water_heating_started" in step_three.writable_fields
    assert workflow.sparse_context is True
    assert workflow.initial_context() == {}
    assert step_three.read_fields == (
        "tea_name",
        "tea_temperature",
        "target_temperature_c",
        "water_filled",
    )
    full_context = {
        "tea_name": "green tea",
        "water_filled": True,
        "water_temperature_current": "80 C",
        "steeping_started_at_us": 123,
        "steeping_complete": False,
    }
    assert set(workflow.context_for_step(workflow.step_by_id(2), full_context)) == {
        "water_filled"
    }
    assert "steeping_started_at_us" not in workflow.context_for_step(step_three, full_context)
    assert "water_temperature_current" not in workflow.context_for_step(final_step, full_context)


async def test_tea_voice_gate_requires_a_wake_phrase_on_every_utterance() -> None:
    config = load_voice_gate_config(_SAMPLE_DIR / "yaml" / "voice_gate.yaml")
    queries: list[tuple[str, str, bool]] = []
    dropped: list[tuple[str, str]] = []

    async def on_query(participant_id: str, text: str, fresh_match: bool) -> None:
        queries.append((participant_id, text, fresh_match))

    async def on_stop(_participant_id: str) -> None:
        return None

    async def on_drop(participant_id: str, text: str) -> None:
        dropped.append((participant_id, text))

    gate = VoiceGate(
        config,
        audio_sink=SimpleNamespace(),
        tts=SimpleNamespace(),
    )
    gate.bind(on_query=on_query, on_stop=on_stop, on_drop=on_drop)

    await gate.feed("alice", "help me make tea")
    await gate.feed("alice", "Agent, help me make tea")
    await gate.feed("alice", "next")
    await gate.feed("alice", "Hey agent, next")
    await gate.feed("alice", "Agent, cancel tea")

    assert config.magic_phrases == ("agent", "hey agent")
    assert config.followup_grace_s == 0.0
    assert "Agent, help me make tea" in config.welcome_message
    assert gate.format_phrase_help() == config.welcome_message
    assert queries == [
        ("alice", "help me make tea", True),
        ("alice", "next", True),
        ("alice", "cancel tea", True),
    ]
    assert dropped == [
        ("alice", "help me make tea"),
        ("alice", "next"),
    ]


def test_launcher_selects_hardware_specific_model_configs() -> None:
    for detected, expected in (
        ("96G_blackwell", "96G_blackwell"),
        ("dual_48G_ada", "dual_48G_ada"),
        ("spark", "96G_blackwell"),
    ):
        processes = sample_main_module._build_processes(detected)
        by_name = {process.name: process for process in processes}

        assert processes[0].name == "omni"
        assert by_name["omni"].config == (f"yaml/{expected}/nemotron_omni_llm_server.yaml")
        assert by_name["stt"].config == f"yaml/{expected}/stt_server.yaml"
        assert by_name["embedding"].config == (f"yaml/{expected}/embedding_server.yaml")


def test_omni_server_forwards_configured_moe_backend(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}
    config = {
        "cuda_visible_devices": "0",
        "moe_backend": "triton",
        "vllm_backend": "docker",
    }

    monkeypatch.setattr(omni_server_module, "setup_logging", lambda *_: None)
    monkeypatch.setattr(
        omni_server_module,
        "load_config",
        lambda: (config, tmp_path, None),
    )
    monkeypatch.setattr(
        omni_server_module,
        "resolve_model_cache",
        lambda *_args, **_kwargs: tmp_path,
    )
    monkeypatch.setattr(
        omni_server_module,
        "setup_hf_env",
        lambda *_args, **_kwargs: "0",
    )
    monkeypatch.setattr(omni_server_module, "gpu_compute_major", lambda: 12)
    monkeypatch.setattr(
        omni_server_module,
        "serve",
        lambda **kwargs: captured.update(kwargs),
    )

    omni_server_module.run()

    serve_args = captured["extra_serve_args"]
    backend_index = serve_args.index("--moe-backend")
    assert serve_args[backend_index + 1] == "triton"


async def test_caption_agent_mechanism_applies_changing_agent_patches(
    tmp_path: Path,
) -> None:
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        yaml.safe_dump(
            {
                "task": {"name": "equipment-check"},
                "runtime": {},
                "steps": [
                    {
                        "id": 0,
                        "name": "Idle",
                        "description": "Wait.",
                        "context_output": {"fields": {}},
                    },
                    {
                        "id": 1,
                        "name": "Read instrument",
                        "description": "Read an arbitrary instrument.",
                        "mechanism": "caption_agent",
                        "vlm_prompt": ("End with PRESSURE_READING and ALARM_ACTIVE lines."),
                        "agent_prompt": "Interpret the latest instrument reading.",
                        "context_output": {
                            "fields": {
                                "pressure_kpa": {
                                    "label": "Pressure",
                                    "type": "number",
                                    "default": 0,
                                },
                                "alarm_active": {
                                    "label": "Alarm active",
                                    "type": "boolean",
                                    "default": False,
                                },
                            }
                        },
                        "advance_when": {
                            "field": "alarm_active",
                            "equals": True,
                        },
                    },
                    {
                        "id": 2,
                        "name": "Finish check",
                        "description": "Finish the equipment check.",
                        "vlm_prompt": "Observe whether the check is finished.",
                        "agent_prompt": "Set finished from visible evidence.",
                        "context_output": {
                            "fields": {
                                "finished": {
                                    "label": "Finished",
                                    "type": "boolean",
                                    "default": False,
                                }
                            }
                        },
                        "advance_when": {
                            "field": "finished",
                            "equals": True,
                        },
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    workflow = WorkflowDefinition.load(workflow_path)
    observations = iter(
        [
            SimpleNamespace(
                text="PRESSURE_READING: 42.5\nALARM_ACTIVE: no",
                frame_pts_us=1,
            ),
            SimpleNamespace(
                text="PRESSURE_READING: 44.0\nALARM_ACTIVE: yes",
                frame_pts_us=2,
            ),
        ]
    )
    agent_contexts: list[dict] = []
    notices: list[tuple[str, str]] = []

    class _Vision:
        async def observe(self, *_args, **_kwargs):
            return next(observations)

        def release(self, _participant_id: str) -> None:
            return None

    class _Agent:
        async def run_step(self, *, context, vlm_observation, **_kwargs):
            agent_contexts.append(dict(context))
            is_complete = "ALARM_ACTIVE: yes" in vlm_observation
            return StepAgentResult(
                context_patch={
                    "pressure_kpa": 44.0 if is_complete else 42.5,
                    "alarm_active": is_complete,
                },
                step_state="complete" if is_complete else "started",
                ready_to_advance=is_complete,
            )

    async def notice(participant_id: str, text: str) -> None:
        notices.append((participant_id, text))

    guide = WorkflowGuide(
        workflow=workflow,
        vision=_Vision(),
        agent=_Agent(),
        notice=notice,
    )
    session = WorkflowSession(
        participant_id="alice",
        step_id=1,
        context=workflow.initial_context(),
        last_reminder_us=guide_module._now_us(),  # noqa: SLF001
    )

    await guide._evaluate(session)  # noqa: SLF001

    assert agent_contexts[0]["pressure_kpa"] == 0
    assert agent_contexts[0]["alarm_active"] is False
    assert session.context["pressure_kpa"] == 42.5
    assert session.context["alarm_active"] is False
    assert session.ready_step_id is None

    await guide._evaluate(session)  # noqa: SLF001

    assert agent_contexts[1]["pressure_kpa"] == 42.5
    assert agent_contexts[1]["alarm_active"] is False
    assert session.context["pressure_kpa"] == 44.0
    assert session.context["alarm_active"] is True
    assert session.ready_step_id == 1
    assert session.step_state == "complete"
    assert len(notices) == 1


async def test_workflow_can_inject_a_different_step_mechanism(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        yaml.safe_dump(
            {
                "task": {"name": "custom-mechanism"},
                "steps": [
                    {"id": 0, "name": "Idle", "description": "Wait."},
                    {
                        "id": 1,
                        "name": "Custom step",
                        "description": "Run custom logic.",
                        "mechanism": "scripted",
                        "agent_prompt": "This prompt belongs to the custom mechanism.",
                        "context_output": {
                            "fields": {"done": {"type": "boolean", "default": False}}
                        },
                        "advance_when": {"field": "done", "equals": True},
                    },
                    {
                        "id": 2,
                        "name": "Later custom step",
                        "description": "Keep the first step from being final.",
                        "mechanism": "scripted",
                        "agent_prompt": "Not evaluated in this test.",
                        "context_output": {
                            "fields": {"finished": {"type": "boolean", "default": False}}
                        },
                        "advance_when": {"field": "finished", "equals": True},
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    workflow = WorkflowDefinition.load(workflow_path)
    calls: list[int] = []

    class _ScriptedMechanism:
        name = "scripted"

        async def run(self, *, step, **_kwargs):
            calls.append(step.id)
            return StepIteration(
                result=StepAgentResult(
                    context_patch={"done": True},
                    step_state="complete",
                    ready_to_advance=True,
                )
            )

    async def notice(_participant_id: str, _text: str) -> None:
        return None

    guide = WorkflowGuide(
        workflow=workflow,
        vision=SimpleNamespace(release=lambda _participant_id: None),
        agent=SimpleNamespace(),
        notice=notice,
        step_mechanisms=StepMechanisms([_ScriptedMechanism()]),
    )
    session = WorkflowSession(
        participant_id="alice",
        step_id=1,
        context=workflow.initial_context(),
    )

    await guide._evaluate(session)  # noqa: SLF001

    assert calls == [1]
    assert session.context["done"] is True
    assert session.ready_step_id == 1
    assert session.step_state == "complete"


def test_sparse_context_carries_values_with_projected_reads_and_partial_writes(
    tmp_path: Path,
) -> None:
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        yaml.safe_dump(
            {
                "task": {"name": "instrument-guide"},
                "context": {
                    "fields": {
                        "durable_fact": {"type": "string"},
                    }
                },
                "steps": [
                    {
                        "id": 0,
                        "name": "Idle",
                        "description": "Wait.",
                        "reads": [],
                        "writes": [],
                    },
                    {
                        "id": 1,
                        "name": "Identify",
                        "description": "Identify the instrument.",
                        "vlm_prompt": "Read its label.",
                        "agent_prompt": "Store the identified instrument.",
                        "reads": [],
                        "writes": ["durable_fact"],
                        "advance_when": {"field": "durable_fact", "exists": True},
                    },
                    {
                        "id": 2,
                        "name": "Measure",
                        "description": "Read the changing measurement.",
                        "vlm_prompt": "End with LIVE_VALUE.",
                        "agent_prompt": "Use the newest reading.",
                        "reads": ["durable_fact"],
                        "writes": {
                            "live_value": {"type": "number", "required": True},
                        },
                        "advance_when": {"field": "live_value", "gte": 10},
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    workflow = WorkflowDefinition.load(workflow_path)
    step = workflow.step_by_id(2)
    context = {
        "durable_fact": "pressure gauge",
        "live_value": 8.0,
        "unrelated_internal_value": "do not expose",
    }

    assert workflow.initial_context() == {}
    assert workflow.context_for_step(step, context) == {
        "durable_fact": "pressure gauge",
        "live_value": 8.0,
    }
    assert set(step.context_schema()["properties"]) == {"live_value"}
    assert "required" not in step.context_schema()
    assert step.mechanism == "caption_agent"
    assert step.agent_tools == ()


def test_templates_and_response_normalization_are_speech_friendly() -> None:
    workflow = _workflow()
    step = workflow.step_by_id(4)
    rendered = render_template(
        ("Started at {{started | local_time}} for {{seconds | duration}} using {{temperature | spoken}}."),
        context={
            "started": "2026-08-03T16:13:00-07:00",
            "seconds": 185,
            "temperature": "93 C",
        },
        step=step,
        task=workflow.task,
    )

    assert "2026-08-03" not in rendered
    assert "4:13 P.M." in rendered
    assert "3 minutes and 5 seconds" in rendered
    assert "93 degrees Celsius" in rendered
    assert speech_text("Heat to 175°F, not 80 C.") == ("Heat to 175 degrees Fahrenheit, not 80 degrees Celsius.")


async def test_answer_agent_can_inspect_a_fresh_view_for_visual_questions() -> None:
    workflow = _workflow()
    step = workflow.step_by_id(4)
    model_calls: list[list] = []
    visual_questions: list[str] = []

    class _Llm:
        async def chat(self, messages, **_kwargs):
            model_calls.append(list(messages))
            if len(model_calls) == 1:
                return SimpleNamespace(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="visual-1",
                            name="inspect_current_view",
                            arguments=('{"question":"What temperature is the kettle showing now?"}'),
                        )
                    ],
                )
            return SimpleNamespace(
                content="The kettle currently shows 82 degrees Celsius.",
                tool_calls=[],
            )

    class _Tools:
        def definitions(self):
            return []

        async def invoke(self, _name, _arguments):
            raise AssertionError("no non-visual tool call expected")

    async def visual_query(question: str) -> dict:
        visual_questions.append(question)
        return {
            "visual_evidence": "The kettle display reads 82 C.",
            "frame_pts_us": 123,
        }

    agent = WorkflowAgent(
        llm=_Llm(),
        tools=_Tools(),
        workflow=workflow,
        answer_prompt=_WORKER_DIR / "tea_making_worker" / "prompts" / "system.txt",
    )
    session = WorkflowSession(
        participant_id="alice",
        step_id=4,
        context={"steeping_started_at_us": 1, "steep_duration_seconds": 180},
    )

    answer = await agent.answer_user(
        transcript="What temperature is the kettle showing now?",
        session=session,
        current_step=step,
        observation_log=[],
        recent_turns=[],
        visual_query=visual_query,
    )

    assert visual_questions == ["What temperature is the kettle showing now?"]
    assert answer == "The kettle currently shows 82 degrees Celsius."
    assert '"visual_evidence": "The kettle display reads 82 C."' in (model_calls[1][-1].content)


def test_visual_question_answer_is_not_reused_as_current_step_observation() -> None:
    observation_log = [
        {
            "step_id": 3,
            "kind": "step_monitor",
            "caption": "WATER_TEMPERATURE: 35 C",
        },
        {
            "step_id": 3,
            "kind": "visual_question",
            "question": "What temperature is it now?",
            "caption": "The display reads 37 C.",
        },
    ]

    latest = agent_module._latest_step_observation(observation_log, 3)  # noqa: SLF001

    assert "35 C" in latest
    assert "37 C" not in latest


async def test_guide_logs_fresh_visual_answers_and_normalizes_them_for_speech() -> None:
    workflow = _workflow()
    inspected: list[str] = []

    class _Vision:
        async def inspect(self, _participant_id, question, **_kwargs):
            inspected.append(question)
            return SimpleNamespace(
                text="The kettle display reads 82 C.",
                frame_pts_us=321,
            )

        def release(self, _participant_id: str) -> None:
            return None

    class _Agent:
        async def classify_intent(self, **_kwargs):
            return NavigationIntent(intent="answer", confidence=0.99)

        async def answer_step(self, *, transcript, visual_query, **_kwargs):
            evidence = await visual_query(transcript)
            assert evidence["visual_evidence"] == "The kettle display reads 82 C."
            return "It reads 82 C."

    async def notice(_participant_id: str, _text: str) -> None:
        return None

    guide = WorkflowGuide(
        workflow=workflow,
        vision=_Vision(),
        agent=_Agent(),
        notice=notice,
    )
    session = WorkflowSession(
        participant_id="alice",
        step_id=4,
        context={"steeping_started_at_us": 1, "steep_duration_seconds": 180},
    )
    guide._sessions["alice"] = session  # noqa: SLF001

    response = await guide.handle_query(
        participant_id="alice",
        text="What does the kettle display show now?",
    )

    assert inspected == ["What does the kettle display show now?"]
    assert response == "It reads 82 degrees Celsius."
    assert session.observation_log[-1]["kind"] == "visual_question"
    assert session.observation_log[-1]["question"] == ("What does the kettle display show now?")


async def test_voice_event_runs_through_mechanism_without_mutating_state() -> None:
    workflow = _workflow()
    events: list[StepEvent] = []

    class _Mechanism:
        name = "caption_agent"

        async def run(self, *, event, **_kwargs):
            events.append(event)
            return StepIteration(
                result=StepAgentResult(
                    context_patch={"tea_name": "incorrect mutation"},
                    assistant_message="Hold the label closer to the camera.",
                )
            )

    class _Agent:
        async def classify_intent(self, **_kwargs):
            return NavigationIntent(intent="answer", confidence=0.99)

    async def notice(_participant_id: str, _text: str) -> None:
        return None

    guide = WorkflowGuide(
        workflow=workflow,
        vision=SimpleNamespace(release=lambda _participant_id: None),
        agent=_Agent(),
        notice=notice,
        step_mechanisms=StepMechanisms([_Mechanism()]),
    )
    session = WorkflowSession(
        participant_id="alice",
        step_id=1,
        context=workflow.initial_context(),
    )
    guide._sessions["alice"] = session  # noqa: SLF001

    response = await guide.handle_query(
        participant_id="alice",
        text="Can you read this label?",
    )

    assert response == "Hold the label closer to the camera."
    assert len(events) == 1
    assert events[0].kind == "voice"
    assert events[0].transcript == "Can you read this label?"
    assert "tea_name" not in session.context


async def test_answer_prompt_is_scoped_without_future_step_state() -> None:
    workflow = _workflow()
    step = workflow.step_by_id(1)
    captured: list = []

    class _Llm:
        async def chat(self, messages, **_kwargs):
            captured.extend(messages)
            return SimpleNamespace(content="Show me the tea label.", tool_calls=[])

    class _Tools:
        def definitions(self):
            return []

    agent = WorkflowAgent(
        llm=_Llm(),
        tools=_Tools(),
        workflow=workflow,
        answer_prompt=(_WORKER_DIR / "tea_making_worker" / "prompts" / "system.txt"),
    )
    session = WorkflowSession(
        participant_id="alice",
        step_id=1,
        context={
            "tea_name": "green tea",
            "steeping_started_at_us": 1_785_798_780_000_000,
            "steeping_complete": True,
        },
    )

    await agent.answer_user(
        transcript="What should I do now?",
        session=session,
        current_step=step,
        observation_log=[],
        recent_turns=[],
    )

    prompt = captured[-1].content
    assert "[Current-step context only]" in prompt
    assert '"tea_name": "green tea"' in prompt
    assert "steeping_started_at_us" not in prompt
    assert "steeping_complete" not in prompt
    assert "navigation_examples" not in prompt
    assert "name=Fill water" not in prompt
    assert "name=Steep the tea" not in prompt


async def test_answer_next_step_tool_is_grounded_in_workflow_yaml() -> None:
    workflow = _workflow()
    step = workflow.step_by_id(1)
    model_calls: list[list] = []

    class _Llm:
        async def chat(self, messages, **_kwargs):
            model_calls.append(list(messages))
            if len(model_calls) == 1:
                return SimpleNamespace(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="next-1",
                            name="get_next_workflow_step",
                            arguments="{}",
                        )
                    ],
                )
            return SimpleNamespace(
                content="Next, fill the kettle or pot with water.",
                tool_calls=[],
            )

    class _Tools:
        def definitions(self):
            return []

    agent = WorkflowAgent(
        llm=_Llm(),
        tools=_Tools(),
        workflow=workflow,
        answer_prompt=(_WORKER_DIR / "tea_making_worker" / "prompts" / "system.txt"),
    )
    session = WorkflowSession(
        participant_id="alice",
        step_id=1,
        context={"tea_name": "green tea"},
    )

    answer = await agent.answer_user(
        transcript="What comes next?",
        session=session,
        current_step=step,
        observation_log=[],
        recent_turns=[],
    )

    assert answer == "Next, fill the kettle or pot with water."
    tool_result = model_calls[1][-1].content
    assert '"name": "Fill water"' in tool_result
    assert "Steep the tea" not in tool_result


async def test_answer_agent_receives_only_read_only_tools() -> None:
    workflow = _workflow()
    offered: list[str] = []

    async def invoke(_arguments):
        return {"ok": True}

    class _Tools:
        def agent_tools(self):
            return [
                tools_module.AgentTool(
                    ToolDef(name="read_sensor", description="Read.", parameters={}),
                    invoke,
                    read_only=True,
                ),
                tools_module.AgentTool(
                    ToolDef(name="change_device", description="Write.", parameters={}),
                    invoke,
                    read_only=False,
                ),
            ]

    class _Llm:
        async def chat(self, _messages, **kwargs):
            offered.extend(tool.name for tool in kwargs["tools"])
            return SimpleNamespace(content="The sensor is available.", tool_calls=[])

    agent = WorkflowAgent(
        llm=_Llm(),
        tools=_Tools(),
        workflow=workflow,
        answer_prompt=(_WORKER_DIR / "tea_making_worker" / "prompts" / "system.txt"),
    )

    await agent.answer_user(
        transcript="Can you check the sensor?",
        session=None,
        current_step=workflow.step_by_id(0),
        observation_log=[],
        recent_turns=[],
    )

    assert "read_sensor" in offered
    assert "change_device" not in offered
    assert "get_recent_vlm_observations" in offered
    assert "inspect_current_view" in offered
    assert "get_next_workflow_step" in offered


def test_yaml_completion_rule_cannot_be_bypassed_by_model_readiness() -> None:
    workflow = _workflow()
    step = workflow.step_by_id(1)
    context = workflow.initial_context()

    assert not workflow.advance_when_met(step, context, ready_to_advance=True)
    context.update(
        tea_name="green tea",
        tea_temperature="175 F",
        target_temperature_c=80,
        steep_time="2 minutes",
        steep_duration_seconds=120,
        context_ready=True,
    )
    assert workflow.advance_when_met(step, context)


def test_final_step_skip_completes_without_inventing_a_start_time() -> None:
    workflow = _workflow()
    step = workflow.step_by_id(4)
    context = workflow.initial_context()

    applied = workflow.apply_skip_defaults(step, context)

    assert applied == {"steeping_complete": True}
    assert "steeping_started_at_us" not in context
    assert "steeping_started_at_iso" not in context


async def test_step_agent_calls_time_tool_to_capture_steeping_start() -> None:
    workflow = _workflow()
    step = workflow.step_by_id(4)
    model_calls: list[list] = []
    tool_calls: list[tuple[str, dict]] = []

    class _Llm:
        async def chat(self, messages, **_kwargs):
            model_calls.append(list(messages))
            if len(model_calls) == 1:
                return SimpleNamespace(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="time-1",
                            name="get_current_time",
                            arguments="{}",
                        )
                    ],
                )
            return SimpleNamespace(
                content=(
                    '{"context":{"steeping_started_at_us":1785798780376000,'
                    '"steeping_started_at_iso":"2026-08-03T16:13:00-07:00"},'
                    '"ready_to_advance":false,"step_state":"started",'
                    '"assistant_message":"","speak":false}'
                ),
                tool_calls=[],
            )

    class _Tools:
        def definitions(self):
            return [
                SimpleNamespace(name="get_current_time"),
                SimpleNamespace(name="get_timer_status"),
            ]

        async def invoke(self, name, arguments):
            tool_calls.append((name, arguments))
            return {
                "epoch_us": 1_785_798_780_376_000,
                "iso": "2026-08-03T16:13:00-07:00",
                "timezone": "PDT",
            }

    agent = WorkflowAgent(
        llm=_Llm(),
        tools=_Tools(),
        workflow=workflow,
        answer_prompt=(_WORKER_DIR / "tea_making_worker" / "prompts" / "system.txt"),
    )
    result = await agent.run_step(
        step=step,
        participant_id="alice",
        context=workflow.initial_context(),
        observation_log=[],
        vlm_observation="tea bag, entering the water\n\nSTEEPING_STARTED: yes",
    )

    assert tool_calls == [("get_current_time", {})]
    assert result.context_patch["steeping_started_at_us"] == 1_785_798_780_376_000
    assert result.context_patch["steeping_started_at_iso"] == ("2026-08-03T16:13:00-07:00")
    assert result.ready_to_advance is False
    assert '"epoch_us": 1785798780376000' in model_calls[1][-1].content


def test_navigation_eval_cases_require_explicit_movement_language() -> None:
    workflow = _workflow()
    cases = yaml.safe_load((_SAMPLE_DIR / "eval" / "cases.yaml").read_text())

    for case in cases["navigation"]:
        proposed = NavigationIntent(
            intent=case["proposed_intent"],
            explicit_command=case["explicit_command"],
            confidence=0.95,
        )
        actual = guide_module._validated_model_intent(  # noqa: SLF001
            case["utterance"],
            proposed,
            workflow,
            active=True,
        )
        assert actual.intent == case["expected_intent"], case["utterance"]


async def test_timer_tool_reports_elapsed_remaining_and_expiration() -> None:
    now = datetime(2026, 8, 3, 16, 13, tzinfo=timezone(timedelta(hours=-7)))

    class _Rag:
        instance_name = "test__retrieve"

    guide_tools = tools_module.GuideTools({"retrieve": _Rag()}, clock=lambda: now)
    started_at_us = int((now - timedelta(seconds=65)).timestamp() * 1_000_000)

    active = await guide_tools.invoke(
        "get_timer_status",
        {
            "started_at_us": started_at_us,
            "duration_seconds": 180,
            "label": "steeping",
        },
    )

    assert active["elapsed_seconds"] == 65
    assert active["remaining_seconds"] == 115
    assert active["expired"] is False

    guide_tools._clock = lambda: now + timedelta(seconds=120)  # noqa: SLF001
    expired = await guide_tools.invoke(
        "get_timer_status",
        {"started_at_us": started_at_us, "duration_seconds": 180},
    )
    assert expired["remaining_seconds"] == 0
    assert expired["expired"] is True


async def test_timer_tool_rejects_a_missing_start_time() -> None:
    class _Rag:
        instance_name = "test__retrieve"

    guide_tools = tools_module.GuideTools({"retrieve": _Rag()})
    status = await guide_tools.invoke(
        "get_timer_status",
        {"started_at_us": 0, "duration_seconds": 180},
    )

    assert status["started"] is False
    assert status["expired"] is False
    assert "positive start timestamp" in status["error"]


async def test_step_agent_uses_timer_tool_to_complete_no_caption_step() -> None:
    workflow = _workflow()
    step = workflow.step_by_id(4)
    model_calls: list[list] = []
    tool_calls: list[tuple[str, dict]] = []

    class _Llm:
        async def chat(self, messages, **_kwargs):
            model_calls.append(list(messages))
            if len(model_calls) == 1:
                return SimpleNamespace(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="timer-1",
                            name="get_timer_status",
                            arguments=(
                                '{"started_at_us":1785798780000000,'
                                '"duration_seconds":180,"label":"steeping"}'
                            ),
                        )
                    ],
                )
            return SimpleNamespace(
                content=(
                    '{"context":{"steeping_complete":true},'
                    '"ready_to_advance":true,"step_state":"complete",'
                    '"assistant_message":"","speak":false}'
                ),
                tool_calls=[],
            )

    class _Tools:
        def definitions(self):
            return [SimpleNamespace(name="get_timer_status")]

        async def invoke(self, name, arguments):
            tool_calls.append((name, arguments))
            return {
                "label": "steeping",
                "elapsed_seconds": 181,
                "remaining_seconds": 0,
                "expired": True,
            }

    agent = WorkflowAgent(
        llm=_Llm(),
        tools=_Tools(),
        workflow=workflow,
        answer_prompt=(_WORKER_DIR / "tea_making_worker" / "prompts" / "system.txt"),
    )
    context = workflow.initial_context()
    context.update(
        steeping_started_at_us=1_785_798_780_000_000,
        steep_duration_seconds=180,
    )

    result = await agent.run_step(
        step=step,
        participant_id="alice",
        context=context,
        observation_log=[],
        vlm_observation="",
    )

    assert tool_calls[0][0] == "get_timer_status"
    assert result.context_patch == {"steeping_complete": True}
    assert result.ready_to_advance is True
    assert '"expired": true' in model_calls[1][-1].content


async def test_answer_agent_uses_timer_tool_for_remaining_time_question() -> None:
    workflow = _workflow()
    step = workflow.step_by_id(4)
    calls = 0
    model_calls: list[list] = []

    class _Llm:
        async def chat(self, messages, **_kwargs):
            nonlocal calls
            calls += 1
            model_calls.append(list(messages))
            if calls == 1:
                return SimpleNamespace(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="timer-answer-1",
                            name="get_timer_status",
                            arguments=(
                                '{"started_at_us":1785798780000000,'
                                '"duration_seconds":180,"label":"steeping"}'
                            ),
                        )
                    ],
                )
            return SimpleNamespace(
                content="About one minute and fifty-five seconds remain.",
                tool_calls=[],
            )

    class _Tools:
        async def timer_status(self, _arguments):
            return {
                "elapsed_seconds": 65,
                "remaining_seconds": 115,
                "expired": False,
            }

        def agent_tools(self):
            return [
                tools_module.AgentTool(
                    ToolDef(
                        name="get_timer_status",
                        description="Read timer status.",
                        parameters={},
                    ),
                    self.timer_status,
                    read_only=True,
                )
            ]

    agent = WorkflowAgent(
        llm=_Llm(),
        tools=_Tools(),
        workflow=workflow,
        answer_prompt=(_WORKER_DIR / "tea_making_worker" / "prompts" / "system.txt"),
    )
    context = workflow.initial_context()
    context.update(
        steeping_started_at_us=1_785_798_780_000_000,
        steep_duration_seconds=180,
    )
    session = WorkflowSession(participant_id="alice", step_id=4, context=context)

    answer = await agent.answer_user(
        transcript="How long do I still need to wait?",
        session=session,
        current_step=step,
        observation_log=[],
        recent_turns=[
            ("How much time has elapsed?", "One minute and five seconds have elapsed."),
        ],
    )

    assert calls == 2
    assert answer == "About one minute and fifty-five seconds remain."
    prompt = model_calls[0][-1].content
    assert "How much time has elapsed?" in prompt
    assert "One minute and five seconds" not in prompt


async def test_merged_steeping_step_stops_vision_and_finishes_workflow() -> None:
    workflow = _workflow()
    now_us = guide_module._now_us()  # noqa: SLF001
    context = workflow.initial_context()
    context.update(
        steeping_started_at_us=now_us - 181_000_000,
        steep_duration_seconds=180,
    )
    notices: list[tuple[str, str]] = []
    agent_calls: list[str] = []

    class _Vision:
        async def observe(self, *_args, **_kwargs):
            raise AssertionError("merged timer phase must not invoke the VLM")

        def release(self, _participant_id: str) -> None:
            return None

    class _Agent:
        async def run_step(self, *, vlm_observation, **_kwargs):
            agent_calls.append(vlm_observation)
            return StepAgentResult(
                context_patch={"steeping_complete": True},
                step_state="complete",
                ready_to_advance=True,
            )

    async def notice(participant_id: str, text: str) -> None:
        notices.append((participant_id, text))

    guide = WorkflowGuide(
        workflow=workflow,
        vision=_Vision(),
        agent=_Agent(),
        notice=notice,
    )
    session = WorkflowSession(participant_id="alice", step_id=4, context=context)
    step = workflow.step_by_id(4)

    assert workflow.condition_met(step.vlm_stop_when, context)
    assert guide._reminder_due(session, step) == ""  # noqa: SLF001
    assert session.reminder_count == 0

    await guide._evaluate(session)  # noqa: SLF001

    assert agent_calls == [""]
    assert session.active is False
    assert session.step_id == 0
    assert session.step_state == "idle"
    assert session.ready_step_id is None
    assert "steeping_complete" not in session.context
    assert notices == [("alice", "The steeping time is up. Remove the tea bag, infuser, or leaves now.")]


async def test_merged_steeping_step_observes_until_start_is_recorded() -> None:
    workflow = _workflow()
    captions: list[str] = []
    notices: list[str] = []

    class _Vision:
        async def observe(self, *_args, **_kwargs):
            return SimpleNamespace(
                text="The tea bag is immersed.\nSTEEPING_STARTED: yes",
                frame_pts_us=10,
            )

        def release(self, _participant_id: str) -> None:
            return None

    class _Agent:
        async def run_step(self, *, vlm_observation, **_kwargs):
            captions.append(vlm_observation)
            return StepAgentResult(
                context_patch={
                    "steeping_started_at_us": 1_785_798_780_376_000,
                    "steeping_started_at_iso": "2026-08-03T16:13:00-07:00",
                },
                step_state="started",
            )

    async def notice(_participant_id: str, text: str) -> None:
        notices.append(text)

    guide = WorkflowGuide(
        workflow=workflow,
        vision=_Vision(),
        agent=_Agent(),
        notice=notice,
    )
    session = WorkflowSession(
        participant_id="alice",
        step_id=4,
        context={"steep_duration_seconds": 180},
        last_reminder_us=0,
    )

    await guide._evaluate(session)  # noqa: SLF001

    assert captions == ["The tea bag is immersed.\nSTEEPING_STARTED: yes"]
    assert session.context["steeping_started_at_us"] == 1_785_798_780_376_000
    assert session.ready_step_id is None
    assert session.reminder_count == 0
    assert notices == []


async def test_completed_step_does_not_repeat_guidance_or_recheck_vision() -> None:
    workflow = _workflow()
    notices: list[tuple[str, str]] = []

    class _Vision:
        async def observe(self, *_args, **_kwargs):
            raise AssertionError("completed step must not invoke the VLM")

        def release(self, _participant_id: str) -> None:
            return None

    class _Agent:
        async def run_step(self, *_args, **_kwargs):
            raise AssertionError("completed step must not invoke the step agent")

    async def notice(participant_id: str, text: str) -> None:
        notices.append((participant_id, text))

    guide = WorkflowGuide(
        workflow=workflow,
        vision=_Vision(),
        agent=_Agent(),
        notice=notice,
    )
    session = WorkflowSession(
        participant_id="alice",
        step_id=1,
        context=workflow.initial_context(),
        ready_step_id=1,
        step_state="complete",
    )

    await guide._evaluate(session)  # noqa: SLF001

    assert session.step_state == "complete"
    assert notices == []


async def test_disconnect_and_reconnect_clear_workflow_state() -> None:
    workflow = _workflow()

    class _Vision:
        async def observe(self, *_args, **_kwargs):
            raise AssertionError("no observation expected")

        def release(self, _participant_id: str) -> None:
            return None

    async def notice(_participant_id: str, _text: str) -> None:
        return None

    guide = WorkflowGuide(
        workflow=workflow,
        vision=_Vision(),
        agent=SimpleNamespace(),
        notice=notice,
    )
    await guide.start("alice")
    session = guide._sessions["alice"]  # noqa: SLF001
    session.context["tea_name"] = "green tea"
    session.step_id = 4
    session.step_state = "needs_input"
    session.observation_log.append({"caption": "old evidence"})
    guide._history["alice"] = [("old request", "old answer")]  # noqa: SLF001

    await guide.release("alice")

    assert "alice" not in guide._sessions  # noqa: SLF001
    assert "alice" not in guide._history  # noqa: SLF001
    assert session.active is False
    assert session.connected is False
    assert session.step_id == 0
    assert session.context == workflow.initial_context()
    assert session.observation_log == []

    await guide.reset("alice")
    restarted_message = await guide.start("alice")
    restarted = guide._sessions["alice"]  # noqa: SLF001

    assert "show me the tea label" in restarted_message.lower()
    assert restarted is not session
    assert restarted.step_id == 1
    assert restarted.context == workflow.initial_context()


async def test_next_on_incomplete_final_step_resets_and_restarts_fresh() -> None:
    workflow = _workflow()

    class _Vision:
        def release(self, _participant_id: str) -> None:
            return None

    async def notice(_participant_id: str, _text: str) -> None:
        return None

    guide = WorkflowGuide(
        workflow=workflow,
        vision=_Vision(),
        agent=SimpleNamespace(),
        notice=notice,
    )
    context = workflow.initial_context()
    context.update(
        tea_name="green tea",
        steeping_started_at_us=123,
    )
    session = WorkflowSession(
        participant_id="alice",
        step_id=4,
        context=context,
        step_state="started",
        observation_log=[{"caption": "old evidence"}],
    )
    guide._sessions["alice"] = session  # noqa: SLF001

    response = await guide.advance("alice")

    assert "using reasonable defaults" in response.lower()
    assert "timer skipped" in response.lower()
    assert session.active is False
    assert session.step_id == 0
    assert session.step_state == "idle"
    assert "tea_name" not in session.context
    assert "steeping_started_at_us" not in session.context
    assert session.observation_log == []

    assert guide.status("alice") == "No guided workflow is active."
    await guide.start("alice")
    restarted = guide._sessions["alice"]  # noqa: SLF001
    assert restarted.active is True
    assert restarted.step_id == 1
    assert "tea_name" not in restarted.context


async def test_cancel_clears_active_state_before_a_fresh_restart() -> None:
    workflow = _workflow()

    class _Vision:
        def release(self, _participant_id: str) -> None:
            return None

    async def notice(_participant_id: str, _text: str) -> None:
        return None

    guide = WorkflowGuide(
        workflow=workflow,
        vision=_Vision(),
        agent=SimpleNamespace(),
        notice=notice,
    )
    await guide.start("alice")
    old_session = guide._sessions["alice"]  # noqa: SLF001
    old_session.step_id = 3
    old_session.context.update(
        tea_name="black tea",
        water_filled=True,
        water_temperature_current="90 C",
    )
    old_session.observation_log.append({"caption": "old evidence"})
    guide._history["alice"] = [("old request", "old answer")]  # noqa: SLF001

    response = await guide.handle_query(participant_id="alice", text="cancel tea")

    assert response == "Guidance stopped."
    assert old_session.active is False
    assert old_session.step_id == 0
    assert old_session.context == workflow.initial_context()
    assert old_session.observation_log == []
    assert "alice" not in guide._history  # noqa: SLF001

    await guide.start("alice")
    restarted = guide._sessions["alice"]  # noqa: SLF001
    assert restarted is not old_session
    assert restarted.step_id == 1
    assert restarted.context == workflow.initial_context()


def test_start_making_tea_trigger_starts_only_while_idle() -> None:
    workflow = _workflow()

    idle_intent = guide_module._local_intent(  # noqa: SLF001
        "Start making tea",
        workflow,
        active=False,
    )
    active_intent = guide_module._local_intent(  # noqa: SLF001
        "Start making tea",
        workflow,
        active=True,
    )

    assert idle_intent is not None
    assert idle_intent.intent == "start"
    assert active_intent is None


async def test_next_after_completed_session_stays_idle_without_agent_call() -> None:
    workflow = _workflow()

    class _Agent:
        async def classify_intent(self, *_args, **_kwargs):
            raise AssertionError("idle next must not invoke the classifier")

        async def answer_user(self, *_args, **_kwargs):
            raise AssertionError("idle next must not invoke the answer agent")

    class _Vision:
        def release(self, _participant_id: str) -> None:
            return None

    async def notice(_participant_id: str, _text: str) -> None:
        return None

    guide = WorkflowGuide(
        workflow=workflow,
        vision=_Vision(),
        agent=_Agent(),
        notice=notice,
    )
    guide._sessions["alice"] = WorkflowSession(  # noqa: SLF001
        participant_id="alice",
        step_id=0,
        context=workflow.initial_context(),
        active=False,
        step_state="idle",
    )

    response = await guide.handle_query(participant_id="alice", text="Next.")

    assert "previous session is finished" in response
    assert "help me make tea" in response
    assert guide.status("alice") == "No guided workflow is active."


async def test_answer_agent_receives_completed_step_state_and_yaml_procedure() -> None:
    workflow = _workflow()
    step = workflow.step_by_id(1)
    captured: list = []

    class _Llm:
        async def chat(self, messages, **_kwargs):
            captured.extend(messages)
            return SimpleNamespace(content="You can proceed.", tool_calls=[])

    class _Tools:
        def definitions(self):
            return []

        async def invoke(self, _name, _arguments):
            raise AssertionError("no tool call expected")

    agent = WorkflowAgent(
        llm=_Llm(),
        tools=_Tools(),
        workflow=workflow,
        answer_prompt=(_WORKER_DIR / "tea_making_worker" / "prompts" / "system.txt"),
    )
    session = WorkflowSession(
        participant_id="alice",
        step_id=1,
        context=workflow.initial_context(),
        ready_step_id=1,
        step_state="complete",
    )

    await agent.answer_user(
        transcript="What happens now?",
        session=session,
        current_step=step,
        observation_log=[],
        recent_turns=[],
    )

    prompt = captured[-1].content
    assert "[Step procedure]" in prompt
    assert step.agent_prompt in prompt
    assert "state=complete" in prompt
    assert "ready=True" in prompt


async def test_answer_prompt_omits_old_assistant_measurements() -> None:
    workflow = _workflow()
    step = workflow.step_by_id(3)
    captured: list = []

    class _Llm:
        async def chat(self, messages, **_kwargs):
            captured.extend(messages)
            return SimpleNamespace(content="The latest reading is 100 C.", tool_calls=[])

    class _Tools:
        def definitions(self):
            return []

    agent = WorkflowAgent(
        llm=_Llm(),
        tools=_Tools(),
        workflow=workflow,
        answer_prompt=(_WORKER_DIR / "tea_making_worker" / "prompts" / "system.txt"),
    )
    context = workflow.initial_context()
    context.update(water_temperature_current="100 C", water_ready=True)
    session = WorkflowSession(
        participant_id="alice",
        step_id=3,
        context=context,
        ready_step_id=3,
        step_state="complete",
    )

    await agent.answer_user(
        transcript="What is the temperature of the water?",
        session=session,
        current_step=step,
        observation_log=[
            {
                "step_id": 3,
                "frame_pts_us": 1,
                "caption": "TEMPERATURE_READING: 59 C\nWATER_READY: no",
            },
            {
                "step_id": 3,
                "frame_pts_us": 2,
                "caption": "TEMPERATURE_READING: 100 C\nWATER_READY: yes",
            },
        ],
        recent_turns=[
            ("What is the temperature?", "The water is currently 59 C."),
        ],
    )

    prompt = captured[-1].content
    assert "The water is currently 59 C." not in prompt
    assert "What is the temperature?" in prompt
    assert '"water_temperature_current": "100 C"' in prompt
    assert "TEMPERATURE_READING: 100 C" in prompt
    assert (
        "TEMPERATURE_READING: 59 C"
        not in prompt.split(
            "[Latest step-monitor observation]",
            maxsplit=1,
        )[1]
    )
