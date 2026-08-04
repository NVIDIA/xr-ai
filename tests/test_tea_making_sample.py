# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the YAML-driven tea-making sample."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE_DIR = _REPO_ROOT / "agent-samples" / "tea-making-sample"
_WORKER_DIR = _SAMPLE_DIR / "worker"
_NANO_DIR = _REPO_ROOT / "ai-services" / "llm" / "nemotron3_nano"
sys.path.insert(0, str(_WORKER_DIR))
sys.path.insert(0, str(_NANO_DIR))

from nemotron3_nano_llm_server import __main__ as nano_server_module  # noqa: E402
from tea_making_worker import agent as agent_module  # noqa: E402
from tea_making_worker import guide as guide_module  # noqa: E402
from tea_making_worker.agent import (  # noqa: E402
    NavigationIntent,
    StepAgentResult,
    WorkflowAgent,
)
from tea_making_worker.guide import WorkflowGuide  # noqa: E402
from tea_making_worker.workflow import (  # noqa: E402
    WorkflowDefinition,
    WorkflowSession,
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


def test_shipped_workflow_uses_nano_cosmos_rag_and_timer_only_step_five() -> None:
    workflow = _workflow()
    models = yaml.safe_load((_SAMPLE_DIR / "yaml" / "models.yaml").read_text())
    worker = yaml.safe_load(
        (_SAMPLE_DIR / "yaml" / "tea_making_worker.yaml").read_text()
    )
    step_five = workflow.step_by_id(5)

    assert models["agent_llm"] == {
        "kind": "preset:nemotron3_nano",
        "base_url": "http://localhost:8107",
    }
    assert models["vlm"] == {
        "kind": "preset:cosmos_vlm",
        "base_url": "http://localhost:8100",
    }
    assert models["embedding"]["kind"] == "preset:nemotron_embedding"
    assert worker["rag_endpoint"] == "tcp://127.0.0.1:8340"
    assert worker["vlm_timeout_s"] == 15.0

    for profile in ("96G_blackwell", "dual_48G_ada", "spark"):
        assert not list((_SAMPLE_DIR / "yaml" / profile).glob("*.yaml"))

    assert not (_SAMPLE_DIR / "yaml" / "rag.yaml").exists()
    assert not (_WORKER_DIR / "tea_making_worker" / "rag.py").exists()
    assert step_five.timer is not None
    assert step_five.vlm_prompt == ""
    assert step_five.timer.completion_field == "steeping_complete"
    step_three = workflow.step_by_id(3)
    state_updates = {
        update.context_field: update for update in step_three.state_updates
    }
    assert set(state_updates) == {"water_temperature_current", "water_ready"}
    assert state_updates["water_temperature_current"].observation_key == (
        "TEMPERATURE_READING"
    )
    assert state_updates["water_temperature_current"].states == (
        "started",
        "needs_input",
        "complete",
    )
    assert state_updates["water_ready"].value_map == {
        "yes": True,
        "no": False,
        "unclear": False,
    }
    assert "water_temperature_current" in {
        field.name for field in step_three.context_fields
    }


def test_launcher_reuses_shared_model_server_stack() -> None:
    processes = sample_main_module._build_processes()
    by_name = {process.name: process for process in processes}

    assert [process.name for process in processes[:4]] == [
        "stt",
        "agent-llm",
        "vlm",
        "embedding",
    ]
    for name, port in {
        "stt": 8103,
        "agent-llm": 8107,
        "vlm": 8100,
        "embedding": 8109,
    }.items():
        assert by_name[name].launch_mode == "reuse"
        assert by_name[name].config is None
        assert by_name[name].port == port

    assert by_name["rag"].launch_mode == "own"
    assert by_name["worker"].launch_mode == "own"


def test_launcher_requires_complete_shared_model_server_stack(monkeypatch) -> None:
    required_ports = set(sample_main_module._REQUIRED_MODEL_PORTS)
    monkeypatch.setattr(
        sample_main_module,
        "_port_open",
        lambda port: port in required_ports,
    )
    sample_main_module._check_model_ports()

    monkeypatch.setattr(
        sample_main_module,
        "_port_open",
        lambda port: port in required_ports - {8100, 8109},
    )
    with pytest.raises(SystemExit) as exc_info:
        sample_main_module._check_model_ports()

    message = str(exc_info.value)
    assert "Cosmos on 8100" in message
    assert "Nemotron Embed on 8109" in message
    assert "agent-samples/model-servers model_servers" in message


def test_nano_agent_server_enables_native_tool_calling(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}
    config = {
        "cuda_visible_devices": "0",
        "model_blackwell": "nvidia/test-nano-blackwell",
        "vllm_backend": "docker",
    }

    monkeypatch.setattr(nano_server_module, "setup_logging", lambda *_: None)
    monkeypatch.setattr(
        nano_server_module,
        "load_config",
        lambda: (config, tmp_path, None),
    )
    monkeypatch.setattr(
        nano_server_module,
        "resolve_model_cache",
        lambda *_args, **_kwargs: tmp_path,
    )
    monkeypatch.setattr(
        nano_server_module,
        "setup_hf_env",
        lambda *_args, **_kwargs: "0",
    )
    monkeypatch.setattr(nano_server_module, "gpu_compute_major", lambda: 12)
    monkeypatch.setattr(
        nano_server_module,
        "_ensure_reasoning_parser",
        lambda *_args, **_kwargs: tmp_path / "nano_v3_reasoning_parser.py",
    )
    monkeypatch.setattr(
        nano_server_module,
        "serve",
        lambda **kwargs: captured.update(kwargs),
    )

    nano_server_module.run()

    serve_args = captured["extra_serve_args"]
    assert captured["model"] == "nvidia/test-nano-blackwell"
    assert captured["port"] == 8107
    assert captured["persistent"] is True
    assert "--enable-auto-tool-choice" in serve_args
    parser_index = serve_args.index("--tool-call-parser")
    assert serve_args[parser_index + 1] == "qwen3_coder"


async def test_state_updates_are_driven_by_arbitrary_workflow_yaml(
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
                        "vlm_prompt": (
                            "End with PRESSURE_READING and ALARM_ACTIVE lines."
                        ),
                        "agent_prompt": "Interpret the latest instrument reading.",
                        "state_updates": [
                            {
                                "context_field": "pressure_kpa",
                                "observation_key": "PRESSURE_READING",
                                "states": ["started", "complete"],
                            },
                            {
                                "context_field": "alarm_active",
                                "observation_key": "ALARM_ACTIVE",
                                "states": ["started", "complete"],
                                "value_map": {"yes": True, "no": False},
                            },
                        ],
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
    step = workflow.step_by_id(1)

    assert step.observation_context_patch(
        "Visible display.\nPRESSURE_READING: 42.5\nALARM_ACTIVE: yes",
        state="started",
    ) == {"pressure_kpa": 42.5, "alarm_active": True}
    assert step.observation_context_patch(
        "PRESSURE_READING: 44.0\nALARM_ACTIVE: no",
        state="complete",
    ) == {"pressure_kpa": 44.0, "alarm_active": False}
    assert step.observation_context_patch(
        "PRESSURE_READING: 45.0\nALARM_ACTIVE: yes",
        state="needs_input",
    ) == {}
    assert step.observation_context_patch(
        "PRESSURE_READING: unavailable\nALARM_ACTIVE: unclear",
        state="started",
    ) == {}

    observations = iter(
        [
            SimpleNamespace(
                text="PRESSURE_READING: 42.5\nALARM_ACTIVE: yes",
                frame_pts_us=1,
            ),
            SimpleNamespace(
                text="PRESSURE_READING: 44.0\nALARM_ACTIVE: no",
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
        async def run_step(self, *, session, **_kwargs):
            agent_contexts.append(dict(session.context))
            return StepAgentResult(
                context_patch={"pressure_kpa": 1.0, "alarm_active": True},
                step_state="started",
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
    )

    await guide._evaluate(session)  # noqa: SLF001

    assert agent_contexts[0]["pressure_kpa"] == 42.5
    assert agent_contexts[0]["alarm_active"] is True
    assert session.context["pressure_kpa"] == 42.5
    assert session.context["alarm_active"] is True
    assert session.ready_step_id == 1
    assert session.step_state == "complete"
    assert len(notices) == 1

    await guide._evaluate(session)  # noqa: SLF001

    assert agent_contexts[1]["pressure_kpa"] == 44.0
    assert agent_contexts[1]["alarm_active"] is False
    assert session.context["pressure_kpa"] == 44.0
    assert session.context["alarm_active"] is False
    assert session.ready_step_id == 1
    assert session.step_state == "complete"
    assert len(notices) == 1


def test_yaml_completion_rule_cannot_be_bypassed_by_model_readiness() -> None:
    workflow = _workflow()
    step = workflow.step_by_id(1)
    context = workflow.initial_context()

    assert not workflow.advance_when_met(step, context, ready_to_advance=True)
    context.update(
        tea_name="green tea",
        tea_temperature="175 F",
        steep_time="2 minutes",
        steep_duration_seconds=120,
        context_ready=True,
    )
    assert workflow.advance_when_met(step, context)


def test_step_four_skip_replaces_zero_timestamp_with_current_time() -> None:
    workflow = _workflow()
    step = workflow.step_by_id(4)
    context = workflow.initial_context()

    applied = workflow.apply_skip_defaults(step, context)

    assert applied.keys() == {"steeping_started_at_us", "steeping_started_at_iso"}
    assert context["steeping_started_at_us"] > 0
    assert context["steeping_started_at_iso"]


def test_vlm_yes_policy_accepts_yaml_boolean_and_string_values() -> None:
    observation = "tea bag, entering the water\n\nSTEEPING_STARTED: yes"

    assert agent_module._tool_policy_met(  # noqa: SLF001
        {"vlm_verdict": "STEEPING_STARTED", "equals": True},
        observation,
    )
    assert agent_module._tool_policy_met(  # noqa: SLF001
        {"vlm_verdict": "STEEPING_STARTED", "equals": "yes"},
        observation,
    )


async def test_vlm_yes_automatically_captures_steeping_start_time() -> None:
    workflow = _workflow()
    step = workflow.step_by_id(4)
    captured_messages: list = []
    tool_calls: list[tuple[str, dict]] = []

    class _Llm:
        async def chat(self, messages, **_kwargs):
            captured_messages.extend(messages)
            return SimpleNamespace(
                content=(
                    '{"context":{"steeping_started_at_us":0,'
                    '"steeping_started_at_iso":""},"ready_to_advance":false,'
                    '"step_state":"started","assistant_message":"","speak":false}'
                ),
                tool_calls=[],
            )

    class _Tools:
        def definitions(self):
            return [SimpleNamespace(name="get_current_time")]

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
        answer_prompt=(
            _WORKER_DIR / "tea_making_worker" / "prompts" / "system.txt"
        ),
    )
    session = WorkflowSession(
        participant_id="alice",
        step_id=4,
        context=workflow.initial_context(),
    )

    result = await agent.run_step(
        step=step,
        session=session,
        vlm_observation="tea bag, entering the water\n\nSTEEPING_STARTED: yes",
    )

    assert tool_calls == [("get_current_time", {})]
    assert result.context_patch["steeping_started_at_us"] == 1_785_798_780_376_000
    assert result.context_patch["steeping_started_at_iso"] == (
        "2026-08-03T16:13:00-07:00"
    )
    assert "1785798780376000" in captured_messages[1].content


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


def test_timer_question_eval_cases_report_elapsed_or_remaining(monkeypatch) -> None:
    workflow = _workflow()
    now_us = 10_000_000_000
    context = workflow.initial_context()
    context.update(
        steeping_started_at_us=now_us - 65_000_000,
        steep_duration_seconds=180,
    )
    session = WorkflowSession(participant_id="alice", step_id=5, context=context)
    cases = yaml.safe_load((_SAMPLE_DIR / "eval" / "cases.yaml").read_text())
    monkeypatch.setattr(guide_module, "_now_us", lambda: now_us)

    for case in cases["timer_questions"]:
        answer = guide_module._time_question_answer(  # noqa: SLF001
            case["utterance"],
            session,
            workflow,
        )
        if case["expected"] == "elapsed":
            assert "1 minute 5 seconds" in answer
        else:
            assert "1 minute 55 seconds" in answer
            assert "left" in answer

    monkeypatch.setattr(guide_module, "_now_us", lambda: now_us + 120_000_000)
    assert "time is up" in guide_module._time_question_answer(  # noqa: SLF001
        "How much longer do I need to wait?",
        session,
        workflow,
    )


def test_timer_questions_do_not_invent_a_missing_start_time() -> None:
    workflow = _workflow()
    context = workflow.initial_context()
    context["steep_duration_seconds"] = 180
    session = WorkflowSession(participant_id="alice", step_id=4, context=context)

    remaining = guide_module._time_question_answer(  # noqa: SLF001
        "How long do I have remaining?",
        session,
        workflow,
    )
    started = guide_module._time_question_answer(  # noqa: SLF001
        "When did I steep the tea?",
        session,
        workflow,
    )

    assert "has not started" in remaining
    assert "3 minutes" in remaining
    assert "no start time is recorded" in started


def test_timer_start_time_question_uses_recorded_timestamp(monkeypatch) -> None:
    workflow = _workflow()
    started_at_us = 1_785_798_780_000_000
    context = workflow.initial_context()
    context.update(
        steeping_started_at_us=started_at_us,
        steep_duration_seconds=180,
    )
    session = WorkflowSession(participant_id="alice", step_id=5, context=context)
    monkeypatch.setattr(
        guide_module,
        "_now_us",
        lambda: started_at_us + 30_000_000,
    )

    answer = guide_module._time_question_answer(  # noqa: SLF001
        "When did I steep the tea?",
        session,
        workflow,
    )

    assert "timer started at" in answer
    assert "time is up" not in answer


async def test_timer_step_expires_without_calling_vision_or_step_agent() -> None:
    workflow = _workflow()
    now_us = guide_module._now_us()  # noqa: SLF001
    context = workflow.initial_context()
    context.update(
        steeping_started_at_us=now_us - 181_000_000,
        steep_duration_seconds=180,
    )
    notices: list[tuple[str, str]] = []

    class _Vision:
        async def observe(self, *_args, **_kwargs):
            raise AssertionError("timer-only step must not invoke the VLM")

        def release(self, _participant_id: str) -> None:
            return None

    class _Agent:
        async def run_step(self, *_args, **_kwargs):
            raise AssertionError("timer-only step must not invoke the step agent")

    async def notice(participant_id: str, text: str) -> None:
        notices.append((participant_id, text))

    guide = WorkflowGuide(
        workflow=workflow,
        vision=_Vision(),
        agent=_Agent(),
        notice=notice,
    )
    session = WorkflowSession(participant_id="alice", step_id=5, context=context)

    await guide._evaluate(session)  # noqa: SLF001

    assert session.active is False
    assert session.step_id == 0
    assert session.step_state == "idle"
    assert session.ready_step_id is None
    assert session.context["steeping_complete"] is False
    assert notices == [
        ("alice", "The steeping time is up. Remove the tea bag, infuser, or leaves now.")
    ]


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


async def test_completed_water_step_silently_updates_observed_state() -> None:
    workflow = _workflow()
    notices: list[tuple[str, str]] = []

    class _Vision:
        async def observe(self, *_args, **_kwargs):
            return SimpleNamespace(
                text="TEMPERATURE_READING: 75 C\nWATER_READY: no",
                frame_pts_us=2_000_000,
            )

        def release(self, _participant_id: str) -> None:
            return None

    class _Agent:
        async def run_step(self, *_args, **_kwargs):
            return StepAgentResult(
                context_patch={
                    "water_temperature_current": "59 C",
                    "water_ready": True,
                    "tea_name": "must not replace existing context",
                },
                step_state="started",
                assistant_message="This must stay silent.",
                speak=True,
            )

    async def notice(participant_id: str, text: str) -> None:
        notices.append((participant_id, text))

    guide = WorkflowGuide(
        workflow=workflow,
        vision=_Vision(),
        agent=_Agent(),
        notice=notice,
    )
    context = workflow.initial_context()
    context.update(
        tea_name="Aged Earl Grey",
        water_temperature_current="59 C",
        water_ready=True,
    )
    session = WorkflowSession(
        participant_id="alice",
        step_id=3,
        context=context,
        ready_step_id=3,
        step_state="complete",
    )

    await guide._evaluate(session)  # noqa: SLF001

    assert session.context["water_temperature_current"] == "75 C"
    assert session.context["water_ready"] is False
    assert session.context["tea_name"] == "Aged Earl Grey"
    assert session.step_state == "complete"
    assert session.ready_step_id == 3
    assert notices == []


async def test_participant_reconnect_preserves_active_workflow_state() -> None:
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

    await guide.release("alice")

    assert session.active is True
    assert session.connected is False
    assert session.step_id == 4
    assert session.context["tea_name"] == "green tea"

    await guide.resume("alice")

    assert session.active is True
    assert session.connected is True
    assert session.step_id == 4


async def test_final_advance_resets_to_idle_and_restart_uses_fresh_context() -> None:
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
        steeping_complete=True,
    )
    session = WorkflowSession(
        participant_id="alice",
        step_id=5,
        context=context,
        ready_step_id=5,
        step_state="complete",
        observation_log=[{"caption": "old evidence"}],
    )
    guide._sessions["alice"] = session  # noqa: SLF001

    response = await guide.advance("alice")

    assert "steeping time is up" in response.lower()
    assert session.active is False
    assert session.step_id == 0
    assert session.step_state == "idle"
    assert session.context["tea_name"] == ""
    assert session.context["steeping_started_at_us"] == 0
    assert session.observation_log == []

    assert guide.status("alice") == "No guided workflow is active."
    await guide.start("alice")
    restarted = guide._sessions["alice"]  # noqa: SLF001
    assert restarted.active is True
    assert restarted.step_id == 1
    assert restarted.context["tea_name"] == ""


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
        answer_prompt=(
            _WORKER_DIR / "tea_making_worker" / "prompts" / "system.txt"
        ),
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


async def test_answer_agent_prioritizes_latest_visual_state_over_old_turns() -> None:
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
        answer_prompt=(
            _WORKER_DIR / "tea_making_worker" / "prompts" / "system.txt"
        ),
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
    assert prompt.index("The water is currently 59 C.") < prompt.index(
        "[Authoritative current state]"
    )
    assert '"water_temperature_current": "100 C"' in prompt
    assert "TEMPERATURE_READING: 100 C" in prompt
    assert "TEMPERATURE_READING: 59 C" not in prompt.split(
        "[Latest VLM observation]",
        maxsplit=1,
    )[1]
