# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""State-machine, native workflow, output, and deployed-eval contracts."""

import json
from pathlib import Path
from typing import Any

import pytest
from eval.benchmark import audit_fixture_leakage
from eval.cases import GUIDE_CASES, VLM_CASES
from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugin_api import Builder, FunctionBaseConfig, FunctionInfo, FunctionRef, register_function
from pydantic import ConfigDict, Field
from visual_task_guide_worker.finger_count import format_finger_count, parse_finger_count
from visual_task_guide_worker.models import GuideAgentRequest, TaskGuideReply, TaskGuideRequest
from visual_task_guide_worker.task_functions import (
    TaskControlFunctionsConfig,
    TaskStateFunctionsConfig,
)
from visual_task_guide_worker.task_store import TaskStore
from visual_task_guide_worker.workflow import TaskGuideWorkflowConfig
from xr_ai_nat.functions.vision import VisionRequest, VisionResult

_TASK = Path(__file__).parents[1] / "tasks/hand-counting"
_SAMPLE = Path(__file__).parents[1]


class _TestVisionConfig(FunctionBaseConfig, name="visual_task_test_vision"):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    queries: Any = Field(exclude=True)


@register_function(config_type=_TestVisionConfig)
async def _test_vision(config: _TestVisionConfig, _builder: Builder):
    async def vision(request: VisionRequest) -> VisionResult:
        config.queries.append(request.query)
        return VisionResult(text="COUNT=2; HANDS=1; CONFIDENCE=high; NOTE=two straight fingers")

    yield FunctionInfo.from_fn(vision, description="Return one test vision result.")


class _TestGuideConfig(FunctionBaseConfig, name="visual_task_test_guide"):
    pass


@register_function(config_type=_TestGuideConfig)
async def _test_guide(_config: _TestGuideConfig, _builder: Builder):
    async def guide(request: GuideAgentRequest) -> TaskGuideReply:
        text = request.latest_observation or "No observation."
        return TaskGuideReply(response=text)

    yield FunctionInfo.from_fn(guide, description="Echo the latest test observation.")


def _store(tmp_path: Path) -> TaskStore:
    return TaskStore(_TASK)


def test_deployed_eval_covers_both_prompts_without_fixture_leakage() -> None:
    audit_fixture_leakage()
    assert GUIDE_CASES and VLM_CASES
    assert all(case["max_words"] <= 30 for case in GUIDE_CASES)
    models = json.loads((_SAMPLE / "yaml/models.local.json").read_text(encoding="utf-8"))
    assert models["models"]["guide_llm"]["adapter"] == {"preset": "nemotron3_nano"}
    assert models["models"]["guide_llm"]["endpoint"]["base_url"] == "http://localhost:8107"
    assert models["models"]["guide_llm"]["deployment"] == {
        "ownership": "reused",
        "service": "agent-llm",
    }
    assert models["models"]["vlm"]["deployment"] == {"ownership": "reused", "service": "vlm"}
    assert models["models"]["embedding"]["deployment"] == {
        "ownership": "reused",
        "service": "embedding",
    }


def test_sample_uses_standard_web_client_without_recorded_video_service() -> None:
    hub_config = (_SAMPLE / "yaml/xr_media_hub.yaml").read_text(encoding="utf-8")
    voice_gate = (_SAMPLE / "yaml/voice_gate.yaml").read_text(encoding="utf-8")
    launcher = (_SAMPLE / "main.py").read_text(encoding="utf-8")
    app = (_SAMPLE / "worker/visual_task_guide_worker/app.py").read_text(encoding="utf-8")
    agent = (_SAMPLE / "worker/visual_task_guide_worker/agent.py").read_text(encoding="utf-8")

    assert "web_client_dir: ../../../client-samples/web\n" in hub_config
    assert "magic_phrases: []\n" in voice_gate
    assert "video_recording:" not in hub_config
    assert "video-memory-service" not in launcher
    assert "services/rag-service" in launcher
    assert "RAGFunctionsConfig" in app
    assert "ChatCompletionConfig" in agent
    assert "ToolCallAgentWorkflowConfig" not in agent
    assert 'text_topic=_OUTPUT_TOPIC' in app


def test_bundled_workflow_counts_from_one_through_ten(tmp_path) -> None:
    store = _store(tmp_path)

    assert [step.id for step in store.steps] == [
        "show-one",
        "show-two",
        "show-three",
        "show-four",
        "show-five",
        "show-six",
        "show-seven",
        "show-eight",
        "show-nine",
        "show-ten",
    ]
    assert all(step.knowledge_files for step in store.steps)


def test_task_store_requires_explicit_state_transitions(tmp_path) -> None:
    store = _store(tmp_path)
    initial = store.progress("alice")
    assert initial.state == "not_started"
    assert store.current_step(initial).id == "show-one"
    assert store.next_step(initial).id == "show-two"
    with pytest.raises(ValueError, match="not started"):
        store.advance("alice")

    started = store.start("alice")
    assert started.state == "running"
    assert store.current_step(started).id == "show-one"
    advanced = store.advance("alice")
    assert store.current_step(advanced).id == "show-two"
    assert store.next_step(advanced).id == "show-three"
    reset = store.reset("alice")
    assert reset.state == "not_started"
    assert store.current_step(reset).id == "show-one"


def test_task_store_completes_only_after_ten_explicit_advances(tmp_path) -> None:
    store = _store(tmp_path)
    store.start("alice")
    for _ in range(9):
        assert store.advance("alice").state == "running"
    completed = store.advance("alice")
    assert completed.state == "completed"
    assert store.current_step(completed) is None
    assert store.next_step(completed) is None


def test_structured_finger_count_is_human_readable() -> None:
    text = "COUNT=2; HANDS=1; CONFIDENCE=high; NOTE=index and middle straight; others folded"
    parsed = parse_finger_count(text)

    assert parsed is not None
    assert parsed.count == 2
    assert format_finger_count(parsed) == (
        "2 extended fingers (high confidence). index and middle straight; others folded."
    )


@pytest.mark.asyncio
async def test_native_task_groups_separate_read_and_mutating_controls(tmp_path) -> None:
    store = _store(tmp_path)
    async with WorkflowBuilder() as builder:
        await builder.add_function_group("task_state", TaskStateFunctionsConfig(store=store))
        await builder.add_function_group("task_control", TaskControlFunctionsConfig(store=store))
        state = await builder.get_function_group("task_state")
        control = await builder.get_function_group("task_control")
        state_functions = await state.get_all_functions()
        control_functions = await control.get_all_functions()
        status = await state_functions["task_state__get_task_status"].ainvoke({"participant_id": "alice"})

    assert set(state_functions) == {"task_state__get_task_status"}
    assert set(control_functions) == {
        "task_control__start_task",
        "task_control__reset_task",
        "task_control__advance_task",
    }
    assert status.progress.state == "not_started"


@pytest.mark.asyncio
async def test_native_workflow_uses_step_specific_on_demand_vision(tmp_path) -> None:
    store = _store(tmp_path)
    queries: list[str] = []
    async with WorkflowBuilder() as builder:
        await builder.add_function("fake_vision", _TestVisionConfig(queries=queries))
        await builder.add_function_group("task_state", TaskStateFunctionsConfig(store=store))
        await builder.add_function_group("task_control", TaskControlFunctionsConfig(store=store))
        await builder.add_function("fake_guide", _TestGuideConfig())
        workflow = await builder.add_function(
            "task_guide_workflow",
            TaskGuideWorkflowConfig(
                vision=FunctionRef("fake_vision"),
                guide_agent=FunctionRef("fake_guide"),
            ),
        )

        ignored = await workflow.ainvoke(TaskGuideRequest(participant_id="alice", text="Our task."))
        started = await workflow.ainvoke(TaskGuideRequest(participant_id="alice", text="Hey agent, start task."))
        unchanged = store.progress("alice")
        advanced = await workflow.ainvoke(TaskGuideRequest(participant_id="alice", text="next step"))
        next_info = await workflow.ainvoke(
            TaskGuideRequest(participant_id="alice", text="What's the next step?")
        )
        after_next_info = store.progress("alice")
        validated = await workflow.ainvoke(
            TaskGuideRequest(participant_id="alice", text="Did I do the step correctly?")
        )
        reset = await workflow.ainvoke(TaskGuideRequest(participant_id="alice", text="reset task"))
        misheard_started = await workflow.ainvoke(TaskGuideRequest(participant_id="alice", text="Dart task."))

    assert ignored.response == "Ready: Show one. Say “start task”."
    assert started.response.startswith("Show one:")
    assert unchanged.step_index == 0
    assert advanced.response.startswith("Show two:")
    assert next_info.response.startswith("Next is Show three:")
    assert after_next_info.step_index == 1
    assert validated.response == "Show two — Yes, I see 2 extended fingers."
    assert "do not assume a target count" in queries[0].casefold()
    assert "2" not in queries[0]
    assert len(queries) == 1
    assert reset.response == "Ready: Show one. Say “start task”."
    assert misheard_started.response.startswith("Show one:")
