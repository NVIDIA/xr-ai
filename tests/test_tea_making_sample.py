# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the native tea-making sample."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from xr_ai_runtime import AgentRuntime
from xr_ai_voice import VoiceParticipantJoined, VoiceParticipantLeft

_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / "agent-samples" / "tea-making-sample"
_WORKER = _SAMPLE / "worker"
sys.path.insert(0, str(_WORKER))

from tea_making_worker.events import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    FOREGROUND_RECORD_TOPIC,
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    ForegroundRecord,
)
from tea_making_worker.file_output import FileOutputAgent  # noqa: E402  # pyright: ignore[reportMissingImports]
from tea_making_worker.spec import load_workflow  # noqa: E402  # pyright: ignore[reportMissingImports]
from tea_making_worker.workflow import GuidanceAgent  # noqa: E402  # pyright: ignore[reportMissingImports]
from tea_making_worker.workflow_state import WorkflowStore  # noqa: E402  # pyright: ignore[reportMissingImports]


def _load_main():
    path = _SAMPLE / "main.py"
    spec = importlib.util.spec_from_file_location("tea_making_sample_main", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_omni_supplies_both_language_and_vision() -> None:
    models = json.loads((_SAMPLE / "yaml/models.local.json").read_text())[
        "models"
    ]

    assert models["llm"]["deployment"]["service"] == "omni"
    assert models["vlm"]["deployment"]["service"] == "omni"
    assert models["llm"]["endpoint"]["base_url"] == "http://localhost:8108"
    assert models["vlm"]["endpoint"]["base_url"] == "http://localhost:8108"
    assert models["vlm"]["adapter"]["capabilities"]["vision"] is True
    assert "cosmos" not in json.dumps(models).lower()


def test_launcher_declares_one_omni_and_no_monitoring_ui(tmp_path: Path) -> None:
    sample_main = _load_main()
    worker_config = sample_main._materialize_worker_config(
        tmp_path,
        "always-on",
        "piper",
    )
    processes, _credentials = sample_main._build_processes(
        worker_config,
        "piper",
    )
    names = [process.name for process in processes]

    assert names[0] == "hub"
    assert names[-1] == "worker"
    assert names.count("omni") == 1
    assert "vlm" not in names
    assert "activity-viewer" not in names
    assert "rag" in names


def test_published_guide_covers_architecture_and_adaptation() -> None:
    guide = (_ROOT / "docs/source/reference/tea-making-sample.md").read_text()

    assert "## Architecture" in guide
    assert "## Source map" in guide
    assert "## Connecting a backend" in guide
    assert "## Adapting the sample" in guide
    assert "## Lifecycle invariants" in guide
    assert "GuidanceAgent" in guide
    assert "BackgroundContextAgent" in guide


def test_workflow_requires_explicit_advancement() -> None:
    workflow = load_workflow(_SAMPLE / "yaml/workflow.yaml")
    store = WorkflowStore(workflow)
    session = store.get("participant-1")

    store.start(session)
    assert session.step_id == "identify"
    assert session.active
    assert "not complete" in store.advance(session, skip=False).lower()
    assert session.step_id == "identify"

    store.advance(session, skip=True)
    assert session.step_id == "fill_water"
    assert session.state["tea_ready"] is True
    assert session.active


def test_workflow_enforces_consecutive_visual_evidence() -> None:
    workflow = load_workflow(_SAMPLE / "yaml/workflow.yaml")
    store = WorkflowStore(workflow)
    session = store.get("participant-2")
    store.start(session)
    store.advance(session, skip=True)
    observation = "The kettle contains water inside with a visible surface and level."

    store.observe(session, observation)
    store.observe(session, observation)
    rejected = store.commit(session, {"water_filled": True}, "")
    assert not rejected.accepted
    assert session.state["water_filled"] is False

    store.observe(session, observation)
    accepted = store.commit(session, {"water_filled": True}, "")
    assert accepted.accepted
    assert accepted.complete
    assert session.state["water_filled"] is True
    assert session.step_id == "fill_water"


def test_guidance_exposes_one_foreground_stack_at_a_time() -> None:
    workflow = load_workflow(_SAMPLE / "yaml/workflow.yaml")
    guidance = GuidanceAgent(
        workflow=workflow,
        llm=SimpleNamespace(),  # type: ignore[arg-type]
        current_frame=SimpleNamespace(),  # type: ignore[arg-type]
        image_query=SimpleNamespace(),  # type: ignore[arg-type]
        rag=SimpleNamespace(),  # type: ignore[arg-type]
    )

    root = {name for name, _tool in guidance.root_tools("participant-3").items()}
    assert root == {"workflow__start", "workflow__status"}
    session = guidance.store.get("participant-3")
    guidance.store.start(session)
    active = guidance.active_tools("participant-3")
    assert active is not None
    active_names = {name for name, _tool in active.items()}
    assert "workflow__start" not in active_names
    assert {
        "workflow__advance",
        "workflow__reset",
        "workflow__restart",
        "workflow__status",
        "current_view",
        "rag_lookup",
    } <= active_names
    assert guidance.active_context("participant-3") is not None


@pytest.mark.asyncio
async def test_file_output_writes_session_bounded_jsonl(tmp_path: Path) -> None:
    files = FileOutputAgent(tmp_path, history_size=4)
    runtime = AgentRuntime()
    runtime.register("files", files)

    async with runtime:
        await runtime.publish(
            PARTICIPANT_JOINED_TOPIC,
            VoiceParticipantJoined(),
            participant_id="glasses/user",
        )
        await runtime.publish(
            FOREGROUND_RECORD_TOPIC,
            ForegroundRecord(
                timestamp_us=10,
                query="help me make tea",
                response="Hold the package in view.",
                tools=["workflow__start"],
            ),
            participant_id="glasses/user",
        )
        await runtime.publish(
            PARTICIPANT_LEFT_TOPIC,
            VoiceParticipantLeft(),
            participant_id="glasses/user",
        )

    path = next(tmp_path.glob("glasses-user-*/foreground.jsonl"))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["type"] == "session"
    assert records[1]["tools"] == ["workflow__start"]
    assert records[-1]["type"] == "session_end"


def test_foreground_prompt_has_route_eval_cases() -> None:
    cases = yaml.safe_load((_SAMPLE / "eval/cases.yaml").read_text())
    assert {case["expected_tool"] for case in cases} == {
        None,
        "application_context__query",
        "change_watch__start",
        "current_view",
        "rag_lookup",
        "transcript__start",
        "video_log__start",
        "workflow__start",
    }
