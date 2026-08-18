# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the native tea-making sample."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from xr_ai_models import ChatResponse, ToolCall
from xr_ai_runtime import AgentRuntime
from xr_ai_tools.image import ImageRegistry
from xr_ai_voice import VoiceParticipantJoined, VoiceParticipantLeft

_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / "agent-samples" / "tea-making-sample"
_WORKER = _SAMPLE / "worker"
sys.path.insert(0, str(_WORKER))

from tea_making_worker.background_context import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    BackgroundContextAgent,
)
from tea_making_worker.change_watch import ChangeWatchAgent  # noqa: E402  # pyright: ignore[reportMissingImports]
from tea_making_worker.config import load_config  # noqa: E402  # pyright: ignore[reportMissingImports]
from tea_making_worker.events import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    FOREGROUND_RECORD_TOPIC,
    GUIDANCE_RECORD_TOPIC,
    PARTICIPANT_CLEANUP_COMPLETE_TOPIC,
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    ForegroundRecord,
    GuidanceRecord,
    ParticipantCleanupComplete,
)
from tea_making_worker.file_output import FileOutputAgent  # noqa: E402  # pyright: ignore[reportMissingImports]
from tea_making_worker.foreground import ForegroundAgent  # noqa: E402  # pyright: ignore[reportMissingImports]
from tea_making_worker.spec import load_workflow  # noqa: E402  # pyright: ignore[reportMissingImports]
from tea_making_worker.transcript import TranscriptAgent  # noqa: E402  # pyright: ignore[reportMissingImports]
from tea_making_worker.video_log import VideoLogAgent  # noqa: E402  # pyright: ignore[reportMissingImports]
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


def test_workflow_rejects_incomplete_completion_and_transitions_atomically() -> None:
    workflow = load_workflow(_SAMPLE / "yaml/workflow.yaml")
    store = WorkflowStore(workflow)
    session = store.get("participant-invalid")
    store.start(session)
    store.observe(session, "Twinings Earl Grey label")

    rejected = store.commit(session, {"tea_ready": True}, "")

    assert rejected.accepted is False
    assert "tea_name" in rejected.message
    assert "tea_ready" not in session.state

    session.state["tea_ready"] = True
    store.advance(session, skip=False)
    assert session.step_id == "fill_water"
    revision = session.revision
    with pytest.raises(ValueError, match="target_temperature_c"):
        store.advance(session, skip=True)
    assert session.step_id == "fill_water"
    assert session.state["water_filled"] is False
    assert session.revision == revision


def test_skipping_steeping_detection_also_skips_the_timer() -> None:
    workflow = load_workflow(_SAMPLE / "yaml/workflow.yaml")
    store = WorkflowStore(workflow)
    session = store.get("participant-skip")
    store.start(session)
    store.advance(session, skip=True)
    store.advance(session, skip=True)
    store.advance(session, skip=True)
    assert session.step_id == "start_steeping"

    message = store.advance(session, skip=True)

    assert session.active is False
    assert session.step_id is None
    assert "guidance complete" in message.lower()
    assert "steeping_started_at_us" not in session.state


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
async def test_completed_step_does_not_invoke_trigger_or_model() -> None:
    workflow = load_workflow(_SAMPLE / "yaml/workflow.yaml")
    calls = 0

    class Llm:
        async def chat(self, *_args, **_kwargs):
            nonlocal calls
            calls += 1
            raise AssertionError("completed steps must not call the model")

    guidance = GuidanceAgent(
        workflow=workflow,
        llm=Llm(),  # type: ignore[arg-type]
        current_frame=SimpleNamespace(),  # type: ignore[arg-type]
        image_query=SimpleNamespace(),  # type: ignore[arg-type]
        rag=SimpleNamespace(),  # type: ignore[arg-type]
    )
    session = guidance.store.get("participant-complete")
    guidance.store.start(session)
    guidance.store.observe(session, "Twinings Earl Grey label")
    result = guidance.store.commit(
        session,
        {
            "tea_name": "Twinings Earl Grey",
            "target_temperature_c": 100,
            "steep_duration_s": 180,
            "guidance_source": "package",
            "tea_ready": True,
        },
        "",
    )
    assert result.complete is True

    await guidance._tick(session)

    assert calls == 0


@pytest.mark.asyncio
async def test_slow_observation_does_not_block_reset() -> None:
    workflow = load_workflow(_SAMPLE / "yaml/workflow.yaml")
    started = asyncio.Event()
    release = asyncio.Event()

    class Llm:
        async def chat(self, *_args, **_kwargs):
            started.set()
            await release.wait()
            return ChatResponse(
                "",
                None,
                [
                    ToolCall(
                        id="commit-1",
                        name="workflow__commit",
                        arguments=json.dumps(
                            {
                                "updates": {
                                    "tea_name": "Twinings Earl Grey",
                                    "target_temperature_c": 100,
                                    "steep_duration_s": 180,
                                    "guidance_source": "package",
                                    "tea_ready": True,
                                },
                                "message": "",
                            }
                        ),
                    )
                ],
                "tool_calls",
                {},
            )

    guidance = GuidanceAgent(
        workflow=workflow,
        llm=Llm(),  # type: ignore[arg-type]
        current_frame=SimpleNamespace(),  # type: ignore[arg-type]
        image_query=SimpleNamespace(),  # type: ignore[arg-type]
        rag=SimpleNamespace(),  # type: ignore[arg-type]
    )
    session = guidance.store.get("participant-slow")
    guidance.store.start(session)

    async def visible_tea(*_args):
        return SimpleNamespace(available=True, value="Twinings Earl Grey label")

    guidance._trigger = visible_tea  # type: ignore[method-assign]
    tick = asyncio.create_task(guidance._tick(session))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    async with asyncio.timeout(0.1):
        async with session.lock:
            guidance.store.reset(session)
    release.set()
    await asyncio.wait_for(tick, timeout=1.0)

    assert session.active is False
    assert session.step_id is None


@pytest.mark.asyncio
async def test_active_workflow_keeps_background_stop_and_status_tools() -> None:
    workflow = load_workflow(_SAMPLE / "yaml/workflow.yaml")
    observed_tools: set[str] = set()

    class Llm:
        async def chat(self, _messages, *, tools, **_kwargs):
            observed_tools.update(tool.name for tool in tools)
            return ChatResponse("I can manage those tasks.", None, None, "stop", {})

    llm = Llm()
    images = SimpleNamespace(
        images=ImageRegistry(),
        get_current_frame=SimpleNamespace(),
    )
    guidance = GuidanceAgent(
        workflow=workflow,
        llm=llm,  # type: ignore[arg-type]
        current_frame=images.get_current_frame,  # type: ignore[arg-type]
        image_query=SimpleNamespace(),  # type: ignore[arg-type]
        rag=SimpleNamespace(),  # type: ignore[arg-type]
    )
    guidance.store.start(guidance.store.get("participant-controls"))
    change_watch = ChangeWatchAgent(
        images=images,  # type: ignore[arg-type]
        vlm=SimpleNamespace(),  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        caption_prompt="Caption.",
        event_prompt="Compare.",
        default_instruction="important changes",
        interval_s=2.0,
    )
    transcript = TranscriptAgent(
        llm=llm,  # type: ignore[arg-type]
        summary_prompt="Summarize.",
        summary_interval_s=120.0,
    )
    video_log = VideoLogAgent(
        images=images,  # type: ignore[arg-type]
        vlm=SimpleNamespace(),  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        caption_prompt="Caption.",
        delta_prompt="Compare.",
        interval_s=2.0,
    )
    foreground = ForegroundAgent(
        llm=llm,  # type: ignore[arg-type]
        images=images,  # type: ignore[arg-type]
        vlm=SimpleNamespace(),  # type: ignore[arg-type]
        rag=SimpleNamespace(),  # type: ignore[arg-type]
        guidance=guidance,
        background_context=BackgroundContextAgent(),
        change_watch=change_watch,
        transcript=transcript,
        video_log=video_log,
        prompt="Route.",
        vlm_timeout_s=15.0,
    )

    await foreground._answer("Is recording running?", "participant-controls")

    assert {
        "change_watch__stop",
        "change_watch__status",
        "transcript__stop",
        "transcript__status",
        "video_log__stop",
        "video_log__status",
    } <= observed_tools


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
        for producer in (
            "foreground",
            "change_watch",
            "transcript",
            "video_log",
        ):
            await runtime.publish(
                PARTICIPANT_CLEANUP_COMPLETE_TOPIC,
                ParticipantCleanupComplete(producer=producer),
                participant_id="glasses/user",
            )
        await runtime.publish(
            GUIDANCE_RECORD_TOPIC,
            GuidanceRecord(
                timestamp_us=11,
                event="participant.left",
                message="",
                state={},
            ),
            participant_id="glasses/user",
        )
        await runtime.publish(
            PARTICIPANT_CLEANUP_COMPLETE_TOPIC,
            ParticipantCleanupComplete(producer="guidance"),
            participant_id="glasses/user",
        )

    path = next(tmp_path.glob("glasses-user-*/foreground.jsonl"))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["type"] == "session"
    assert records[1]["tools"] == ["workflow__start"]
    assert records[-1]["type"] == "session_end"
    guidance = [
        json.loads(line)
        for line in next(tmp_path.glob("glasses-user-*/guidance.jsonl"))
        .read_text()
        .splitlines()
    ]
    assert guidance[-2]["event"] == "participant.left"
    assert guidance[-1]["type"] == "session_end"


def test_default_prompts_come_from_packaged_files(tmp_path: Path) -> None:
    config = load_config(_SAMPLE / "yaml/tea_making_worker.yaml")
    prompt_dir = _WORKER / "tea_making_worker/prompts"
    assert config.foreground_prompt == (
        prompt_dir / "foreground_prompt.txt"
    ).read_text().strip()
    assert config.video_delta_prompt == (
        prompt_dir / "video_delta_prompt.txt"
    ).read_text().strip()

    override = tmp_path / "worker.yaml"
    override.write_text("foreground_prompt: Explicit override\n")
    assert load_config(override).foreground_prompt == "Explicit override"


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
