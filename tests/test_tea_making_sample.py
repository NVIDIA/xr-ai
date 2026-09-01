# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the native tea-making sample."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import nemo_relay
import pytest
import yaml
from xr_ai_models import ChatMessage, ChatResponse, ToolCall
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, subscribe
from xr_ai_tools import ToolSet
from xr_ai_tools.image import ImageReference, ImageRegistry
from xr_ai_tools.tool_calling import ToolCallRecord, ToolLoopResult
from xr_ai_tools.vision import ImageQueryChunk, ImageQueryResult
from xr_ai_voice import (
    VOICE_CONTRIBUTION_TOPIC,
    UserQuery,
    VoiceOutput,
    VoiceParticipantJoined,
    VoiceParticipantLeft,
)
from xr_ai_web_events import WEB_EVENT_TOPIC, WebEvent, WebEventsAgent

_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / "agent-samples" / "tea-making-sample"
_WORKER = _SAMPLE / "worker"
sys.path.insert(0, str(_WORKER))

import tea_making_worker.foreground as foreground_module  # noqa: E402  # pyright: ignore[reportMissingImports]
from tea_making_worker.app import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    _CLIENT_TEXT_TOPIC,
    _ParticipantVoiceAggregationAgent,
    _relay_event_log,
)
from tea_making_worker.background_context import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    BackgroundContextAgent,
)
from tea_making_worker.change_watch import ChangeWatchAgent  # noqa: E402  # pyright: ignore[reportMissingImports]
from tea_making_worker.config import load_config  # noqa: E402  # pyright: ignore[reportMissingImports]
from tea_making_worker.events import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    BACKGROUND_FACT_TOPIC,
    CHANGE_WATCH_RECORD_TOPIC,
    FOREGROUND_RECORD_TOPIC,
    GUIDANCE_NOTICE_TOPIC,
    GUIDANCE_RECORD_TOPIC,
    PARTICIPANT_CLEANUP_COMPLETE_TOPIC,
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    TRANSCRIPT_RECORD_TOPIC,
    VIDEO_LOG_RECORD_TOPIC,
    BackgroundFact,
    ChangeWatchRecord,
    ForegroundRecord,
    GuidanceNotice,
    GuidanceRecord,
    ParticipantCleanupComplete,
    TranscriptRecord,
    VideoLogRecord,
)
from tea_making_worker.file_output import FileOutputAgent  # noqa: E402  # pyright: ignore[reportMissingImports]
from tea_making_worker.foreground import ForegroundAgent  # noqa: E402  # pyright: ignore[reportMissingImports]
from tea_making_worker.images import ParticipantImageAgent  # noqa: E402  # pyright: ignore[reportMissingImports]
from tea_making_worker.spec import load_workflow  # noqa: E402  # pyright: ignore[reportMissingImports]
from tea_making_worker.transcript import TranscriptAgent  # noqa: E402  # pyright: ignore[reportMissingImports]
from tea_making_worker.video_log import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    VideoLogAgent,
    _is_no_change,
)
from tea_making_worker.web_events import TeaWebEventsAgent  # noqa: E402  # pyright: ignore[reportMissingImports]
from tea_making_worker.workflow import GuidanceAgent  # noqa: E402  # pyright: ignore[reportMissingImports]
from tea_making_worker.workflow_state import WorkflowStore  # noqa: E402  # pyright: ignore[reportMissingImports]
from tea_making_worker.workflow_tools import CurrentViewRequest  # noqa: E402  # pyright: ignore[reportMissingImports]


def _load_main():
    path = _SAMPLE / "main.py"
    spec = importlib.util.spec_from_file_location("tea_making_sample_main", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_omni_supplies_both_language_and_vision() -> None:
    models = json.loads((_SAMPLE / "yaml/models.local.json").read_text())["models"]

    assert models["llm"]["deployment"]["service"] == "omni"
    assert models["vlm"]["deployment"]["service"] == "omni"
    assert models["llm"]["endpoint"]["base_url"] == "http://localhost:8108"
    assert models["vlm"]["endpoint"]["base_url"] == "http://localhost:8108"
    assert models["vlm"]["adapter"]["capabilities"]["vision"] is True
    assert models["vlm"]["adapter"]["reasoning_field"] == "reasoning_content"
    assert all(
        model["deployment"]["ownership"] == "reused"
        for model in models.values()
    )
    assert "cosmos" not in json.dumps(models).lower()


def test_launcher_declares_one_omni_and_no_monitoring_ui(tmp_path: Path) -> None:
    sample_main = _load_main()
    worker_config = sample_main._materialize_worker_config(
        tmp_path,
    )
    processes = sample_main._build_processes(worker_config)
    names = [process.name for process in processes]

    assert names[0] == "hub"
    assert names[-1] == "worker"
    assert names.count("omni") == 1
    assert "vlm" not in names
    assert "activity-viewer" not in names
    assert "rag" in names
    assert all(
        process.launch_mode == "reuse"
        for process in processes
        if process.name in {"stt", "omni", "embedding", "tts"}
    )


def test_launcher_only_exposes_web_events_override() -> None:
    sample_main = _load_main()

    default = sample_main._parse_args([])
    exposed = sample_main._parse_args(["--expose-web-events"])

    assert default.expose_web_events is False
    assert exposed.expose_web_events is True
    assert _CLIENT_TEXT_TOPIC == "agent.response"


def test_web_event_config_defaults_and_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(_SAMPLE / "yaml/tea_making_worker.yaml")

    assert config.web_events_host == "127.0.0.1"
    assert config.web_events_port == 8092
    assert config.web_events_max_events == 5_000

    invalid = tmp_path / "invalid-web-events.yaml"
    invalid.write_text("web_events_host: ''\n")
    with pytest.raises(ValueError, match="web_events_host"):
        load_config(invalid)
    invalid.write_text("web_events_port: 70000\n")
    with pytest.raises(ValueError, match="web_events_port"):
        load_config(invalid)
    invalid.write_text("web_events_max_events: 0\n")
    with pytest.raises(ValueError, match="web_events_max_events"):
        load_config(invalid)

    sample_main = _load_main()
    exposed_config = sample_main._materialize_worker_config(
        tmp_path,
        expose_web_events=True,
    )
    assert load_config(exposed_config).web_events_host == "0.0.0.0"
    assert (
        load_config(exposed_config).voice_gate_yaml
        == _SAMPLE / "yaml/voice_gate.yaml"
    )

    always_on_source = tmp_path / "always-on-worker.yaml"
    always_on_source.write_text(
        (_SAMPLE / "yaml/tea_making_worker.yaml")
        .read_text()
        .replace("voice_gate.yaml", "voice_gate.always-on.yaml"),
        encoding="utf-8",
    )
    monkeypatch.setattr(sample_main, "_WORKER_CONFIG", always_on_source)
    always_on_config = sample_main._materialize_worker_config(tmp_path)
    assert load_config(always_on_config).voice_gate_yaml == (
        tmp_path / "voice_gate.always-on.yaml"
    )


@pytest.mark.asyncio
async def test_typed_events_are_projected_to_web_topics() -> None:
    captured: list[tuple[WebEvent, str | None]] = []

    class Capture(Agent):
        def __init__(self) -> None:
            super().__init__()

        @subscribe(WEB_EVENT_TOPIC)
        async def event(self, event: WebEvent, ctx: RuntimeContext) -> None:
            captured.append((event, ctx.metadata.participant_id))

    runtime = AgentRuntime()
    runtime.register("web-event-adapter", TeaWebEventsAgent())
    runtime.register("capture", Capture())
    participant_id = "participant-web"
    publications = (
        (
            FOREGROUND_RECORD_TOPIC,
            ForegroundRecord(timestamp_us=1, query="What now?", response="Pour."),
        ),
        (
            GUIDANCE_RECORD_TOPIC,
            GuidanceRecord(
                timestamp_us=2,
                event="step.enter",
                message="Fill the kettle.",
            ),
        ),
        (
            GUIDANCE_NOTICE_TOPIC,
            GuidanceNotice(timestamp_us=3, text="The water is ready."),
        ),
        (
            BACKGROUND_FACT_TOPIC,
            BackgroundFact(
                timestamp_us=4,
                application="transcript",
                text="User mentioned Earl Grey.",
            ),
        ),
        (
            BACKGROUND_FACT_TOPIC,
            BackgroundFact(
                timestamp_us=5,
                application="change_watch",
                text="A hand was raised.",
            ),
        ),
        (
            BACKGROUND_FACT_TOPIC,
            BackgroundFact(
                timestamp_us=6,
                application="video_log",
                text="A person entered the room.",
            ),
        ),
        (
            CHANGE_WATCH_RECORD_TOPIC,
            ChangeWatchRecord(timestamp_us=7, record_type="started"),
        ),
        (
            CHANGE_WATCH_RECORD_TOPIC,
            ChangeWatchRecord(
                timestamp_us=8,
                record_type="observation",
                caption="A hand remains raised.",
            ),
        ),
        (
            TRANSCRIPT_RECORD_TOPIC,
            TranscriptRecord(
                timestamp_us=9,
                record_type="utterance",
                text="Start steeping.",
            ),
        ),
        (
            VIDEO_LOG_RECORD_TOPIC,
            VideoLogRecord(timestamp_us=10, record_type="observation", caption="A cup."),
        ),
        (PARTICIPANT_JOINED_TOPIC, VoiceParticipantJoined()),
        (PARTICIPANT_LEFT_TOPIC, VoiceParticipantLeft()),
    )

    async with runtime:
        for topic, event in publications:
            await runtime.publish(topic, event, participant_id=participant_id)

    assert [event.topic for event, _participant in captured] == [
        "foreground",
        "guidance.events",
        "guidance.notices",
        "background.change-watch",
        "background.video-log",
        "background.change-watch",
        "background.transcript",
        "participant.lifecycle",
        "participant.lifecycle",
    ]
    assert captured[3][0].payload == {"text": "A hand was raised."}
    assert captured[4][0].payload == {"text": "A person entered the room."}
    assert {participant for _event, participant in captured} == {participant_id}
    assert captured[-2][0].payload == {"event": "joined"}
    assert captured[-1][0].payload == {"event": "left"}


@pytest.mark.asyncio
async def test_change_watch_rejects_prose_as_invalid_model_output() -> None:
    chat = AsyncMock(
        return_value=ChatResponse("The hand is raised.", None, None, "stop", {})
    )

    agent = object.__new__(ChangeWatchAgent)
    agent._llm = SimpleNamespace(chat=chat)  # type: ignore[attr-defined]
    agent._event_prompt = "Compare and commit."  # type: ignore[attr-defined]

    with pytest.raises(ValueError):
        await agent._decide(
            SimpleNamespace(instruction="raised hands", captions=["Hands are down."]),
            "A hand is raised.",
        )

    assert chat.await_count == 1


@pytest.mark.asyncio
async def test_change_watch_accepts_typed_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def execute(_name, arguments, handler, *_args, **_kwargs):
        return await handler(arguments)

    monkeypatch.setattr("xr_ai_tools.tools.typed.tool_execute", execute)
    chat = AsyncMock(
        return_value=ChatResponse(
            "",
            None,
            [
                ToolCall(
                    id="change-commit",
                    name="change_watch__commit",
                    arguments=json.dumps(
                        {"important": True, "summary": "A hand was raised."}
                    ),
                )
            ],
            "tool_calls",
            {},
        )
    )

    agent = object.__new__(ChangeWatchAgent)
    agent._llm = SimpleNamespace(chat=chat)  # type: ignore[attr-defined]
    agent._event_prompt = "Compare and commit."  # type: ignore[attr-defined]

    decision = await agent._decide(
        SimpleNamespace(instruction="raised hands", captions=["Hands are down."]),
        "A hand is raised.",
    )

    assert decision is not None
    assert decision.important is True
    assert decision.summary == "A hand was raised."
    assert chat.await_count == 1


@pytest.mark.asyncio
async def test_video_log_rejects_prose_as_invalid_model_output() -> None:
    chat = AsyncMock(
        return_value=ChatResponse("A person entered.", None, None, "stop", {})
    )

    agent = object.__new__(VideoLogAgent)
    agent._llm = SimpleNamespace(chat=chat)  # type: ignore[attr-defined]
    agent._delta_prompt = "Compare and commit."  # type: ignore[attr-defined]
    agent._history_size = 5  # type: ignore[attr-defined]

    with pytest.raises(ValueError):
        await agent._generate_delta(
            SimpleNamespace(captions=["The room is empty."]),
            "A person is in the room.",
        )

    assert chat.await_count == 1


@pytest.mark.asyncio
async def test_video_log_accepts_typed_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def execute(_name, arguments, handler, *_args, **_kwargs):
        return await handler(arguments)

    monkeypatch.setattr("xr_ai_tools.tools.typed.tool_execute", execute)
    chat = AsyncMock(
        return_value=ChatResponse(
            "",
            None,
            [
                ToolCall(
                    id="video-commit",
                    name="video_log__commit",
                    arguments=json.dumps({"delta": "A person entered."}),
                )
            ],
            "tool_calls",
            {},
        )
    )

    agent = object.__new__(VideoLogAgent)
    agent._llm = SimpleNamespace(chat=chat)  # type: ignore[attr-defined]
    agent._delta_prompt = "Compare and commit."  # type: ignore[attr-defined]
    agent._history_size = 5  # type: ignore[attr-defined]

    delta = await agent._generate_delta(
        SimpleNamespace(captions=["The room is empty."]),
        "A person is in the room.",
    )

    assert delta is not None
    assert delta.delta == "A person entered."
    assert chat.await_count == 1


@pytest.mark.parametrize(
    "delta",
    (
        "No meaningful visual change.",
        "No meaningful visual changes; the scene remains stable.",
        "There was no meaningful visual change.",
    ),
)
def test_video_log_recognizes_no_change_variants(delta: str) -> None:
    assert _is_no_change(delta)
    assert not _is_no_change("A person entered the room.")
    assert not _is_no_change(
        "No meaningful visual change, but a person entered the room."
    )


@pytest.mark.asyncio
async def test_change_watch_records_classifier_failure_with_caption() -> None:
    participant_id = "participant-change-error"
    agent = object.__new__(ChangeWatchAgent)
    agent._states = {
        participant_id: SimpleNamespace(
            instruction="raised hands",
            captions=deque(["Hands are down."]),
            lock=asyncio.Lock(),
        )
    }
    agent._caption_prompt = "Describe the focused visual state."
    agent._images = SimpleNamespace(
        get_current_frame=SimpleNamespace(
            execute=AsyncMock(
                return_value=SimpleNamespace(image=ImageReference(uri="frame.jpg"))
            )
        )
    )
    agent._query_image = SimpleNamespace(
        execute=AsyncMock(
            return_value=ImageQueryResult(text="A hand is raised.", available=True)
        )
    )
    agent._decide = AsyncMock(side_effect=ValueError("invalid classifier output"))
    agent._publish_error = AsyncMock()

    await agent._observe(participant_id)

    error = agent._publish_error.await_args
    assert error.args[0] == participant_id
    assert error.args[1] is agent._states[participant_id]
    assert error.args[3] == "invalid classifier output"
    assert error.kwargs == {"caption": "A hand is raised."}


@pytest.mark.asyncio
async def test_video_log_records_classifier_failure_with_caption() -> None:
    participant_id = "participant-video-error"
    agent = object.__new__(VideoLogAgent)
    agent._states = {
        participant_id: SimpleNamespace(
            captions=deque(["The room is empty."]),
            lock=asyncio.Lock(),
        )
    }
    agent._caption_prompt = "Describe the current scene."
    agent._images = SimpleNamespace(
        get_current_frame=SimpleNamespace(
            execute=AsyncMock(
                return_value=SimpleNamespace(image=ImageReference(uri="frame.jpg"))
            )
        )
    )
    agent._query_image = SimpleNamespace(
        execute=AsyncMock(
            return_value=ImageQueryResult(
                text="A person entered the room.",
                available=True,
            )
        )
    )
    agent._generate_delta = AsyncMock(
        side_effect=ValueError("invalid classifier output")
    )
    agent._publish_error = AsyncMock()

    await agent._observe(participant_id)

    error = agent._publish_error.await_args
    assert error.args[0] == participant_id
    assert error.args[2] == "invalid classifier output"
    assert error.kwargs == {"caption": "A person entered the room."}


@pytest.mark.asyncio
async def test_video_log_first_frame_is_a_baseline_observation() -> None:
    participant_id = "participant-video-baseline"
    state = SimpleNamespace(captions=deque(), lock=asyncio.Lock())
    agent = object.__new__(VideoLogAgent)
    agent._states = {participant_id: state}
    agent._caption_prompt = "Describe the current scene."
    agent._images = SimpleNamespace(
        get_current_frame=SimpleNamespace(
            execute=AsyncMock(
                return_value=SimpleNamespace(image=ImageReference(uri="frame.jpg"))
            )
        )
    )
    agent._query_image = SimpleNamespace(
        execute=AsyncMock(
            return_value=ImageQueryResult(text="The room is empty.", available=True)
        )
    )
    agent._generate_delta = AsyncMock()
    agent._publish_record = AsyncMock()

    await agent._observe(participant_id)

    record = agent._publish_record.await_args.args[1]
    assert record.record_type == "observation"
    assert record.caption == "The room is empty."
    assert record.delta == ""
    assert list(state.captions) == ["The room is empty."]
    agent._generate_delta.assert_not_awaited()


@pytest.mark.asyncio
async def test_web_viewer_surrounds_runtime_lifecycle() -> None:
    runtime = AgentRuntime()
    viewer = runtime.register(
        "web-events",
        WebEventsAgent(host="127.0.0.1", port=0, max_events=8),
    )
    runtime.register("web-event-adapter", TeaWebEventsAgent())

    assert not viewer.running
    async with viewer:
        assert viewer.running
        async with runtime:
            await runtime.publish(
                PARTICIPANT_JOINED_TOPIC,
                VoiceParticipantJoined(),
                participant_id="participant-live",
            )
        assert viewer.running
    assert not viewer.running


@pytest.mark.asyncio
async def test_participant_leave_releases_voice_aggregation_state() -> None:
    aggregation = object.__new__(_ParticipantVoiceAggregationAgent)
    aggregation.release = AsyncMock()
    ctx = SimpleNamespace(metadata=SimpleNamespace(participant_id="participant-voice"))

    await aggregation.participant_left(
        VoiceParticipantLeft(),
        ctx,  # type: ignore[arg-type]
    )

    aggregation.release.assert_awaited_once_with("participant-voice")


def test_published_guide_covers_architecture_and_adaptation() -> None:
    guide = (_ROOT / "docs/source/reference/tea-making-sample.md").read_text()

    assert "## Architecture" in guide
    assert "## Source map" in guide
    assert "## Connecting a backend" in guide
    assert "## Live inspection and durable output" in guide
    assert "## Adapting the sample" in guide
    assert "## Lifecycle invariants" in guide
    assert "GuidanceAgent" in guide
    assert "BackgroundContextAgent" in guide
    assert "TeaWebEventsAgent" in guide


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


def test_skipping_completed_step_preserves_verified_state() -> None:
    workflow = load_workflow(_SAMPLE / "yaml/workflow.yaml")
    store = WorkflowStore(workflow)
    session = store.get("participant-verified")
    store.start(session)
    store.observe(session, "Twinings Earl Grey label")
    accepted = store.commit(
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
    assert accepted.complete is True

    message = store.advance(session, skip=True)

    assert session.step_id == "fill_water"
    assert session.state["tea_name"] == "Twinings Earl Grey"
    assert session.state["target_temperature_c"] == 100
    assert session.state["guidance_source"] == "package"
    assert "generic" not in message.lower()


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
        "current_view",
        "rag_lookup",
    } <= active_names
    assert guidance.active_context("participant-3") is not None


def test_guidance_answers_active_structure_without_tools() -> None:
    workflow = load_workflow(_SAMPLE / "yaml/workflow.yaml")
    guidance = GuidanceAgent(
        workflow=workflow,
        llm=SimpleNamespace(),  # type: ignore[arg-type]
        current_frame=SimpleNamespace(),  # type: ignore[arg-type]
        image_query=SimpleNamespace(),  # type: ignore[arg-type]
        rag=SimpleNamespace(),  # type: ignore[arg-type]
    )
    session = guidance.store.get("participant-readonly")
    guidance.store.start(session)
    guidance.store.advance(session, skip=True)

    instructions = guidance._active_readonly_answer(
        "participant-readonly",
        "What are the tea instructions?",
    )
    next_step = guidance._active_readonly_answer(
        "participant-readonly",
        "What is the next step?",
    )

    assert instructions is not None
    assert "Fill a kettle" in instructions
    assert "Heat the water to 93 degrees Celsius" in instructions
    assert next_step == "Heat the water to 93 degrees Celsius."


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Move on to the following tea step.", "workflow__advance"),
        ("End this tea-making session.", "workflow__reset"),
        ("Begin the tea instructions again from the first step.", "workflow__restart"),
        ("What is the next step?", None),
        ("Please stop monitoring tea in the background.", None),
        ("Advance my career.", None),
    ],
)
def test_workflow_controls_require_a_direct_guide_command(
    query: str,
    expected: str | None,
) -> None:
    assert foreground_module._requested_workflow_control(query) == expected


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
    )

    await foreground._answer("Is recording running?", "participant-controls")

    active_tools = guidance.active_tools("participant-controls")
    assert active_tools is not None
    current_view = active_tools.get("current_view")

    assert {
        "change_watch__stop",
        "change_watch__status",
        "transcript__stop",
        "transcript__status",
        "video_log__stop",
        "video_log__status",
    } <= observed_tools
    assert current_view is not None and current_view.return_direct is False


@pytest.mark.asyncio
async def test_active_current_view_preserves_structured_question() -> None:
    workflow = load_workflow(_SAMPLE / "yaml/workflow.yaml")
    observed: list[str] = []

    class CurrentFrame:
        async def execute(self, _request):
            return SimpleNamespace(image=ImageReference(uri="frame.jpg"))

    class ImageQuery:
        async def execute(self, request):
            observed.append(request.query)
            return ImageQueryResult(text="The display reads 85 Celsius.")

    guidance = GuidanceAgent(
        workflow=workflow,
        llm=SimpleNamespace(),  # type: ignore[arg-type]
        current_frame=CurrentFrame(),  # type: ignore[arg-type]
        image_query=ImageQuery(),  # type: ignore[arg-type]
        rag=SimpleNamespace(),  # type: ignore[arg-type]
    )
    guidance.store.start(guidance.store.get("participant-question"))
    active_tools = guidance.active_tools("participant-question")
    assert active_tools is not None
    current_view = active_tools.get("current_view")
    assert current_view is not None

    result = await current_view.handler(
        CurrentViewRequest(question="Read only the kettle temperature and unit.")
    )

    assert result.text == "The display reads 85 Celsius."
    assert observed == ["Read only the kettle temperature and unit."]
    assert current_view.return_direct is False


@pytest.mark.asyncio
async def test_mixed_tool_turn_with_streamed_current_view_is_already_spoken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = (
        ToolCallRecord(
            call=ToolCall(
                id="rag-1",
                name="rag_lookup",
                arguments='{"query":"Earl Grey"}',
            ),
            message=ChatMessage(
                role="tool",
                content="Reference result",
                tool_call_id="rag-1",
            ),
            return_direct=False,
        ),
        ToolCallRecord(
            call=ToolCall(
                id="view-1",
                name="current_view",
                arguments='{"question":"What do you see?"}',
            ),
            message=ChatMessage(
                role="tool",
                content="A kettle is visible.",
                tool_call_id="view-1",
            ),
            return_direct=True,
        ),
    )

    async def run_loop(*_args, **_kwargs):
        return ToolLoopResult(
            content="A kettle is visible.",
            messages=(),
            tool_calls=records,
            iterations=2,
            return_direct=True,
        )

    monkeypatch.setattr(foreground_module, "run_tool_loop", run_loop)
    foreground = object.__new__(ForegroundAgent)
    foreground._guidance = SimpleNamespace(
        active_context=lambda _pid: None,
        _active_readonly_answer=lambda _pid, _query: None,
    )
    foreground._root_tools = lambda *_args, **_kwargs: ToolSet(())
    foreground._prompt = "Route."
    foreground._llm = SimpleNamespace()

    response, calls, spoken = await foreground._answer(
        "Use the reference, then tell me what you see.",
        "participant-mixed-view",
    )

    assert response == "A kettle is visible."
    assert calls == ["rag_lookup", "current_view"]
    assert spoken is True


@pytest.mark.asyncio
async def test_streaming_current_view_uses_question_and_times_out() -> None:
    observed: list[str] = []
    published: list[VoiceOutput] = []

    class CurrentFrame:
        async def execute(self, _request):
            return SimpleNamespace(image=ImageReference(uri="frame.jpg"))

    class Vision:
        def stream(self, request):
            observed.append(request.query)

            async def chunks():
                await asyncio.sleep(1)
                yield ImageQueryChunk(text="late")

            return chunks()

    class Context:
        metadata = SimpleNamespace(message_id="view-timeout")

        async def publish(self, topic, message):
            assert topic is VOICE_CONTRIBUTION_TOPIC
            published.append(message)

    foreground = object.__new__(ForegroundAgent)
    foreground._images = SimpleNamespace(get_current_frame=CurrentFrame())
    foreground._vision = Vision()
    foreground._vlm_timeout_s = 0.01
    tool = foreground._current_view_tool(
        "participant-timeout",
        ctx=Context(),  # type: ignore[arg-type]
        timestamp_us=42,
    )

    result = await tool.handler(
        CurrentViewRequest(question="What number is on the display?")
    )

    assert result.available is False
    assert "vision timeout" in result.text
    assert observed == ["What number is on the display?"]
    assert published[0].text == result.text
    assert published[0].interrupt is True
    assert published[-1].text == ""
    assert published[-1].final is True


@pytest.mark.parametrize(
    ("chunks", "expected_text", "available"),
    [
        (["\n", "A kettle is visible."], "A kettle is visible.", True),
        (
            ["\n", "  "],
            "Unable to inspect the current frame because the vision model returned no "
            "description.",
            False,
        ),
    ],
)
@pytest.mark.asyncio
async def test_streaming_current_view_handles_leading_whitespace(
    chunks: list[str], expected_text: str, available: bool
) -> None:
    published: list[VoiceOutput] = []

    class CurrentFrame:
        async def execute(self, _request):
            return SimpleNamespace(image=ImageReference(uri="frame.jpg"))

    class Vision:
        def stream(self, _request):
            async def responses():
                for text in chunks:
                    yield ImageQueryChunk(text=text)

            return responses()

    class Context:
        metadata = SimpleNamespace(message_id="view-whitespace")

        async def publish(self, topic, message):
            assert topic is VOICE_CONTRIBUTION_TOPIC
            published.append(message)

    foreground = object.__new__(ForegroundAgent)
    foreground._images = SimpleNamespace(get_current_frame=CurrentFrame())
    foreground._vision = Vision()
    foreground._vlm_timeout_s = 1.0

    result = await foreground._stream_current_view(
        "What do you see?",
        "participant-whitespace",
        Context(),  # type: ignore[arg-type]
        timestamp_us=42,
    )

    assert result == ImageQueryResult(text=expected_text, available=available)
    assert [output.text for output in published] == [expected_text, ""]
    assert published[0].interrupt is True
    assert published[-1].final is True


@pytest.mark.asyncio
async def test_foreground_record_failure_does_not_suppress_speech() -> None:
    published: list[tuple[object, object]] = []

    class Context:
        metadata = SimpleNamespace(
            participant_id="participant-disk-full",
            message_id="query-1",
        )

        async def publish(self, topic, message):
            if topic is FOREGROUND_RECORD_TOPIC:
                raise OSError("disk full")
            published.append((topic, message))

    foreground = object.__new__(ForegroundAgent)

    async def answer(*_args, **_kwargs):
        return "Your water is ready.", ["clock__timer"], False

    foreground._answer = answer
    await foreground._run_turn(
        UserQuery(text="Is it ready?", timestamp_us=50),
        Context(),  # type: ignore[arg-type]
    )

    assert len(published) == 1
    topic, message = published[0]
    assert topic is VOICE_CONTRIBUTION_TOPIC
    assert isinstance(message, VoiceOutput)
    assert message.text == "Your water is ready."


@pytest.mark.asyncio
async def test_guidance_record_failure_does_not_suppress_notice() -> None:
    workflow = load_workflow(_SAMPLE / "yaml/workflow.yaml")
    guidance = GuidanceAgent(
        workflow=workflow,
        llm=SimpleNamespace(),  # type: ignore[arg-type]
        current_frame=SimpleNamespace(),  # type: ignore[arg-type]
        image_query=SimpleNamespace(),  # type: ignore[arg-type]
        rag=SimpleNamespace(),  # type: ignore[arg-type]
    )
    session = guidance.store.get("participant-guidance-disk-full")
    guidance.store.start(session)
    session.notices.append("The tea package is identified.")
    delivered: list[tuple[object, object]] = []

    class Runtime:
        running = True

        async def publish(self, topic, message, **_kwargs):
            delivered.append((topic, message))
            raise OSError("disk full")

    guidance._runtime = Runtime()  # type: ignore[assignment]

    await guidance._flush(session)

    assert delivered
    assert delivered[0][0] is GUIDANCE_NOTICE_TOPIC
    assert isinstance(delivered[0][1], GuidanceNotice)
    assert any(topic is GUIDANCE_RECORD_TOPIC for topic, _message in delivered)
    assert guidance.store.drain_notices(session) == ()
    assert guidance.store.drain_events(session) == ()


@pytest.mark.asyncio
async def test_images_release_after_every_producer_finishes_cleanup() -> None:
    released: list[str] = []
    images = object.__new__(ParticipantImageAgent)
    images._cleanup = {}
    images._leaving_generation = {}
    images.get_current_frame = SimpleNamespace(release=released.append)
    ctx = SimpleNamespace(
        metadata=SimpleNamespace(
            participant_id="participant-images",
            message_id="leave-1",
        )
    )
    await images.participant_left(
        VoiceParticipantLeft(),
        ctx,  # type: ignore[arg-type]
    )
    producers = (
        "guidance",
        "foreground",
        "change_watch",
        "transcript",
        "video_log",
    )

    for producer in producers[:-1]:
        await images.participant_cleanup_complete(
            ParticipantCleanupComplete(generation="leave-1", producer=producer),
            ctx,  # type: ignore[arg-type]
        )
    assert released == []

    await images.participant_cleanup_complete(
        ParticipantCleanupComplete(generation="leave-1", producer=producers[-1]),
        ctx,  # type: ignore[arg-type]
    )

    assert released == ["participant-images"]
    assert images._cleanup == {}


@pytest.mark.asyncio
async def test_cleanup_quorum_does_not_mix_rejoin_generations() -> None:
    participant_id = "participant-rejoin"
    producers = (
        "guidance",
        "foreground",
        "change_watch",
        "transcript",
        "video_log",
    )

    def context(message_id: str):
        return SimpleNamespace(
            metadata=SimpleNamespace(
                participant_id=participant_id,
                message_id=message_id,
            )
        )

    files = object.__new__(FileOutputAgent)
    files._sessions_lock = asyncio.Lock()
    files._sessions = {participant_id: object()}
    files._cleanup = {}
    files._leaving_generation = {}
    files._closed = set()
    files._state = AsyncMock(return_value=files._sessions[participant_id])

    async def close_participant(
        closing_participant_id: str,
        *,
        expected_generation: str | None = None,
    ) -> None:
        if (
            expected_generation is not None
            and files._leaving_generation.get(closing_participant_id)
            != expected_generation
        ):
            return
        files._sessions.pop(closing_participant_id, None)
        files._cleanup.pop(closing_participant_id, None)
        files._leaving_generation.pop(closing_participant_id, None)

    files._close_participant = close_participant
    await files.participant_joined(
        VoiceParticipantJoined(),
        context("join-1"),  # type: ignore[arg-type]
    )
    await files.participant_left(
        VoiceParticipantLeft(),
        context("leave-1"),  # type: ignore[arg-type]
    )
    for producer in producers[:-1]:
        await files.participant_cleanup_complete(
            ParticipantCleanupComplete(generation="leave-1", producer=producer),
            context("cleanup-old"),  # type: ignore[arg-type]
        )

    await files.participant_joined(
        VoiceParticipantJoined(),
        context("join-2"),  # type: ignore[arg-type]
    )
    await files.participant_cleanup_complete(
        ParticipantCleanupComplete(generation="leave-1", producer=producers[-1]),
        context("cleanup-stale"),  # type: ignore[arg-type]
    )
    await files.participant_left(
        VoiceParticipantLeft(),
        context("leave-2"),  # type: ignore[arg-type]
    )
    for producer in producers[:-1]:
        await files.participant_cleanup_complete(
            ParticipantCleanupComplete(generation="leave-2", producer=producer),
            context("cleanup-new"),  # type: ignore[arg-type]
        )

    assert participant_id in files._sessions
    await files.participant_cleanup_complete(
        ParticipantCleanupComplete(generation="leave-1", producer=producers[-1]),
        context("cleanup-stale-again"),  # type: ignore[arg-type]
    )
    assert participant_id in files._sessions

    await files.participant_cleanup_complete(
        ParticipantCleanupComplete(generation="leave-2", producer=producers[-1]),
        context("cleanup-final"),  # type: ignore[arg-type]
    )
    assert participant_id not in files._sessions

    released: list[str] = []
    images = object.__new__(ParticipantImageAgent)
    images._cleanup = {}
    images._leaving_generation = {}
    images.get_current_frame = SimpleNamespace(release=released.append)
    await images.participant_joined(
        VoiceParticipantJoined(),
        context("join-images"),  # type: ignore[arg-type]
    )
    await images.participant_cleanup_complete(
        ParticipantCleanupComplete(generation="leave-old", producer="guidance"),
        context("cleanup-images-old"),  # type: ignore[arg-type]
    )
    await images.participant_left(
        VoiceParticipantLeft(),
        context("leave-images-new"),  # type: ignore[arg-type]
    )
    for producer in producers:
        generation = "leave-old" if producer == "video_log" else "leave-images-new"
        await images.participant_cleanup_complete(
            ParticipantCleanupComplete(generation=generation, producer=producer),
            context("cleanup-images"),  # type: ignore[arg-type]
        )
    assert released == []

    await images.participant_cleanup_complete(
        ParticipantCleanupComplete(
            generation="leave-images-new",
            producer="video_log",
        ),
        context("cleanup-images-final"),  # type: ignore[arg-type]
    )
    assert released == [participant_id]


@pytest.mark.asyncio
async def test_rejoin_wins_race_with_ready_cleanup() -> None:
    participant_id = "participant-rejoin-race"
    producers = (
        "guidance",
        "foreground",
        "change_watch",
        "transcript",
        "video_log",
    )

    def context(message_id: str):
        return SimpleNamespace(
            metadata=SimpleNamespace(
                participant_id=participant_id,
                message_id=message_id,
            )
        )

    files = object.__new__(FileOutputAgent)
    files._sessions_lock = asyncio.Lock()
    original_state = SimpleNamespace(active=True)
    files._sessions = {participant_id: original_state}
    files._cleanup = {}
    files._leaving_generation = {}
    files._closed = set()
    files._state = AsyncMock(return_value=original_state)
    await files.participant_joined(
        VoiceParticipantJoined(),
        context("join-1"),  # type: ignore[arg-type]
    )
    await files.participant_left(
        VoiceParticipantLeft(),
        context("leave-1"),  # type: ignore[arg-type]
    )
    for producer in producers[:-1]:
        await files.participant_cleanup_complete(
            ParticipantCleanupComplete(generation="leave-1", producer=producer),
            context("cleanup"),  # type: ignore[arg-type]
        )

    close_started = asyncio.Event()
    continue_close = asyncio.Event()
    close_participant = files._close_participant

    async def delayed_close(
        closing_participant_id: str,
        *,
        expected_generation: str | None = None,
    ) -> None:
        close_started.set()
        await continue_close.wait()
        await close_participant(
            closing_participant_id,
            expected_generation=expected_generation,
        )

    files._close_participant = delayed_close  # type: ignore[method-assign]
    final_cleanup = asyncio.create_task(
        files.participant_cleanup_complete(
            ParticipantCleanupComplete(
                generation="leave-1",
                producer=producers[-1],
            ),
            context("cleanup-final"),  # type: ignore[arg-type]
        )
    )
    await asyncio.wait_for(close_started.wait(), timeout=1.0)
    await files.participant_joined(
        VoiceParticipantJoined(),
        context("join-2"),  # type: ignore[arg-type]
    )
    continue_close.set()
    await asyncio.wait_for(final_cleanup, timeout=1.0)

    assert files._sessions[participant_id] is original_state
    assert participant_id not in files._closed
    assert participant_id not in files._leaving_generation
    assert original_state.active


@pytest.mark.parametrize("case", ["initial", "complete_when", "state_on_skip"])
def test_workflow_rejects_mistyped_declared_state_values(
    tmp_path: Path,
    case: str,
) -> None:
    raw = yaml.safe_load((_SAMPLE / "yaml/workflow.yaml").read_text())
    if case == "initial":
        raw["state"]["water_filled"]["initial"] = "false"
    elif case == "complete_when":
        raw["steps"][1]["complete_when"]["water_filled"] = "true"
    else:
        raw["steps"][0]["state_on_skip"]["target_temperature_c"] = "93"
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="must be"):
        load_workflow(path)


@pytest.mark.parametrize(("value", "expected"), [("false", False), ("true", True)])
def test_workflow_parses_quoted_complete_on_skip_boolean(
    tmp_path: Path,
    value: str,
    expected: bool,
) -> None:
    raw = yaml.safe_load((_SAMPLE / "yaml/workflow.yaml").read_text())
    raw["steps"][0]["complete_on_skip"] = value
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(raw))

    workflow = load_workflow(path)

    assert workflow.steps[raw["steps"][0]["id"]].complete_on_skip is expected


def test_workflow_rejects_invalid_complete_on_skip_boolean(tmp_path: Path) -> None:
    raw = yaml.safe_load((_SAMPLE / "yaml/workflow.yaml").read_text())
    raw["steps"][0]["complete_on_skip"] = "sometimes"
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="complete_on_skip must be a boolean"):
        load_workflow(path)


@pytest.mark.asyncio
async def test_file_output_writes_session_bounded_jsonl(tmp_path: Path) -> None:
    files = FileOutputAgent(tmp_path, history_size=4)
    runtime = AgentRuntime()
    runtime.register("files", files)
    leave_generation: str | None = None

    class LeaveObserver(Agent):
        @subscribe(PARTICIPANT_LEFT_TOPIC)
        async def participant_left(
            self,
            _event: VoiceParticipantLeft,
            ctx: RuntimeContext,
        ) -> None:
            nonlocal leave_generation
            leave_generation = ctx.metadata.message_id

    runtime.register("leave-observer", LeaveObserver())

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
        assert leave_generation is not None
        for producer in (
            "foreground",
            "change_watch",
            "transcript",
            "video_log",
        ):
            await runtime.publish(
                PARTICIPANT_CLEANUP_COMPLETE_TOPIC,
                ParticipantCleanupComplete(
                    generation=leave_generation,
                    producer=producer,
                ),
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
            ParticipantCleanupComplete(
                generation=leave_generation,
                producer="guidance",
            ),
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


@pytest.mark.asyncio
async def test_relay_event_log_buffers_and_excludes_stream_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = None

    def register(_name, registered_callback):
        nonlocal callback
        callback = registered_callback

    async def flush_async() -> None:
        return None

    monkeypatch.setattr(nemo_relay.subscribers, "register", register)
    monkeypatch.setattr(nemo_relay.subscribers, "flush_async", flush_async)
    monkeypatch.setattr(nemo_relay.subscribers, "deregister", lambda _name: None)

    class Event:
        def __init__(self, name: str) -> None:
            self.kind = "mark"
            self.name = name

        def to_json(self) -> str:
            return json.dumps({"name": self.name})

    async with _relay_event_log(tmp_path):
        assert callback is not None
        callback(Event("llm.chunk"))
        callback(Event("turn.summary"))

    events = [
        yaml.safe_load(line)
        for line in (tmp_path / "relay-events.jsonl").read_text().splitlines()
    ]
    names = [event["name"] for event in events]
    assert "llm.chunk" not in names
    assert "turn.summary" in names


def test_default_prompts_come_from_packaged_files(tmp_path: Path) -> None:
    config = load_config(_SAMPLE / "yaml/tea_making_worker.yaml")
    prompt_dir = _WORKER / "tea_making_worker/prompts"
    assert (
        config.foreground_prompt
        == (prompt_dir / "foreground_prompt.txt").read_text().strip()
    )
    assert (
        config.video_delta_prompt
        == (prompt_dir / "video_delta_prompt.txt").read_text().strip()
    )
    assert "change_watch__commit exactly once" in config.change_watch_event_prompt
    assert "tool-only visual-change classifier" in config.change_watch_event_prompt
    assert "return no assistant text" in config.change_watch_event_prompt
    assert "Continued presence" in config.change_watch_event_prompt
    assert "video_log__commit exactly once" in config.video_delta_prompt
    assert "tool-only visual-delta classifier" in config.video_delta_prompt
    assert "return no assistant text" in config.video_delta_prompt
    assert "transcript__commit_summary exactly once" in config.transcript_summary_prompt

    override = tmp_path / "worker.yaml"
    override.write_text("foreground_prompt: Explicit override\n")
    assert load_config(override).foreground_prompt == "Explicit override"

    matching_prompt = tmp_path / "matching-prompt.txt"
    matching_prompt.write_text((prompt_dir / "foreground_prompt.txt").read_text())
    override.write_text("foreground_prompt_file: matching-prompt.txt\n")
    matching_config = load_config(override)
    assert matching_config.foreground_prompt == config.foreground_prompt


def test_foreground_prompt_has_route_eval_cases() -> None:
    cases = yaml.safe_load((_SAMPLE / "eval/cases.yaml").read_text())
    common_prompt = (
        _WORKER / "tea_making_worker/prompts/foreground_prompt.txt"
    ).read_text()
    idle_prompt = (
        _WORKER / "tea_making_worker/prompts/foreground_idle.txt"
    ).read_text()
    active_prompt = (
        _WORKER / "tea_making_worker/prompts/foreground_active.txt"
    ).read_text()
    legacy_refusal = "I can only help with the active tea guide right now."
    idle_model_prompt = f"{common_prompt}\n\n{idle_prompt}".lower()
    active_model_prompt = f"{common_prompt}\n\n{active_prompt}".lower()
    assert legacy_refusal not in common_prompt
    assert legacy_refusal not in idle_prompt
    for worked_example_term in ("holding", "color", "shirt", "clothing"):
        assert worked_example_term not in idle_model_prompt
        assert worked_example_term not in active_model_prompt
    assert "general-purpose assistant" in idle_prompt
    assert "Answer anything relevant to tea making" in active_prompt
    assert "output only a brief refusal" in active_prompt

    root_cases = [case for case in cases if case.get("route", "root") == "root"]
    assert {case["expected_tool"] for case in root_cases} == {
        None,
        "application_context__query",
        "change_watch__start",
        "change_watch__stop",
        "current_view",
        "rag_lookup",
        "transcript__start",
        "video_log__start",
        "workflow__start",
    }
    active_cases = [case for case in cases if case.get("route") == "active"]
    assert {case["expected_tool"] for case in active_cases} == {
        None,
        "change_watch__start",
        "clock__timer",
        "current_view",
        "rag_lookup",
        "workflow__advance",
        "workflow__reset",
        "workflow__restart",
    }
    unrelated_cases = [
        case for case in active_cases if case["name"].startswith("active-rejects-")
    ]
    assert len(unrelated_cases) >= 4
    assert all(case["expected_tool"] is None for case in unrelated_cases)
    assert all("expected_response" not in case for case in unrelated_cases)

    positive_active_names = {
        case["name"]
        for case in active_cases
        if case.get("forbidden_response") == legacy_refusal
        or case["expected_tool"] in {
            "change_watch__start",
            "current_view",
        }
    }
    assert {
        "active-answers-current-step-question",
        "active-routes-workflow-status",
        "active-routes-step-visual-question",
        "active-starts-background-watch",
    } <= positive_active_names

    idle_unrelated = next(
        case
        for case in root_cases
        if case["name"] == "idle-answers-unrelated-general-knowledge"
    )
    assert idle_unrelated["forbidden_response"] == legacy_refusal
    idle_visual = next(
        case
        for case in root_cases
        if case["name"] == "idle-routes-visible-shirt-color"
    )
    assert idle_visual["expected_tool"] == "current_view"
    assert idle_visual["forbidden_response"] == legacy_refusal
    active_visual = next(
        case
        for case in active_cases
        if case["name"] == "active-rejects-unrelated-visual-question"
    )
    assert active_visual["expected_tool"] is None
    assert "expected_response_pattern" in active_visual
    assert "expected_response" not in active_visual


def _foreground_for_route_test(prompt: str) -> ForegroundAgent:
    images = SimpleNamespace(images=ImageRegistry())
    foreground = ForegroundAgent(
        llm=SimpleNamespace(),  # type: ignore[arg-type]
        images=images,  # type: ignore[arg-type]
        vlm=SimpleNamespace(),  # type: ignore[arg-type]
        rag=SimpleNamespace(),  # type: ignore[arg-type]
        guidance=SimpleNamespace(
            active_context=lambda participant_id: (
                None if participant_id == "idle" else '{"step":"fill_water"}'
            ),
            active_tools=lambda _participant_id: ToolSet(()),
        ),  # type: ignore[arg-type]
        background_context=SimpleNamespace(),  # type: ignore[arg-type]
        change_watch=SimpleNamespace(),  # type: ignore[arg-type]
        transcript=SimpleNamespace(),  # type: ignore[arg-type]
        video_log=SimpleNamespace(),  # type: ignore[arg-type]
        prompt=prompt,
    )
    foreground._root_tools = lambda *_args, **_kwargs: ToolSet(())
    foreground._background_tools = lambda *_args, **_kwargs: ToolSet(())
    return foreground


def test_foreground_route_appends_policy_through_constructor() -> None:
    config = load_config(_SAMPLE / "yaml/tea_making_worker.yaml")
    foreground = _foreground_for_route_test(config.foreground_prompt)

    idle_prompt, _, idle_route = foreground._prepare_route(
        "idle", ctx=None, timestamp_us=None
    )
    active_prompt, _, active_route = foreground._prepare_route(
        "active", ctx=None, timestamp_us=None
    )
    assert idle_route == "root"
    assert "general-purpose assistant" in idle_prompt
    assert active_route == "tea"
    assert "general-purpose assistant" not in active_prompt
    assert "Answer anything relevant to tea making" in active_prompt
    assert "output only a brief refusal" in active_prompt


def test_foreground_route_appends_policy_to_prompt_override(tmp_path: Path) -> None:
    override = tmp_path / "worker.yaml"
    override.write_text("foreground_prompt: Explicit override\n")
    config = load_config(override)
    foreground = _foreground_for_route_test(config.foreground_prompt)

    idle_prompt, _, idle_route = foreground._prepare_route(
        "idle", ctx=None, timestamp_us=None
    )
    active_prompt, _, active_route = foreground._prepare_route(
        "active", ctx=None, timestamp_us=None
    )
    assert idle_route == "root"
    assert idle_prompt.startswith("Explicit override\n\n")
    assert "general-purpose assistant" in idle_prompt
    assert active_route == "tea"
    assert active_prompt.startswith("Explicit override\n\n")
    assert "Answer anything relevant to tea making" in active_prompt
    assert "output only a brief refusal" in active_prompt
    assert active_prompt.endswith(
        'Active tea guide:\n{"step":"fill_water"}'
    )
