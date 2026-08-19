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
from unittest.mock import AsyncMock

import pytest
import yaml
from xr_ai_models import ChatResponse, ToolCall
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, subscribe
from xr_ai_tools.image import ImageReference, ImageRegistry
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

from tea_making_worker.app import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    _CLIENT_TEXT_TOPIC,
    _ParticipantVoiceAggregationAgent,
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
from tea_making_worker.video_log import VideoLogAgent  # noqa: E402  # pyright: ignore[reportMissingImports]
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


def test_launcher_defaults_to_wake_word_and_allows_always_on() -> None:
    sample_main = _load_main()

    default = sample_main._parse_args(["--tts-mode", "piper"])
    explicit = sample_main._parse_args(
        ["--voice-mode", "always-on", "--tts-mode", "piper"]
    )

    assert default is not None and default.voice_mode == "wake-word"
    assert default is not None and default.expose_web_events is False
    assert explicit is not None and explicit.voice_mode == "always-on"
    exposed = sample_main._parse_args(
        ["--tts-mode", "piper", "--expose-web-events"]
    )
    assert exposed is not None and exposed.expose_web_events is True
    assert _CLIENT_TEXT_TOPIC == "agent.response"


def test_web_event_config_defaults_and_validation(tmp_path: Path) -> None:
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
        "wake-word",
        "piper",
        expose_web_events=True,
    )
    assert load_config(exposed_config).web_events_host == "0.0.0.0"


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
            CHANGE_WATCH_RECORD_TOPIC,
            ChangeWatchRecord(timestamp_us=5, record_type="started"),
        ),
        (
            TRANSCRIPT_RECORD_TOPIC,
            TranscriptRecord(
                timestamp_us=6,
                record_type="utterance",
                text="Start steeping.",
            ),
        ),
        (
            VIDEO_LOG_RECORD_TOPIC,
            VideoLogRecord(timestamp_us=7, record_type="observation", caption="A cup."),
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
        "background.facts",
        "background.change-watch",
        "background.transcript",
        "background.video-log",
        "participant.lifecycle",
        "participant.lifecycle",
    ]
    assert {participant for _event, participant in captured} == {participant_id}
    assert captured[-2][0].payload == {"event": "joined"}
    assert captured[-1][0].payload == {"event": "left"}


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
    images.get_current_frame = SimpleNamespace(release=released.append)
    ctx = SimpleNamespace(metadata=SimpleNamespace(participant_id="participant-images"))
    producers = (
        "guidance",
        "foreground",
        "change_watch",
        "transcript",
        "video_log",
    )

    for producer in producers[:-1]:
        await images.participant_cleanup_complete(
            ParticipantCleanupComplete(producer=producer),
            ctx,  # type: ignore[arg-type]
        )
    assert released == []

    await images.participant_cleanup_complete(
        ParticipantCleanupComplete(producer=producers[-1]),
        ctx,  # type: ignore[arg-type]
    )

    assert released == ["participant-images"]
    assert images._cleanup == {}


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
    assert (
        config.foreground_prompt
        == (prompt_dir / "foreground_prompt.txt").read_text().strip()
    )
    assert (
        config.video_delta_prompt
        == (prompt_dir / "video_delta_prompt.txt").read_text().strip()
    )

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
