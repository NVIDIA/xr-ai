# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the native tea-making sample."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import nemo_relay
import pytest
import yaml
from xr_ai_hub import FrameUnavailable
from xr_ai_interaction import ParticipantJoined, ParticipantLeft, ResponseOutput
from xr_ai_models import ChatResponse, DecisionAnswer, DecisionResponse, ToolCall
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, subscribe
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.current_frame import CurrentFrameRequest, ImageFrame
from xr_ai_tools.image import ImageReference, ImageRegistry
from xr_ai_tools.vision import ImageQueryChunk, ImageQueryResult
from xr_ai_voice import VOICE_OUTPUT_TOPIC
from xr_ai_web_events import WEB_EVENT_TOPIC, WebEvent, WebEventsAgent

_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / "agent-samples" / "tea-making-sample"
_WORKER = _SAMPLE / "worker"
sys.path.insert(0, str(_WORKER))

import tea_making_worker.agent as foreground_module  # noqa: E402  # pyright: ignore[reportMissingImports]
from tea_making_worker import SOPAgent  # noqa: E402  # pyright: ignore[reportMissingImports]
from tea_making_worker.app import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    _CLIENT_TEXT_TOPIC,
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
from tea_making_worker.images import ParticipantImageAgent  # noqa: E402  # pyright: ignore[reportMissingImports]
from tea_making_worker.sop import load_workflow  # noqa: E402  # pyright: ignore[reportMissingImports]
from tea_making_worker.sop._engine import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    GuidanceAgent,
    _TriggerResult,
)
from tea_making_worker.sop._state import WorkflowStore  # noqa: E402  # pyright: ignore[reportMissingImports]
from tea_making_worker.sop._tools import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    CurrentViewRequest,
    TemperatureThresholdRequest,
    participant_current_view_tool,
    temperature_threshold_tool,
)
from tea_making_worker.transcript import TranscriptAgent  # noqa: E402  # pyright: ignore[reportMissingImports]
from tea_making_worker.video_log import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    VideoLogAgent,
    _is_no_change,
)
from tea_making_worker.web_events import TeaWebEventsAgent  # noqa: E402  # pyright: ignore[reportMissingImports]


class _SOPFixture(SOPAgent):
    """Test fixture accepting a separately constructed execution engine."""

    def __init__(self, *, guidance, **kwargs):
        super().__init__(workflow=guidance.workflow, image_query=guidance._image_query, **kwargs)
        self._guidance = guidance
        guidance._record_event = self._remember_event
        _attach_router(self)


def _attach_router(sop):
    from tea_making_worker.conversation._applications import _destinations
    from xr_ai_routing import AgentRouter, AgentTurn

    runtime = AgentRuntime()
    records = []
    control = []
    publish = runtime.publish

    async def record(topic, message, **kwargs):
        if topic is FOREGROUND_RECORD_TOPIC:
            records.append(message)
        return await publish(topic, message, **kwargs)

    runtime.publish = record

    async def emit(_turn, _route, text):
        control.append(text)

    router = AgentRouter(_destinations(sop), sop._decisions, output=emit)
    sop._router = router
    original_stop = sop.stop
    opened = False
    sequence = 0

    async def ask(query, participant_id, ctx=None, *, timestamp_us=None):
        nonlocal opened, sequence
        if sop._runtime is None:
            sop.bind_runtime(runtime)
        if sop._runtime is runtime and not runtime.running:
            await runtime.__aenter__()
            opened = True
        sequence += 1
        before = len(records)
        result = await router.handle(AgentTurn(participant_id, f"test:{sequence}", query, timestamp_us=timestamp_us))
        rows = records[before:]
        return result.text or (control[-1] if control else ""), (rows[-1].tools if rows else []), False

    async def stop():
        await router.stop()
        await original_stop()
        if opened:
            await runtime.__aexit__(None, None, None)

    sop._answer = ask
    sop.stop = stop
    return sop


async def _stream_view(owner, query, participant_id, ctx, *, timestamp_us):
    from tea_making_worker.conversation._applications import _QuestionDestination
    from xr_ai_interaction import ResponseOutputs
    from xr_ai_routing import AgentTurn

    outputs = ResponseOutputs(ctx, VOICE_OUTPUT_TOPIC)
    destination = _QuestionDestination(owner, "observation")
    async with outputs.reply(participant_id, request_id=ctx.metadata.message_id, timestamp_us=timestamp_us) as reply:
        text, _calls = await destination._camera(
            AgentTurn(participant_id, ctx.metadata.message_id, query, timestamp_us=timestamp_us), reply,
        )
    return ImageQueryResult(text=text, available=not text.startswith(("camera failed", "Unable to inspect")))

def _manual_workflow(path):
    """Manual policy for tests that exercise explicit next and stable completed steps."""
    return replace(load_workflow(path), advancement="manual")


def _load_main():
    path = _SAMPLE / "main.py"
    spec = importlib.util.spec_from_file_location("tea_making_sample_main", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_omni_supplies_both_language_and_vision() -> None:
    models = json.loads((_SAMPLE / "yaml/models.local.json").read_text())["models"]

    assert models["llm"]["endpoint"]["base_url"] == "http://localhost:8108"
    assert models["vlm"]["endpoint"]["base_url"] == "http://localhost:8108"
    assert models["vlm"]["adapter"]["capabilities"]["vision"] is True
    assert models["vlm"]["adapter"]["reasoning_field"] == "reasoning_content"
    assert all("deployment" not in model for model in models.values())
    assert "cosmos" not in json.dumps(models).lower()


def test_materialized_config_stays_inside_runtime_dir(tmp_path: Path) -> None:
    sample_main = _load_main()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()

    assert sample_main._materialize_worker_config(runtime_dir) == runtime_dir / "tea_making_worker.yaml"

    escaped = tmp_path / "escaped"
    escaped.mkdir()
    (escaped / "tea_making_worker.yaml").symlink_to(tmp_path / "elsewhere.yaml")
    with pytest.raises(ValueError, match="is outside"):
        sample_main._materialize_worker_config(escaped)


def test_launcher_declares_only_sample_owned_processes(tmp_path: Path) -> None:
    sample_main = _load_main()
    worker_config = sample_main._materialize_worker_config(
        tmp_path,
    )
    processes = sample_main._build_processes(worker_config)
    names = [process.name for process in processes]

    assert names == ["hub", "rag", "worker"]


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
    assert config.web_events_port == 8093
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
    assert load_config(exposed_config).voice_gate_yaml == _SAMPLE / "yaml/voice_gate.yaml"

    custom_gate = tmp_path / "custom-voice.yaml"
    custom_gate.write_text("mode: hybrid\nmagic_phrases: [hey agent]\n", encoding="utf-8")
    custom_source = tmp_path / "custom-worker.yaml"
    custom_source.write_text(
        (_SAMPLE / "yaml/tea_making_worker.yaml").read_text().replace("voice_gate.yaml", "custom-voice.yaml"),
        encoding="utf-8",
    )
    monkeypatch.setattr(sample_main, "_WORKER_CONFIG", custom_source)
    custom_config = sample_main._materialize_worker_config(tmp_path)
    assert load_config(custom_config).voice_gate_yaml == custom_gate


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
        (PARTICIPANT_JOINED_TOPIC, ParticipantJoined()),
        (PARTICIPANT_LEFT_TOPIC, ParticipantLeft()),
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
@pytest.mark.parametrize("choice", ["rejected", "unknown"])
async def test_background_no_change_skips_generation(choice: str) -> None:
    config = load_config(_SAMPLE / "yaml/tea_making_worker.yaml")
    for cls, method, decision in [
        (ChangeWatchAgent, "_decide", config.change_watch_decision),
        (VideoLogAgent, "_generate_delta", config.video_delta_decision),
    ]:
        agent = object.__new__(cls)
        agent._decisions = _decisions(choice)
        agent._decision = decision
        agent._llm = SimpleNamespace(chat=AsyncMock(side_effect=AssertionError("no generation")))
        agent._history_size = 5
        result = await getattr(agent, method)(
            SimpleNamespace(instruction="changes", captions=["A cup"]), "The same cup"
        )
        assert getattr(result, "important", False) is False
        agent._llm.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_background_positive_change_generates_only_summary() -> None:
    config = load_config(_SAMPLE / "yaml/tea_making_worker.yaml")
    for cls, method, decision in [
        (ChangeWatchAgent, "_decide", config.change_watch_decision),
        (VideoLogAgent, "_generate_delta", config.video_delta_decision),
    ]:
        agent = object.__new__(cls)
        agent._decisions = _decisions("accepted")
        agent._decision = decision
        agent._llm = SimpleNamespace(
            chat=AsyncMock(return_value=ChatResponse("A person entered.", None, None, "stop", {}))
        )
        agent._history_size = 5
        agent._event_prompt = agent._delta_prompt = "Describe the proven transition."
        result = await getattr(agent, method)(
            SimpleNamespace(instruction="changes", captions=["Empty room"]), "A person"
        )
        assert getattr(result, "summary", getattr(result, "delta", "")) == "A person entered."
        agent._llm.chat.assert_awaited_once()
        assert "tools" not in agent._llm.chat.call_args.kwargs


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
    assert not _is_no_change("No meaningful visual change, but a person entered the room.")


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
            execute=AsyncMock(return_value=SimpleNamespace(image=ImageReference(uri="frame.jpg")))
        )
    )
    agent._query_image = SimpleNamespace(
        execute=AsyncMock(return_value=ImageQueryResult(text="A hand is raised.", available=True))
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
            execute=AsyncMock(return_value=SimpleNamespace(image=ImageReference(uri="frame.jpg")))
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
    agent._generate_delta = AsyncMock(side_effect=ValueError("invalid classifier output"))
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
            execute=AsyncMock(return_value=SimpleNamespace(image=ImageReference(uri="frame.jpg")))
        )
    )
    agent._query_image = SimpleNamespace(
        execute=AsyncMock(return_value=ImageQueryResult(text="The room is empty.", available=True))
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
                ParticipantJoined(),
                participant_id="participant-live",
            )
        assert viewer.running
    assert not viewer.running


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
    workflow = _manual_workflow(_SAMPLE / "yaml/workflow.yaml")
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
    workflow = _manual_workflow(_SAMPLE / "yaml/workflow.yaml")
    store = WorkflowStore(workflow)
    session = store.get("participant-invalid")
    store.start(session)
    store.observe(session, "accepted")

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
    workflow = _manual_workflow(_SAMPLE / "yaml/workflow.yaml")
    store = WorkflowStore(workflow)
    session = store.get("participant-verified")
    store.start(session)
    store.observe(session, "accepted")
    accepted = store.commit(
        session,
        {
            "tea_name": "Twinings Earl Grey",
            "target_temperature_c": 100,
            "steep_duration_s": 180,
            "guidance_source": "retrieved",
            "temperature_source": "document",
            "duration_source": "document",
            "tea_ready": True,
        },
        "",
    )
    assert accepted.complete is True

    message = store.advance(session, skip=True)

    assert session.step_id == "fill_water"
    assert session.state["tea_name"] == "Twinings Earl Grey"
    assert session.state["target_temperature_c"] == 100
    assert session.state["guidance_source"] == "retrieved"
    assert "generic" not in message.lower()


def test_skipping_steeping_detection_also_skips_the_timer() -> None:
    workflow = _manual_workflow(_SAMPLE / "yaml/workflow.yaml")
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


def test_workflow_enforces_consecutive_model_judgments() -> None:
    workflow = _manual_workflow(_SAMPLE / "yaml/workflow.yaml")
    store = WorkflowStore(workflow)
    session = store.get("participant-2")
    store.start(session)
    store.advance(session, skip=True)
    store.advance(session, skip=True)

    store.observe(session, "accepted")
    rejected = store.commit(session, {"temperature_ready": True}, "")
    assert not rejected.accepted
    assert session.state["temperature_ready"] is False

    store.observe(session, "accepted")
    accepted = store.commit(session, {"temperature_ready": True}, "")
    assert accepted.accepted
    assert accepted.complete
    assert session.state["temperature_ready"] is True
    assert session.step_id == "heat_water"


@pytest.mark.asyncio
async def test_threshold_helper_is_independent_but_heating_step_verifies_target() -> None:
    workflow = _manual_workflow(_SAMPLE / "yaml/workflow.yaml")
    tool = temperature_threshold_tool()

    at_threshold = await tool.execute(
        TemperatureThresholdRequest(
            reading=50,
            unit="celsius",
            threshold_c=50,
        )
    )
    above_threshold = await tool.execute(
        TemperatureThresholdRequest(
            reading=55,
            unit="celsius",
            threshold_c=50,
        )
    )

    assert at_threshold.above is False
    assert above_threshold.above is True
    assert workflow.step("heat_water").observation.evaluation.function == "temperature__verify"


def _commit_response(
    updates: dict[str, bool | int | float | str],
    *,
    evidence: str = "rejected",
) -> ChatResponse:
    return ChatResponse(
        "",
        None,
        [
            ToolCall(
                id="commit-1",
                name="workflow__commit",
                arguments=json.dumps(
                    {
                        "updates": updates,
                        "message": "",
                        "evidence": evidence,
                    }
                ),
            )
        ],
        "tool_calls",
        {},
    )


def _guidance_for_observation(decisions: object | None = None) -> GuidanceAgent:
    return GuidanceAgent(
        workflow=_manual_workflow(_SAMPLE / "yaml/workflow.yaml"),
        decisions=decisions or SimpleNamespace(),  # type: ignore[arg-type]
        llm=SimpleNamespace(),  # type: ignore[arg-type]
        current_frame=SimpleNamespace(),  # type: ignore[arg-type]
        image_query=SimpleNamespace(),  # type: ignore[arg-type]
        rag=SimpleNamespace(),  # type: ignore[arg-type]
    )


def _decisions(*choices: str):
    pending = iter(choices)

    async def decide(_state, questions, **_kwargs):
        choice = next(pending)
        return DecisionResponse(
            {
                key: DecisionAnswer(
                    "command" if key == "command" else choice,
                    1.0,
                    {label: float(label == choice) for label in question.criteria},
                )
                for key, question in questions.items()
            },
            "fake",
            {},
            {},
        )

    return SimpleNamespace(decide=AsyncMock(side_effect=decide))


async def _observe_caption(guidance, participant_id, caption):
    session = guidance.store.get(participant_id)
    await guidance._observe(
        session, guidance.workflow.step(session.step_id), caption, dict(session.state), session.revision
    )


@pytest.mark.asyncio
async def test_water_choice_has_distinct_unknown_and_negative_outcomes() -> None:
    decisions = _decisions("accepted", "unknown", "rejected", "accepted", "accepted")
    guidance = _guidance_for_observation(decisions)
    _activate_test_step(guidance, "fill_water")
    session = guidance.store.get("active")
    for expected_hits in [1, 1, 0, 1, 2]:
        await _observe_caption(guidance, "active", "Fresh visual description")
        assert session.evidence_hits == expected_hits
    assert session.state["water_filled"] is True
    assert decisions.decide.await_count == 5


@pytest.mark.asyncio
async def test_heating_uses_native_temperature_comparison() -> None:
    guidance = _guidance_for_observation()
    _activate_test_step(guidance, "heat_water")
    session = guidance.store.get("active")
    session.state["target_temperature_c"] = 80
    for reading, unit, hits in [(79, "celsius", 0), (80, "celsius", 1), (176, "fahrenheit", 2)]:
        await _observe_caption(guidance, "active", json.dumps({"reading": reading, "unit": unit}))
        assert session.evidence_hits == hits
    assert session.state["temperature_ready"] is True


@pytest.mark.asyncio
async def test_heating_unknown_preserves_and_negative_resets_evidence() -> None:
    guidance = _guidance_for_observation()
    _activate_test_step(guidance, "heat_water")
    session = guidance.store.get("active")
    session.state["target_temperature_c"] = 80
    for reading, unit, hits in [(80, "celsius", 1), (58, None, 1), (79, "celsius", 0), (81, "celsius", 1)]:
        await _observe_caption(guidance, "active", json.dumps({"reading": reading, "unit": unit}))
        assert session.evidence_hits == hits
    assert session.state["temperature_ready"] is False


@pytest.mark.asyncio
async def test_malformed_extraction_cannot_mutate_state_or_evidence() -> None:
    guidance = _guidance_for_observation()
    _activate_test_step(guidance, "heat_water")
    session = guidance.store.get("active")
    await _observe_caption(guidance, "active", '{"reading":93,"unit":"celsius"}')
    for invalid in [
        '{"temperature_ready":true}',
        '{"reading":"55","unit":"celsius"}',
        '{"reading":NaN,"unit":"celsius"}',
    ]:
        with pytest.raises(ValueError):
            await _observe_caption(guidance, "active", invalid)
    assert session.evidence_hits == 1
    assert session.state["temperature_ready"] is False


@pytest.mark.asyncio
async def test_steeping_clock_is_bound_by_code_only_after_confirmation() -> None:
    guidance = _guidance_for_observation(_decisions("accepted", "accepted"))
    _activate_test_step(guidance, "start_steeping")
    session = guidance.store.get("active")
    await _observe_caption(guidance, "active", "Visible water and tea-water contact")
    assert "steeping_started_at_us" not in session.state
    await _observe_caption(guidance, "active", "Visible water and tea-water contact")
    assert session.state["steeping_started"] is True
    assert session.state["steeping_started_at_us"] > 0


def test_guidance_exposes_one_foreground_stack_at_a_time() -> None:
    workflow = _manual_workflow(_SAMPLE / "yaml/workflow.yaml")
    guidance = GuidanceAgent(
        decisions=SimpleNamespace(),  # type: ignore[arg-type]
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

    guidance.store.advance(session, skip=True)
    fill_tools = guidance.active_tools("participant-3")
    assert fill_tools is not None
    fill_names = {name for name, _tool in fill_tools.items()}
    assert "current_view" in fill_names
    assert "rag_lookup" not in fill_names


def test_active_context_remains_sparse_for_legacy_callers() -> None:
    workflow = _manual_workflow(_SAMPLE / "yaml/workflow.yaml")
    guidance = GuidanceAgent(
        decisions=SimpleNamespace(),  # type: ignore[arg-type]
        workflow=workflow,
        llm=SimpleNamespace(),  # type: ignore[arg-type]
        current_frame=SimpleNamespace(),  # type: ignore[arg-type]
        image_query=SimpleNamespace(),  # type: ignore[arg-type]
        rag=SimpleNamespace(),  # type: ignore[arg-type]
    )
    session = guidance.store.get("participant-guide-context")
    guidance.store.start(session)
    guidance.store.advance(session, skip=True)

    context = json.loads(guidance.active_context("participant-guide-context") or "{}")

    assert context["step"] == {"id": "fill_water", "title": "Fill the heating vessel"}
    assert context["workflow"] == "This workflow guides tea making."
    assert context["state"] == {
        "tea_name": "generic tea",
        "target_temperature_c": 93,
        "steep_duration_s": 180,
        "water_filled": False,
    }
    assert "public_guide" not in context


def test_workflow_control_authorization_is_declarative() -> None:
    workflow = _manual_workflow(_SAMPLE / "yaml/workflow.yaml")
    assert set(workflow.foreground.idle_question.criteria) == {"start", "status", "other"}
    assert {"advance", "skip", "reset", "restart"} <= workflow.foreground.active_question.criteria.keys()
    assert set(workflow.foreground.authorization.criteria) == {"command", "not_command", "unknown"}


def test_foreground_current_view_uses_first_person_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []
    image_query = foreground_module.StreamingImageQueryTool

    def capture_image_query(**kwargs):
        captured.append(kwargs["system_prompt"])
        return image_query(**kwargs)

    monkeypatch.setattr(
        foreground_module,
        "StreamingImageQueryTool",
        capture_image_query,
    )

    _foreground_for_route_test("shared", step_id="identify")

    assert captured == [load_config(None).current_view_prompt]


def test_conversation_tools_do_not_offer_workflow_mutation() -> None:
    for step in _manual_workflow(_SAMPLE / "yaml/workflow.yaml").steps.values():
        foreground = _foreground_for_route_test("shared", step_id=step.id)
        names = {name for name, _ in foreground._question_tools("active").items()}
        assert names == set(step.voice.tools)
        assert not any(name.startswith("workflow__") for name in names)


@pytest.mark.asyncio
async def test_completed_step_does_not_invoke_trigger_or_model() -> None:
    workflow = _manual_workflow(_SAMPLE / "yaml/workflow.yaml")
    calls = 0

    class Llm:
        async def chat(self, *_args, **_kwargs):
            nonlocal calls
            calls += 1
            raise AssertionError("completed steps must not call the model")

    guidance = GuidanceAgent(
        decisions=SimpleNamespace(),  # type: ignore[arg-type]
        workflow=workflow,
        llm=Llm(),  # type: ignore[arg-type]
        current_frame=SimpleNamespace(),  # type: ignore[arg-type]
        image_query=SimpleNamespace(),  # type: ignore[arg-type]
        rag=SimpleNamespace(),  # type: ignore[arg-type]
    )
    session = guidance.store.get("participant-complete")
    guidance.store.start(session)
    guidance.store.observe(session, "accepted")
    result = guidance.store.commit(
        session,
        {
            "tea_name": "Twinings Earl Grey",
            "target_temperature_c": 100,
            "steep_duration_s": 180,
            "guidance_source": "retrieved",
            "temperature_source": "document",
            "duration_source": "document",
            "tea_ready": True,
        },
        "",
    )
    assert result.complete is True

    await guidance._tick(session)

    assert calls == 0


@pytest.mark.asyncio
async def test_slow_observation_does_not_block_reset() -> None:
    workflow = _manual_workflow(_SAMPLE / "yaml/workflow.yaml")
    started = asyncio.Event()
    release = asyncio.Event()

    class Decisions:
        async def decide(self, _state, questions, **_kwargs):
            started.set()
            await release.wait()
            return await _decisions("accepted").decide(_state, questions)

    guidance = GuidanceAgent(
        decisions=Decisions(),  # type: ignore[arg-type]
        workflow=workflow,
        llm=SimpleNamespace(),  # type: ignore[arg-type]
        current_frame=SimpleNamespace(),  # type: ignore[arg-type]
        image_query=SimpleNamespace(),  # type: ignore[arg-type]
        rag=SimpleNamespace(),  # type: ignore[arg-type]
    )
    session = guidance.store.get("participant-slow")
    guidance.store.start(session)
    guidance.store.advance(session, skip=True)

    async def visible_tea(*_args):
        return _TriggerResult(available=True, value="Twinings Earl Grey label", frame_key=None)

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
    workflow = _manual_workflow(_SAMPLE / "yaml/workflow.yaml")
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
        decisions=SimpleNamespace(),  # type: ignore[arg-type]
        workflow=workflow,
        llm=llm,  # type: ignore[arg-type]
        current_frame=images.get_current_frame,  # type: ignore[arg-type]
        image_query=SimpleNamespace(),  # type: ignore[arg-type]
        rag=SimpleNamespace(),  # type: ignore[arg-type]
    )
    guidance.store.start(guidance.store.get("participant-controls"))
    change_watch = ChangeWatchAgent(
        decisions=SimpleNamespace(),
        decision=load_config(_SAMPLE / "yaml/tea_making_worker.yaml").change_watch_decision,
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
        decisions=SimpleNamespace(),
        decision=load_config(_SAMPLE / "yaml/tea_making_worker.yaml").video_delta_decision,
        images=images,  # type: ignore[arg-type]
        vlm=SimpleNamespace(),  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        caption_prompt="Caption.",
        delta_prompt="Compare.",
        interval_s=2.0,
    )
    foreground = _SOPFixture(
        decisions=SimpleNamespace(),  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        images=images,  # type: ignore[arg-type]
        vlm=SimpleNamespace(),  # type: ignore[arg-type]
        rag=SimpleNamespace(),  # type: ignore[arg-type]
        guidance=guidance,
        application_tools=lambda pid: ToolSet(
            tool for provider in (BackgroundContextAgent(), change_watch, transcript, video_log)
            for _, tool in provider.participant_tools(pid).items()
        ),
        prompt="Route.",
        current_view_prompt=load_config(None).current_view_prompt,
    )

    observed_tools.update(name for name, _ in foreground._background_tools("participant-controls").items())

    active_tools = guidance.active_tools("participant-controls")
    assert active_tools is not None
    current_view = active_tools.get("current_view")

    assert observed_tools >= {
        "application_context__query",
        "transcript__start",
        "transcript__stop",
        "transcript__status",
    }
    assert current_view is not None and current_view.return_direct is False


@pytest.mark.asyncio
async def test_active_current_view_preserves_structured_question() -> None:
    workflow = _manual_workflow(_SAMPLE / "yaml/workflow.yaml")
    observed: list[str] = []

    class CurrentFrame:
        async def execute(self, _request):
            return SimpleNamespace(image=ImageReference(uri="frame.jpg"))

    class ImageQuery:
        async def execute(self, request):
            observed.append(request.query)
            return ImageQueryResult(text="The display reads 85 Celsius.")

    guidance = GuidanceAgent(
        decisions=SimpleNamespace(),  # type: ignore[arg-type]
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

    result = await current_view.handler(CurrentViewRequest(question="Read only the kettle temperature and unit."))

    assert result.text == "The display reads 85 Celsius."
    assert observed == ["Read only the kettle temperature and unit."]
    assert current_view.return_direct is False


@pytest.mark.asyncio
async def test_direct_view_is_streamed_without_second_generation() -> None:
    fg = _routed_foreground(_ScriptedDecisions(("destination", "observation")))
    fg._images.get_current_frame = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(
        image=fg._images.images.put(b"pixels"),
    )))
    async def tokens(_request):
        yield ImageQueryChunk(text="A cup.")
    fg._vision = SimpleNamespace(stream=tokens)
    result, calls, spoken = await fg._answer(
        "What is visible?", "reader", SimpleNamespace(metadata=SimpleNamespace(message_id="view"))
    )
    assert (result, calls) == ("A cup.", ["current_view"])
    fg._llm.chat.assert_not_awaited()
    await fg.stop()


@pytest.mark.parametrize("error_type", [FrameUnavailable, ValueError, TypeError, RuntimeError])
async def test_current_view_tool_recovers_only_camera_unavailability(error_type) -> None:
    error = error_type("camera failed")

    async def fail(_request: CurrentFrameRequest) -> ImageFrame:
        raise error

    frames = Tool("camera_failure", "Select a frame.", CurrentFrameRequest, ImageFrame, fail)
    tool = participant_current_view_tool("alice", frames, None)  # type: ignore[arg-type]
    if error_type is FrameUnavailable:
        result = await tool.execute(CurrentViewRequest(question="Visible?"))
        assert not result.available and "camera failed" in result.text
    else:
        with pytest.raises(error_type) as raised:
            await tool.execute(CurrentViewRequest(question="Visible?"))
        assert raised.value is error


@pytest.mark.parametrize("error_type", [FrameUnavailable, ValueError, TypeError, RuntimeError])
async def test_foreground_current_view_recovers_only_camera_unavailability(error_type) -> None:
    error = error_type("camera failed")
    published = []

    async def fail(_request: CurrentFrameRequest) -> ImageFrame:
        raise error

    async def publish(_topic, message, *, participant_id):
        published.append(message)

    frames = Tool("camera_failure", "Select a frame.", CurrentFrameRequest, ImageFrame, fail)
    foreground = object.__new__(_SOPFixture)
    foreground._images = SimpleNamespace(get_current_frame=frames)
    foreground._vlm_timeout_s = 15
    ctx = SimpleNamespace(metadata=SimpleNamespace(message_id="turn"), publish=publish)
    call = _stream_view(foreground,"Visible?", "alice", ctx, timestamp_us=1)
    if error_type is FrameUnavailable:
        result = await call
        assert not result.available and result.text == str(error)
        assert [output.text for output in published] == [str(error), ""]
        assert [output.final for output in published] == [False, True]
    else:
        with pytest.raises(error_type) as raised:
            await call
        assert raised.value is error
        assert not published


@pytest.mark.asyncio
async def test_streaming_current_view_uses_question_and_times_out() -> None:
    observed: list[str] = []
    published: list[ResponseOutput] = []

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

        async def publish(self, topic, message, *, participant_id=None, **kwargs):
            assert topic is VOICE_OUTPUT_TOPIC
            published.append(message)

    foreground = object.__new__(_SOPFixture)
    foreground._images = SimpleNamespace(get_current_frame=CurrentFrame())
    foreground._vision = Vision()
    foreground._vlm_timeout_s = 0.01
    with pytest.raises(TimeoutError):
        await _stream_view(foreground, "Read the display.", "participant-timeout", Context(), timestamp_us=20)

    assert observed == ["Read the display."]
    assert published == []


@pytest.mark.parametrize(
    ("chunks", "expected_text", "available"),
    [
        (["\n", "A kettle is visible."], "A kettle is visible.", True),
        (
            ["\n", "  "],
            "Unable to inspect the current frame because the vision model returned no description.",
            False,
        ),
    ],
)
@pytest.mark.asyncio
async def test_streaming_current_view_handles_leading_whitespace(
    chunks: list[str], expected_text: str, available: bool
) -> None:
    published: list[ResponseOutput] = []

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

        async def publish(self, topic, message, *, participant_id=None, **kwargs):
            assert topic is VOICE_OUTPUT_TOPIC
            published.append(message)

    foreground = object.__new__(_SOPFixture)
    foreground._images = SimpleNamespace(get_current_frame=CurrentFrame())
    foreground._vision = Vision()
    foreground._vlm_timeout_s = 1.0

    result = await _stream_view(foreground,
        "What do you see?",
        "participant-whitespace",
        Context(),  # type: ignore[arg-type]
        timestamp_us=42,
    )

    assert result == ImageQueryResult(text=expected_text, available=available)
    assert [output.text for output in published] == [expected_text, ""]
    assert {output.request_id for output in published} == {"view-whitespace"}
    assert published[-1].final is True


@pytest.mark.asyncio
async def test_foreground_record_failure_does_not_suppress_speech() -> None:
    published: list[tuple[object, object]] = []

    class Context:
        metadata = SimpleNamespace(
            participant_id="participant-disk-full",
            message_id="query-1",
        )

        async def publish(self, topic, message, *, participant_id=None, **kwargs):
            if topic is FOREGROUND_RECORD_TOPIC:
                raise OSError("disk full")
            published.append((topic, message))

    foreground = object.__new__(_SOPFixture)
    foreground._guidance = SimpleNamespace(store=SimpleNamespace(find=lambda _participant: None))
    from xr_ai_conversation import ContextManager

    foreground._context = ContextManager(SimpleNamespace())

    async def answer(*_args, **_kwargs):
        return "Your water is ready.", ["clock__timer"], False

    from xr_ai_interaction import ResponseOutputs
    from xr_ai_routing import AgentTurn
    foreground._responses = ResponseOutputs(Context(), VOICE_OUTPUT_TOPIC)
    foreground._runtime = Context()
    async def produce(reply):
        await reply.send("Your water is ready.")
        return "Your water is ready.", ["clock__timer"]
    with pytest.raises(OSError, match="disk full"):
        await foreground._respond(AgentTurn("participant-disk-full", "query-1", "Is it ready?", timestamp_us=50),
                                  AsyncMock(), produce)

    assert [topic for topic, _message in published] == [VOICE_OUTPUT_TOPIC] * 2
    message = published[0][1]
    assert isinstance(message, ResponseOutput)
    assert message.kind == "reply"
    assert message.request_id == "query-1"
    assert message.text == "Your water is ready."
    assert published[1][1].final is True


@pytest.mark.asyncio
async def test_guidance_record_failure_does_not_suppress_notice() -> None:
    workflow = _manual_workflow(_SAMPLE / "yaml/workflow.yaml")
    guidance = GuidanceAgent(
        decisions=SimpleNamespace(),  # type: ignore[arg-type]
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
        ParticipantLeft(),
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
            and files._leaving_generation.get(closing_participant_id) != expected_generation
        ):
            return
        files._sessions.pop(closing_participant_id, None)
        files._cleanup.pop(closing_participant_id, None)
        files._leaving_generation.pop(closing_participant_id, None)

    files._close_participant = close_participant
    await files.participant_joined(
        ParticipantJoined(),
        context("join-1"),  # type: ignore[arg-type]
    )
    await files.participant_left(
        ParticipantLeft(),
        context("leave-1"),  # type: ignore[arg-type]
    )
    for producer in producers[:-1]:
        await files.participant_cleanup_complete(
            ParticipantCleanupComplete(generation="leave-1", producer=producer),
            context("cleanup-old"),  # type: ignore[arg-type]
        )

    await files.participant_joined(
        ParticipantJoined(),
        context("join-2"),  # type: ignore[arg-type]
    )
    await files.participant_cleanup_complete(
        ParticipantCleanupComplete(generation="leave-1", producer=producers[-1]),
        context("cleanup-stale"),  # type: ignore[arg-type]
    )
    await files.participant_left(
        ParticipantLeft(),
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
        ParticipantJoined(),
        context("join-images"),  # type: ignore[arg-type]
    )
    await images.participant_cleanup_complete(
        ParticipantCleanupComplete(generation="leave-old", producer="guidance"),
        context("cleanup-images-old"),  # type: ignore[arg-type]
    )
    await images.participant_left(
        ParticipantLeft(),
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
        ParticipantJoined(),
        context("join-1"),  # type: ignore[arg-type]
    )
    await files.participant_left(
        ParticipantLeft(),
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
        ParticipantJoined(),
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
        _manual_workflow(path)


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

    workflow = _manual_workflow(path)

    assert workflow.steps[raw["steps"][0]["id"]].complete_on_skip is expected


def test_workflow_rejects_invalid_complete_on_skip_boolean(tmp_path: Path) -> None:
    raw = yaml.safe_load((_SAMPLE / "yaml/workflow.yaml").read_text())
    raw["steps"][0]["complete_on_skip"] = "sometimes"
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="complete_on_skip must be a boolean"):
        _manual_workflow(path)


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
            _event: ParticipantLeft,
            ctx: RuntimeContext,
        ) -> None:
            nonlocal leave_generation
            leave_generation = ctx.metadata.message_id

    runtime.register("leave-observer", LeaveObserver())

    async with runtime:
        await runtime.publish(
            PARTICIPANT_JOINED_TOPIC,
            ParticipantJoined(),
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
            ParticipantLeft(),
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
        json.loads(line) for line in next(tmp_path.glob("glasses-user-*/guidance.jsonl")).read_text().splitlines()
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

    events = [yaml.safe_load(line) for line in (tmp_path / "relay-events.jsonl").read_text().splitlines()]
    names = [event["name"] for event in events]
    assert "llm.chunk" not in names
    assert "turn.summary" in names


def test_default_prompts_come_from_packaged_files(tmp_path: Path) -> None:
    config = load_config(_SAMPLE / "yaml/tea_making_worker.yaml")
    prompt_dir = _WORKER / "tea_making_worker/prompts"
    current_view_prompt = (prompt_dir / "current_view.txt").read_text().strip()
    assert config.foreground_prompt == (prompt_dir / "foreground_prompt.txt").read_text().strip()
    assert config.current_view_prompt == current_view_prompt
    assert config.video_delta_prompt == (prompt_dir / "video_delta_prompt.txt").read_text().strip()
    assert "newly visible change" in config.change_watch_event_prompt
    assert "newly visible change" in config.change_watch_event_prompt
    assert "tool calls" in config.change_watch_event_prompt
    assert "continued presence" in config.change_watch_decision.criteria["rejected"].lower()
    assert "material transition" in config.video_delta_prompt
    assert "material transition" in config.video_delta_prompt
    assert "tool calls" in config.video_delta_prompt
    assert "supplied new utterances" in config.transcript_summary_prompt
    assert "user's first-person view" in current_view_prompt
    assert "only visible evidence" in current_view_prompt
    assert "what am I looking at" not in current_view_prompt.lower()

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
    common_prompt = (_WORKER / "tea_making_worker/prompts/foreground_prompt.txt").read_text()
    idle_prompt = _manual_workflow(_SAMPLE / "yaml/workflow.yaml").foreground.conversation_prompt
    workflow = _manual_workflow(_SAMPLE / "yaml/workflow.yaml")
    active_prompt = "\n".join(
        (
            workflow.foreground.active_question.instructions,
            workflow.foreground.active_prompt,
            *(step.voice.prompt for step in workflow.steps.values()),
        )
    )
    legacy_refusal = "I can only help with the active tea guide right now."
    idle_model_prompt = f"{common_prompt}\n\n{idle_prompt}".lower()
    active_model_prompt = f"{common_prompt}\n\n{active_prompt}".lower()
    assert legacy_refusal not in common_prompt
    assert legacy_refusal not in idle_prompt
    for worked_example_term in ("holding", "color", "shirt", "clothing"):
        assert worked_example_term not in idle_model_prompt
        assert worked_example_term not in active_model_prompt
    assert "general-purpose assistant" in idle_prompt
    assert "answer tea-making questions" in active_prompt.lower()
    assert "decline" in active_prompt.lower() and "unrelated" in active_prompt.lower()

    root_cases = [case for case in cases if case.get("kind") != "observation" and case.get("route", "root") == "root"]
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
        "application_context__query",
        "change_watch__start",
        "change_watch__stop",
        "current_view",
        "workflow__advance",
        "workflow__reset",
        "workflow__restart",
        "workflow__status",
    }
    unrelated_cases = [case for case in active_cases if case.get("expected_workflow_scope") is False]
    assert len(unrelated_cases) >= 4
    assert all(case["expected_tool"] in {None, "current_view"} for case in unrelated_cases)
    assert all("expected_response" not in case for case in unrelated_cases)
    advance_cases = [case for case in active_cases if case["expected_tool"] == "workflow__advance"]
    assert advance_cases
    assert all(isinstance(case.get("expected_skip"), bool) for case in advance_cases)
    observation_cases = [case for case in cases if case.get("kind") == "observation"]
    assert len(observation_cases) >= 4
    assert all(case["expected_updates"] == {} for case in observation_cases)

    positive_active_names = {
        case["name"]
        for case in active_cases
        if case.get("forbidden_response") == legacy_refusal
        or case["expected_tool"]
        in {
            "change_watch__start",
            "current_view",
            "workflow__status",
        }
    }
    assert {
        "active-answers-current-step-question",
        "active-routes-workflow-status",
        "active-routes-step-visual-question",
        "active-starts-background-watch",
    } <= positive_active_names

    idle_unrelated = next(case for case in root_cases if case["name"] == "idle-answers-unrelated-general-knowledge")
    assert idle_unrelated["forbidden_response"] == legacy_refusal
    idle_visual = next(case for case in root_cases if case["name"] == "idle-routes-visible-shirt-color")
    assert idle_visual["expected_tool"] == "current_view"
    assert idle_visual["forbidden_response"] == legacy_refusal
    first_person_view = next(case for case in root_cases if case["name"] == "idle-routes-first-person-view")
    assert first_person_view["expected_tool"] == "current_view"
    active_visual = next(
        case
        for case in active_cases
        if case.get("legacy_case", case["name"]) == "active-rejects-unrelated-visual-question"
    )
    assert active_visual["expected_tool"] == "current_view"
    assert active_visual["expected_workflow_scope"] is False
    assert "expected_response" not in active_visual


def _activate_test_step(guidance: GuidanceAgent, step_id: str) -> None:
    session = guidance.store.get("active")
    guidance.store.start(session)
    while session.step_id != step_id:
        assert session.step_id is not None
        if session.step_id == "start_steeping" and step_id == "steep_timer":
            guidance.store.observe(session, "accepted")
            guidance.store.observe(session, "accepted")
            committed = guidance.store.commit(
                session,
                {"steeping_started_at_us": 1, "steeping_started": True},
                "",
            )
            assert committed.complete
            guidance.store.advance(session, skip=False)
        else:
            guidance.store.advance(session, skip=True)


def _foreground_for_route_test(
    prompt: str,
    *,
    step_id: str = "fill_water",
) -> _SOPFixture:
    workflow = _manual_workflow(_SAMPLE / "yaml/workflow.yaml")
    guidance = GuidanceAgent(
        decisions=SimpleNamespace(),  # type: ignore[arg-type]
        workflow=workflow,
        llm=SimpleNamespace(),  # type: ignore[arg-type]
        current_frame=SimpleNamespace(),  # type: ignore[arg-type]
        image_query=SimpleNamespace(),  # type: ignore[arg-type]
        rag=SimpleNamespace(),  # type: ignore[arg-type]
    )
    _activate_test_step(guidance, step_id)
    images = SimpleNamespace(images=ImageRegistry(), get_current_frame=SimpleNamespace())
    foreground = _SOPFixture(
        decisions=SimpleNamespace(),  # type: ignore[arg-type]
        llm=SimpleNamespace(),  # type: ignore[arg-type]
        images=images,  # type: ignore[arg-type]
        vlm=SimpleNamespace(),  # type: ignore[arg-type]
        rag=SimpleNamespace(),  # type: ignore[arg-type]
        guidance=guidance,
        application_tools=lambda _pid: ToolSet(()),
        prompt=prompt,
        current_view_prompt=load_config(None).current_view_prompt,
    )
    foreground._root_tools = lambda *_args, **_kwargs: ToolSet(())
    return foreground




def test_workflow_tool_typo_fails_when_guidance_is_built(tmp_path: Path) -> None:
    raw = yaml.safe_load((_SAMPLE / "yaml/workflow.yaml").read_text())
    raw["steps"][0]["voice"]["tools"].append("current_veiw")
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="current_veiw"):
        GuidanceAgent(
            decisions=SimpleNamespace(),  # type: ignore[arg-type]
            workflow=_manual_workflow(path),
            llm=SimpleNamespace(),  # type: ignore[arg-type]
            current_frame=SimpleNamespace(),  # type: ignore[arg-type]
            image_query=SimpleNamespace(),  # type: ignore[arg-type]
            rag=SimpleNamespace(),  # type: ignore[arg-type]
        )






def _identity_guidance(responses, *, passages=None):
    from xr_ai_tools.rag import RetrievedChunk, RetrieveRequest, RetrieveResult

    guidance = _guidance_for_observation()
    text = "White tea: brew at 82 C for 3 minutes."
    result = RetrieveResult(
        results=[RetrievedChunk(text=text, source="reference.md", score=0.9)] if passages is None else passages
    )
    guidance._rag = SimpleNamespace(retrieve=Tool(
        "retrieve", "Retrieve", RetrieveRequest, RetrieveResult, AsyncMock(return_value=result)
    ))
    guidance._llm = SimpleNamespace(chat=AsyncMock(side_effect=[
        ChatResponse(json.dumps(item), None, None, "stop", {}) for item in responses
    ]))
    return guidance


@pytest.mark.asyncio
async def test_identification_reads_identity_then_retrieves_unit_bearing_settings() -> None:
    quote = "White tea: brew at 82 C for 3 minutes."
    guidance = _identity_guidance([{
        "values": {"target_temperature_c": {"value": 82, "unit": "celsius"},
                   "steep_duration_s": {"value": 3, "unit": "minutes"}},
        "sources": {name: {"passage": 0, "quote": quote} for name in ("target_temperature_c", "steep_duration_s")},
    }])
    session = guidance.store.get("identification")
    guidance.store.start(session)
    await _observe_caption(guidance, session.participant_id, '{"tea_name":"Northwind White Tea"}')
    assert session.state["tea_name"] == "Northwind White Tea"
    assert session.state["target_temperature_c"] == 82
    assert session.state["steep_duration_s"] == 180
    assert session.state["tea_ready"] is True
    for name in ("temperature_source", "duration_source"):
        source = json.loads(session.state[name])
        assert source == {"kind": "retrieved", "source": "reference.md", "quote": quote}
    assert guidance._llm.chat.await_count == 1
    visual_question = guidance.workflow.step("identify").observation.extraction_question("Read identity")
    assert '"required": ["tea_name"]' in visual_question
    assert "target_temperature_c" not in visual_question


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", [None, "null", "UNKNOWN", "unreadable"])
async def test_unreadable_identity_cannot_be_supplied_by_retrieval(identity) -> None:
    guidance = _identity_guidance([])
    session = guidance.store.get("no-identity")
    guidance.store.start(session)
    await _observe_caption(guidance, session.participant_id, json.dumps({"tea_name": identity}))
    assert "tea_name" not in session.state
    assert "tea_ready" not in session.state
    assert session.evidence_hits == 0
    guidance._llm.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_retrieval_preserves_supported_value_and_generates_only_missing_field() -> None:
    quote = "White tea: brew at 82 C for 3 minutes."
    guidance = _identity_guidance([
        {"values": {"target_temperature_c": {"value": 82, "unit": "celsius"}},
         "sources": {"target_temperature_c": {"passage": 0, "quote": quote}}},
        {"values": {"steep_duration_s": {"value": 2, "unit": "minutes"}}},
    ])
    session = guidance.store.get("partial")
    guidance.store.start(session)
    await _observe_caption(guidance, session.participant_id, '{"tea_name":"Northwind White Tea"}')
    assert session.state["target_temperature_c"] == 82
    assert session.state["steep_duration_s"] == 120
    assert json.loads(session.state["temperature_source"])["kind"] == "retrieved"
    assert json.loads(session.state["duration_source"])["kind"] == "generated"
    fallback = json.loads(guidance._llm.chat.await_args.args[0][-1].content)
    assert set(fallback["output_schema"]["properties"]["values"]["properties"]) == {"steep_duration_s"}


@pytest.mark.asyncio
async def test_identity_survives_failed_resolution_and_later_unreadable_frame() -> None:
    guidance = _identity_guidance([], passages=[])
    guidance._llm.chat.side_effect = RuntimeError("service unavailable")
    session = guidance.store.get("retry")
    guidance.store.start(session)
    with pytest.raises(RuntimeError, match="unavailable"):
        await _observe_caption(guidance, session.participant_id, '{"tea_name":"Northwind White Tea"}')
    assert session.state["tea_name"] == "Northwind White Tea"
    assert not session.state.get("tea_ready")
    guidance._llm.chat.side_effect = [ChatResponse(json.dumps({"values": {
        "target_temperature_c": {"value": 85, "unit": "celsius"},
        "steep_duration_s": {"value": 120, "unit": "seconds"},
    }}), None, None, "stop", {})]
    await _observe_caption(guidance, session.participant_id, '{"tea_name":null}')
    assert session.state["tea_name"] == "Northwind White Tea"
    assert session.state["tea_ready"]
    assert json.loads(session.state["temperature_source"])["kind"] == "generated"


@pytest.mark.asyncio
async def test_unverifiable_retrieval_quote_uses_explicit_generated_fallback() -> None:
    guidance = _identity_guidance([
        {"values": {"target_temperature_c": {"value": 99, "unit": "celsius"}},
         "sources": {"target_temperature_c": {"passage": 0, "quote": "invented quote"}}},
        {"values": {"target_temperature_c": {"value": 80, "unit": "celsius"},
                    "steep_duration_s": {"value": 120, "unit": "seconds"}}},
    ])
    session = guidance.store.get("unsupported")
    guidance.store.start(session)
    await _observe_caption(guidance, session.participant_id, '{"tea_name":"Northwind White Tea"}')
    assert session.state["target_temperature_c"] == 80
    assert json.loads(session.state["temperature_source"])["kind"] == "generated"


@pytest.mark.asyncio
@pytest.mark.parametrize("values", [
    {"tea_name": "fabricated identity"},
    {"target_temperature_c": {"value": float("nan"), "unit": "celsius"}},
    {"steep_duration_s": {"value": 3, "unit": "unknown"}},
])
async def test_invalid_fallback_cannot_change_identity_or_complete(values) -> None:
    guidance = _identity_guidance([{"values": values}], passages=[])
    session = guidance.store.get("bad-fallback")
    guidance.store.start(session)
    with pytest.raises(ValueError):
        await _observe_caption(guidance, session.participant_id, '{"tea_name":"Northwind White Tea"}')
    assert session.state["tea_name"] == "Northwind White Tea"
    assert not session.state.get("tea_ready")


@pytest.mark.asyncio
async def test_timer_completion_has_no_model_dependency() -> None:
    guidance = _guidance_for_observation()
    _activate_test_step(guidance, "steep_timer")
    session = guidance.store.get("active")
    await guidance._tick(session)
    assert session.state["steeping_complete"] is True
    assert session.step_id == "steep_timer"
    assert session.active


@pytest.mark.asyncio
async def test_repeated_camera_frame_cannot_satisfy_two_confirmations() -> None:
    guidance = _guidance_for_observation(_decisions("accepted", "accepted"))
    _activate_test_step(guidance, "fill_water")
    session = guidance.store.get("active")
    registry = ImageRegistry()
    frame = ImageFrame(
        image=registry.put(b"fake pixels"),
        timestamp_us=10,
        sequence=1,
        width=10,
        height=10,
        participant_id="active",
        track_id="camera",
    )
    guidance._current_frame = SimpleNamespace(execute=AsyncMock(return_value=frame))
    guidance._image_query = SimpleNamespace(execute=AsyncMock(return_value=ImageQueryResult(
        text='{"observable":true,"description":"Water is visible."}'
    )))
    await guidance._tick(session)
    session.next_tick = 0
    await guidance._tick(session)
    assert session.evidence_hits == 1
    assert session.state["water_filled"] is False
    assert guidance._image_query.execute.await_count == 1
    guidance._current_frame.execute.return_value = frame.model_copy(update={"sequence": 2, "timestamp_us": 20})
    session.next_tick = 0
    await guidance._tick(session)
    assert session.state["water_filled"] is True


@pytest.mark.asyncio
async def test_observer_is_not_bound_to_tea_steps_or_field_names(tmp_path: Path) -> None:
    raw = {
        "task": {"name": "inspect payload", "start_step": "inspect"},
        "state": {"payload_present": {"type": "boolean", "initial": False}},
        "steps": [
            {
                "id": "inspect",
                "reads": [],
                "writes": ["payload_present"],
                "trigger": {
                    "function": "current_view",
                    "interval_s": 1,
                    "arguments": {"question": "Inspect the container."},
                },
                "observation": {
                    "kind": "choice",
                    "instructions": "Judge container contents.",
                    "criteria": {"accepted": "Visible payload", "rejected": "Empty", "unknown": "Obscured"},
                    "updates": {"payload_present": True},
                },
                "voice": {"prompt": "Explain the inspection", "tools": []},
                "complete_when": {"payload_present": True},
                "next": None,
            }
        ],
    }
    path = tmp_path / "inspection.yaml"
    path.write_text(yaml.safe_dump(raw))
    guidance = GuidanceAgent(
        workflow=_manual_workflow(path),
        decisions=_decisions("accepted"),
        llm=SimpleNamespace(),
        current_frame=SimpleNamespace(),
        image_query=SimpleNamespace(),
        rag=SimpleNamespace(),
    )
    session = guidance.store.get("inspection")
    guidance.store.start(session)
    await _observe_caption(guidance, session.participant_id, "A payload is visible.")
    assert session.state == {"payload_present": True}
    assert session.step_id == "inspect"


@pytest.mark.parametrize(
    "bad_update",
    [
        {"water_filled": "$observation.imaginary"},
        {"water_filled": "$state.imaginary"},
        {"water_filled": "$clock.now_us"},
        {"tea_ready": True},
    ],
)
def test_workflow_rejects_invalid_observation_bindings(tmp_path: Path, bad_update) -> None:
    raw = yaml.safe_load((_SAMPLE / "yaml/workflow.yaml").read_text())
    raw["steps"][1]["observation"]["updates"] = bad_update
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        _manual_workflow(path)


@pytest.mark.asyncio
async def test_transcript_summary_uses_one_generation_without_tools() -> None:
    agent = object.__new__(TranscriptAgent)
    agent._llm = SimpleNamespace(
        chat=AsyncMock(
            return_value=ChatResponse(
                "The user asked about steeping time.",
                None,
                None,
                "stop",
                {},
            )
        )
    )
    agent._summary_prompt = "Summarize only the supplied new utterances."
    summary = await agent._generate_summary(("How long do I steep it?",))
    assert summary.summary == "The user asked about steeping time."
    agent._llm.chat.assert_awaited_once()
    assert "tools" not in agent._llm.chat.call_args.kwargs


class _ScriptedDecisions:
    def __init__(self, *choices):
        self.choices = deque(choices)
        self.calls = []

    async def decide(self, state, questions, **_kwargs):
        key, choice = self.choices.popleft()
        assert set(questions) == {key}
        selected = choice.choice if isinstance(choice, DecisionAnswer) else choice
        assert selected in questions[key].criteria
        self.calls.append((state, questions))
        answer = (
            choice
            if isinstance(choice, DecisionAnswer)
            else DecisionAnswer(
                choice,
                1,
                {label: float(label == choice) for label in questions[key].criteria},
            )
        )
        return DecisionResponse({key: answer}, "fake", {}, {})


def _routed_foreground(decisions) -> _SOPFixture:
    config = load_config(_SAMPLE / "yaml/tea_making_worker.yaml")
    llm = SimpleNamespace(
        chat=AsyncMock(
            return_value=ChatResponse(
                "The next step is heating the water.",
                None,
                None,
                "stop",
                {},
            )
        )
    )
    images = SimpleNamespace(images=ImageRegistry(), get_current_frame=SimpleNamespace())
    guidance = GuidanceAgent(
        workflow=_manual_workflow(config.workflow_config),
        decisions=decisions,
        llm=llm,
        current_frame=images.get_current_frame,
        image_query=SimpleNamespace(),
        rag=SimpleNamespace(),
    )
    providers = (
        BackgroundContextAgent(),
        ChangeWatchAgent(
            images=images,
            vlm=SimpleNamespace(),
            llm=llm,
            decisions=decisions,
            decision=config.change_watch_decision,
            caption_prompt=config.change_watch_caption_prompt,
            event_prompt=config.change_watch_event_prompt,
            default_instruction=config.change_watch_default_instruction,
            interval_s=2,
        ),
        TranscriptAgent(llm=llm, summary_prompt=config.transcript_summary_prompt),
        VideoLogAgent(
            images=images,
            vlm=SimpleNamespace(),
            llm=llm,
            decisions=decisions,
            decision=config.video_delta_decision,
            caption_prompt=config.video_caption_prompt,
            delta_prompt=config.video_delta_prompt,
            interval_s=2,
        ),
    )
    return _SOPFixture(
        llm=llm,
        decisions=decisions,
        images=images,
        vlm=SimpleNamespace(),
        rag=SimpleNamespace(),
        guidance=guidance,
        application_tools=lambda pid: ToolSet(
            tool for provider in providers for _, tool in provider.participant_tools(pid).items()
        ),
        prompt=config.foreground_prompt,
        current_view_prompt=config.current_view_prompt,
    )


@pytest.mark.asyncio
async def test_focused_workflow_hands_off_side_request_and_resumes_without_top_level_routing() -> None:
    decisions = _ScriptedDecisions(
        ("destination", "workflow"),
        ("action", "start"),
        ("authorization", "command"),
        ("action", "handoff"),
        ("destination", "transcript"),
        ("action", "stop"),
        ("authorization", "command"),
        ("action", "answer"),
    )
    fg = _routed_foreground(decisions)
    await fg._answer("Start the guide.", "focused")
    session = fg._guidance.store.get("focused")
    assert session.active and session.step_id == "identify"
    assert fg._llm.chat.await_count == 0
    await fg._answer("Stop the transcript recorder.", "focused")
    assert session.active and session.step_id == "identify"
    before = len(decisions.calls)
    result, calls, _ = await fg._answer("What is the next step?", "focused")
    assert result == "The next step is heating the water."
    assert calls == []
    assert all(set(questions) == {"action"} for _state, questions in decisions.calls[before:])
    assert not any(t.name.startswith("workflow__") for t in (fg._llm.chat.call_args.kwargs.get("tools") or ()))
    await fg.stop()


@pytest.mark.asyncio
async def test_native_skip_is_code_bound_and_stop_releases_router_focus() -> None:
    decisions = _ScriptedDecisions(
        ("destination", "workflow"),
        ("action", "start"),
        ("authorization", "command"),
        ("action", "skip"),
        ("authorization", "command"),
        ("action", "reset"),
        ("authorization", "command"),
        ("destination", "conversation"),
    )
    fg = _routed_foreground(decisions)
    await fg._answer("Start.", "controls")
    await fg._answer("Skip this step.", "controls")
    session = fg._guidance.store.get("controls")
    assert session.step_id == "fill_water" and session.state["tea_ready"] is True
    await fg._answer("Exit the guide.", "controls")
    assert not session.active
    await fg._answer("Hello.", "controls")
    assert not decisions.choices
    await fg.stop()


@pytest.mark.asyncio
async def test_rejected_command_authorization_cannot_mutate_workflow() -> None:
    decisions = _ScriptedDecisions(
        ("destination", "workflow"),
        ("action", "start"),
        ("authorization", "command"),
        ("action", "restart"),
        ("authorization", "not_command"),
        ("destination", "conversation"),
    )
    fg = _routed_foreground(decisions)
    await fg._answer("Start.", "authorization")
    session = fg._guidance.store.get("authorization")
    revision = session.revision
    await fg._answer("Can this walkthrough be reset?", "authorization")
    assert session.revision == revision
    assert session.active
    assert fg._llm.chat.await_count == 1
    await fg.stop()


@pytest.mark.asyncio
async def test_side_conversation_does_not_inherit_workflow_state_or_prompt() -> None:
    decisions = _ScriptedDecisions(
        ("destination", "workflow"),
        ("action", "start"),
        ("authorization", "command"),
        ("action", "handoff"),
        ("destination", "conversation"),
    )
    fg = _routed_foreground(decisions)
    await fg._answer("Start tea guidance.", "scope")
    await fg._answer("Why does ice float?", "scope")
    messages = fg._llm.chat.call_args.args[0]
    assert fg._policy.active_prompt not in messages[0].content
    assert json.loads(messages[-1].content)["state"] == {}
    assert fg._guidance.store.get("scope").active
    await fg.stop()


@pytest.mark.asyncio
async def test_explicit_camera_handoff_remains_available_during_timer_step() -> None:
    decisions = _ScriptedDecisions(
        ("destination", "workflow"),
        ("action", "start"),
        ("authorization", "command"),
        ("action", "handoff"),
        ("destination", "observation"),
    )
    fg = _routed_foreground(decisions)
    await fg._answer("Start tea guidance.", "active")
    _activate_test_step(fg._guidance, "steep_timer")
    assert fg._question_tools("active").get("current_view") is not None
    fg._images.get_current_frame = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(
        image=fg._images.images.put(b"pixels"),
    )))
    async def tokens(_request):
        yield ImageQueryChunk(text="A blue sign.")
    fg._vision = SimpleNamespace(stream=tokens)
    result, calls, spoken = await fg._answer(
        "Read the sign.", "active", SimpleNamespace(metadata=SimpleNamespace(message_id="camera"))
    )
    assert (result, calls) == ("A blue sign.", ["current_view"])
    assert fg._guidance.store.get("active").step_id == "steep_timer"
    await fg.stop()


@pytest.mark.asyncio
async def test_failed_classification_can_retry_the_same_frame_without_double_counting() -> None:
    decisions = _decisions("accepted")
    successful = decisions.decide.side_effect
    failures = 0

    async def decide(*args, **kwargs):
        nonlocal failures
        if not failures:
            failures += 1
            raise TimeoutError("transient decision failure")
        return await successful(*args, **kwargs)

    decisions.decide.side_effect = decide
    guidance = _guidance_for_observation(decisions)
    _activate_test_step(guidance, "fill_water")
    session = guidance.store.get("active")
    registry = ImageRegistry()
    frame = ImageFrame(
        image=registry.put(b"fake pixels"),
        timestamp_us=10,
        sequence=1,
        width=10,
        height=10,
        participant_id="active",
        track_id="camera",
    )
    guidance._current_frame = SimpleNamespace(execute=AsyncMock(return_value=frame))
    guidance._image_query = SimpleNamespace(execute=AsyncMock(return_value=ImageQueryResult(
        text='{"observable":true,"description":"Visible water."}'
    )))
    with pytest.raises(TimeoutError):
        await guidance._tick(session)
    assert session.observed_frame is None
    session.next_tick = 0
    await guidance._tick(session)
    session.next_tick = 0
    await guidance._tick(session)
    assert session.evidence_hits == 1
    assert guidance._image_query.execute.await_count == 2


@pytest.mark.asyncio
async def test_conversation_can_inspect_and_verify_through_the_standard_tool_loop() -> None:
    fg = _routed_foreground(SimpleNamespace())
    _activate_test_step(fg._guidance, "heat_water")
    frame = ImageFrame(
        image=fg._images.images.put(b"fake pixels"), timestamp_us=10, sequence=1,
        width=10, height=10, participant_id="active",
    )
    fg._guidance._current_frame = SimpleNamespace(execute=AsyncMock(return_value=frame))
    fg._guidance._image_query = SimpleNamespace(execute=AsyncMock(return_value=ImageQueryResult(text="57 celsius")))
    fg._llm.chat.side_effect = [
        ChatResponse("", None, [ToolCall(
            "view-1", "current_view", '{"question":"Read the current temperature"}'
        )], "tool_calls", {}),
        ChatResponse("", None, [ToolCall(
            "verify-1", "temperature__verify", '{"reading":57,"unit":"celsius"}'
        )], "tool_calls", {}),
        ChatResponse("The water is 57 degrees, below the target.", None, None, "stop", {}),
    ]
    revision = fg._guidance.store.get("active").revision
    text, calls, _ = await fg._generate_answer("Is it hot enough?", "active", workflow_context=True)
    assert text == "The water is 57 degrees, below the target."
    assert calls == ["current_view", "temperature__verify"]
    messages = fg._llm.chat.await_args.args[0]
    assert json.loads(messages[-1].content) == {"reading_c": 57.0, "target_c": 93.0, "ready": False}
    assert fg._guidance.store.get("active").revision == revision
    await fg.stop()


@pytest.mark.asyncio
async def test_minority_command_score_cannot_authorize_mutation() -> None:
    decisions = _ScriptedDecisions(
        ("destination", "workflow"),
        ("action", "start"),
        ("authorization", "command"),
        ("action", "restart"),
        ("authorization", DecisionAnswer("command", 0.43, {"command": 0.43, "not_command": 0.38, "unknown": 0.19})),
        ("destination", "conversation"),
    )
    fg = _routed_foreground(decisions)
    await fg._answer("Start tea guidance.", "minority")
    session = fg._guidance.store.get("minority")
    revision = session.revision
    await fg._answer("Would it be possible to start again?", "minority")
    assert session.revision == revision
    assert session.active
    assert not any(t.name.startswith("workflow__") for t in (fg._llm.chat.call_args.kwargs.get("tools") or ()))
    state, _questions = next(
        (state, question)
        for state, question in decisions.calls
        if "authorization" in question and state.get("proposed_action", {}).get("operation") == "workflow__restart"
    )
    assert state["proposed_action"]["operation"] == "workflow__restart"
    assert state["proposed_action"]["target"] == "tea making"
    assert "first step" in state["proposed_action"]["description"]
    await fg.stop()


@pytest.mark.asyncio
async def test_partial_settings_and_provenance_survive_an_unresolved_fallback():
    quote = "White tea: brew at 82 C for 3 minutes."
    guidance = _identity_guidance([
        {"values": {"target_temperature_c": {"value": 82, "unit": "celsius"}},
         "sources": {"target_temperature_c": {"passage": 0, "quote": quote}}},
        {"values": {"steep_duration_s": None}},
        {"values": {}},
        {"values": {"steep_duration_s": {"value": 2, "unit": "minutes"}}},
    ])
    session = guidance.store.get("partial-retry")
    guidance.store.start(session)
    await _observe_caption(guidance, session.participant_id, '{"tea_name":"Northwind White Tea"}')
    assert session.state["target_temperature_c"] == 82
    assert not session.state.get("tea_ready")
    original_source = session.state["temperature_source"]
    await _observe_caption(guidance, session.participant_id, '{"tea_name":null}')
    assert session.state["tea_ready"]
    assert session.state["target_temperature_c"] == 82
    assert session.state["temperature_source"] == original_source
    assert json.loads(session.state["duration_source"])["kind"] == "generated"
    payload = json.loads(guidance._llm.chat.await_args.args[0][-1].content)
    assert set(payload["output_schema"]["properties"]["values"]["properties"]) == {"steep_duration_s"}
