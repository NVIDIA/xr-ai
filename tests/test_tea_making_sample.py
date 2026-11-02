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
from xr_ai_interaction import ParticipantJoined, ParticipantLeft
from xr_ai_models import ChatResponse, DecisionAnswer, DecisionResponse
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, subscribe
from xr_ai_tools.image import ImageReference
from xr_ai_tools.vision import ImageQueryResult
from xr_ai_web_events import WEB_EVENT_TOPIC, WebEvent, WebEventsAgent

_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / "agent-samples" / "tea-making-sample"
_WORKER = _SAMPLE / "worker"
sys.path.insert(0, str(_WORKER))
from tea_making_worker.agents.change_watch import ChangeWatchAgent  # noqa: E402
from tea_making_worker.agents.file_output import FileOutputAgent  # noqa: E402
from tea_making_worker.agents.images import ParticipantImageAgent  # noqa: E402
from tea_making_worker.agents.video_log import VideoLogAgent, _is_no_change  # noqa: E402
from tea_making_worker.agents.web_events import TeaWebEventsAgent  # noqa: E402
from tea_making_worker.app import _CLIENT_TEXT_TOPIC, _relay_event_log  # noqa: E402
from tea_making_worker.config import load_config  # noqa: E402
from tea_making_worker.events import (  # noqa: E402
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
    assert all(("deployment" not in model for model in models.values()))
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
    worker_config = sample_main._materialize_worker_config(tmp_path)
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


def test_web_event_config_defaults_and_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config(_SAMPLE / "yaml/tea_making_worker.yaml")
    assert config.web_events_host == "127.0.0.1"
    assert config.web_events_port == 8093
    assert config.web_events_max_events == 5000
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
    exposed_config = sample_main._materialize_worker_config(tmp_path, expose_web_events=True)
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
        (FOREGROUND_RECORD_TOPIC, ForegroundRecord(timestamp_us=1, query="What now?", response="Pour.")),
        (GUIDANCE_RECORD_TOPIC, GuidanceRecord(timestamp_us=2, event="step.enter", message="Fill the kettle.")),
        (GUIDANCE_NOTICE_TOPIC, GuidanceNotice(timestamp_us=3, text="The water is ready.")),
        (
            BACKGROUND_FACT_TOPIC,
            BackgroundFact(timestamp_us=4, application="transcript", text="User mentioned Earl Grey."),
        ),
        (BACKGROUND_FACT_TOPIC, BackgroundFact(timestamp_us=5, application="change_watch", text="A hand was raised.")),
        (
            BACKGROUND_FACT_TOPIC,
            BackgroundFact(timestamp_us=6, application="video_log", text="A person entered the room."),
        ),
        (CHANGE_WATCH_RECORD_TOPIC, ChangeWatchRecord(timestamp_us=7, record_type="started")),
        (
            CHANGE_WATCH_RECORD_TOPIC,
            ChangeWatchRecord(timestamp_us=8, record_type="observation", caption="A hand remains raised."),
        ),
        (TRANSCRIPT_RECORD_TOPIC, TranscriptRecord(timestamp_us=9, record_type="utterance", text="Start steeping.")),
        (VIDEO_LOG_RECORD_TOPIC, VideoLogRecord(timestamp_us=10, record_type="observation", caption="A cup.")),
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
            instruction="raised hands", captions=deque(["Hands are down."]), lock=asyncio.Lock()
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
    agent._states = {participant_id: SimpleNamespace(captions=deque(["The room is empty."]), lock=asyncio.Lock())}
    agent._caption_prompt = "Describe the current scene."
    agent._images = SimpleNamespace(
        get_current_frame=SimpleNamespace(
            execute=AsyncMock(return_value=SimpleNamespace(image=ImageReference(uri="frame.jpg")))
        )
    )
    agent._query_image = SimpleNamespace(
        execute=AsyncMock(return_value=ImageQueryResult(text="A person entered the room.", available=True))
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
    viewer = runtime.register("web-events", WebEventsAgent(host="127.0.0.1", port=0, max_events=8))
    runtime.register("web-event-adapter", TeaWebEventsAgent())
    assert not viewer.running
    async with viewer:
        assert viewer.running
        async with runtime:
            await runtime.publish(PARTICIPANT_JOINED_TOPIC, ParticipantJoined(), participant_id="participant-live")
        assert viewer.running
    assert not viewer.running


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


@pytest.mark.asyncio
async def test_images_release_after_every_producer_finishes_cleanup() -> None:
    released: list[str] = []
    images = object.__new__(ParticipantImageAgent)
    images._cleanup = {}
    images._leaving_generation = {}
    images.get_current_frame = SimpleNamespace(release=released.append)
    ctx = SimpleNamespace(metadata=SimpleNamespace(participant_id="participant-images", message_id="leave-1"))
    await images.participant_left(ParticipantLeft(), ctx)
    producers = ("guidance", "foreground", "change_watch", "transcript", "video_log")
    for producer in producers[:-1]:
        await images.participant_cleanup_complete(
            ParticipantCleanupComplete(generation="leave-1", producer=producer), ctx
        )
    assert released == []
    await images.participant_cleanup_complete(
        ParticipantCleanupComplete(generation="leave-1", producer=producers[-1]), ctx
    )
    assert released == ["participant-images"]
    assert images._cleanup == {}


@pytest.mark.asyncio
async def test_cleanup_quorum_does_not_mix_rejoin_generations() -> None:
    participant_id = "participant-rejoin"
    producers = ("guidance", "foreground", "change_watch", "transcript", "video_log")

    def context(message_id: str):
        return SimpleNamespace(metadata=SimpleNamespace(participant_id=participant_id, message_id=message_id))

    files = object.__new__(FileOutputAgent)
    files._sessions_lock = asyncio.Lock()
    files._sessions = {participant_id: object()}
    files._cleanup = {}
    files._leaving_generation = {}
    files._closed = set()
    files._state = AsyncMock(return_value=files._sessions[participant_id])

    async def close_participant(closing_participant_id: str, *, expected_generation: str | None = None) -> None:
        if (
            expected_generation is not None
            and files._leaving_generation.get(closing_participant_id) != expected_generation
        ):
            return
        files._sessions.pop(closing_participant_id, None)
        files._cleanup.pop(closing_participant_id, None)
        files._leaving_generation.pop(closing_participant_id, None)

    files._close_participant = close_participant
    await files.participant_joined(ParticipantJoined(), context("join-1"))
    await files.participant_left(ParticipantLeft(), context("leave-1"))
    for producer in producers[:-1]:
        await files.participant_cleanup_complete(
            ParticipantCleanupComplete(generation="leave-1", producer=producer), context("cleanup-old")
        )
    await files.participant_joined(ParticipantJoined(), context("join-2"))
    await files.participant_cleanup_complete(
        ParticipantCleanupComplete(generation="leave-1", producer=producers[-1]), context("cleanup-stale")
    )
    await files.participant_left(ParticipantLeft(), context("leave-2"))
    for producer in producers[:-1]:
        await files.participant_cleanup_complete(
            ParticipantCleanupComplete(generation="leave-2", producer=producer), context("cleanup-new")
        )
    assert participant_id in files._sessions
    await files.participant_cleanup_complete(
        ParticipantCleanupComplete(generation="leave-1", producer=producers[-1]), context("cleanup-stale-again")
    )
    assert participant_id in files._sessions
    await files.participant_cleanup_complete(
        ParticipantCleanupComplete(generation="leave-2", producer=producers[-1]), context("cleanup-final")
    )
    assert participant_id not in files._sessions
    released: list[str] = []
    images = object.__new__(ParticipantImageAgent)
    images._cleanup = {}
    images._leaving_generation = {}
    images.get_current_frame = SimpleNamespace(release=released.append)
    await images.participant_joined(ParticipantJoined(), context("join-images"))
    await images.participant_cleanup_complete(
        ParticipantCleanupComplete(generation="leave-old", producer="guidance"), context("cleanup-images-old")
    )
    await images.participant_left(ParticipantLeft(), context("leave-images-new"))
    for producer in producers:
        generation = "leave-old" if producer == "video_log" else "leave-images-new"
        await images.participant_cleanup_complete(
            ParticipantCleanupComplete(generation=generation, producer=producer), context("cleanup-images")
        )
    assert released == []
    await images.participant_cleanup_complete(
        ParticipantCleanupComplete(generation="leave-images-new", producer="video_log"), context("cleanup-images-final")
    )
    assert released == [participant_id]


@pytest.mark.asyncio
async def test_rejoin_wins_race_with_ready_cleanup() -> None:
    participant_id = "participant-rejoin-race"
    producers = ("guidance", "foreground", "change_watch", "transcript", "video_log")

    def context(message_id: str):
        return SimpleNamespace(metadata=SimpleNamespace(participant_id=participant_id, message_id=message_id))

    files = object.__new__(FileOutputAgent)
    files._sessions_lock = asyncio.Lock()
    original_state = SimpleNamespace(active=True)
    files._sessions = {participant_id: original_state}
    files._cleanup = {}
    files._leaving_generation = {}
    files._closed = set()
    files._state = AsyncMock(return_value=original_state)
    await files.participant_joined(ParticipantJoined(), context("join-1"))
    await files.participant_left(ParticipantLeft(), context("leave-1"))
    for producer in producers[:-1]:
        await files.participant_cleanup_complete(
            ParticipantCleanupComplete(generation="leave-1", producer=producer), context("cleanup")
        )
    close_started = asyncio.Event()
    continue_close = asyncio.Event()
    close_participant = files._close_participant

    async def delayed_close(closing_participant_id: str, *, expected_generation: str | None = None) -> None:
        close_started.set()
        await continue_close.wait()
        await close_participant(closing_participant_id, expected_generation=expected_generation)

    files._close_participant = delayed_close
    final_cleanup = asyncio.create_task(
        files.participant_cleanup_complete(
            ParticipantCleanupComplete(generation="leave-1", producer=producers[-1]), context("cleanup-final")
        )
    )
    await asyncio.wait_for(close_started.wait(), timeout=1.0)
    await files.participant_joined(ParticipantJoined(), context("join-2"))
    continue_close.set()
    await asyncio.wait_for(final_cleanup, timeout=1.0)
    assert files._sessions[participant_id] is original_state
    assert participant_id not in files._closed
    assert participant_id not in files._leaving_generation
    assert original_state.active


@pytest.mark.asyncio
async def test_file_output_writes_session_bounded_jsonl(tmp_path: Path) -> None:
    files = FileOutputAgent(tmp_path, history_size=4)
    runtime = AgentRuntime()
    runtime.register("files", files)
    leave_generation: str | None = None

    class LeaveObserver(Agent):
        @subscribe(PARTICIPANT_LEFT_TOPIC)
        async def participant_left(self, _event: ParticipantLeft, ctx: RuntimeContext) -> None:
            nonlocal leave_generation
            leave_generation = ctx.metadata.message_id

    runtime.register("leave-observer", LeaveObserver())
    async with runtime:
        await runtime.publish(PARTICIPANT_JOINED_TOPIC, ParticipantJoined(), participant_id="glasses/user")
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
        await runtime.publish(PARTICIPANT_LEFT_TOPIC, ParticipantLeft(), participant_id="glasses/user")
        assert leave_generation is not None
        for producer in ("foreground", "change_watch", "transcript", "video_log"):
            await runtime.publish(
                PARTICIPANT_CLEANUP_COMPLETE_TOPIC,
                ParticipantCleanupComplete(generation=leave_generation, producer=producer),
                participant_id="glasses/user",
            )
        await runtime.publish(
            GUIDANCE_RECORD_TOPIC,
            GuidanceRecord(timestamp_us=11, event="participant.left", message="", state={}),
            participant_id="glasses/user",
        )
        await runtime.publish(
            PARTICIPANT_CLEANUP_COMPLETE_TOPIC,
            ParticipantCleanupComplete(generation=leave_generation, producer="guidance"),
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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


def test_default_prompt_policies_come_from_main_worker_yaml(tmp_path: Path) -> None:
    raw = yaml.safe_load((_SAMPLE / "yaml/tea_making_worker.yaml").read_text())
    config = load_config(None)
    names = (
        "conversation_system_prompt",
        "current_view_prompt",
        "change_watch_caption_prompt",
        "change_watch_event_prompt",
        "transcript_summary_prompt",
        "video_caption_prompt",
        "video_delta_prompt",
    )
    for name in names:
        assert getattr(config, name) == raw[name].strip()
    assert config.sop.advancement == "automatic"
    assert config.models_config == _SAMPLE / "yaml/models.local.json"
    assert config.change_watch_decision.criteria == raw["change_watch_decision"]["criteria"]
    assert config.video_delta_decision.criteria == raw["video_delta_decision"]["criteria"]
    assert not (_WORKER / "tea_making_worker/prompts").exists()
    raw["conversation_system_prompt"] = "Explicit override"
    path = tmp_path / "worker.yaml"
    path.write_text(yaml.safe_dump(raw))
    override = load_config(path)
    assert override.conversation_system_prompt == "Explicit override"
    assert override.current_view_prompt == config.current_view_prompt


@pytest.mark.parametrize(
    "name",
    [
        "conversation_system_prompt",
        "current_view_prompt",
        "change_watch_caption_prompt",
        "change_watch_event_prompt",
        "transcript_summary_prompt",
        "video_caption_prompt",
        "video_delta_prompt",
        "change_watch_decision",
        "video_delta_decision",
    ],
)
def test_selected_yaml_has_no_hidden_prompt_or_decision_fallback(tmp_path, name):
    raw = yaml.safe_load((_SAMPLE / "yaml/tea_making_worker.yaml").read_text())
    del raw[name]
    path = tmp_path / "worker.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=name):
        load_config(path)


@pytest.mark.parametrize("name", ["conversation_system_prompt", "current_view_prompt"])
def test_prompt_file_options_are_replaced_by_inline_yaml(tmp_path, name):
    raw = yaml.safe_load((_SAMPLE / "yaml/tea_making_worker.yaml").read_text())
    raw[name + "_file"] = "legacy.txt"
    path = tmp_path / "worker.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=name + "_file"):
        load_config(path)


def test_installed_worker_default_uses_sample_yaml_or_requires_explicit_config(tmp_path, monkeypatch):
    import tea_making_worker.config as config_module

    monkeypatch.setattr(config_module, "_PACKAGE", tmp_path / "site-packages/tea_making_worker")
    monkeypatch.chdir(_SAMPLE)
    assert load_config(None).sop.advancement == "automatic"
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="--config"):
        load_config(None)
    explicit = load_config(_SAMPLE / "yaml/tea_making_worker.yaml")
    assert explicit.current_view_prompt
