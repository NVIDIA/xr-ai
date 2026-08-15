# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the native background monitoring sample."""

from __future__ import annotations

import json
import sys
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from xr_ai_models import ChatResponse, ToolCall
from xr_ai_runtime import AgentRuntime
from xr_ai_tools import Tool
from xr_ai_tools.current_frame import CurrentFrameRequest, ImageFrame
from xr_ai_tools.image import ImageReference
from xr_ai_tools.tool_calling import handle_tool_call, tool_definitions
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryResult
from xr_ai_voice import UserQuery, VoiceParticipantLeft

_REPO = Path(__file__).resolve().parents[1]
_SAMPLE = _REPO / "agent-samples" / "background-monitoring-sample"
_WORKER = _SAMPLE / "worker"
sys.path.insert(0, str(_WORKER))

from background_monitoring_worker.config import load_config  # noqa: E402  # pyright: ignore[reportMissingImports]
from background_monitoring_worker.events import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    FOREGROUND_RECORD_TOPIC,
    MONITOR_RECORD_TOPIC,
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    USER_QUERY_TOPIC,
    ForegroundRecord,
    MonitorRecord,
    ParticipantJoined,
)
from background_monitoring_worker.file_output import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    FileOutputAgent,
    MonitoringHistoryRequest,
)
from background_monitoring_worker.foreground import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    CURRENT_VIEW_TOOL,
    FOREGROUND_TOOL_DEFS,
    LAB_INSTRUMENTS_START_TOOL,
    LAB_INSTRUMENTS_STOP_TOOL,
    RECENT_VISUAL_HISTORY_TOOL,
    VISUAL_MONITOR_START_TOOL,
    VISUAL_MONITOR_STOP_TOOL,
    ForegroundAgent,
)
from background_monitoring_worker.images import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    ParticipantImageAgent,
)
from background_monitoring_worker.monitor import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    MonitorAgent,
    MonitoringRequest,
    StartMonitoringRequest,
    parse_monitor_response,
)
from background_monitoring_worker.qr_instruments import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    QRInstrumentAgent,
)
from background_monitoring_worker.transcript import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    TranscriptAgent,
)


def _fake_endpoint() -> SimpleNamespace:
    return SimpleNamespace(
        on_frame=lambda _callback: None,
        on_participant=lambda _callback: None,
    )


def _make_images(endpoint: SimpleNamespace | None = None) -> ParticipantImageAgent:
    return ParticipantImageAgent(
        endpoint=endpoint or _fake_endpoint(),  # type: ignore[arg-type]
        frame_max_age_s=2.0,
        frame_timeout_s=5.0,
    )


def _make_monitor(images: ParticipantImageAgent | None = None) -> MonitorAgent:
    return MonitorAgent(
        images=images or _make_images(),
        vlm=SimpleNamespace(),  # type: ignore[arg-type]
        prompt="Observe.",
        interval_s=5.0,
    )


def _make_qr(images: ParticipantImageAgent | None = None) -> QRInstrumentAgent:
    return QRInstrumentAgent(
        images=images or _make_images(),
        vlm=SimpleNamespace(),  # type: ignore[arg-type]
        interval_s=5.0,
    )


def test_sample_uses_named_native_agents_and_shared_connection_client() -> None:
    project = tomllib.loads((_WORKER / "pyproject.toml").read_text())
    dependencies = set(project["project"]["dependencies"])
    package = _WORKER / "background_monitoring_worker"

    assert project["project"]["scripts"]["background_monitoring_worker"] == (
        "background_monitoring_worker.__main__:run"
    )
    assert {
        "app.py",
        "events.py",
        "file_output.py",
        "foreground.py",
        "images.py",
        "monitor.py",
        "qr_instruments.py",
        "transcript.py",
    } <= {path.name for path in package.glob("*.py")}
    assert "xr-ai-agent-runtime" in dependencies
    assert "xr-ai-tools[frames,image-editing,qr-code,vision]" in dependencies
    assert "xr-ai-voice" in dependencies
    assert "xr-ai-nat" not in dependencies
    assert "xr-ai-pipecat" not in dependencies
    assert all("mcp" not in dependency.lower() for dependency in dependencies)
    hub = yaml.safe_load((_SAMPLE / "yaml" / "xr_media_hub.yaml").read_text())
    assert hub["enable_token_server"] is True
    assert (
        (_SAMPLE / "yaml" / hub["web_client_dir"]).resolve()
        == _REPO / "client-samples" / "web"
    )
    assert not any(path.name == "web" for path in _SAMPLE.iterdir())


def test_config_loads_packaged_prompts_and_file_output_defaults() -> None:
    config = load_config(_SAMPLE / "yaml" / "background_monitoring_worker.yaml")

    assert config.models_config == _SAMPLE / "yaml" / "models.local.json"
    assert config.voice_gate_yaml == _SAMPLE / "yaml" / "voice_gate.yaml"
    assert config.artifacts_dir == _SAMPLE / "artifacts"
    assert config.monitor_interval_s == 5.0
    assert "Previous caption" not in config.monitor_prompt
    assert "current_view" in config.foreground_prompt


def test_monitor_and_foreground_share_participant_image_acquisition(tmp_path: Path) -> None:
    images = _make_images()
    vlm = SimpleNamespace()
    monitor = _make_monitor(images)
    qr_instruments = _make_qr(images)
    foreground = ForegroundAgent(
        llm=SimpleNamespace(),  # type: ignore[arg-type]
        images=images,
        vlm=vlm,  # type: ignore[arg-type]
        files=FileOutputAgent(tmp_path, history_size=2),
        monitor=monitor,
        qr_instruments=qr_instruments,
        prompt="Answer.",
    )

    assert monitor._images is images
    assert foreground._images is images
    assert qr_instruments._images is images
    assert monitor.query_image is not foreground.query_image
    assert {tool.name for tool in monitor.tools} == {
        "query_image",
        "start_monitoring",
        "stop_monitoring",
        "monitoring_status",
    }
    assert {tool.name for tool in foreground.tools} == {"query_image"}


@pytest.mark.asyncio
async def test_monitor_controls_are_participant_scoped_and_idempotent() -> None:
    monitor = _make_monitor()
    runtime = AgentRuntime()
    runtime.register("monitor", monitor)

    async with runtime:
        monitor.bind_runtime(runtime)
        started = await monitor.start_monitoring.execute(
            StartMonitoringRequest(
                participant_id="participant-1",
                instruction="packages near the doorway",
            )
        )
        repeated = await monitor.start_monitoring.execute(
            StartMonitoringRequest(
                participant_id="participant-1",
                instruction="a different request",
            )
        )
        running = await monitor.monitoring_status.execute(
            MonitoringRequest(participant_id="participant-1")
        )
        stopped = await monitor.stop_monitoring.execute(
            MonitoringRequest(participant_id="participant-1")
        )
        stopped_again = await monitor.stop_monitoring.execute(
            MonitoringRequest(participant_id="participant-1")
        )

        assert started.active is True
        assert started.instruction == "packages near the doorway"
        assert repeated.active is True
        assert repeated.instruction == started.instruction
        assert running.active is True
        assert stopped.active is False
        assert stopped_again.message == "Background monitoring is not running."

        await monitor.stop()


def test_monitor_response_is_strict_and_normalizes_baselines() -> None:
    baseline = parse_monitor_response(
        '```json\n{"caption":"A closed door.","changed":true,"summary":"Door closed."}\n```',
        baseline=True,
    )
    unchanged = parse_monitor_response(
        '{"caption":"A closed door.","changed":false,"summary":"ignored"}',
        baseline=False,
    )

    assert baseline.caption == "A closed door."
    assert baseline.changed is False
    assert baseline.summary == ""
    assert unchanged.summary == ""
    with pytest.raises(ValueError):
        parse_monitor_response(
            '{"caption":"A person entered.","changed":true,"summary":""}',
            baseline=False,
        )
    with pytest.raises(ValueError):
        parse_monitor_response("not json", baseline=False)


@pytest.mark.asyncio
async def test_file_output_records_transcript_monitor_and_foreground(tmp_path: Path) -> None:
    files = FileOutputAgent(tmp_path, history_size=2)
    runtime = AgentRuntime()
    runtime.register("files", files)
    runtime.register("transcript", TranscriptAgent())
    now = time.time_ns() // 1_000

    async with runtime:
        await runtime.publish(
            PARTICIPANT_JOINED_TOPIC,
            ParticipantJoined(timestamp_us=now),
            participant_id="glasses/user",
        )
        await runtime.publish(
            USER_QUERY_TOPIC,
            UserQuery(text="What changed?", timestamp_us=now),
            participant_id="glasses/user",
        )
        for index in range(3):
            await runtime.publish(
                MONITOR_RECORD_TOPIC,
                MonitorRecord(
                    timestamp_us=now + index,
                    record_type="observation",
                    caption=f"scene {index}",
                ),
                participant_id="glasses/user",
            )
        await runtime.publish(
            FOREGROUND_RECORD_TOPIC,
            ForegroundRecord(
                timestamp_us=now,
                query="What changed?",
                response="A bag appeared.",
                tools=["read_monitoring_history"],
            ),
            participant_id="glasses/user",
        )
        history = await files.read_monitoring_history.execute(
            MonitoringHistoryRequest(participant_id="glasses/user", limit=20)
        )
        await runtime.publish(
            PARTICIPANT_LEFT_TOPIC,
            VoiceParticipantLeft(),
            participant_id="glasses/user",
        )
        await runtime.publish(
            MONITOR_RECORD_TOPIC,
            MonitorRecord(
                timestamp_us=now + 4,
                record_type="observation",
                caption="late record",
            ),
            participant_id="glasses/user",
        )

    assert [item.caption for item in history.observations] == ["scene 1", "scene 2"]
    sessions = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(sessions) == 1
    for name in ("monitor.jsonl", "transcript.jsonl", "foreground.jsonl"):
        records = [json.loads(line) for line in (sessions[0] / name).read_text().splitlines()]
        assert records[0]["type"] == "session"
        assert records[-1]["type"] == "session_end"
    transcript = (sessions[0] / "transcript.jsonl").read_text()
    assert "What changed?" in transcript


@pytest.mark.asyncio
async def test_foreground_injects_participant_into_current_frame_tool(tmp_path: Path) -> None:
    frame_requests: list[CurrentFrameRequest] = []
    image_requests: list[ImageQueryRequest] = []

    async def select_frame(request: CurrentFrameRequest) -> ImageFrame:
        frame_requests.append(request)
        return ImageFrame(
            image=ImageReference(uri="xr-image://frame-1"),
            timestamp_us=1,
            width=640,
            height=480,
            sequence=1,
            participant_id=request.participant_id,
        )

    async def query_image(request: ImageQueryRequest) -> ImageQueryResult:
        image_requests.append(request)
        return ImageQueryResult(text="A blue notebook.")

    images = _make_images()
    agent = ForegroundAgent(
        llm=SimpleNamespace(),  # type: ignore[arg-type]
        images=images,
        vlm=SimpleNamespace(),  # type: ignore[arg-type]
        files=FileOutputAgent(tmp_path, history_size=2),
        monitor=_make_monitor(images),
        qr_instruments=_make_qr(images),
        prompt="Answer briefly.",
    )
    images.get_current_frame = Tool(
        "get_current_frame",
        "Select a frame.",
        CurrentFrameRequest,
        ImageFrame,
        select_frame,
    )
    agent.query_image = Tool(
        "query_image",
        "Query an image.",
        ImageQueryRequest,
        ImageQueryResult,
        query_image,
    )
    result = await handle_tool_call(
        ToolCall(id="call-1", name=CURRENT_VIEW_TOOL, arguments='{"question":"Color?"}'),
        agent._participant_tools("participant-7"),
    )

    assert tool_definitions(agent._participant_tools("participant-7")) == FOREGROUND_TOOL_DEFS
    assert json.loads(result.message.content)["text"] == "A blue notebook."
    assert frame_requests == [CurrentFrameRequest(participant_id="participant-7")]
    assert image_requests == [
        ImageQueryRequest(
            image=ImageReference(uri="xr-image://frame-1"),
            query="Color?",
        )
    ]


@pytest.mark.asyncio
async def test_foreground_background_control_returns_direct(tmp_path: Path) -> None:
    class Llm:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, _messages, **_kwargs):
            self.calls += 1
            return ChatResponse(
                content="",
                reasoning=None,
                tool_calls=[
                    ToolCall(
                        id="call-start",
                        name=VISUAL_MONITOR_START_TOOL,
                        arguments='{"instruction":"the doorway"}',
                    )
                ],
                finish_reason="tool_calls",
                raw={},
            )

    llm = Llm()
    images = _make_images()
    monitor = _make_monitor(images)
    qr_instruments = _make_qr(images)
    runtime = AgentRuntime()
    runtime.register("monitor", monitor)

    async with runtime:
        monitor.bind_runtime(runtime)
        agent = ForegroundAgent(
            llm=llm,  # type: ignore[arg-type]
            images=images,
            vlm=SimpleNamespace(),  # type: ignore[arg-type]
            files=FileOutputAgent(tmp_path, history_size=2),
            monitor=monitor,
            qr_instruments=qr_instruments,
            prompt="Route one request.",
        )
        response, tools = await agent._answer(
            "Watch the doorway.",
            "participant-2",
        )
        status = await monitor.monitoring_status.execute(
            MonitoringRequest(participant_id="participant-2")
        )

        assert response == "Background monitoring started. Monitoring: the doorway."
        assert tools == [VISUAL_MONITOR_START_TOOL]
        assert llm.calls == 1
        assert status.active is True

        await monitor.stop()


@pytest.mark.asyncio
async def test_foreground_hides_monitor_controls_without_explicit_intent(
    tmp_path: Path,
) -> None:
    class Llm:
        def __init__(self) -> None:
            self.tool_names: set[str] = set()

        async def chat(self, _messages, *, tools, **_kwargs):
            self.tool_names = {tool.name for tool in tools}
            return ChatResponse("I heard you.", None, None, "stop", {})

    llm = Llm()
    images = _make_images()
    agent = ForegroundAgent(
        llm=llm,  # type: ignore[arg-type]
        images=images,
        vlm=SimpleNamespace(),  # type: ignore[arg-type]
        files=FileOutputAgent(tmp_path, history_size=2),
        monitor=_make_monitor(images),
        qr_instruments=_make_qr(images),
        prompt="Route one request.",
    )

    response, used = await agent._answer("To Peter.", "participant-2")

    assert response == "I heard you."
    assert used == []
    assert llm.tool_names.isdisjoint(
        {
            VISUAL_MONITOR_START_TOOL,
            VISUAL_MONITOR_STOP_TOOL,
            LAB_INSTRUMENTS_START_TOOL,
            LAB_INSTRUMENTS_STOP_TOOL,
        }
    )


@pytest.mark.asyncio
async def test_foreground_tool_loop_returns_model_answer_and_tool_audit(tmp_path: Path) -> None:
    class Llm:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return ChatResponse(
                    content="",
                    reasoning=None,
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name=RECENT_VISUAL_HISTORY_TOOL,
                            arguments='{"limit":2}',
                        )
                    ],
                    finish_reason="tool_calls",
                    raw={},
                )
            assert messages[-1].role == "tool"
            return ChatResponse("Nothing material changed.", None, None, "stop", {})

    files = FileOutputAgent(tmp_path, history_size=2)
    runtime = AgentRuntime()
    runtime.register("files", files)
    await runtime.start()
    images = _make_images()
    agent = ForegroundAgent(
        llm=Llm(),  # type: ignore[arg-type]
        images=images,
        vlm=SimpleNamespace(),  # type: ignore[arg-type]
        files=files,
        monitor=_make_monitor(images),
        qr_instruments=_make_qr(images),
        prompt="Answer briefly.",
    )

    try:
        await runtime.publish(
            PARTICIPANT_JOINED_TOPIC,
            ParticipantJoined(timestamp_us=1),
            participant_id="participant-4",
        )
        response, tools = await agent._answer("What changed?", "participant-4")
    finally:
        await runtime.stop()

    assert response == "Nothing material changed."
    assert tools == [RECENT_VISUAL_HISTORY_TOOL]


def test_foreground_prompt_has_non_overlapping_routing_eval_cases() -> None:
    prompt = (
        _WORKER
        / "background_monitoring_worker"
        / "prompts"
        / "foreground_prompt.txt"
    ).read_text().lower()
    cases = yaml.safe_load((_SAMPLE / "eval" / "cases.yaml").read_text())

    assert {case["expected_tool"] for case in cases} == {
        None,
        "current_view",
        "recent_visual_history",
        "visual_monitor__start",
        "visual_monitor__stop",
        "monitoring_status",
        "lab_instruments__read",
        "lab_instruments__start",
        "lab_instruments__stop",
    }
    assert all(case["query"].lower() not in prompt for case in cases)
