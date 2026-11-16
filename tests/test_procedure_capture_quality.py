# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify optional setup, independent alert clocks, transport and capture lifetime."""

# ruff: noqa: F811
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError
from test_procedure_capture_guide import demo, joined, packet, speak  # noqa: F401
from xr_ai_hub import DataMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-samples/procedure-capture-guide/worker"))
from procedure_capture_guide_worker.capture import agent as agent_module  # noqa: E402
from procedure_capture_guide_worker.capture import preflight  # noqa: E402
from procedure_capture_guide_worker.capture.agent import CaptureAgent  # noqa: E402
from procedure_capture_guide_worker.capture.quality import (  # noqa: E402
    TELEMETRY_TOPIC,
    NotificationSchedule,
    QualityConfig,
    QualityFeedback,
    QualityState,
    Telemetry,
)


def telemetry(**updates):
    return Telemetry.model_validate(
        {
            "version": 2,
            "token": "a" * 32,
            "sequence": 1,
            "mode": "live",
            "profile": "camera-1",
            "track_id": "camera",
            "streaming": True,
            "quality": "good",
            **updates,
        }
    )


def frame(seq, width=1280, height=720, track="camera"):
    return SimpleNamespace(
        seq=seq, sequence=seq, timestamp_us=seq, width=width, height=height, track_id=track, participant_id="user"
    )


class Scene:
    def __init__(self):
        self.now, self.seq = 0.0, 0
        self.state = QualityState(QualityConfig(), 2, clock=lambda: self.now)

    def second(self, fps=20, width=1280, height=720, *, send_telemetry=True, **updates):
        if send_telemetry:
            self.state.telemetry_received(telemetry(**updates))
        start = self.now
        for i in range(fps):
            self.now = start + i / fps
            self.seq += 1
            self.state.frame(frame(self.seq, width, height))
        self.now = start + 1
        return self.state.evaluate()


@pytest.mark.parametrize(
    "values,expected",
    [
        ({"quality": "poor"}, {"poor_network"}),
        ({"quality": "lost"}, {"poor_network"}),
        ({"fps": 2}, {"low_frame_rate"}),
        ({"width": 640, "height": 360}, {"low_resolution"}),
        ({"fps": 0}, {"video_stalled"}),
        (
            {"fps": 2, "width": 320, "height": 180, "quality": "poor"},
            {"low_frame_rate", "low_resolution", "poor_network"},
        ),
        ({"width": 720, "height": 1280}, set()),
        ({"fps": 0, "mode": "off", "streaming": False, "track_id": ""}, set()),
        ({"fps": 0, "mode": "on-demand", "streaming": False, "track_id": ""}, set()),
        ({"send_telemetry": False}, set()),
    ],
)
def test_media_conditions_and_recovery(values, expected):
    scene = Scene()
    for _ in range(12):
        assert not scene.second()
    for _ in range(12):
        conditions = scene.second(**values)
    assert conditions == expected
    for _ in range(12):
        conditions = scene.second()
    assert not conditions


def test_profile_reset_duplicate_frames_and_snapshot_rate():
    scene = Scene()
    for _ in range(12):
        scene.state.sample(frame(1))
        assert not scene.second()
    assert scene.state.metrics["received_fps"] == 20
    assert scene.state.metrics["saved_fps"] < 2
    count = len(scene.state.frames)
    scene.state.frame(frame(scene.seq))
    scene.state.frame(frame(scene.seq + 1, track="old"))
    assert len(scene.state.frames) == count
    for _ in range(12):
        assert not scene.second(fps=8, profile="new-camera", width=960, height=540)


def test_missing_telemetry_does_not_invent_network_or_live_rate():
    state = QualityState(QualityConfig(), 2)
    state.sample(frame(1))
    assert not state.evaluate()
    assert state.metrics["received_fps"] is None


def test_requested_snapshots_stall_and_recover():
    scene = Scene()
    scene.state.telemetry_received(telemetry(mode="on-demand", streaming=False))
    scene.state.sample(None)
    scene.now = 3
    assert scene.state.evaluate() == {"video_stalled"}
    scene.state.sample(frame(1))
    assert not scene.state.evaluate()


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_repeat_interval_must_be_finite_and_positive(value):
    with pytest.raises(ValidationError):
        QualityConfig(repeat_interval_s=value)


@pytest.mark.parametrize("interval", [10, 7])
def test_independent_notification_clocks_and_recovery(interval):
    schedule = NotificationSchedule(interval)
    network, rate = {"poor_network"}, {"low_frame_rate"}
    assert schedule.due(network, 10) == ["poor_network"]
    schedule.sent(network, 10)
    assert schedule.due(network | rate, 15) == ["low_frame_rate"]
    schedule.sent(rate, 15)
    assert schedule.due(network | rate, 10 + interval) == ["poor_network"]
    schedule.sent(network, 10 + interval)
    assert schedule.due(network | rate, 15 + interval) == ["low_frame_rate"]
    assert schedule.due(set(), 100) == []


def test_simultaneous_conditions_share_message_and_repeat_time():
    schedule = NotificationSchedule(10)
    conditions = {"poor_network", "low_frame_rate"}
    assert set(schedule.due(conditions, 10)) == conditions
    schedule.sent(conditions, 10)
    assert not schedule.due(conditions, 19.999)
    assert set(schedule.due(conditions, 20)) == conditions
    schedule.sent(conditions, 20)
    assert not schedule.due(set(), 21)
    assert not schedule.due(conditions, 22)  # Flapping cannot bypass the cooldown.


@pytest.mark.parametrize("simultaneous", [False, True])
async def test_monitor_submits_audio_on_independent_deadlines(simultaneous):
    now = [0.0]
    ticks, resume, spoken = asyncio.Queue(), asyncio.Queue(), []

    async def sleep(_seconds):
        ticks.put_nowait(True)
        await resume.get()

    async def publish(_topic, message, **_kwargs):
        if message.kind == "update":
            spoken.append((now[0], message.text))

    monitor = QualityFeedback(
        "user",
        SimpleNamespace(send_return_data=AsyncMock()),
        SimpleNamespace(capture_quality=AsyncMock()),
        QualityConfig(),
        2,
        clock=lambda: now[0],
        sleep=sleep,
    )
    monitor.start(SimpleNamespace(publish=publish))
    await ticks.get()
    conditions = {"poor_network", "low_frame_rate"} if simultaneous else {"poor_network"}
    monitor.state.evaluate = Mock(side_effect=lambda: set(conditions))
    try:
        for value in [10, 15, 19, 20, 25]:
            now[0] = value
            if value == 15:
                conditions.add("low_frame_rate")
            resume.put_nowait(True)
            await ticks.get()
        assert [t for t, _ in spoken] == ([10, 20] if simultaneous else [10, 15, 20, 25])
        if simultaneous:
            assert all("network" in text and "frame rate" in text for _, text in spoken)
        else:
            assert ["network" in text for _, text in spoken] == [True, False, True, False]
    finally:
        await monitor.stop()
    assert monitor.task is None


async def wait_for(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.005)


def install_quality(demo):
    endpoint = SimpleNamespace(send_return_data=AsyncMock())
    demo.capture = CaptureAgent(
        demo.capture_backend,
        quality_config=QualityConfig(poll_interval_s=0.01, repeat_interval_s=0.05),
        endpoint=endpoint,
    )
    demo.capture_backend.on_sample = demo.capture.on_quality_sample
    return endpoint


async def test_lease_rejects_replays_and_previous_connection(demo):
    install_quality(demo)
    await demo.capture.receive(joined())
    monitor = demo.capture._connections["user"].quality

    async def send(token, seq):
        data = telemetry(token=token, sequence=seq)
        await demo.capture.on_quality_data(DataMessage("user", TELEMETRY_TOPIC, 1, data.model_dump_json().encode()))

    await send("wrong", 99)
    assert monitor.latest is None
    await send(monitor.token, 2)
    await send(monitor.token, 1)
    assert monitor.latest.sequence == 2
    await demo.capture.receive(joined(session="replacement"))
    await send(monitor.token, 3)
    assert demo.capture._connections["user"].quality.latest is None


async def test_audio_notifications_never_stop_recording_and_stop_drains_them(demo):
    install_quality(demo)
    await demo.capture.receive(joined())
    await speak(demo, "start capture")
    state = demo.capture_backend._sessions["user"]
    monitor = demo.capture._connections["user"].quality
    monitor.state.telemetry_received(telemetry(quality="poor", mode="off", streaming=False))
    await wait_for(lambda: sum(c.args[1].kind == "update" for c in demo.output.call_args_list) >= 2)
    assert demo.capture_backend.is_capturing("user")
    await speak(demo, "Narration while the connection is poor")
    assert state.transcript_count == 1
    await speak(demo, "stop capture")
    assert monitor.task is None and not demo.capture_backend.is_capturing("user")
    saved = packet(state)
    assert saved["status"] == "complete" and saved["counts"]["frames"] > 0
    diagnostics = [json.loads(line) for line in (state.directory / "quality.jsonl").read_text().splitlines()]
    assert any(row["kind"] == "warning_submitted" for row in diagnostics)
    before = len(demo.output.call_args_list)
    await asyncio.sleep(0.08)
    assert len(demo.output.call_args_list) == before


@pytest.mark.parametrize("end", ["start", "disconnect", "close"])
async def test_setup_is_optional_nonblocking_and_cancelled_on_lifecycle_change(demo, monkeypatch, end):
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    monkeypatch.setattr(agent_module, "check_setup", blocked)
    await demo.capture.receive(joined())
    await speak(demo, "check setup")
    await entered.wait()
    assert not demo.capture_backend.is_capturing("user")
    if end == "start":
        await speak(demo, "start capture")
        assert demo.capture_backend.is_capturing("user")
    elif end == "disconnect":
        event = joined()
        await demo.capture.receive(
            type(event)("user", False, event.pts_us + 1, participant_session_id=event.participant_session_id)
        )
    else:
        await demo.capture.close()
    await asyncio.wait_for(cancelled.wait(), 1)
    assert not any("Network quality" in c.args[1].text for c in demo.output.call_args_list if c.args[1].text)


async def test_setup_audio_and_capture_command_not_saved_as_narration(demo, monkeypatch):
    monkeypatch.setattr(
        agent_module, "check_setup", AsyncMock(return_value=({}, "Network quality is good. Video is 20 fps."))
    )
    await demo.capture.receive(joined())
    await speak(demo, "check setup")
    await demo.capture._connections["user"].preflight_task
    assert any("Network quality is good" in c.args[1].text for c in demo.output.call_args_list if c.args[1].text)
    assert not list((demo.root / "sessions").glob("*/packet.json"))
    await speak(demo, "start capture")
    state = demo.capture_backend._sessions["user"]
    await speak(demo, "A narrated action")
    await speak(demo, "stop capture")
    assert packet(state)["counts"]["transcripts"] == 1


async def test_setup_reports_unavailable_immediately_without_telemetry():
    async with asyncio.timeout(0.1):
        report, speech = await preflight.check_setup()
    assert report == {
        "network_quality": "unknown",
        "fps": None,
        "fps_source": "unavailable",
        "resolution": None,
        "conditions": [],
    }
    assert "frame rate is unavailable" in speech and "resolution is unavailable" in speech


def setup_monitor(**updates):
    monitor = QualityFeedback("user", Mock(), Mock(), QualityConfig(), 2)
    monitor.telemetry(
        DataMessage(
            "user",
            TELEMETRY_TOPIC,
            1,
            telemetry(token=monitor.token, sent_fps=30.0, sent_width=1280, sent_height=720, **updates)
            .model_dump_json()
            .encode(),
        )
    )
    return monitor


async def test_setup_reports_current_metrics_without_sampling_or_sleeping(monkeypatch):
    monitor = setup_monitor(quality="poor")
    # A healthy check uses already received telemetry and must not start a timer.
    sleep = AsyncMock(side_effect=AssertionError("setup must not sleep"))
    monkeypatch.setattr(asyncio, "sleep", sleep)
    async with asyncio.timeout(0.1):
        report, speech = await preflight.check_setup(monitor)
    assert report["network_quality"] == "poor"
    assert report["fps"] == 30
    assert report["fps_source"] == "client_video"
    assert report["resolution"] == (1280, 720)
    assert report["conditions"] == ["poor_network"]
    assert "camera is sending 30.0" in speech
    assert monitor.state is None
    sleep.assert_not_awaited()


async def test_setup_uses_recent_received_resolution_and_rejects_stale_data():
    monitor = setup_monitor()
    monitor.frame(frame(1, width=640, height=360))
    report, _ = await preflight.check_setup(monitor)
    assert report["resolution"] == (640, 360)
    assert report["conditions"] == ["low_resolution"]
    monitor.latest_at -= 10
    monitor.latest_frame_at -= 10
    report, _ = await preflight.check_setup(monitor)
    assert report["network_quality"] == "unknown"
    assert report["fps"] is None and report["resolution"] is None


async def test_setup_on_demand_does_not_report_stale_video_stats():
    monitor = setup_monitor(mode="on-demand", streaming=False)
    report, speech = await preflight.check_setup(monitor)
    assert report["network_quality"] == "good"
    assert report["fps"] is None and report["resolution"] is None
    assert "frame rate is unavailable" in speech


async def test_setup_does_not_reopen_stopped_or_replaced_connection(demo, monkeypatch):
    entered = asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(agent_module, "check_setup", blocked)
    await demo.capture.receive(joined())
    await speak(demo, "check setup")
    await entered.wait()
    old = demo.capture._connections["user"].preflight_task
    await demo.capture.receive(joined(session="new"))
    await asyncio.gather(old, return_exceptions=True)
    assert old.cancelled()
    assert demo.capture._connections["user"].preflight_task is None


def test_browser_media_telemetry():
    script = Path(__file__).parent / "javascript/capture_quality.test.mjs"
    result = subprocess.run(["node", "--test", str(script)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


async def test_media_alert_reaches_voice_text_for_tts(demo):
    from pipecat.frames.frames import TextFrame
    from xr_ai_voice._processors.io import _VoiceIOProcessor

    install_quality(demo)
    voice = _VoiceIOProcessor(AsyncMock())
    voice.push_frame = AsyncMock()

    async def output(_topic, message, *, participant_id, source=None):
        if source and source.startswith("capture-quality:"):
            if message.kind == "update":
                await voice.enqueue_update(participant_id, message.text, source=source, urgent=message.urgent)
            elif message.kind == "withdraw":
                await voice.withdraw_updates(participant_id, source=source)

    demo.output.side_effect = output
    try:
        await demo.capture.receive(joined())
        await speak(demo, "start capture")
        monitor = demo.capture._connections["user"].quality
        monitor.state.telemetry_received(telemetry(quality="poor", mode="off", streaming=False))
        await wait_for(lambda: any(isinstance(c.args[0], TextFrame) for c in voice.push_frame.call_args_list))
        text = next(c.args[0] for c in voice.push_frame.call_args_list if isinstance(c.args[0], TextFrame))
        assert "network connection is poor" in text.text
        assert text.transport_destination == "user"
        assert demo.capture_backend.is_capturing("user")
        await speak(demo, "stop capture")
    finally:
        await voice._shutdown()


async def test_every_idle_setup_request_gets_a_reply(demo, monkeypatch):
    await demo.capture.receive(joined())
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def check(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await release.wait()
        return {}, f"Setup report {calls}"

    monkeypatch.setattr(agent_module, "check_setup", check)
    await speak(demo, "check setup")
    await entered.wait()
    await speak(demo, "check setup")
    release.set()
    await demo.capture._connections["user"].preflight_task
    assert calls == 2
    replies = [c.args[1].text for c in demo.output.call_args_list]
    assert "Setup report 1" in replies and "Setup report 2" in replies
