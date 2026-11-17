# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify optional setup, independent alert clocks, transport and capture lifetime."""

# ruff: noqa: F811
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError
from test_procedure_capture_guide import announcements, capture_context, demo, joined, packet, speak  # noqa: F401
from xr_ai_hub import MediaStatus

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-samples/procedure-capture-guide/worker"))
from procedure_capture_guide_worker.capture import preflight  # noqa: E402
from procedure_capture_guide_worker.capture.agent import CaptureAgent  # noqa: E402
from procedure_capture_guide_worker.capture.quality import (  # noqa: E402
    NotificationSchedule,
    QualityConfig,
    QualityFeedback,
    QualityState,
)


def status(**updates):
    return MediaStatus(**{
        "participant_id": "user", "participant_session_id": "session",
        "pts_us": 1, "video_expected": True, "quality": "good", **updates,
    })


def frame(seq, width=1280, height=720, track="camera"):
    return SimpleNamespace(
        seq=seq, sequence=seq, timestamp_us=seq, width=width, height=height, track_id=track, participant_id="user"
    )


class Scene:
    def __init__(self):
        self.now, self.seq = 0.0, 0
        self.state = QualityState(QualityConfig(), 2, clock=lambda: self.now)

    def second(self, fps=20, width=1280, height=720, *, send_status=True, **updates):
        if send_status:
            self.state.status_received(status(**updates))
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
        ({"fps": 0, "video_expected": False}, set()),
        ({"fps": 0, "video_expected": False}, set()),
        ({"send_status": False}, set()),
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


def test_track_change_duplicate_frames_and_snapshot_rate():
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
    # Stopping and republishing the camera resets cadence and resolution baselines.
    scene.state.status_received(status(video_expected=False))
    for _ in range(12):
        assert not scene.second(fps=8, width=960, height=540)


def test_missing_status_does_not_invent_network_or_live_rate():
    state = QualityState(QualityConfig(), 2)
    state.sample(frame(1))
    assert not state.evaluate()
    assert state.metrics["received_fps"] is None


def test_requested_snapshots_stall_and_recover():
    scene = Scene()
    scene.state.status_received(status(video_expected=False))
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
        "session",
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


def install_quality(demo, *, enabled=True):
    demo.capture = CaptureAgent(
        demo.capture_backend, runtime=demo.runtime,
        quality_config=QualityConfig(enabled=enabled, poll_interval_s=0.01, repeat_interval_s=0.05),
    )
    demo.capture_backend.on_sample = demo.capture.on_quality_sample


async def test_status_rejects_replays_and_previous_connection(demo):
    install_quality(demo)
    await demo.capture.receive(joined())
    monitor = demo.capture._connections["user"].quality

    async def send(session, timestamp):
        await demo.capture.on_media_status(status(participant_session_id=session, pts_us=timestamp))

    await send("wrong", 99)
    assert monitor.preview.status is None
    await send(monitor.session_id, 2)
    await send(monitor.session_id, 1)
    assert monitor.preview.status.pts_us == 2
    await demo.capture.receive(joined(session="replacement"))
    await send(monitor.session_id, 3)
    assert demo.capture._connections["user"].quality.preview.status is None


async def test_audio_notifications_never_stop_recording_and_stop_drains_them(demo):
    install_quality(demo)
    await demo.capture.receive(joined())
    await speak(demo, "start capture")
    state = demo.capture_backend._sessions["user"]
    monitor = demo.capture._connections["user"].quality
    monitor.state.status_received(status(quality="poor", video_expected=False))
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


@pytest.mark.parametrize("ending", ["disconnect", "restart", "close"])
async def test_quality_lifetime_drains_on_leave_restart_and_shutdown(demo, ending):
    install_quality(demo)
    event = joined()
    await demo.capture.receive(event)
    monitor = demo.capture._connections["user"].quality
    data = status(participant_id=monitor.pid, participant_session_id=monitor.session_id, quality="poor")
    await demo.capture.on_media_status(data)
    await speak(demo, "start capture")
    state = demo.capture_backend._sessions["user"]
    await wait_for(lambda: any("network connection is poor" in text for text in announcements(demo)))
    first_task = monitor.task
    if ending == "disconnect":
        await demo.capture.receive(
            type(event)("user", False, event.pts_us + 1, participant_session_id=event.participant_session_id)
        )
    elif ending == "restart":
        await speak(demo, "stop capture")
        assert first_task.done()
        await speak(demo, "start capture")
        assert monitor.task is not first_task
        await speak(demo, "stop capture")
    else:
        await demo.capture.close()
    assert first_task.done()
    assert monitor.task is None and monitor.state is None
    assert packet(state)["status"] == "complete"
    assert not demo.capture_backend.is_capturing("user")


@pytest.mark.parametrize("command", ["Check setup", "Check set up.", "Check set-up!"])
async def test_setup_audio_and_capture_command_not_saved_as_narration(demo, command):
    install_quality(demo)
    await demo.capture.receive(joined())
    await speak(demo, command)
    assert any("Network quality is unknown" in text for text in announcements(demo))
    assert not list((demo.root / "sessions").glob("*/packet.json"))
    await speak(demo, "start capture")
    state = demo.capture_backend._sessions["user"]
    await speak(demo, "A narrated action")
    await speak(demo, "stop capture")
    assert packet(state)["counts"]["transcripts"] == 1


async def test_quality_disabled_hides_setup_and_leaves_capture_available(demo):
    from xr_ai_interaction import ParticipantJoined

    install_quality(demo, enabled=False)
    await demo.capture.receive(joined())
    ctx = capture_context(demo)
    await demo.capture.participant_joined(ParticipantJoined(), ctx)
    assert len(announcements(demo)) == 1
    assert "start capture" in announcements(demo)[0]
    assert "check setup" not in announcements(demo)[0]
    await speak(demo, "check setup")
    assert len(announcements(demo)) == 1
    await speak(demo, "start capture")
    assert demo.capture_backend.is_capturing("user")
    await speak(demo, "stop capture")


def test_setup_reports_latest_received_fps_resolution_and_fresh_grade():
    scene = Scene()
    monitor = QualityFeedback("user", "session", Mock(), QualityConfig(), 2, clock=lambda: scene.now)
    text = preflight.check_setup(monitor)
    assert "Network quality is unknown" in text
    assert "frame rate is unavailable" in text and "resolution is unavailable" in text
    monitor.preview.status_received(status(quality="poor"))
    for i in range(10):
        scene.now = i / 20
        monitor.frame(frame(i, width=640, height=360))
    text = preflight.check_setup(monitor)
    assert "Received video is 20.0 frames per second." in text
    assert "640 by 360" in text and "resolution is low" in text
    assert "Network quality is poor" in text
    assert monitor.state is None
    scene.now = 5
    text = preflight.check_setup(monitor)
    assert "Network quality is unknown" in text
    assert "frame rate is unavailable" in text and "resolution is unavailable" in text


def test_setup_on_demand_does_not_report_old_live_video():
    monitor = QualityFeedback("user", "session", Mock(), QualityConfig(), 2)
    monitor.preview.status_received(status())
    monitor.frame(frame(1))
    assert "1280 by 720" in preflight.check_setup(monitor)
    monitor.preview.status_received(status(video_expected=False))
    text = preflight.check_setup(monitor)
    assert "Network quality is good" in text
    assert "frame rate is unavailable" in text and "resolution is unavailable" in text


async def test_initial_alert_waits_for_start_acknowledgement(demo, monkeypatch):
    from test_voice_playback_state import _PID, _one_reply, _pipeline

    install_quality(demo)
    await demo.capture.receive(joined(_PID))
    monitor = demo.capture._connections[_PID].quality
    data = status(participant_id=monitor.pid, participant_session_id=monitor.session_id, quality="poor")
    await demo.capture.on_media_status(data)
    async with _pipeline(monkeypatch) as (voice, _, tts, endpoint, _gate):
        warning = asyncio.Event()

        async def output(_topic, message, *, participant_id, source=None):
            if message.kind == "reply" and message.text:
                await voice.enqueue_reply(participant_id, _one_reply(message.text), request_id=message.request_id)
                if message.text == "Capture started.":
                    await asyncio.wait_for(endpoint.first_chunk.wait(), 2)
            elif message.kind == "update":
                await voice.enqueue_update(participant_id, message.text, source=source, urgent=message.urgent)
                warning.set()
            elif message.kind == "withdraw" and source:
                await voice.withdraw_updates(participant_id, source=source)

        demo.output.side_effect = output
        await speak(demo, "start capture", pid=_PID)
        await asyncio.wait_for(warning.wait(), 2)
        alert = next(c.args[1] for c in demo.output.call_args_list if c.args[1].kind == "update")
        assert not alert.urgent
        assert tts.calls == ["Capture started."]
        endpoint.release.set()
        await wait_for(lambda: any("network connection is poor" in text for text in tts.calls))
        assert demo.capture_backend.is_capturing(_PID)
        await speak(demo, "stop capture", pid=_PID)


@pytest.mark.parametrize("earlier_setup", [False, True])
async def test_queued_capture_and_setup_reply_in_input_order(demo, monkeypatch, earlier_setup):
    install_quality(demo)
    await demo.capture.receive(joined())
    await speak(demo, "start capture")
    first = demo.capture_backend._sessions["user"]
    entered, release = asyncio.Event(), asyncio.Event()
    finalize = demo.capture_backend._finalize

    async def delayed(state, **kwargs):
        if state is first:
            entered.set()
            await release.wait()
        await finalize(state, **kwargs)

    monkeypatch.setattr(demo.capture_backend, "_finalize", delayed)
    await speak(demo, "stop capture", wait=False)
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if earlier_setup:
            await speak(demo, "check setup", wait=False)
        await speak(demo, "start capture", wait=False)
        await speak(demo, "stop capture", wait=False)
        await speak(demo, "check setup", wait=False)
        await speak(demo, "check setup", wait=False)
    finally:
        release.set()
    await asyncio.wait_for(asyncio.shield(demo.capture._tails["user"]), 3)
    packets = list((demo.root / "sessions").glob("*/packet.json"))
    assert len(packets) == 2
    assert all(json.loads(path.read_text())["status"] == "complete" for path in packets)
    assert not demo.capture_backend.is_capturing("user")
    replies = ["setup" if text.startswith("Network quality") else text for text in announcements(demo)]
    pair = ["Capture started.", "Capture stopped."]
    assert replies == pair + (["setup"] if earlier_setup else []) + pair + ["setup", "setup"]


@pytest.mark.parametrize("staggered", [False, True])
async def test_pending_media_alerts_replace_repeats_without_erasing_other_conditions(monkeypatch, staggered):
    from pipecat.frames.frames import UserStartedSpeakingFrame, UserStoppedSpeakingFrame
    from test_voice_playback_state import _PID, _pipeline

    now = [0.0]
    conditions = {"poor_network"}
    ticks, resume = asyncio.Queue(), asyncio.Queue()

    async def sleep(_seconds):
        ticks.put_nowait(True)
        await resume.get()

    async with _pipeline(monkeypatch) as (voice, worker, tts, endpoint, _gate):
        endpoint.release.set()
        started = UserStartedSpeakingFrame()
        started.transport_source = _PID
        await worker.queue_frame(started)
        await wait_for(lambda: _PID in voice._user_speaking)

        async def publish(_topic, message, *, participant_id, source):
            if message.kind == "update":
                await voice.enqueue_update(participant_id, message.text, source=source, urgent=message.urgent)
            elif message.kind == "withdraw":
                await voice.withdraw_updates(participant_id, source=source)

        monitor = QualityFeedback(
            _PID, "session", SimpleNamespace(capture_quality=AsyncMock()), QualityConfig(), 2,
            clock=lambda: now[0], sleep=sleep,
        )
        monitor.start(SimpleNamespace(publish=publish))
        monitor.state.evaluate = lambda: set(conditions)
        try:
            await ticks.get()  # First warning at t=0 remains queued behind narration.
            await voice.enqueue_update(_PID, "Unrelated notice.", source="other-producer")
            for value in ([5, 10, 15, 20] if staggered else [10, 20]):
                now[0] = value
                if staggered and value == 5:
                    conditions.add("low_frame_rate")
                resume.put_nowait(True)
                await ticks.get()
            assert not tts.calls
            stopped = UserStoppedSpeakingFrame()
            stopped.transport_source = _PID
            await worker.queue_frame(stopped)
            await wait_for(lambda: bool(tts.calls) and not voice._floor)
            spoken = " ".join(tts.calls)
            assert spoken.count("The network connection is poor.") == 1
            assert spoken.count("The video frame rate has dropped.") == int(staggered)
            assert spoken.count("Unrelated notice.") == 1
            # Replacing unheard copies must not disable reminders once heard.
            now[0] = 30
            resume.put_nowait(True)
            await ticks.get()
            await wait_for(lambda: " ".join(tts.calls).count("The network connection is poor.") == 2)
            await wait_for(lambda: not voice._floor)
            spoken = " ".join(tts.calls)
            assert spoken.count("The network connection is poor.") == 2
            assert spoken.count("The video frame rate has dropped.") == 2 * int(staggered)
        finally:
            await monitor.stop()
