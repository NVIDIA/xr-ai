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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-samples/procedure-capture-guide/worker"))
from procedure_capture_guide_worker.capture import preflight  # noqa: E402
from procedure_capture_guide_worker.capture.agent import CaptureAgent  # noqa: E402
from procedure_capture_guide_worker.capture.quality import (  # noqa: E402
    NotificationSchedule,
    QualityConfig,
    QualityFeedback,
    QualityState,
)


def frame(seq, width=1280, height=720, track="camera", pid="user"):
    return SimpleNamespace(
        seq=seq, sequence=seq, timestamp_us=seq, width=width, height=height, track_id=track, participant_id=pid
    )


class Scene:
    def __init__(self):
        self.now, self.seq = 0.0, 0
        self.state = QualityState(QualityConfig(), clock=lambda: self.now)

    def second(self, fps=20, width=1280, height=720, track="camera"):
        start = self.now
        for i in range(fps):
            self.now = start + i / fps
            self.seq += 1
            self.state.frame(frame(self.seq, width, height, track))
        self.now = start + 1
        return self.state.evaluate()


@pytest.mark.parametrize("values,expected", [
    ({"fps": 2}, {"low_frame_rate"}),
    ({"width": 640, "height": 360}, {"low_resolution"}),
    ({"fps": 0}, {"low_frame_rate"}),
    ({"fps": 2, "width": 320, "height": 180}, {"low_frame_rate", "low_resolution"}),
    ({"width": 720, "height": 1280}, set()),
])
def test_video_conditions_and_recovery(values, expected):
    scene = Scene()
    for _ in range(12):
        assert not scene.second()
    for _ in range(12):
        conditions = scene.second(**values)
    assert conditions == expected
    for _ in range(12):
        conditions = scene.second()
    assert not conditions


@pytest.mark.parametrize("fps,width,height", [(1, 1280, 720), (20, 320, 240), (1, 320, 240), (2, 640, 360)])
def test_stable_low_rate_and_resolution_do_not_warn(fps, width, height):
    scene = Scene()
    for _ in range(12):
        assert not scene.second(fps=fps, width=width, height=height)
    assert scene.state.baseline_fps == fps


@pytest.mark.parametrize("values,expected", [
    ({"fps": 11}, set()),
    ({"fps": 10}, {"low_frame_rate"}),
    ({"width": 640, "height": 720}, {"low_resolution"}),
    ({"width": 641, "height": 720}, set()),
])
def test_half_initial_value_is_the_warning_threshold(values, expected):
    scene = Scene()
    for _ in range(12):
        scene.second()
    for _ in range(12):
        conditions = scene.second(**values)
    assert conditions == expected


def test_track_change_and_duplicate_frames():
    scene = Scene()
    for _ in range(12):
        assert not scene.second()
    assert scene.state.metrics["received_fps"] == 20
    count = len(scene.state.frames)
    scene.state.frame(frame(scene.seq))
    scene.state.frame(frame(scene.seq + 1, track="old"))
    assert len(scene.state.frames) == count
    # A new track replaces a stopped stream and gets its own baselines.
    scene.now += 3
    for _ in range(12):
        assert not scene.second(fps=8, width=960, height=540, track="new")
    assert scene.state.track_id == "new"
    assert scene.state.metrics["received_fps"] == 8


def test_no_frames_means_unknown_not_zero_fps_or_an_alert():
    scene = Scene()
    scene.now = 60
    assert not scene.state.evaluate()
    assert scene.state.metrics["received_fps"] is None
    assert scene.state.setup_video(scene.now) == (None, None)


def test_stopped_stream_is_a_rate_drop_after_the_stall_threshold():
    scene = Scene()
    scene.state.frame(frame(1))
    scene.now = 2.99
    assert not scene.state.evaluate()
    scene.now = 3
    assert scene.state.evaluate() == {"low_frame_rate"}
    assert scene.state.metrics["received_fps"] == 0
    assert scene.state.setup_video(scene.now) == (None, None)
    assert not scene.second()


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_repeat_interval_must_be_finite_and_positive(value):
    with pytest.raises(ValidationError):
        QualityConfig(repeat_interval_s=value)


@pytest.mark.parametrize("interval", [10, 7])
def test_independent_notification_clocks_and_recovery(interval):
    schedule = NotificationSchedule(interval)
    resolution, rate = {"low_resolution"}, {"low_frame_rate"}
    assert schedule.due(resolution, 10) == ["low_resolution"]
    schedule.sent(resolution, 10)
    assert schedule.due(resolution | rate, 15) == ["low_frame_rate"]
    schedule.sent(rate, 15)
    assert schedule.due(resolution | rate, 10 + interval) == ["low_resolution"]
    schedule.sent(resolution, 10 + interval)
    assert schedule.due(resolution | rate, 15 + interval) == ["low_frame_rate"]
    assert schedule.due(set(), 100) == []


def test_simultaneous_conditions_share_message_and_repeat_time():
    schedule = NotificationSchedule(10)
    conditions = {"low_resolution", "low_frame_rate"}
    assert set(schedule.due(conditions, 10)) == conditions
    schedule.sent(conditions, 10)
    assert not schedule.due(conditions, 19.999)
    assert set(schedule.due(conditions, 20)) == conditions
    schedule.sent(conditions, 20)
    assert not schedule.due(set(), 21)
    assert not schedule.due(conditions, 22)  # Flapping cannot bypass the cooldown.


@pytest.mark.parametrize("simultaneous", [False, True])
@pytest.mark.parametrize("drop_ratio", [0.5, 0.25])
async def test_monitor_submits_audio_on_independent_deadlines(simultaneous, drop_ratio):
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
        SimpleNamespace(capture_quality=AsyncMock()),
        QualityConfig(frame_rate_drop_ratio=drop_ratio, resolution_drop_ratio=drop_ratio),
        clock=lambda: now[0],
        sleep=sleep,
    )
    monitor.start(SimpleNamespace(publish=publish))
    await ticks.get()
    conditions = {"low_resolution", "low_frame_rate"} if simultaneous else {"low_resolution"}
    monitor.state.evaluate = Mock(side_effect=lambda: set(conditions))
    try:
        for value in [10, 15, 19, 20, 25]:
            now[0] = value
            if value == 15:
                conditions.add("low_frame_rate")
            resume.put_nowait(True)
            await ticks.get()
        assert [t for t, _ in spoken] == ([10, 20] if simultaneous else [10, 15, 20, 25])
        assert all(f"at least {100 * (1 - drop_ratio):.0f} percent" in text for _, text in spoken)
        if simultaneous:
            assert all("resolution" in text and "frame rate" in text for _, text in spoken)
        else:
            assert ["resolution" in text for _, text in spoken] == [True, False, True, False]
    finally:
        await monitor.stop()
    assert monitor.task is None


async def wait_for(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.005)


async def resolution_drop(demo, monitor):
    await demo.capture.on_quality_frame(frame(1, pid=monitor.pid))
    await wait_for(lambda: monitor.state.baseline_pixels is not None)
    await demo.capture.on_quality_frame(frame(2, width=320, height=240, pid=monitor.pid))


def install_quality(demo, *, enabled=True):
    demo.capture = CaptureAgent(
        demo.capture_backend, runtime=demo.runtime,
        quality_config=QualityConfig(enabled=enabled, poll_interval_s=0.01, repeat_interval_s=0.05),
    )


async def test_reconnect_and_participants_have_separate_frame_observations(demo):
    install_quality(demo)
    await demo.capture.receive(joined())
    await demo.capture.receive(joined("bob", "bob-session"))
    await demo.capture.on_quality_frame(frame(1))
    assert not demo.capture._connections["user"].quality.state.evaluate()
    await demo.capture.on_quality_frame(frame(2, width=320, height=240))
    assert demo.capture._connections["user"].quality.state.evaluate() == {"low_resolution"}
    assert not demo.capture._connections["bob"].quality.state.evaluate()
    await demo.capture.receive(joined(session="replacement"))
    assert "resolution is unavailable" in preflight.check_setup(demo.capture._connections["user"].quality)


async def test_audio_notifications_never_stop_recording_and_stop_drains_them(demo):
    install_quality(demo)
    await demo.capture.receive(joined())
    await speak(demo, "start capture")
    state = demo.capture_backend._sessions["user"]
    monitor = demo.capture._connections["user"].quality
    await resolution_drop(demo, monitor)
    await wait_for(lambda: sum(c.args[1].kind == "update" for c in demo.output.call_args_list) >= 2)
    assert demo.capture_backend.is_capturing("user")
    await speak(demo, "Narration while the resolution is low")
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
    await demo.capture.on_quality_frame(frame(1, pid=monitor.pid))
    await speak(demo, "start capture")
    state = demo.capture_backend._sessions["user"]
    await resolution_drop(demo, monitor)
    await wait_for(lambda: any("image resolution has dropped" in text for text in announcements(demo)))
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
    assert monitor.task is None
    assert packet(state)["status"] == "complete"
    assert not demo.capture_backend.is_capturing("user")


@pytest.mark.parametrize("command", ["Check setup", "Check set up.", "Check set-up!"])
async def test_setup_audio_and_capture_command_not_saved_as_narration(demo, command):
    install_quality(demo)
    await demo.capture.receive(joined())
    await speak(demo, command)
    assert any("frame rate is unavailable" in text for text in announcements(demo))
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


def test_setup_reports_latest_received_fps_and_resolution():
    scene = Scene()
    monitor = QualityFeedback("user", Mock(), QualityConfig(), clock=lambda: scene.now)
    text = preflight.check_setup(monitor)
    assert "frame rate is unavailable" in text and "resolution is unavailable" in text
    monitor.frame(frame(0))
    text = preflight.check_setup(monitor)
    assert "1280 by 720" in text and "frame rate is unavailable" in text
    for i in range(1, 11):
        scene.now = i / 20
        monitor.frame(frame(i, width=640, height=360))
    text = preflight.check_setup(monitor)
    assert "Received video is 20.0 frames per second." in text
    assert text.startswith("Received video is 20.0 frames per second. Image resolution is 640 by 360 pixels.")
    assert "Audio signal-to-noise ratio is unavailable" in text
    assert monitor.task is None
    scene.now = 5
    text = preflight.check_setup(monitor)
    assert "frame rate is unavailable" in text and "resolution is unavailable" in text


async def test_initial_alert_waits_for_start_acknowledgement(demo, monkeypatch):
    from test_voice_playback_state import _PID, _one_reply, _pipeline

    install_quality(demo)
    await demo.capture.receive(joined(_PID))
    monitor = demo.capture._connections[_PID].quality
    await demo.capture.on_quality_frame(frame(1, pid=monitor.pid))
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
        await resolution_drop(demo, monitor)
        await asyncio.wait_for(warning.wait(), 2)
        alert = next(c.args[1] for c in demo.output.call_args_list if c.args[1].kind == "update")
        assert not alert.urgent
        assert tts.calls == ["Capture started."]
        endpoint.release.set()
        await wait_for(lambda: any("image resolution has dropped" in text for text in tts.calls))
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
    replies = ["setup" if "frame rate is unavailable" in text else text for text in announcements(demo)]
    pair = ["Capture started.", "Capture stopped."]
    assert replies == pair + (["setup"] if earlier_setup else []) + pair + ["setup", "setup"]


@pytest.mark.parametrize("staggered", [False, True])
@pytest.mark.parametrize("recover", [False, True])
async def test_pending_media_alerts_replace_repeats_without_erasing_other_conditions(monkeypatch, staggered, recover):
    from pipecat.frames.frames import UserStartedSpeakingFrame, UserStoppedSpeakingFrame
    from test_voice_playback_state import _PID, _pipeline

    resolution_warning = (
        "The image resolution has dropped by at least 50 percent in pixel count from its initial value."
    )
    rate_warning = "The video frame rate has dropped by at least 50 percent from its initial value."
    now = [0.0]
    conditions = {"low_resolution", "low_frame_rate"} if recover else {"low_resolution"}
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
            if message.replace_pending:
                await voice.replace_pending_update(participant_id, message.text, source=source)
            elif message.kind == "update":
                await voice.enqueue_update(participant_id, message.text, source=source, urgent=message.urgent)
            elif message.kind == "withdraw":
                await voice.withdraw_updates(participant_id, source=source)

        monitor = QualityFeedback(
            _PID, SimpleNamespace(capture_quality=AsyncMock()), QualityConfig(),
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
            if recover:
                now[0] = 21
                conditions.discard("low_resolution")
                resume.put_nowait(True)
                await ticks.get()
            assert not tts.calls
            stopped = UserStoppedSpeakingFrame()
            stopped.transport_source = _PID
            await worker.queue_frame(stopped)
            await wait_for(lambda: bool(tts.calls) and not voice._floor)
            spoken = " ".join(tts.calls)
            assert spoken.count(resolution_warning) == int(not recover)
            assert spoken.count(rate_warning) == int(staggered or recover)
            assert spoken.count("Unrelated notice.") == 1
            # Replacing unheard copies must not disable reminders once heard.
            now[0] = 30
            resume.put_nowait(True)
            await ticks.get()
            await wait_for(lambda: " ".join(tts.calls).count(rate_warning if recover else resolution_warning) == 2)
            await wait_for(lambda: not voice._floor)
            spoken = " ".join(tts.calls)
            assert spoken.count(resolution_warning) == 2 * int(not recover)
            assert spoken.count(rate_warning) == 2 * int(staggered or recover)
            if staggered and not recover:
                calls = len(tts.calls)
                now[0] = 31
                conditions.discard("low_resolution")
                resume.put_nowait(True)
                await ticks.get()
                await asyncio.sleep(.02)
                assert len(tts.calls) == calls
        finally:
            await monitor.stop()


async def test_recovery_does_not_replay_an_already_heard_combined_warning(monkeypatch):
    from test_voice_playback_state import _PID, _pipeline

    now, conditions = [0.0], {"low_resolution", "low_frame_rate"}
    ticks, resume = asyncio.Queue(), asyncio.Queue()

    async def sleep(_):
        ticks.put_nowait(True)
        await resume.get()

    async with _pipeline(monkeypatch) as (voice, worker, tts, endpoint, _gate):
        endpoint.release.set()

        async def publish(_topic, message, *, participant_id, source):
            if message.replace_pending:
                await voice.replace_pending_update(participant_id, message.text, source=source)
            elif message.kind == "update":
                await voice.enqueue_update(participant_id, message.text, source=source)
            else:
                await voice.withdraw_updates(participant_id, source=source)

        monitor = QualityFeedback(_PID, SimpleNamespace(capture_quality=AsyncMock()), QualityConfig(),
                                  clock=lambda: now[0], sleep=sleep)
        monitor.state.evaluate = lambda: set(conditions)
        monitor.start(SimpleNamespace(publish=publish))
        try:
            await ticks.get()
            await wait_for(lambda: bool(tts.calls) and not voice._floor)
            calls = list(tts.calls)
            assert "resolution" in " ".join(calls) and "frame rate" in " ".join(calls)
            now[0] = 1
            conditions.remove("low_resolution")
            resume.put_nowait(True)
            await ticks.get()
            await asyncio.sleep(.02)
            assert tts.calls == calls
            now[0] = 10
            resume.put_nowait(True)
            await ticks.get()
            await wait_for(lambda: len(tts.calls) > len(calls) and not voice._floor)
            assert "resolution" not in " ".join(tts.calls[len(calls):])
        finally:
            await monitor.stop()


@pytest.mark.parametrize("partial_recovery", [False, True])
@pytest.mark.parametrize("applied", [False, True])
async def test_stop_during_warning_recovery_withdraws_the_owned_source(monkeypatch, partial_recovery, applied):
    from pipecat.frames.frames import UserStartedSpeakingFrame, UserStoppedSpeakingFrame
    from test_voice_playback_state import _PID, _pipeline

    conditions = {"low_resolution", "low_frame_rate"}
    ticks, resume = asyncio.Queue(), asyncio.Queue()
    publishing = asyncio.Event()
    block_recovery = False
    cleanup_sources = []

    async def sleep(_):
        ticks.put_nowait(True)
        await resume.get()

    async with _pipeline(monkeypatch) as (voice, worker, tts, endpoint, _gate):
        endpoint.release.set()
        started = UserStartedSpeakingFrame()
        started.transport_source = _PID
        await worker.queue_frame(started)
        await wait_for(lambda: _PID in voice._user_speaking)

        async def publish(_topic, message, *, participant_id, source):
            nonlocal block_recovery
            recovering = block_recovery and (message.replace_pending or message.kind == "withdraw")
            if recovering:
                block_recovery = False
                publishing.set()
                if not applied:
                    await asyncio.Event().wait()
            if message.replace_pending:
                await voice.replace_pending_update(participant_id, message.text, source=source)
            elif message.kind == "update":
                await voice.enqueue_update(participant_id, message.text, source=source)
            else:
                if not recovering:
                    cleanup_sources.append(source)
                await voice.withdraw_updates(participant_id, source=source)
            if recovering:
                await asyncio.Event().wait()

        monitor = QualityFeedback(_PID, SimpleNamespace(capture_quality=AsyncMock()), QualityConfig(),
                                  clock=lambda: 0, sleep=sleep)
        monitor.state.evaluate = lambda: set(conditions)
        monitor.start(SimpleNamespace(publish=publish))
        try:
            await asyncio.wait_for(ticks.get(), 2)
            assert len(voice._pending_updates[_PID]) == 1
            source = voice._pending_updates[_PID][0].source
            await voice.enqueue_update(_PID, "Unrelated notice.", source="other-producer")
            conditions.clear()
            if partial_recovery:
                conditions.add("low_frame_rate")
            block_recovery = True
            resume.put_nowait(True)
            await asyncio.wait_for(publishing.wait(), 2)
            await asyncio.wait_for(monitor.stop(), 2)
            assert cleanup_sources == [source]
            assert [update.source for update in voice._pending_updates[_PID]] == ["other-producer"]
            stopped = UserStoppedSpeakingFrame()
            stopped.transport_source = _PID
            await worker.queue_frame(stopped)
            await wait_for(lambda: bool(tts.calls) and not voice._floor)
            assert " ".join(tts.calls) == "Unrelated notice."
            assert monitor.task is None
        finally:
            await monitor.stop()
