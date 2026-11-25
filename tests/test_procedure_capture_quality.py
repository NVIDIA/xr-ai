# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise capture quality from metadata through packet and voice ownership."""

# Pytest discovers the imported demo fixture.
# ruff: noqa: F811
import asyncio
import io
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image, ImageDraw, ImageFilter
from pipecat.frames.frames import TextFrame
from pydantic import ValidationError
from test_procedure_capture_guide import demo, joined, packet, speak  # noqa: F401
from xr_ai_hub import DataMessage, FrameSignal, ParticipantEvent, PixelFormat
from xr_ai_tools.current_frame import ImageFrame
from xr_ai_voice._processors.io import _VoiceIOProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-samples/procedure-capture-guide/worker"))
from procedure_capture_guide_worker.capture import quality as quality_module  # noqa: E402
from procedure_capture_guide_worker.capture.agent import CaptureAgent  # noqa: E402
from procedure_capture_guide_worker.capture.quality import (  # noqa: E402
    TELEMETRY_TOPIC,
    QualityConfig,
    QualityState,
    Telemetry,
)


def telemetry(**updates):
    return Telemetry.model_validate({"version": 1, "token": "a" * 32, "sequence": 1, "mode": "live",
                                    "profile": "camera-1", "track_id": "camera", "streaming": True,
                                    "quality": "good", **updates})


def frame(seq, width=1280, height=720, track="camera"):
    return SimpleNamespace(seq=seq, width=width, height=height, track_id=track, participant_id="user")


class Scene:
    def __init__(self):
        self.now = 0.0
        self.seq = 0
        self.state = QualityState(QualityConfig(), 2, clock=lambda: self.now)

    def second(self, fps=20, width=1280, height=720, *, offsets=None, send_telemetry=True, **updates):
        if send_telemetry:
            self.state.telemetry_received(telemetry(**updates))
        start = self.now
        for offset in (offsets if offsets is not None else [i / fps for i in range(fps)]):
            self.now = start + offset
            self.seq += 1
            self.state.frame(frame(self.seq, width, height))
        self.now = start + 1
        return self.state.evaluate()


# Durations are simulated seconds with the production policy's default values.
# Each row has a healthy baseline, an impairment, and healthy recovery. Raw RTT
# and loss remain diagnostic inputs: LiveKit's quality grade owns network scoring.
_IMPAIRMENTS = [
    ("healthy", {}, 20, set(), None),
    ("poor_network", {"quality": "poor", "uplink_loss_fraction": 0.15,
                      "uplink_rtt_ms": 900.0}, 20, {"poor_network"}, 6),
    ("brief_network_dip", {"quality": "poor"}, 4, set(), None),
    ("lost_quality_with_media", {"quality": "lost"}, 20, {"poor_network"}, 6),
    ("ninety_percent_frame_loss", {"fps": 2}, 20, {"low_frame_rate"}, 8),
    ("resolution_720_to_360", {"width": 640, "height": 360}, 20, {"low_resolution"}, 6),
    ("brief_resolution_dip", {"width": 320, "height": 180}, 4, set(), None),
    ("bandwidth_collapse", {"fps": 2, "width": 320, "height": 180,
                            "quality": "poor", "limitation": "bandwidth"}, 20,
     {"poor_network", "low_frame_rate", "low_resolution"}, 6),
    ("video_only_outage", {"fps": 0}, 20, {"video_stalled"}, 3),
    ("brief_video_dropout", {"fps": 0}, 2, set(), None),
    ("bursty_arrivals", {"offsets": [0.8 + i / 200 for i in range(20)]}, 20, set(), None),
    ("portrait_rotation", {"width": 720, "height": 1280}, 20, set(), None),
    ("intentional_camera_off", {"fps": 0, "mode": "off", "streaming": False,
                                "track_id": ""}, 20, set(), None),
    ("on_demand_between_images", {"fps": 0, "mode": "on-demand", "streaming": False,
                                   "track_id": ""}, 20, set(), None),
    ("missing_telemetry", {"send_telemetry": False}, 20, set(), None),
    ("high_rtt_without_bad_grade", {"uplink_rtt_ms": 1200.0}, 20, set(), None),
]


@pytest.mark.parametrize("name,impairment,duration,expected,first_alert", _IMPAIRMENTS,
                         ids=[case[0] for case in _IMPAIRMENTS])
def test_impairment_timeline(tmp_path, name, impairment, duration, expected, first_alert):
    scene = Scene()
    trace = []
    for phase, length, values in (("baseline", 12, {}), ("impaired", duration, impairment),
                                  ("recovered", 25, {})):
        for second in range(1, length + 1):
            conditions = scene.second(**values)
            trace.append({"phase": phase, "second": second, "elapsed_s": scene.now,
                          "conditions": sorted(conditions), "metrics": scene.state.metrics})
    result = {"scenario": name, "policy": scene.state.config.model_dump(), "trace": trace}
    (tmp_path / "quality-timeline.json").write_text(json.dumps(result, indent=2))
    impaired = [row for row in trace if row["phase"] == "impaired"]
    assert set(impaired[-1]["conditions"]) == expected
    first = next((row["second"] for row in impaired if row["conditions"]), None)
    assert first == first_alert
    if not expected:
        assert all(not row["conditions"] for row in trace)
    assert not trace[-1]["conditions"], "recovery must clear all active conditions"


def test_low_quality_from_start_and_repeated_brief_fluctuations():
    scene = Scene()
    for _ in range(20):
        conditions = scene.second(fps=1, width=320, height=180)
    assert conditions == {"low_frame_rate", "low_resolution"}
    scene = Scene()
    for _ in range(12):
        scene.second()
    for _ in range(8):
        for _ in range(3):
            assert not scene.second(quality="poor", width=640, height=360)
        for _ in range(3):
            assert not scene.second()


def test_partial_telemetry_loss_and_short_video_delivery_gaps():
    scene = Scene()
    for _ in range(12):
        scene.second()
    # Drop two out of three telemetry messages; the remaining samples are
    # recent enough to establish sustained poor quality without guessing.
    for second in range(15):
        conditions = scene.second(send_telemetry=second % 3 == 0, quality="poor")
    assert conditions == {"poor_network"}
    for _ in range(10):
        conditions = scene.second(send_telemetry=False)
    assert not conditions
    scene = Scene()
    for _ in range(12):
        scene.second()
    # Half the frames arrive in alternate-second bursts. Rolling FPS fluctuates
    # around the relative boundary, rather than staying below it for five seconds.
    for second in range(30):
        assert not scene.second(fps=20 if second % 2 else 0)


def test_live_video_degradation_recovery_and_rotation():
    scene = Scene()
    for _ in range(12):
        assert not scene.second()
    assert scene.state.baseline_fps == 20
    for _ in range(12):
        conditions = scene.second(fps=2, width=640, height=360)
    assert conditions == {"low_frame_rate", "low_resolution"}
    for _ in range(16):
        conditions = scene.second(width=720, height=1280)
    assert not conditions


def test_stall_requires_expected_stream_and_fresh_telemetry():
    scene = Scene()
    scene.second()
    for _ in range(4):
        result = scene.second(fps=0)
    assert "video_stalled" in result
    # Missing telemetry is unknown, not proof that a camera remains enabled.
    scene.now += 5
    assert not scene.state.evaluate()
    for mode in ("off", "on-demand"):
        for _ in range(20):
            assert not scene.second(fps=0, mode=mode, streaming=False, track_id="")


def test_profile_change_rebaselines_and_repeated_metadata_is_not_new_video():
    scene = Scene()
    for _ in range(12):
        scene.second()
    for _ in range(12):
        assert not scene.second(fps=8, width=960, height=540, profile="camera-2")
    before = len(scene.state.frames)
    for _ in range(50):
        scene.state.frame(frame(scene.seq))
        scene.state.frame(frame(scene.seq + 1, track="old-camera"))
    assert len(scene.state.frames) == before


def test_network_transient_debounce_recovery_and_unknown():
    scene = Scene()
    for _ in range(4):
        assert not scene.second(quality="poor")
    assert not scene.second(quality="good")
    for _ in range(7):
        result = scene.second(quality="poor")
    assert "poor_network" in result
    for _ in range(3):
        assert "poor_network" in scene.second(quality="good")
    assert not scene.second(quality="unknown")


def test_on_demand_reports_failed_requests_without_fps_warnings():
    scene = Scene()
    for _ in range(7):
        scene.state.telemetry_received(telemetry(mode="on-demand", streaming=False, track_id=""))
        scene.state.sample(None)
        scene.now += 1
        result = scene.state.evaluate()
    assert result == {"image_unavailable"}
    for _ in range(12):
        scene.state.telemetry_received(telemetry(mode="on-demand", streaming=False, track_id=""))
        scene.state.sample(frame(1))
        scene.now += 1
        result = scene.state.evaluate()
    assert not result


@pytest.mark.parametrize("value", [{"degrade_s": float("nan")}, {"enabled": "false"},
                                   {"drop_ratio": 1}, {"min_short_edge": -1}, {"typo": True}])
def test_invalid_quality_configuration_rejected(value):
    with pytest.raises(ValidationError):
        QualityConfig.model_validate(value)


async def wait_for(predicate):
    async with asyncio.timeout(4):
        while not predicate():
            await asyncio.sleep(0.01)


def install_quality(demo):
    endpoint = SimpleNamespace(send_return_data=AsyncMock())
    demo.capture = CaptureAgent(demo.capture_backend, quality_config=QualityConfig(degrade_s=0.01), endpoint=endpoint)
    demo.capture_backend.on_sample = demo.capture.on_quality_sample
    return endpoint


async def test_telemetry_lease_rejects_other_connections_and_reordered_data(demo):
    install_quality(demo)
    await demo.capture.receive(joined())
    old = demo.capture._connections["user"].quality
    async def send(token, sequence, pid="user"):
        value = telemetry(token=token, sequence=sequence)
        await demo.capture.on_quality_data(DataMessage(pid, TELEMETRY_TOPIC, 1, value.model_dump_json().encode()))
    await send("wrong", 90)
    assert old.latest is None
    await send(old.token, 2)
    await send(old.token, 1)
    assert old.latest.sequence == 2
    await demo.capture.receive(joined(session="new-session"))
    new = demo.capture._connections["user"].quality
    await send(old.token, 3)
    await send(new.token, 4, pid="another-user")
    assert new.latest is None
    await demo.capture.on_quality_data(DataMessage("user", TELEMETRY_TOPIC, 1, b"not-json"))
    assert new.latest is None


async def test_warning_is_withdrawn_on_stop_and_cannot_leak_into_next_capture(demo):
    install_quality(demo)
    await demo.capture.receive(joined())
    io = _VoiceIOProcessor(AsyncMock())
    io._user_speaking.add("user")
    io.push_frame = AsyncMock()

    async def output(topic, message, *, participant_id, source=None):
        if message.kind == "update" and source and source.startswith("capture-quality:"):
            await io.enqueue_update(participant_id, message.text, source=source, urgent=message.urgent)
        elif message.kind == "withdraw" and source:
            await io.withdraw_updates(participant_id, source=source)

    demo.output.side_effect = output
    await speak(demo, "start capture")
    recording = demo.capture_backend._sessions["user"]
    monitor = demo.capture._connections["user"].quality
    now = [0.0]
    monitor.state.clock = lambda: now[0]
    monitor.state.telemetry_received(telemetry(mode="off", streaming=False, quality="poor"))
    monitor.state.evaluate()
    now[0] = 1
    await wait_for(lambda: bool(io._pending_updates.get("user")))
    await speak(demo, "Place the wheel on the axle")
    await speak(demo, "stop capture")
    assert not io._pending_updates.get("user")
    assert monitor.task is None
    io._user_speaking.clear()
    await io._start_next("user")
    io.push_frame.assert_not_awaited()
    events = [json.loads(line) for line in (recording.directory / "quality.jsonl").read_text().splitlines()]
    assert any(e["kind"] == "warning_submitted" for e in events)
    assert events[-1]["kind"] == "monitor_stopped"
    assert packet(recording)["files"]["quality"] == "quality.jsonl"
    assert packet(recording)["status"] == "complete"
    assert "connection is unstable" not in (recording.directory / "transcript.jsonl").read_text()
    await speak(demo, "start capture")
    assert monitor.task is not None and not monitor.task.done()
    assert not monitor.state.active
    await speak(demo, "stop capture")


async def test_disconnect_drains_monitor_before_sealing_packet(demo):
    install_quality(demo)
    await demo.capture.receive(joined())
    await speak(demo, "start capture")
    monitor = demo.capture._connections["user"].quality
    recording = demo.capture_backend._sessions["user"]
    await demo.capture.receive(ParticipantEvent("user", False, 10, participant_session_id="session"))
    assert monitor.task is None
    assert packet(recording)["status"] == "complete"


def test_browser_capture_quality_behavior():
    path = Path(__file__).parent / "javascript/capture_quality.test.mjs"
    result = subprocess.run(["node", str(path)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


async def test_stop_accepted_during_startup_never_arms_quality_audio(demo):
    endpoint = install_quality(demo)
    await demo.capture.receive(joined())
    entered, release = asyncio.Event(), asyncio.Event()
    original = demo.capture_backend.start_capture

    async def delayed(*args, **kwargs):
        entered.set()
        await release.wait()
        await original(*args, **kwargs)

    demo.capture_backend.start_capture = delayed
    await speak(demo, "start capture", wait=False)
    await entered.wait()
    await speak(demo, "stop capture", wait=False)
    release.set()
    await demo.capture._tails["user"]
    monitor = demo.capture._connections["user"].quality
    assert monitor.task is None
    assert all(not json.loads(call.args[0].data)["active"] for call in endpoint.send_return_data.call_args_list)
    packets = list((demo.root / "sessions").glob("*/packet.json"))
    assert json.loads(packets[0].read_text())["status"] == "complete"


async def test_recovery_withdraws_queued_warning_without_stopping_capture(demo):
    install_quality(demo)
    await demo.capture.receive(joined())
    await speak(demo, "start capture")
    monitor = demo.capture._connections["user"].quality
    now = [0.0]
    monitor.state.clock = lambda: now[0]
    monitor.state.telemetry_received(telemetry(mode="off", streaming=False, quality="poor"))
    monitor.state.evaluate()
    now[0] = 1
    await wait_for(lambda: any(call.args[1].kind == "update" for call in demo.output.call_args_list))
    demo.output.reset_mock()
    monitor.state.telemetry_received(telemetry(mode="off", streaming=False, quality="unknown"))
    await wait_for(lambda: any(call.args[1].kind == "withdraw" for call in demo.output.call_args_list))
    assert demo.capture_backend.is_capturing("user")
    await speak(demo, "stop capture")


def test_snapshot_sampling_is_not_used_as_incoming_fps():
    scene = Scene()
    for _ in range(12):
        scene.state.sample(frame(1))  # Deliberately one saved snapshot per second.
        assert not scene.second(fps=20)
    assert scene.state.metrics["received_fps"] == 20
    assert scene.state.metrics["saved_fps"] < 2


async def test_video_stall_escalates_without_waiting_for_network_warning_cooldown(demo):
    install_quality(demo)
    await demo.capture.receive(joined())
    await speak(demo, "start capture")
    monitor = demo.capture._connections["user"].quality
    now = [0.0]
    monitor.state.clock = lambda: now[0]
    monitor.state.telemetry_received(telemetry(quality="poor"))
    monitor.state.frame(frame(1))
    monitor.state.evaluate()
    now[0] = 1
    await wait_for(lambda: any(call.args[1].kind == "update" for call in demo.output.call_args_list))
    assert not next(call.args[1] for call in demo.output.call_args_list if call.args[1].kind == "update").urgent
    now[0] = 4
    monitor.state.telemetry_received(telemetry(quality="poor"))
    await wait_for(lambda: any(call.args[1].kind == "update" and call.args[1].urgent
                              for call in demo.output.call_args_list))
    assert demo.capture_backend.is_capturing("user")
    await speak(demo, "stop capture")


@pytest.mark.parametrize("condition,changes", [
    ("poor_network", {"quality": "poor", "uplink_loss_fraction": 0.2, "uplink_rtt_ms": 800.0}),
    ("low_frame_rate", {"fps": 2}),
    ("low_resolution", {"width": 640, "height": 360}),
    ("video_stalled", {"fps": 0}),
    ("image_unavailable", {"mode": "on-demand", "streaming": False, "track_id": "", "fail": True}),
])
async def test_impairments_through_capture_voice_and_packet(demo, monkeypatch, condition, changes):
    """Replay over a minute per fault without replacing the monitor or voice queue.

    Only the monitor clock/sleep is stepped. Real participant callbacks, schema
    validation, warning selection, voice processing, and packet writes still run.
    STT, models, network transport, and the physical speaker are outside this test.
    """
    now = [0.0]
    sleeping, resume = asyncio.Queue(), asyncio.Queue()

    async def sleep(_delay):
        await sleeping.put(True)
        await resume.get()

    monkeypatch.setattr(quality_module, "time", SimpleNamespace(monotonic=lambda: now[0], time_ns=time.time_ns))
    monkeypatch.setattr(quality_module, "asyncio", SimpleNamespace(
        create_task=asyncio.create_task, gather=asyncio.gather, timeout=asyncio.timeout, sleep=sleep,
    ))
    endpoint = SimpleNamespace(send_return_data=AsyncMock())
    demo.capture = CaptureAgent(demo.capture_backend, quality_config=QualityConfig(), endpoint=endpoint)
    demo.capture_backend.on_sample = demo.capture.on_quality_sample
    voice = _VoiceIOProcessor(AsyncMock())
    voice.push_frame = AsyncMock()
    voice._user_speaking.add("user")

    async def output(topic, message, *, participant_id, source=None):
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
        recording = demo.capture_backend._sessions["user"]
        monitor.state.clock = lambda: now[0]
        await asyncio.wait_for(sleeping.get(), 4)
        sequence = 0
        trace = []
        for phase, length, values in (("baseline", 12, {}), ("impaired", 80, changes), ("recovery", 25, {})):
            for second in range(1, length + 1):
                fps, width, height = values.get("fps", 20), values.get("width", 1280), values.get("height", 720)
                data = telemetry(token=monitor.token, sequence=int(now[0]) + 1,
                                 **{k: v for k, v in values.items() if k not in {"fps", "width", "height", "fail"}})
                await demo.capture.on_quality_data(DataMessage("user", TELEMETRY_TOPIC, 1,
                                                               data.model_dump_json().encode()))
                start = now[0]
                for i in range(fps):
                    now[0] = start + i / fps
                    sequence += 1
                    await demo.capture.on_quality_frame(FrameSignal(
                        slot=0, seq=sequence, pts_us=int(now[0] * 1e6), width=width, height=height,
                        fmt=PixelFormat.RGB24, data_sz=width * height * 3, participant_id="user", track_id="camera",
                    ))
                if values.get("fail"):
                    demo.capture.on_quality_sample("user", None)
                now[0] = start + 1
                resume.put_nowait(True)
                await asyncio.wait_for(sleeping.get(), 4)
                assert demo.capture_backend.is_capturing("user")
                trace.append({"phase": phase, "second": second, "elapsed_s": now[0],
                              "conditions": sorted(monitor.state.active)})
                # Ordinary feedback must wait until the narrator pauses; a stall
                # is urgent and must reach the speech processor even during speech.
                if phase == "impaired" and second == 10:
                    delivered = [call.args[0] for call in voice.push_frame.call_args_list
                                 if isinstance(call.args[0], TextFrame)]
                    assert bool(delivered) == (condition == "video_stalled")
                    voice._user_speaking.clear()
                    await voice._start_next("user")
                    await wait_for(lambda: any(isinstance(call.args[0], TextFrame)
                                              for call in voice.push_frame.call_args_list))
        assert not monitor.state.active
        assert not voice._pending_updates.get("user")
        await speak(demo, "Place the part on the table")
        await speak(demo, "stop capture")
        events = [json.loads(line) for line in (recording.directory / "quality.jsonl").read_text().splitlines()]
        warnings = [event for event in events if event["kind"] == "warning_submitted"]
        assert [event["conditions"] for event in warnings] == [[condition], [condition]]
        assert packet(recording)["status"] == "complete"
        transcript = (recording.directory / "transcript.jsonl").read_text().splitlines()
        narration = [json.loads(line)["text"] for line in transcript]
        assert narration == ["Place the part on the table"]
        assert monitor.task is None
        (demo.root / "quality-delivery.json").write_text(json.dumps(
            {"condition": condition, "trace": trace, "warnings": warnings, "packet_status": "complete"}, indent=2,
        ))
    finally:
        await demo.capture.close()
        await voice._shutdown()


async def test_saved_image_dimensions_and_pixel_quality_limit(demo):
    """Exercise JPEG saving; blur/darkness alone are not metadata quality signals."""
    endpoint = SimpleNamespace(send_return_data=AsyncMock())
    demo.capture = CaptureAgent(demo.capture_backend, quality_config=QualityConfig(), endpoint=endpoint)
    demo.capture_backend.on_sample = demo.capture.on_quality_sample
    original = Image.new("RGB", (1280, 720), "white")
    drawing = ImageDraw.Draw(original)
    for x in range(0, 1280, 40):
        drawing.rectangle((x, 240, x + 20, 480), fill="black")
    variants = {
        "sharp": original,
        "blurred": original.filter(ImageFilter.GaussianBlur(20)),
        "dark": Image.new("RGB", original.size, (2, 2, 2)),
        "low_resolution": original.resize((320, 180)),
    }
    try:
        await demo.capture.receive(joined())
        for sequence, (name, image) in enumerate(variants.items(), 1):
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            demo.frames.execute.return_value = ImageFrame(
                image=demo.capture_backend._images.put(buffer.getvalue(), owner="user"),
                width=image.width, height=image.height, timestamp_us=time.time_ns() // 1000,
                sequence=sequence, participant_id="user", track_id="camera",
            )
            if sequence == 1:
                await speak(demo, "start capture")
                recording = demo.capture_backend._sessions["user"]
            await wait_for(lambda: recording.latest_frame is not None and recording.latest_frame.sequence == sequence)
            saved = recording.directory / recording.latest_frame.path
            with Image.open(saved) as stored:
                assert stored.size == image.size
            scene = Scene()
            for _ in range(12):
                conditions = scene.second(width=image.width, height=image.height)
            assert conditions == ({"low_resolution"} if name == "low_resolution" else set())
        await speak(demo, "stop capture")
        assert packet(recording)["counts"]["frames"] == len(variants)
        assert packet(recording)["status"] == "complete"
    finally:
        await demo.capture.close()
