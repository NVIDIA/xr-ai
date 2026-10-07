# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture-only SOP sample: camera boundaries and durable source packets."""

import asyncio
import importlib.util
import io
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image
from xr_ai_hub import DataMessage, ParticipantEvent, VideoTrackEvent
from xr_ai_hub._capture import CAPTURE_STT_TOPIC
from xr_ai_models import load_models_config
from xr_ai_tools.current_frame import ImageFrame
from xr_ai_tools.image import ImageRegistry

_SAMPLE = Path(__file__).resolve().parents[1] / "agent-samples/sop-sample"
sys.path.insert(0, str(_SAMPLE / "worker"))

from sop_sample_worker.config import load_config  # noqa: E402
from sop_sample_worker.lifecycle import CameraRecording  # noqa: E402
from sop_sample_worker.recorder import RecorderAgent, _export_narration  # noqa: E402


def test_config_and_cli():
    from device_io_hub.capture.config import load_capture_config

    worker = load_config(_SAMPLE / "yaml/worker.yaml")
    capture = load_capture_config(_SAMPLE / "yaml/media_capture.yaml")
    assert worker.media_capture_dir == Path(capture.out_dir)
    assert (capture.profile, capture.session_mode, capture.max_total_bytes) == ("raw", "explicit", 0)
    assert (worker.capture_fps, worker.caption_interval_s) == (2, 5)
    load_models_config(worker.models_config)
    spec = importlib.util.spec_from_file_location("sop_main", _SAMPLE / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._parser().parse_args(["--capture"]).capture
    assert module._parser().parse_args(["--replay", "any guide"]).replay == "any guide"
    for args in ([], ["--replay"], ["--capture", "--replay", "guide"]):
        with pytest.raises(SystemExit):
            module._parser().parse_args(args)
    assert [process.name for process in module.PROCESSES] == ["hub", "capture", "worker"]


def test_narration_is_all_user_speech_without_command_filtering(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"transcript": "speech.jsonl"}))
    rows = [
        {"source": "agent", "pts_us": 1, "text": "ignored"},
        {"source": "user", "pts_us": 2, "text": "Start recording"},
        {"source": "user", "pts_us": 3, "text": "Place the cup"},
        {"source": "user", "pts_us": 4, "text": "Stop recording"},
    ]
    (tmp_path / "speech.jsonl").write_text("\n".join(map(json.dumps, rows)))
    destination = tmp_path / "transcript.jsonl"
    assert _export_narration(manifest, destination) == 3
    assert [json.loads(line)["text"] for line in destination.read_text().splitlines()] == [
        "Start recording",
        "Place the cup",
        "Stop recording",
    ]
    assert _export_narration(manifest, destination) == 3


async def test_capture_voice_gate_does_not_interrupt_or_acknowledge_stop():
    from xr_ai_voicegate import VoiceGate, load_voice_gate_config

    config = load_voice_gate_config(_SAMPLE / "yaml/voice_gate.yaml")
    gate = VoiceGate(config, tts=Mock(), audio_sink=Mock())
    query, stop = AsyncMock(), AsyncMock()
    gate.bind(on_query=query, on_stop=stop)
    for text in ("stop", "stop recording", "start recording", "be quiet"):
        assert not gate._matches_partial_stop(text)
        await gate.feed("user", text)
    stop.assert_not_awaited()
    assert [call.args[1] for call in query.await_args_list] == [
        "stop",
        "stop recording",
        "start recording",
        "be quiet",
    ]


@pytest.fixture
async def demo(tmp_path, monkeypatch):
    from device_io_hub.capture._service import CaptureService
    from device_io_hub.capture.config import CaptureConfig

    # Use the real service's control, transcript, and finalization path; no
    # media frames enter its GPU encoder in this CPU regression fixture.
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", SimpleNamespace())
    service = CaptureService(
        CaptureConfig(
            out_dir=str(tmp_path / "captures"),
            session_mode="explicit",
            max_total_bytes=0,
        )
    )
    endpoint = Mock(send_return_data=AsyncMock(side_effect=service._on_agent_data))
    image = io.BytesIO()
    Image.new("RGB", (8, 8), "blue").save(image, format="JPEG")
    images = ImageRegistry()
    frames = Mock(
        execute=AsyncMock(
            return_value=ImageFrame(
                image=images.put(image.getvalue(), owner="user"),
                width=8,
                height=8,
                timestamp_us=time.time_ns() // 1000,
                sequence=1,
                participant_id="user",
                track_id="camera",
            )
        )
    )
    captioner = Mock(
        execute=AsyncMock(
            return_value=SimpleNamespace(
                available=True,
                text=json.dumps(
                    {"activity": "Setup", "phase": "Arrange", "caption": "A blue block", "delta": "Initial view"}
                ),
            )
        )
    )
    recorder = RecorderAgent(
        sessions_dir=tmp_path / "sessions",
        capture_endpoint=endpoint,
        media_capture_dir=tmp_path / "captures",
        current_frame=frames,
        images=images,
        query_image=captioner,
        capture_fps=2,
        caption_interval_s=5,
    )
    camera = CameraRecording(recorder)
    try:
        yield SimpleNamespace(
            recorder=recorder, camera=camera, endpoint=endpoint, service=service, root=tmp_path, frames=frames
        )
    finally:
        await camera.close()
        await service.stop()


def joined(pid="user", session="session"):
    return ParticipantEvent(pid, True, 1, participant_session_id=session)


def video(active=True, pid="user", track="camera", session="session"):
    return VideoTrackEvent(pid, track, active, time.time_ns() // 1000, session)


async def test_camera_not_connection_controls_multiple_recordings(demo):
    await demo.camera.receive(joined())
    assert not demo.recorder.is_recording("user")
    for _ in range(2):
        await demo.camera.receive(video())
        state = demo.recorder._sessions["user"]
        await demo.camera.receive(video())  # roster duplicate is not another packet
        await demo.endpoint.send_return_data(
            DataMessage(
                "user",
                CAPTURE_STT_TOPIC,
                time.time_ns() // 1000,
                b"Place the cup",
            )
        )
        async with asyncio.timeout(2):
            while state.caption_count == 0:
                await asyncio.sleep(0.01)
        await demo.camera.receive(video(False))
        packet = json.loads((state.directory / "packet.json").read_text())
        assert packet["status"] == packet["narration_status"] == "complete"
        assert packet["counts"] == {"frames": 1, "transcripts": 1, "captions": 1}
        assert (state.directory / "summary.md").is_file()
        assert list((state.directory / "frames").glob("*.jpg"))
        assert Path(packet["media_capture"]["manifest"]).is_file()
        assert not demo.recorder.is_recording("user")
    assert len(list((demo.root / "sessions").iterdir())) == 2


async def test_disconnect_and_old_events_cannot_restart_recording(demo):
    await demo.camera.receive(joined())
    await demo.camera.receive(video())
    await demo.camera.receive(ParticipantEvent("user", False, 2, participant_session_id="session"))
    await demo.camera.receive(video())
    assert not demo.recorder.is_recording("user")
    await demo.camera.receive(joined(session="new"))
    await demo.camera.receive(video(session="new"))
    await demo.camera.receive(video(False))  # stale old connection
    assert demo.recorder.is_recording("user")


async def test_tracks_and_participants_are_independent(demo):
    await demo.camera.receive(joined())
    await demo.camera.receive(joined("other"))
    await demo.camera.receive(video())
    await demo.camera.receive(video(track="second"))
    await demo.camera.receive(video(pid="other"))
    await demo.camera.receive(video(False))
    assert demo.recorder.is_recording("user")
    await demo.camera.receive(video(False, track="second"))
    assert not demo.recorder.is_recording("user")
    assert demo.recorder.is_recording("other")


async def test_fast_restart_waits_for_previous_manifest_even_if_handler_cancelled(demo, monkeypatch):
    await demo.camera.receive(joined())
    await demo.camera.receive(video())
    old = demo.recorder._sessions["user"]
    entered, release = asyncio.Event(), asyncio.Event()
    original = demo.recorder._wait_for_media

    async def wait(state):
        entered.set()
        await release.wait()
        await original(state)

    monkeypatch.setattr(demo.recorder, "_wait_for_media", wait)
    finishing = asyncio.create_task(demo.camera.receive(video(False)))
    await asyncio.wait_for(entered.wait(), 2)
    finishing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await finishing
    restarting = asyncio.create_task(demo.camera.receive(video()))
    try:
        await asyncio.sleep(0.02)
        assert not restarting.done()
        assert demo.recorder._sessions["user"] is old
    finally:
        release.set()
        await asyncio.wait_for(restarting, 2)
    assert demo.recorder._sessions["user"] is not old
    assert json.loads((old.directory / "packet.json").read_text())["status"] == "complete"


async def test_slow_media_finalization_keeps_restart_blocked(demo, monkeypatch):
    import sop_sample_worker.recorder as module

    monkeypatch.setattr(module, "_MEDIA_FINALIZE_WARNING_S", 0.02)
    await demo.camera.receive(joined())
    await demo.camera.receive(video())
    old = demo.recorder._sessions["user"]
    entered, release = asyncio.Event(), asyncio.Event()
    original = demo.service._finish_session_owned

    async def delayed(*args, **kwargs):
        entered.set()
        await release.wait()
        await original(*args, **kwargs)

    monkeypatch.setattr(demo.service, "_finish_session_owned", delayed)
    finishing = asyncio.create_task(demo.camera.receive(video(False)))
    await asyncio.wait_for(entered.wait(), 2)
    restarting = asyncio.create_task(demo.camera.receive(video()))
    try:
        await asyncio.sleep(0.15)  # Exceed the accelerated manifest deadline.
        assert not finishing.done()
        assert not restarting.done()
        assert demo.recorder._sessions["user"] is old
        assert len(list((demo.root / "sessions").iterdir())) == 1
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(finishing, restarting), 2)

    new = demo.recorder._sessions["user"]
    assert new is not old
    assert demo.service._recorder.has_session("user")
    await demo.endpoint.send_return_data(
        DataMessage("user", CAPTURE_STT_TOPIC, time.time_ns() // 1000, b"Second recording")
    )
    await demo.camera.receive(video(False))
    packet = json.loads((new.directory / "packet.json").read_text())
    assert packet["status"] == packet["narration_status"] == "complete"
    assert packet["counts"]["transcripts"] == 1
    manifest = Path(packet["media_capture"]["manifest"])
    assert manifest.is_relative_to(new.media_directory)
    assert json.loads(manifest.read_text())["target"] == new.session_id
    assert json.loads((new.directory / "transcript.jsonl").read_text())["text"] == "Second recording"


async def test_stop_during_caption_append_seals_consistent_hierarchy(demo, monkeypatch):
    import sop_sample_worker.recorder as module

    entered, release = threading.Event(), threading.Event()
    original = module._append_jsonl

    def delayed(path, record):
        if path.name == "captions.jsonl":
            entered.set()
            assert release.wait(5)
        original(path, record)

    monkeypatch.setattr(module, "_append_jsonl", delayed)
    await demo.camera.receive(joined())
    await demo.camera.receive(video())
    state = demo.recorder._sessions["user"]
    finishing = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        finishing = asyncio.create_task(demo.camera.receive(video(False)))
        async with asyncio.timeout(2):
            while not any(task.cancelled() for task in state.tasks):
                await asyncio.sleep(0.001)
        assert not finishing.done()
    finally:
        release.set()
        if finishing is not None:
            await asyncio.wait_for(finishing, 2)

    captions = [json.loads(line) for line in (state.directory / "captions.jsonl").read_text().splitlines()]
    packet = json.loads((state.directory / "packet.json").read_text())
    assert packet["status"] == "complete"
    assert packet["counts"]["captions"] == len(captions) == 1
    assert len(packet["hierarchy"]) == 1
    activity = packet["hierarchy"][0]
    assert activity["caption_count"] == 1
    assert activity["phases"][0]["summary"] == captions[0]["caption"]
    summary = (state.directory / "summary.md").read_text()
    assert "Waiting for the first visual caption" not in summary
    assert captions[0]["caption"] in summary


async def test_restart_retries_if_capture_is_closing_after_manifest(demo, monkeypatch):
    await demo.camera.receive(joined())
    await demo.camera.receive(video())
    old = demo.recorder._sessions["user"]
    entered, release = asyncio.Event(), asyncio.Event()
    original = demo.service._render

    async def delayed(bundle):
        entered.set()
        await release.wait()
        await original(bundle)

    # end_session has published the manifest, but the service still rejects starts.
    monkeypatch.setattr(demo.service, "_render", delayed)
    finishing = asyncio.create_task(demo.camera.receive(video(False)))
    await asyncio.wait_for(entered.wait(), 2)
    restarting = asyncio.create_task(demo.camera.receive(video()))
    try:
        await asyncio.wait_for(finishing, 2)
        await asyncio.sleep(0.1)
        assert old.media_manifest is not None
        assert not restarting.done()
        # A participant waiting for acceptance must not hold the global lock.
        await demo.camera.receive(joined("other"))
        await asyncio.wait_for(demo.camera.receive(video(pid="other")), 2)
        assert demo.service._recorder.has_session("other")
    finally:
        release.set()
        await asyncio.wait_for(restarting, 2)
    new = demo.recorder._sessions["user"]
    assert new is not old
    assert demo.service._recorder.has_session("user")
    await demo.camera.receive(video(False))
    packet = json.loads((new.directory / "packet.json").read_text())
    assert packet["status"] == "complete"
    assert Path(packet["media_capture"]["manifest"]).is_relative_to(new.media_directory)


async def test_shutdown_drains_inflight_writes_before_final_packet(demo, monkeypatch):
    import sop_sample_worker.recorder as module

    await demo.camera.receive(joined())
    await demo.camera.receive(video())
    state = demo.recorder._sessions["user"]
    entered, release = threading.Event(), threading.Event()
    original = module._atomic_json

    def write(path, packet):
        if packet["status"] == "recording":
            entered.set()
            assert release.wait(5)
        original(path, packet)

    monkeypatch.setattr(module, "_atomic_json", write)

    async def write_view():
        async with state.lock:
            await demo.recorder._write_packet(state)

    state.tasks.append(asyncio.create_task(write_view()))
    assert await asyncio.to_thread(entered.wait, 2)
    closing = asyncio.create_task(demo.camera.close())
    try:
        await asyncio.sleep(0.02)
        assert not closing.done()
    finally:
        release.set()
        await asyncio.wait_for(closing, 2)
    assert json.loads((state.directory / "packet.json").read_text())["status"] == "complete"
    await demo.camera.receive(video())
    assert not demo.recorder.is_recording("user")


async def test_app_finalizes_after_voice_shutdown_before_capture_endpoint_closes(demo, monkeypatch):
    from dataclasses import replace

    from sop_sample_worker import app
    from xr_ai_voice._session import _VoiceSession

    receiving = asyncio.Event()
    receiver_cancelled = asyncio.Event()
    capture_closed = False
    voice_closed = False
    callbacks = {}

    async def send(message):
        assert not capture_closed
        await demo.service._on_agent_data(message)

    async def receive_forever():
        receiving.set()
        try:
            await asyncio.Future()
        finally:
            receiver_cancelled.set()

    capture_endpoint = SimpleNamespace(
        on_participant=lambda cb: callbacks.update(participant=cb),
        on_video_track=lambda cb: callbacks.update(video=cb),
        run=receive_forever,
        wait_until_running=receiving.wait,
        send_return_data=AsyncMock(side_effect=send),
        # ProcessorEndpoint.stop does not wake its blocked ZMQ receive.
        stop=Mock(),
    )

    def close_capture():
        nonlocal capture_closed
        packets = list((demo.root / "app-sessions/sessions").glob("*/packet.json"))
        assert len(packets) == 1
        packet = json.loads(packets[0].read_text())
        assert packet["status"] == packet["narration_status"] == "complete"
        assert voice_closed
        capture_closed = True

    capture_endpoint.close = close_capture
    monkeypatch.setattr(app, "ProcessorEndpoint", lambda **kwargs: capture_endpoint)
    monkeypatch.setattr(app, "CurrentFrameTool", lambda **kwargs: demo.frames)
    monkeypatch.setattr(app, "setup_logging", lambda *args: None)
    for name in ("make_stt", "make_tts", "make_vlm"):
        monkeypatch.setattr(app, name, lambda *args: Mock(health=AsyncMock(return_value=True), close=AsyncMock()))

    def shutdown_voice():
        nonlocal voice_closed
        voice_closed = True

    transport = SimpleNamespace(endpoint=Mock(), shutdown=shutdown_voice)
    real_voice = app.VoiceAgent
    monkeypatch.setattr(app, "VoiceAgent", lambda **kwargs: real_voice(transport=transport, **kwargs))

    async def run_session(self, *args, **kwargs):
        assert self.vad.stop_probe_after_s == 0
        await callbacks["participant"](joined())
        await callbacks["video"](video())
        # End the voice run while the participant and camera are still active.
        # VoiceAgent's real finally path closes only its own transport.
        await send(DataMessage("user", CAPTURE_STT_TOPIC, time.time_ns() // 1000, b"Place a cup"))

    monkeypatch.setattr(_VoiceSession, "run", run_session)
    config = replace(
        load_config(_SAMPLE / "yaml/worker.yaml"),
        artifacts_dir=demo.root / "app-sessions",
        media_capture_dir=demo.root / "captures",
    )
    task = asyncio.create_task(app.run_app(config))
    try:
        done, _ = await asyncio.wait({task}, timeout=2)
        assert task in done, "capture receiver blocked application shutdown"
        await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert capture_closed
    assert receiver_cancelled.is_set()
