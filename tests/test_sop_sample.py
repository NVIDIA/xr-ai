# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Voice-controlled SOP recording and independent optional shared capture."""

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
from xr_ai_hub import AudioChunk, DataMessage, FrameUnavailable, ParticipantEvent
from xr_ai_hub._capture import CAPTURE_STT_TOPIC
from xr_ai_models import load_models_config
from xr_ai_runtime import AgentRuntime
from xr_ai_tools.current_frame import ImageFrame
from xr_ai_tools.image import ImageRegistry
from xr_ai_voice import VOICE_TRANSCRIPT_TOPIC, VoiceTranscript

_SAMPLE = Path(__file__).resolve().parents[1] / "agent-samples/sop-sample"
sys.path.insert(0, str(_SAMPLE / "worker"))

from sop_sample_worker.config import load_config  # noqa: E402
from sop_sample_worker.lifecycle import CaptureRecording  # noqa: E402
from sop_sample_worker.recorder import RecorderAgent  # noqa: E402


def test_config_and_cli(monkeypatch):
    from device_io_hub.capture.config import load_capture_config

    worker = load_config(_SAMPLE / "yaml/worker.yaml")
    capture = load_capture_config(_SAMPLE / "yaml/media_capture.yaml")
    assert not hasattr(worker, "media_capture_dir")
    assert (capture.profile, capture.session_mode, capture.max_total_bytes) == ("demo", "participant", 0)
    assert (worker.capture_fps, worker.caption_interval_s) == (2, 5)
    load_models_config(worker.models_config)
    spec = importlib.util.spec_from_file_location("sop_main", _SAMPLE / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not module._parser().parse_args([]).capture
    assert module._parser().parse_args(["--capture"]).capture
    assert [p.name for p in module.PROCESSES] == ["hub", "worker"]
    assert [p.name for p in module._build_processes(capture=True)] == ["hub", "capture", "worker"]
    stack = Mock()
    monkeypatch.setattr(module, "run_stack", stack)
    for args, expected in (([], ["hub", "worker"]), (["--capture"], ["hub", "capture", "worker"])):
        module.run(args)
        assert [p.name for p in stack.call_args.args[0]] == expected


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
    assert len(query.await_args_list) == 4


@pytest.fixture
async def demo(tmp_path):
    image = io.BytesIO()
    Image.new("RGB", (8, 8), "blue").save(image, format="JPEG")
    images = ImageRegistry()
    frames = Mock(execute=AsyncMock(return_value=ImageFrame(
        image=images.put(image.getvalue(), owner="user"), width=8, height=8,
        timestamp_us=time.time_ns() // 1000, sequence=1, participant_id="user", track_id="camera",
    )))
    captioner = Mock(execute=AsyncMock(return_value=SimpleNamespace(
        available=True, text=json.dumps(
            {"activity": "Setup", "phase": "Arrange", "caption": "A blue block", "delta": "Initial view"}
        ),
    )))
    recorder = RecorderAgent(
        sessions_dir=tmp_path / "sessions", current_frame=frames, images=images,
        query_image=captioner, capture_fps=2, caption_interval_s=5,
    )
    capture = CaptureRecording(recorder)
    try:
        yield SimpleNamespace(recorder=recorder, capture=capture, root=tmp_path, frames=frames, captioner=captioner)
    finally:
        await capture.close()


def joined(pid="user", session="session", pts_us=1):
    return ParticipantEvent(pid, True, pts_us, participant_session_id=session)


async def speak(demo, text, pid="user", timestamp_us=None, wait=True):
    timestamp_us = time.time_ns() // 1000 if timestamp_us is None else timestamp_us
    await demo.capture.transcript(
        VoiceTranscript(text=text, timestamp_us=timestamp_us),
        SimpleNamespace(metadata=SimpleNamespace(participant_id=pid)),
    )
    if wait and (task := demo.capture._tails.get(pid)):
        await asyncio.shield(task)


def packet(state):
    return json.loads((state.directory / "packet.json").read_text())


def narration(state):
    return [json.loads(line)["text"] for line in (state.directory / "transcript.jsonl").read_text().splitlines()]


async def test_join_is_idle_and_voice_can_record_multiple_sessions(demo):
    event = joined()
    await demo.capture.receive(event)
    await speak(demo, "Before recording")
    await speak(demo, "stop recording")
    assert not list((demo.root / "sessions").iterdir())
    demo.frames.execute.assert_not_awaited()
    demo.captioner.execute.assert_not_awaited()
    for _ in range(2):
        await speak(demo, " Start  RECORDING! ")
        state = demo.recorder._sessions["user"]
        await speak(demo, "start recording")  # Duplicate command, not narration or another packet.
        await demo.capture.receive(event)
        assert demo.recorder._sessions["user"] is state
        await speak(demo, "Place the cup")
        async with asyncio.timeout(2):
            while state.caption_count == 0:
                await asyncio.sleep(0.001)
        await speak(demo, "Stop recording.")
        assert packet(state)["status"] == packet(state)["narration_status"] == "complete"
        assert packet(state)["counts"] == {"frames": 1, "transcripts": 1, "captions": 1}
        assert packet(state)["stop_command"]["text"] == "Stop recording."
        assert narration(state) == ["Place the cup"]
        await speak(demo, "After recording")
        await demo.capture.receive(event)
        assert not demo.recorder.is_recording("user")
        assert narration(state) == ["Place the cup"]
    assert len(list((demo.root / "sessions").iterdir())) == 2


async def test_disconnect_and_old_events_cannot_stop_new_recording(demo):
    await demo.capture.receive(joined())
    await speak(demo, "start recording")
    old = demo.recorder._sessions["user"]
    await speak(demo, "First procedure")
    await demo.capture.receive(ParticipantEvent("user", False, 2, participant_session_id="session"))
    assert packet(old)["status"] == "complete"
    await demo.capture.receive(joined(session="new", pts_us=100))
    assert not demo.recorder.is_recording("user")
    await speak(demo, "start recording", timestamp_us=101)
    await demo.capture.receive(ParticipantEvent("user", False, 3, participant_session_id="session"))
    await speak(demo, "stop recording", timestamp_us=50)
    assert demo.recorder.is_recording("user")

@pytest.mark.parametrize("stop_command", [False, True])
async def test_app_drains_voice_transcripts_and_recordings_before_endpoint_closes(demo, monkeypatch, stop_command):
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
        assert message.topic == CAPTURE_STT_TOPIC

    async def receive_forever():
        receiving.set()
        try:
            await asyncio.Future()
        finally:
            receiver_cancelled.set()

    capture_endpoint = SimpleNamespace(
        on_participant=lambda cb: callbacks.update(participant=cb),
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
        assert packet["counts"]["transcripts"] == 1
        assert voice_closed
        capture_closed = True

    capture_endpoint.close = close_capture
    monkeypatch.setattr(app, "ProcessorEndpoint", lambda **kwargs: capture_endpoint)
    monkeypatch.setattr(app, "ImageRegistry", lambda: demo.recorder._images)
    monkeypatch.setattr(app, "CurrentFrameTool", lambda **kwargs: demo.frames)
    monkeypatch.setattr(app, "setup_logging", lambda *args: None)
    for name in ("make_stt", "make_tts", "make_vlm"):
        monkeypatch.setattr(app, name, lambda *args: Mock(health=AsyncMock(return_value=True), close=AsyncMock()))

    def shutdown_voice():
        nonlocal voice_closed
        voice_closed = True

    transport = SimpleNamespace(endpoint=Mock(), shutdown=shutdown_voice, send_return_data=AsyncMock(side_effect=send))
    real_voice = app.VoiceAgent
    voices = []

    def make_voice(**kwargs):
        voice = real_voice(transport=transport, **kwargs)
        voices.append(voice)
        return voice

    monkeypatch.setattr(app, "VoiceAgent", make_voice)

    async def run_session(self, *args, **kwargs):
        assert self.vad.stop_probe_after_s == 0
        await callbacks["participant"](joined())
        # End the voice run while the participant is still connected.
        # VoiceAgent's real finally path closes only its own transport.
        await voices[0]._publish_transcript("user", "Start recording.", time.time_ns() // 1000)
        async with asyncio.timeout(2):
            while not list((demo.root / "app-sessions/sessions").glob("*/packet.json")):
                await asyncio.sleep(0.001)
        await voices[0]._publish_transcript("user", "Place a cup", time.time_ns() // 1000)
        # Delivered final transcripts must drain into the packet; speech still
        # awaiting STT at transport shutdown is outside this guarantee.
        await voices[0]._transcript_queue.join()
        if stop_command:
            # Exercise VoiceAgent's real transcript queue, shared transcript
            # publication, and the app's registered capture subscriber.
            await voices[0]._publish_transcript("user", "Stop recording.", time.time_ns() // 1000)
            async with asyncio.timeout(2):
                while True:
                    packets = list((demo.root / "app-sessions/sessions").glob("*/packet.json"))
                    if packets and json.loads(packets[0].read_text())["status"] == "complete":
                        break
                    await asyncio.sleep(0.001)
            packet = json.loads(packets[0].read_text())
            assert packet["counts"]["transcripts"] == 1
            assert packet["stop_command"]["text"] == "Stop recording."
            assert not voice_closed

    monkeypatch.setattr(_VoiceSession, "run", run_session)
    config = replace(
        load_config(_SAMPLE / "yaml/worker.yaml"),
        artifacts_dir=demo.root / "app-sessions",
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


async def test_participants_are_independent(demo):
    for pid in ("user", "other"):
        await demo.capture.receive(joined(pid))
        await speak(demo, "start recording", pid)
        await speak(demo, pid + " narration", pid)
    other = demo.recorder._sessions["other"]
    await speak(demo, "stop recording")
    assert not demo.recorder.is_recording("user")
    assert demo.recorder.is_recording("other")
    assert narration(other) == ["other narration"]


@pytest.mark.parametrize("text", [
    "stop", "stop capture", "do not stop recording", "say stop recording", "stop recording now",
    "start recording the next step", "do not start recording",
])
async def test_only_standalone_commands_control_recording(demo, text):
    await demo.capture.receive(joined())
    await speak(demo, text)
    assert not demo.recorder.is_recording("user")
    await speak(demo, "start recording")
    state = demo.recorder._sessions["user"]
    await speak(demo, text)
    assert demo.recorder.is_recording("user")
    await speak(demo, "stop recording")
    assert narration(state) == [text]


async def test_failed_narration_write_cannot_seal_a_complete_packet(demo, monkeypatch):
    import sop_sample_worker.recorder as module

    await demo.capture.receive(joined())
    await speak(demo, "start recording")
    state = demo.recorder._sessions["user"]
    original = module._append_jsonl

    def fail_transcript(path, row):
        if path.name == "transcript.jsonl":
            raise OSError("write failed")
        original(path, row)

    monkeypatch.setattr(module, "_append_jsonl", fail_transcript)
    await speak(demo, "Place the cup")
    await speak(demo, "stop recording")
    assert packet(state)["status"] == "incomplete"
    assert packet(state)["narration_status"] == "failed"


async def test_no_camera_frames_does_not_end_narration(demo):
    demo.frames.execute.side_effect = FrameUnavailable("no camera available")
    await demo.capture.receive(joined())
    await speak(demo, "start recording")
    state = demo.recorder._sessions["user"]
    async with asyncio.timeout(2):
        while not (state.directory / "errors.jsonl").exists():
            await asyncio.sleep(0.001)
    await speak(demo, "Explain the procedure without video")
    await speak(demo, "stop recording")
    assert packet(state)["status"] == "complete"
    assert packet(state)["counts"] == {"frames": 0, "transcripts": 1, "captions": 0}


async def test_optional_media_capture_is_independent_of_sop_commands(demo, monkeypatch):
    from device_io_hub.capture._service import CaptureService
    from device_io_hub.capture.config import CaptureConfig

    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", SimpleNamespace())
    service = CaptureService(CaptureConfig(
        out_dir=str(demo.root / "captures"), session_mode="participant", max_total_bytes=0,
    ))
    runtime = AgentRuntime()
    runtime.register("capture", demo.capture)
    try:
        async with runtime:
            await service._on_participant(joined(pts_us=time.time_ns() // 1000))
            await demo.capture.receive(joined())
            assert service._recorder.has_session("user")
            assert not demo.recorder.is_recording("user")
            for text in ("Before", "start recording", "Place the cup", "stop recording", "After"):
                ts = time.time_ns() // 1000
                await service._on_agent_data(DataMessage("user", CAPTURE_STT_TOPIC, ts, text.encode()))
                await runtime.publish(
                    VOICE_TRANSCRIPT_TOPIC, VoiceTranscript(text=text, timestamp_us=ts), participant_id="user",
                )
                await asyncio.shield(demo.capture._tails["user"])
            assert service._recorder.has_session("user")  # SOP stop cannot stop shared capture.
            audio = AudioChunk(time.time_ns() // 1000, 16000, 1, 160, b"\x00" * 640, "user", "microphone")
            await service._on_device_audio(audio)
            await service._on_participant(ParticipantEvent(
                "user", False, time.time_ns() // 1000, participant_session_id="session",
            ))
    finally:
        await service.stop()
    manifests = list((demo.root / "captures").rglob("manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    shared_lines = (manifests[0].parent / manifest["transcript"]).read_text().splitlines()
    rows = [json.loads(line)["text"] for line in shared_lines]
    assert rows == ["Before", "start recording", "Place the cup", "stop recording", "After"]
    assert (manifests[0].parent / "audio/device.f32le").read_bytes() == audio.data
    sop = next((demo.root / "sessions").iterdir())
    sop_lines = (sop / "transcript.jsonl").read_text().splitlines()
    assert [json.loads(line)["text"] for line in sop_lines] == ["Place the cup"]


@pytest.mark.parametrize("boundary", ["restart", "disconnect", "shutdown"])
async def test_queued_restart_and_narration_are_ordered_without_media_wait(demo, monkeypatch, boundary):
    await demo.capture.receive(joined())
    await speak(demo, "start recording")
    old = demo.recorder._sessions["user"]
    entered, release = asyncio.Event(), asyncio.Event()
    original = demo.recorder._finalize

    async def delayed(state, **kwargs):
        if state is old:
            entered.set()
            await release.wait()
        await original(state, **kwargs)

    monkeypatch.setattr(demo.recorder, "_finalize", delayed)
    await speak(demo, "First", wait=False)
    finishing = asyncio.create_task(speak(demo, "stop recording"))
    await asyncio.wait_for(entered.wait(), 2)
    finishing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await finishing
    await speak(demo, "start recording", wait=False)
    await speak(demo, "Second", wait=False)
    closing = None
    if boundary == "disconnect":
        closing = asyncio.ensure_future(demo.capture.receive(
            ParticipantEvent("user", False, time.time_ns() // 1000, participant_session_id="session")
        ))
    elif boundary == "shutdown":
        closing = asyncio.create_task(demo.capture.close())
        await asyncio.sleep(0)
    await demo.capture.receive(joined("other"))
    if boundary != "shutdown":
        await speak(demo, "start recording", "other")
    try:
        assert demo.recorder._sessions["user"] is old
    finally:
        release.set()
        await asyncio.wait_for(asyncio.shield(demo.capture._tails["user"]), 2)
        if closing:
            await asyncio.wait_for(closing, 2)
    assert narration(old) == ["First"]
    if boundary == "restart":
        new = demo.recorder._sessions["user"]
        assert new is not old
        assert narration(new) == ["Second"]
    else:
        assert not demo.recorder.is_recording("user")
        assert len(list((demo.root / "sessions").glob("*-user"))) == 1


async def test_on_demand_images_and_republished_stream_are_distinct(demo, monkeypatch):
    async def idle(_state):
        await asyncio.Future()

    monkeypatch.setattr(demo.recorder, "_capture_loop", idle)
    monkeypatch.setattr(demo.recorder, "_caption_loop", idle)
    await demo.capture.receive(joined())
    await speak(demo, 'start recording')
    state = demo.recorder._sessions["user"]
    image = demo.frames.execute.return_value
    # All on-demand snapshots use sequence zero. A live track can reset its
    # sequence too; neither should discard a different timestamp or source.
    for index, (track, timestamp, sequence) in enumerate([
        ("", 100, 0), ("", 200, 0), ("camera", 300, 0), ("replacement", 300, 0), ("", 400, 0),
    ], start=1):
        demo.frames.execute.return_value = image.model_copy(update={
            "track_id": track, "timestamp_us": timestamp, "sequence": sequence,
        })
        await demo.recorder._capture(state)
        await demo.recorder._caption(state)
        await demo.recorder._capture(state)  # Same observation is not new evidence.
        await demo.recorder._caption(state)
        assert state.frame_count == state.caption_count == index
    demo.frames.execute.side_effect = FrameUnavailable("live camera disabled")
    with pytest.raises(FrameUnavailable):
        await demo.recorder._capture(state)
    await demo.recorder._caption(state)
    assert state.caption_count == 5  # No repeated caption of the stale last frame.
    assert demo.recorder.is_recording("user")
    await speak(demo, 'stop recording')
    packet = json.loads((state.directory / "packet.json").read_text())
    assert packet["counts"]["frames"] == packet["counts"]["captions"] == 5


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
    await demo.capture.receive(joined())
    await speak(demo, 'start recording')
    state = demo.recorder._sessions["user"]
    finishing = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        finishing = asyncio.create_task(speak(demo, 'stop recording'))
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


async def test_shutdown_drains_inflight_writes_before_final_packet(demo, monkeypatch):
    import sop_sample_worker.recorder as module

    await demo.capture.receive(joined())
    await speak(demo, 'start recording')
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
    closing = asyncio.create_task(demo.capture.close())
    try:
        await asyncio.sleep(0.02)
        assert not closing.done()
    finally:
        release.set()
        await asyncio.wait_for(closing, 2)
    assert json.loads((state.directory / "packet.json").read_text())["status"] == "complete"
    assert not demo.recorder.is_recording("user")
