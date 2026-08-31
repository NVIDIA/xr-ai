# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import types
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from device_io_hub.capture._compositor import compose_caption
from device_io_hub.capture._recorder import (
    SessionRecorder,
    _safe_name,
    _StereoWaveWriter,
)
from device_io_hub.capture._service import (
    CaptureService,
    _FrameWorker,
    _invalid_audio_reason,
)
from device_io_hub.capture.config import CaptureConfig, load_capture_config
from xr_ai_hub import AudioChunk, DataMessage, FrameData, FrameSignal, PixelFormat
from xr_ai_hub._capture import CAPTURE_STT_TOPIC, CAPTURE_TTS_TOPIC


def _frame(
    *,
    pts_us: int = 1_000_000,
    width: int = 64,
    height: int = 32,
    track_id: str = "camera",
) -> FrameData:
    y = np.full(width * height, 96, dtype=np.uint8)
    u = np.full(width * height // 4, 128, dtype=np.uint8)
    v = np.full(width * height // 4, 128, dtype=np.uint8)
    return FrameData(
        seq=1,
        pts_us=pts_us,
        width=width,
        height=height,
        fmt=PixelFormat.I420,
        data=np.concatenate((y, u, v)).tobytes(),
        participant_id="alice",
        track_id=track_id,
    )


def _audio(direction: str, *, pts_us: int = 1_000_000) -> AudioChunk:
    values = np.full(480, 0.25 if direction == "device" else -0.5, dtype=np.float32)
    return AudioChunk(
        pts_us=pts_us,
        sample_rate=48_000,
        channels=1,
        samples=values.size,
        data=values.tobytes(),
        participant_id="alice",
        track_id="mic" if direction == "device" else "tts",
    )


def test_compositor_preserves_sensor_pixels_and_appends_caption_panel() -> None:
    frame = _frame(width=640, height=320)

    output, width, height = compose_caption(
        frame,
        "Agent: hello",
        data_feed=("DEVICE sensor.state: open", "AGENT scene.update: complete"),
        max_lines=2,
    )

    assert width > frame.width
    assert height > frame.height
    assert width % 2 == 0
    assert height % 2 == 0
    luma = output[:height]
    chroma = output[height:]
    assert np.all(luma[:frame.height, :frame.width] == 96)
    assert np.any(luma[frame.height:, :frame.width] == 235)
    assert np.any(luma[:, frame.width:] == 235)
    assert np.all(chroma[:frame.height // 2, :frame.width] == 128)


def test_stereo_wave_aligns_device_left_and_agent_right(tmp_path: Path) -> None:
    path = tmp_path / "conversation.wav"
    writer = _StereoWaveWriter(path, start_us=1_000_000, sample_rate=48_000)
    writer.add("device", _audio("device"))
    writer.add("agent", _audio("agent"))
    writer.close()

    with wave.open(str(path), "rb") as stream:
        assert stream.getnchannels() == 2
        assert stream.getframerate() == 48_000
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").reshape(-1, 2)
    assert np.all(samples[:, 0] > 0)
    assert np.all(samples[:, 1] < 0)


def test_stereo_wave_ignores_arrival_jitter_and_bursts(tmp_path: Path) -> None:
    path = tmp_path / "conversation.wav"
    writer = _StereoWaveWriter(path, start_us=1_000_000, sample_rate=48_000)
    device = _audio("device")
    agent = _audio("agent")
    writer.add("device", device)
    writer.add("device", _audio("device", pts_us=1_510_900))
    writer.add("agent", agent)
    writer.add("agent", _audio("agent", pts_us=1_000_100))
    writer.close()

    with wave.open(str(path), "rb") as stream:
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").reshape(-1, 2)

    assert samples.shape == (960, 2)
    assert np.all(samples[:, 0] > 0)
    assert np.all(samples[:, 1] < 0)


def test_stereo_wave_retains_real_conversation_gap(tmp_path: Path) -> None:
    path = tmp_path / "conversation.wav"
    writer = _StereoWaveWriter(path, start_us=1_000_000, sample_rate=48_000)
    writer.add("agent", _audio("agent"))
    writer.add("agent", _audio("agent", pts_us=1_510_000))
    writer.close()

    with wave.open(str(path), "rb") as stream:
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").reshape(-1, 2)

    assert np.all(samples[:480, 1] < 0)
    assert np.all(samples[480:24_480, 1] == 0)
    assert np.all(samples[24_480:, 1] < 0)


def test_capture_rejects_malformed_audio_before_recording() -> None:
    valid = _audio("device")
    invalid = _audio("device")
    invalid.data = invalid.data[:-4]

    assert _invalid_audio_reason(valid) is None
    assert _invalid_audio_reason(invalid) == "expected 1920 PCM bytes, got 1916"


@pytest.mark.asyncio
async def test_frame_worker_samples_before_requesting_pixels() -> None:
    class Endpoint:
        def __init__(self) -> None:
            self.requested: list[FrameSignal] = []
            self.changed = asyncio.Event()

        async def request_frame(self, signal: FrameSignal):
            self.requested.append(signal)
            if len(self.requested) == 2:
                self.changed.set()
            return None

    class Recorder:
        def __init__(self) -> None:
            self.dropped: list[int] = []

        def note_video_drop(self, _participant_id: str, pts_us: int) -> None:
            self.dropped.append(pts_us)

    endpoint = Endpoint()
    recorder = Recorder()
    with ThreadPoolExecutor(max_workers=1) as executor:
        worker = _FrameWorker(
            endpoint=endpoint,  # type: ignore[arg-type]
            recorder=recorder,  # type: ignore[arg-type]
            executor=executor,
            participant_id="alice",
            track_id="camera",
            queue_size=4,
            sample_fps=30,
            on_failure=lambda _task: None,
        )
        for seq, pts_us in enumerate((1_000_000, 1_010_000, 1_040_000)):
            worker.submit(FrameSignal(
                slot=0,
                seq=seq,
                pts_us=pts_us,
                width=64,
                height=32,
                fmt=PixelFormat.I420,
                data_sz=3_072,
                participant_id="alice",
                track_id="camera",
            ))
        await asyncio.wait_for(endpoint.changed.wait(), 1.0)
        await worker.close()

    assert [signal.pts_us for signal in endpoint.requested] == [1_000_000, 1_040_000]
    assert recorder.dropped == [1_000_000, 1_040_000]


def test_capture_config_resolves_output_and_caption_duration(tmp_path: Path) -> None:
    path = tmp_path / "capture.yaml"
    path.write_text(
        "out_dir: artifacts\noverlay_seconds: 8\n",
        encoding="utf-8",
    )

    config = load_capture_config(path)

    assert config.out_dir == str((tmp_path / "artifacts").resolve())
    assert config.overlay_seconds == 8


def test_session_bundle_uses_nvenc_packets_and_preserves_raw_streams(
    tmp_path: Path,
    monkeypatch,
) -> None:
    encoded_inputs: list[np.ndarray] = []

    class FakeEncoder:
        def Encode(self, frame, _params):
            encoded_inputs.append(frame.copy())
            return [
                {
                    "data": (
                        b"\x00\x00\x00\x01\x67\x64\x00\x1f\xac\x00\x00\x00\x01\x68\xee\x3c\x80\x00\x00\x00\x01\x65frame"
                    ),
                    "timestamp": 1_000_000,
                    "picture_type": 3,
                }
            ]

        def EndEncode(self):
            return []

    fake_module = types.SimpleNamespace(
        CreateEncoder=lambda *_args, **_kwargs: FakeEncoder(),
        NV_ENC_PIC_PARAMS=type("NV_ENC_PIC_PARAMS", (), {}),
    )
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", fake_module)
    config = CaptureConfig(
        out_dir=str(tmp_path),
        sample_fps=30,
        max_total_bytes=0,
    )
    recorder = SessionRecorder(config)
    recorder.begin_session("alice", 1_000_000)
    recorder.record_data(
        "agent",
        DataMessage("alice", "agent.response", 1_000_000, b"Hello from the agent"),
    )
    recorder.record_data(
        "device",
        DataMessage("alice", "sensor.state", 1_000_000, b"Door open"),
    )
    recorder.record_data(
        "agent",
        DataMessage("alice", "agent.large", 1_000_000, b"x" * 4_096),
    )
    recorder.record_voice_caption(
        "user",
        DataMessage("alice", CAPTURE_STT_TOPIC, 1_000_000, b"What is this?"),
    )
    recorder.record_audio("device", _audio("device"))
    recorder.record_audio("agent", _audio("agent"))
    recorder.record_video(_frame())
    recorder.record_voice_caption(
        "agent",
        DataMessage("alice", CAPTURE_TTS_TOPIC, 1_040_000, b"This is a door."),
    )
    recorder.record_video(_frame(pts_us=1_050_000))
    session_state = recorder._sessions["alice"]
    assert tuple(session_state.data_feed)[:2] == (
        "AGENT agent.response: Hello from the agent",
        "DEVICE sensor.state: Door open",
    )
    assert session_state.data_feed[-1] == "AGENT agent.large: " + "x" * 1_024
    assert session_state.caption == "AGENT: This is a door."
    packets = recorder._sessions["alice"].video["camera"]._packets
    assert [packet.pts_us for packet in packets] == [1_000_000, 1_050_000]
    recorder.end_session("alice", 1_100_000)

    session = next(path for path in tmp_path.iterdir() if path.is_dir())
    assert stat.S_IMODE(session.stat().st_mode) & 0o077 == 0
    assert not list(session.rglob("*.pending"))
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["participant_id"] == "alice"
    assert manifest["audio"]["channels"] == {"left": "device", "right": "agent"}
    assert manifest["video_tracks"]["camera"][0]["num_frames"] == 2
    segment = manifest["video_tracks"]["camera"][0]
    assert segment["path"].endswith(".mkv")
    assert segment["audio_embedded"] is True
    muxed = (session / segment["path"]).read_bytes()
    assert muxed.startswith(b"\x1a\x45\xdf\xa3")
    assert b"V_MPEG4/ISO/AVC" in muxed
    assert b"A_PCM/INT/LIT" in muxed
    assert (session / segment["raw_path"]).read_bytes().startswith(b"\x00\x00\x00\x01")
    assert (session / "audio" / "device.f32le").read_bytes() == _audio("device").data
    assert (session / "audio" / "agent.f32le").read_bytes() == _audio("agent").data
    events = [json.loads(line) for line in (session / "events.jsonl").read_text().splitlines()]
    assert any(event.get("text") == "Hello from the agent" for event in events)
    assert any(
        event.get("kind") == "voice_caption"
        and event.get("source") == "agent"
        and event.get("text") == "This is a door."
        for event in events
    )
    assert encoded_inputs
    assert np.any(encoded_inputs[0][_frame().height:] == 235)


def test_safe_names_do_not_alias_distinct_track_ids() -> None:
    assert _safe_name("cam/1") != _safe_name("cam_1")


def test_retention_counts_incomplete_capture_directories(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", types.SimpleNamespace())
    incomplete = tmp_path / "1_incomplete"
    incomplete.mkdir()
    (incomplete / "events.jsonl").write_bytes(b"x" * 100)
    complete = tmp_path / "2_complete"
    complete.mkdir()
    (complete / "manifest.json").write_text("{}")
    (complete / "payload").write_bytes(b"y" * 100)
    os.utime(incomplete, ns=(1, 1))
    os.utime(complete, ns=(2, 2))

    SessionRecorder(CaptureConfig(
        out_dir=str(tmp_path),
        max_total_bytes=102,
    ))

    assert not incomplete.exists()
    assert complete.exists()


def test_session_bundle_merges_resolution_and_track_segments_into_one_video(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeEncoder:
        def Encode(self, _frame, _params):
            return [{
                "data": (
                    b"\x00\x00\x00\x01\x67\x64\x00\x1f\xac"
                    b"\x00\x00\x00\x01\x68\xee\x3c\x80"
                    b"\x00\x00\x00\x01\x65frame"
                ),
                "picture_type": 3,
            }]

        def EndEncode(self):
            return []

    fake_module = types.SimpleNamespace(
        CreateEncoder=lambda *_args, **_kwargs: FakeEncoder(),
        NV_ENC_PIC_PARAMS=type("NV_ENC_PIC_PARAMS", (), {}),
    )
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", fake_module)
    recorder = SessionRecorder(CaptureConfig(
        out_dir=str(tmp_path),
        sample_fps=30,
        max_total_bytes=0,
    ))
    recorder.begin_session("alice", 1_000_000)
    recorder.record_video(_frame(pts_us=1_000_000, width=64, height=32))
    recorder.record_video(_frame(pts_us=1_050_000, width=128, height=64))
    recorder.record_video(_frame(
        pts_us=1_100_000,
        width=128,
        height=64,
        track_id="replacement-camera",
    ))
    recorder.end_session("alice", 1_150_000)

    session = next(path for path in tmp_path.iterdir() if path.is_dir())
    manifest = json.loads((session / "manifest.json").read_text())
    videos = [
        video
        for track in manifest["video_tracks"].values()
        for video in track
    ]
    assert len(videos) == 1
    assert videos[0]["num_frames"] == 3
    assert videos[0]["source_track_ids"] == ["camera", "replacement-camera"]
    assert videos[0]["encoded_dimensions"] == [
        {"width": 224, "height": 112},
        {"width": 288, "height": 144},
    ]
    assert [path.name for path in (session / "video").glob("*.mkv")] == [
        "session.mkv"
    ]
    assert [path.name for path in (session / "video").glob("*.264")] == [
        "session.264"
    ]


@pytest.mark.asyncio
async def test_capture_service_observes_both_sides_of_media_hub(
    hub,
    hub_addrs,
    make_connector,
    make_processor,
    tmp_path: Path,
    monkeypatch,
) -> None:
    encoded_inputs: list[np.ndarray] = []

    class FakeEncoder:
        def Encode(self, frame, _params):
            encoded_inputs.append(frame.copy())
            return [
                {
                    "data": (
                        b"\x00\x00\x00\x01\x67\x64\x00\x1f\xac\x00\x00\x00\x01\x68\xee\x3c\x80\x00\x00\x00\x01\x65frame"
                    ),
                    "timestamp": 1_000_000,
                    "picture_type": 3,
                }
            ]

        def EndEncode(self):
            return []

    monkeypatch.setitem(
        sys.modules,
        "PyNvVideoCodec",
        types.SimpleNamespace(
            CreateEncoder=lambda *_args, **_kwargs: FakeEncoder(),
            NV_ENC_PIC_PARAMS=type("NV_ENC_PIC_PARAMS", (), {}),
        ),
    )
    pull, publish = hub_addrs
    service = CaptureService(CaptureConfig(
        hub_push_addr=pull,
        hub_sub_addr=publish,
        out_dir=str(tmp_path),
        frame_queue_size=1,
        encoder_workers=1,
        max_total_bytes=0,
    ))
    await service.start()
    connector = make_connector(connector_id="capture-test")
    processor = make_processor()
    returned_data: list[DataMessage] = []

    async def record_return(message: DataMessage) -> None:
        returned_data.append(message)

    connector.on_return_data(record_return)
    connector_task: asyncio.Task | None = None
    try:
        await connector.register()
        connector_task = asyncio.create_task(connector.run())
        await asyncio.sleep(0.05)
        await connector.notify_participant_joined("alice", pts_us=1_000_000)
        for _ in range(40):
            if "alice" in service._endpoint.subscribed_participants:
                break
            await asyncio.sleep(0.025)
        assert "alice" in service._endpoint.subscribed_participants
        await asyncio.sleep(0.05)

        await connector.push_audio(_audio("device"))
        await connector.push_data(DataMessage("alice", "sensor.state", 1_000_000, b"open"))
        await connector.push_frame(
            data=_frame().data,
            width=_frame().width,
            height=_frame().height,
            fmt=_frame().fmt,
            pts_us=_frame().pts_us,
            participant_id="alice",
            track_id="camera",
        )
        await processor.send_return_audio(_audio("agent"))
        await processor.send_return_data(
            DataMessage("alice", "agent.response", 1_000_000, b"Captured response"),
        )
        await processor.send_return_data(
            DataMessage("alice", CAPTURE_STT_TOPIC, 1_000_000, b"User transcript"),
        )
        await processor.send_return_data(
            DataMessage("alice", CAPTURE_TTS_TOPIC, 1_010_000, b"Spoken response"),
        )

        for _ in range(80):
            sessions = [path for path in tmp_path.iterdir() if path.is_dir()]
            if sessions and encoded_inputs and b"Spoken response" in (
                sessions[0] / "events.jsonl"
            ).read_bytes():
                break
            await asyncio.sleep(0.025)
        assert encoded_inputs
        for _ in range(20):
            if returned_data:
                break
            await asyncio.sleep(0.025)

        await connector.notify_participant_left("alice", pts_us=1_100_000)
        for _ in range(80):
            sessions = [path for path in tmp_path.iterdir() if path.is_dir()]
            if sessions and (sessions[0] / "manifest.json").is_file():
                break
            await asyncio.sleep(0.025)
        session = next(path for path in tmp_path.iterdir() if path.is_dir())
        manifest = json.loads((session / "manifest.json").read_text())
        assert manifest["video_tracks"]["camera"]
        assert (session / "audio" / "device.f32le").stat().st_size > 0
        assert (session / "audio" / "agent.f32le").stat().st_size > 0
        events = (session / "events.jsonl").read_text()
        assert '"direction":"device"' in events
        assert '"direction":"agent"' in events
        assert '"kind":"voice_caption"' in events
        returned_topics = [message.topic for message in returned_data]
        assert "agent.response" in returned_topics
        assert CAPTURE_STT_TOPIC not in returned_topics
        assert CAPTURE_TTS_TOPIC not in returned_topics
    finally:
        connector.stop()
        if connector_task is not None:
            connector_task.cancel()
            await asyncio.gather(connector_task, return_exceptions=True)
        await service.stop()
