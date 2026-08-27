# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest
from device_io_hub.capture._compositor import compose_caption
from device_io_hub.capture._recorder import SessionRecorder, _StereoWaveWriter
from device_io_hub.capture._service import CaptureService
from device_io_hub.capture.config import CaptureConfig, load_capture_config
from xr_ai_hub import AudioChunk, DataMessage, FrameData, PixelFormat


def _frame(*, pts_us: int = 1_000_000) -> FrameData:
    width, height = 64, 32
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
        track_id="camera",
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
    frame = _frame()

    output, width, height = compose_caption(frame, "Agent: hello", max_lines=2)

    assert width == frame.width
    assert height > frame.height
    assert height % 2 == 0
    assert np.all(output[:frame.height] == 96)
    assert np.any(output[frame.height:height] == 235)
    assert np.all(output[height:height + frame.height // 2] == 128)


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


def test_capture_config_resolves_output_and_normalizes_topics(tmp_path: Path) -> None:
    path = tmp_path / "capture.yaml"
    path.write_text(
        "out_dir: artifacts\noverlay_topics:\n  - agent.response\n",
        encoding="utf-8",
    )

    config = load_capture_config(path)

    assert config.out_dir == str((tmp_path / "artifacts").resolve())
    assert config.overlay_topics == ("agent.response",)


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
        overlay_topics=("agent.response",),
        max_total_bytes=0,
    )
    recorder = SessionRecorder(config)
    recorder.begin_session("alice", 1_000_000)
    recorder.record_data(
        "agent",
        DataMessage("alice", "agent.response", 1_000_000, b"Hello from the agent"),
    )
    recorder.record_audio("device", _audio("device"))
    recorder.record_audio("agent", _audio("agent"))
    recorder.record_video(_frame())
    recorder.record_video(_frame(pts_us=1_050_000))
    packets = recorder._sessions["alice"].video["camera"]._packets
    assert [packet.pts_us for packet in packets] == [1_000_000, 1_050_000]
    recorder.end_session("alice", 1_100_000)

    session = next(path for path in tmp_path.iterdir() if path.is_dir())
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
    assert encoded_inputs
    assert np.any(encoded_inputs[0][_frame().height:] == 235)


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
    try:
        await connector.register()
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

        for _ in range(80):
            sessions = [path for path in tmp_path.iterdir() if path.is_dir()]
            if sessions and encoded_inputs and b"Captured response" in (
                sessions[0] / "events.jsonl"
            ).read_bytes():
                break
            await asyncio.sleep(0.025)
        assert encoded_inputs

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
    finally:
        await service.stop()
