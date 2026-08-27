# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GPU integration tests for ``device_io_hub.video._recorder.VideoRecorder``.

These exercise the real NVENC path end-to-end: feed synthetic NV12 frames in,
assert that an H.264 chunk + JSON sidecar land on disk. They are skipped on
hosts without PyNvVideoCodec or an NVENC-capable GPU.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# PyNvVideoCodec initialises NVENC at import time, so a missing
# libnvidia-encode.so.1 raises RuntimeError (not ImportError) — importorskip
# would let it escape and break collection on CI boxes without NVENC.
try:
    import PyNvVideoCodec  # noqa: F401  (import-only — used to detect NVENC availability)
except (ImportError, RuntimeError, OSError) as exc:
    pytest.skip(f"PyNvVideoCodec unavailable: {exc}", allow_module_level=True)

from xr_ai_hub import AudioChunk, DataMessage, FrameData, FrameSignal, PixelFormat, SlotView  # noqa: E402

from video_memory_service.frames import decode_h264  # noqa: E402
from video_memory_service.service import VideoMemoryService  # noqa: E402
from video_memory_service.store import ChunkStore  # noqa: E402
from device_io_hub.capture._recorder import SessionRecorder  # noqa: E402
from device_io_hub.capture.config import CaptureConfig  # noqa: E402
from device_io_hub.video import VideoRecorder, VideoRecorderConfig  # noqa: E402

pytestmark = [pytest.mark.asyncio, pytest.mark.gpu]


# ── helpers ───────────────────────────────────────────────────────────────────


def _nv12_gradient(width: int, height: int, seed: int = 0) -> bytes:
    """Build a deterministic NV12 buffer (Y plane + interleaved UV plane).

    Y is a vertical gradient that shifts per ``seed`` so successive frames
    aren't identical (otherwise NVENC may emit zero-byte frames after the
    first IDR — fine for the encoder, but pointless for the test).
    """
    rows = np.arange(height, dtype=np.int64)
    y    = ((rows + seed) & 0xFF).astype(np.uint8)
    y    = np.broadcast_to(y[:, None], (height, width)).copy()
    uv   = np.full((height // 2, width), 128, dtype=np.uint8)
    return np.concatenate([y.ravel(), uv.ravel()]).tobytes()


def _make_view(buf: bytes, *, width: int, height: int,
               pid: str = "test_pid", tid: str = "test_track",
               seq: int = 0) -> SlotView:
    sig = FrameSignal(
        slot=0, seq=seq, pts_us=seq * 33_000,
        width=width, height=height, fmt=PixelFormat.NV12,
        data_sz=len(buf),
        participant_id=pid, track_id=tid,
    )
    return SlotView(data=memoryview(buf), signal=sig)


def _make_recorder(out_dir: str) -> VideoRecorder:
    """Build a recorder with the rate-limit effectively disabled, and
    pre-flight an NVENC session so the test can skip cleanly on hosts
    where the lib loads but no GPU is reachable (e.g. CI without
    ``/dev/nvidia*``).

    The hub's default ``sample_fps=30`` means ``_min_interval ≈ 33 ms``;
    pushing frames in a tight loop would silently drop ~all of them. Tests
    bump ``sample_fps`` so every frame actually reaches NVENC.
    """
    cfg = VideoRecorderConfig(
        out_dir=out_dir,
        chunk_frames=15,
        sample_fps=1000.0,
        bitrate=2_000_000,
    )
    try:
        recorder = VideoRecorder(cfg)
        probe = recorder._create_encoder(640, 480)
        try:
            probe.EndEncode()
        except Exception:  # best-effort teardown if NVENC went away mid-probe
            pass
        del probe
    except Exception as e:
        pytest.skip(f"NVENC unavailable on this host: {e}")
    return recorder


# ── tests ─────────────────────────────────────────────────────────────────────


async def test_record_synthetic_frames():
    """Feed 30 NV12 frames → expect at least one .h264 chunk + matching .json."""
    width, height = 640, 480
    pid = "synthetic_pid"

    with tempfile.TemporaryDirectory() as out_dir:
        recorder = _make_recorder(out_dir)

        for i in range(30):
            buf  = _nv12_gradient(width, height, seed=i)
            view = _make_view(buf, width=width, height=height, pid=pid, seq=i)
            await recorder.on_frame(view)

        recorder.close_participant(pid)

        # The recorder maps the raw pid through _safe_name() — here that's
        # identity, but find it via .identity sidecar to be safe.
        out = Path(out_dir)
        pid_dirs = [p for p in out.iterdir() if p.is_dir()]
        assert pid_dirs, "no participant subdirectory created"
        pid_dir = pid_dirs[0]

        chunks = sorted(pid_dir.glob("*.264"))
        sidecars = sorted(pid_dir.glob("*.json"))
        assert chunks, f"no .h264 chunk written to {pid_dir}"
        assert sidecars, f"no .json sidecar written to {pid_dir}"

        # One sidecar per chunk, paired by stem.
        chunk_stems    = {c.stem for c in chunks}
        sidecar_stems  = {s.stem for s in sidecars}
        assert chunk_stems == sidecar_stems, (
            f"chunk/sidecar mismatch: {chunk_stems} vs {sidecar_stems}"
        )

        # H.264 Annex B start code on the first chunk.
        first = chunks[0].read_bytes()
        assert first.startswith(b"\x00\x00\x00\x01"), (
            f"first chunk doesn't begin with NAL start code: {first[:8]!r}"
        )

        meta = json.loads(sidecars[0].read_text())
        assert meta["width"]      == width
        assert meta["height"]     == height
        assert meta["num_frames"] >  0
        assert meta["size_bytes"] == len(first)
        assert meta["end_us"]     >= meta["start_us"]

        service = VideoMemoryService(
            store=ChunkStore(out),
            out_dir=out / "frames",
            gpu_id=0,
        )
        frame = await service.dispatch(
            "get_historical_frame",
            {
                "participant_id": pid,
                "start_us": max(1, int(meta["start_us"])),
            },
        )
        with Image.open(frame["image"]["uri"]) as image:
            assert image.format == "PNG"
            assert image.size == (width, height)


async def test_resolution_change_surfaces_error():
    """Resolution change must either keep recording in a new chunk OR mark
    the track failed and stop silently dropping frames."""
    pid = "resize_pid"

    with tempfile.TemporaryDirectory() as out_dir:
        recorder = _make_recorder(out_dir)

        for i in range(10):
            buf  = _nv12_gradient(640, 480, seed=i)
            view = _make_view(buf, width=640, height=480, pid=pid, seq=i)
            await recorder.on_frame(view)

        for i in range(10, 20):
            buf  = _nv12_gradient(1280, 720, seed=i)
            view = _make_view(buf, width=1280, height=720, pid=pid, seq=i)
            await recorder.on_frame(view)

        # Grab the track encoder before close_participant() pops it.
        keys = [k for k in recorder._encoders if k[0] == pid]
        assert keys, "expected one track encoder for the test participant"
        enc = recorder._encoders[keys[0]]

        recorder.close_participant(pid)

        out = Path(out_dir)
        pid_dirs = [p for p in out.iterdir() if p.is_dir()]
        assert pid_dirs, "no participant subdirectory created"
        pid_dir = pid_dirs[0]

        sidecars = sorted(pid_dir.glob("*.json"))
        assert sidecars, "expected at least one sidecar even after a resolution change"
        metas = [json.loads(s.read_text()) for s in sidecars]
        resolutions = {(m["width"], m["height"]) for m in metas}

        # Either path is acceptable; both prove there is no silent drop:
        # `failed` short-circuits subsequent frames loudly, otherwise the
        # encoder was rebuilt at the new resolution and produced new chunks.
        assert (640, 480) in resolutions
        if not enc.failed:
            assert (1280, 720) in resolutions


async def test_media_capture_composites_caption_with_real_nvenc():
    """The session-capture path must feed valid contiguous NV12 into NVENC."""
    width, height = 640, 480
    with tempfile.TemporaryDirectory() as out_dir:
        _make_recorder(out_dir)  # pre-flight NVENC and skip only for unavailable hardware
        config = CaptureConfig(
            out_dir=out_dir,
            sample_fps=30,
            max_total_bytes=0,
        )
        recorder = SessionRecorder(config)
        recorder.begin_session("gpu_capture", 1_000_000)
        recorder.record_data(
            "agent",
            DataMessage(
                "gpu_capture",
                "agent.response",
                1_000_000,
                b"NVENC caption test",
            ),
        )
        for index in range(4):
            frame = FrameData(
                seq=index,
                pts_us=1_000_000 + index * 34_000,
                width=width,
                height=height,
                fmt=PixelFormat.NV12,
                data=_nv12_gradient(width, height, seed=index),
                participant_id="gpu_capture",
                track_id="camera",
            )
            recorder.record_video(frame)
        samples = np.full(480, 0.25, dtype=np.float32)
        for direction, value in (("device", samples), ("agent", -samples)):
            recorder.record_audio(
                direction,
                AudioChunk(
                    pts_us=1_000_000,
                    sample_rate=48_000,
                    channels=1,
                    samples=samples.size,
                    data=value.tobytes(),
                    participant_id="gpu_capture",
                    track_id=direction,
                ),
            )
        recorder.end_session("gpu_capture", 1_140_000)

        session = next(path for path in Path(out_dir).iterdir() if path.is_dir())
        manifest = json.loads((session / "manifest.json").read_text())
        segment = manifest["video_tracks"]["camera"][0]
        assert segment["width"] > width
        assert segment["height"] > height
        assert segment["audio_embedded"] is True
        demuxer = PyNvVideoCodec.CreateDemuxer(str(session / segment["path"]))
        assert demuxer.GetVideoStreamId() >= 0
        assert demuxer.GetAudioStreamId() >= 0
        packet_types = set()
        video_pts = set()
        while True:
            packet = demuxer.DemuxNoSkipAudio()
            if packet.bsl == 0:
                break
            packet_types.add("video" if packet.is_video else "audio")
            if packet.is_video:
                video_pts.add(packet.pts)
        assert packet_types == {"video", "audio"}
        assert len(video_pts) > 1
        encoded = (session / segment["raw_path"]).read_bytes()
        frames = decode_h264(encoded, gpu_id=0)
        assert frames
        assert frames[0].shape == (
            segment["height"] * 3 // 2,
            segment["width"],
        )
