# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GPU integration tests for ``xr_media_hub.video._recorder.VideoRecorder``.

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

# PyNvVideoCodec initialises NVENC at import time, so a missing
# libnvidia-encode.so.1 raises RuntimeError (not ImportError) — importorskip
# would let it escape and break collection on CI boxes without NVENC.
try:
    import PyNvVideoCodec  # noqa: F401  (import-only — used to detect NVENC availability)
except (ImportError, RuntimeError, OSError) as exc:
    pytest.skip(f"PyNvVideoCodec unavailable: {exc}", allow_module_level=True)

from xr_ai_agent import FrameSignal, PixelFormat, SlotView  # noqa: E402

from xr_media_hub.video import VideoRecorder, VideoRecorderConfig  # noqa: E402

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
