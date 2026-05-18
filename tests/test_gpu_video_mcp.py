# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end GPU tests for the video-mcp server.

Two complementary scenarios:

1. ``test_get_frame_from_time_returns_valid_png`` — historical path:
   synthesises an NVENC chunk on disk and queries ``get_frame_from_time``.
2. ``test_get_latest_frame_via_live_hub`` — realtime path: rounds a
   synthetic NV12 frame through a live hub to ``get_latest_frame``.

IPC mechanics (FRAME_SIGNAL / FRAME_REQUEST / FRAME_DATA) live in
``video_mcp_server/__main__.py``. Skipped when PyNvVideoCodec or NVENC/
NVDEC hardware is missing.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import socket
import subprocess
import sys
import time
import uuid

import numpy as np
import pytest
import yaml

# PyNvVideoCodec loads libnvidia-encode.so.1 at import time and raises
# RuntimeError when NVENC drivers are absent (e.g. on CI runners).
try:
    import PyNvVideoCodec as nvc
except (ImportError, RuntimeError, OSError) as exc:
    pytest.skip(f"PyNvVideoCodec unavailable: {exc}", allow_module_level=True)

PIL_Image = pytest.importorskip("PIL.Image")
pytest.importorskip("fastmcp")

from fastmcp import Client as McpClient  # noqa: E402
from xr_ai_agent import PixelFormat  # noqa: E402
from xr_media_hub.ipc import ConnectorEndpoint  # noqa: E402


pytestmark = [pytest.mark.asyncio, pytest.mark.gpu]


# ── chunk synthesis ─────────────────────────────────────────────────────────


_WIDTH, _HEIGHT, _FRAMES, _FPS, _BITRATE = 320, 240, 10, 30, 1_500_000


def _synthetic_nv12(idx: int) -> np.ndarray:
    """Return one ``(H*3//2, W)`` uint8 NV12 frame with a diagonal stripe
    that drifts per call so the encoder sees real spatial and temporal
    entropy — without it H.264 collapses each frame to a tiny keyframe."""
    yy, xx   = np.indices((_HEIGHT, _WIDTH))
    y_plane  = ((xx + yy + idx * 8 + idx * 20 + 16) % 240).astype(np.uint8)
    uv_plane = np.full((_HEIGHT // 2, _WIDTH), 128, dtype=np.uint8)
    return np.concatenate([y_plane, uv_plane], axis=0)


def _encode_chunk(out_dir: pathlib.Path, pid: str, start_us: int) -> dict:
    """Encode ``_FRAMES`` synthetic NV12 frames into ``<start_us>.264`` plus
    matching JSON sidecar inside ``out_dir/<pid>/``. Returns the sidecar dict.

    Mirrors ``server-runtime/xr_media_hub/video/_recorder.py``: same encoder
    options, same NV12 layout, same ``.identity`` + ``<start_us>.json``
    layout the video-mcp ``ChunkStore`` reads back.
    """
    pid_dir = out_dir / pid
    pid_dir.mkdir(parents=True, exist_ok=True)
    (pid_dir / ".identity").write_text(pid, encoding="utf-8")

    encoder = nvc.CreateEncoder(
        _WIDTH, _HEIGHT, "NV12",
        True,  # usecpuinputbuffer
        gpu_id=0, codec="h264",
        preset="P4", tuning_info="high_quality",
        rc="vbr", fps=_FPS,
        bitrate=_BITRATE, maxbitrate=_BITRATE,
        bf=0, repeat_sps_pps=1,
    )

    buf = bytearray()
    for i in range(_FRAMES):
        chunk = encoder.Encode(_synthetic_nv12(i))
        if chunk:
            buf.extend(chunk)
    flushed = encoder.EndEncode()
    if flushed:
        buf.extend(flushed)

    h264_path = pid_dir / f"{start_us}.264"
    h264_path.write_bytes(bytes(buf))

    # end_us must be strictly > start_us so the ratio math in
    # get_frame_from_time picks a non-zero frame index.
    end_us = start_us + int(_FRAMES * 1_000_000 / _FPS)
    meta = {
        "start_us":   start_us,
        "end_us":     end_us,
        "num_frames": _FRAMES,
        "width":      _WIDTH,
        "height":     _HEIGHT,
        "size_bytes": len(buf),
    }
    (pid_dir / f"{start_us}.json").write_text(json.dumps(meta))
    return meta


# ── server lifecycle ────────────────────────────────────────────────────────


def _free_port() -> int:
    """Bind/release pattern: kernel won't immediately reuse the port, so by
    the time the server starts a few hundred ms later it's still ours."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _wait_ready(ready_file: pathlib.Path, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ready_file.exists():
            return
        if proc.poll() is not None:
            raise RuntimeError(
                f"video_mcp_server exited early with code {proc.returncode} "
                f"before touching the ready-file"
            )
        await asyncio.sleep(0.1)
    raise TimeoutError(f"video_mcp_server did not become ready within {timeout}s")


# ── test ────────────────────────────────────────────────────────────────────


async def test_get_frame_from_time_returns_valid_png(tmp_path: pathlib.Path) -> None:
    pid       = f"gpu_test_{uuid.uuid4().hex[:8]}"
    rec_dir   = tmp_path / "recordings"
    out_dir   = tmp_path / "queries"
    rec_dir.mkdir()
    out_dir.mkdir()

    start_us = int(time.time() * 1_000_000)
    try:
        meta = _encode_chunk(rec_dir, pid, start_us)
    except Exception as exc:  # noqa: BLE001
        # No NVENC hardware (or driver mismatch) — skip cleanly per the
        # task brief; we only care about the path when the GPU is present.
        pytest.skip(f"NVENC unavailable: {exc!r}")

    port      = _free_port()
    cfg_path  = tmp_path / "video_mcp_server.yaml"
    ready     = tmp_path / "video_mcp.ready"
    cfg_path.write_text(yaml.safe_dump({
        "recordings_dir": str(rec_dir),
        "out_dir":        str(out_dir),
        # Unique per-test IPC sockets so the server's ProcessorEndpoint
        # binds without colliding with a real hub on the dev box.
        "hub_pub":        f"ipc://{tmp_path}/hub_pub",
        "hub_push":       f"ipc://{tmp_path}/hub_push",
        "host":           "127.0.0.1",
        "port":           port,
        "gpu_id":         0,
    }))

    proc = subprocess.Popen(
        [sys.executable, "-m", "video_mcp_server",
         "--config", str(cfg_path), "--ready-file", str(ready)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        await _wait_ready(ready, proc, timeout=30.0)

        url = f"http://127.0.0.1:{port}/mcp"
        # Anchor inside the chunk window with second_ago=0 — forces the
        # NVDEC path (live IPC is bypassed when reference_time_us > 0).
        anchor_us = (meta["start_us"] + meta["end_us"]) // 2
        async with McpClient(url) as client:
            res = await client.call_tool(
                "get_frame_from_time",
                {"participant_id": pid, "second_ago": 0,
                 "reference_time_us": anchor_us},
            )

        # fastmcp Client returns a CallToolResult whose ``.data`` holds the
        # tool's JSON return when present; older shapes expose the same
        # payload under ``structured_content``.
        payload = getattr(res, "data", None) or getattr(res, "structured_content", None)
        assert isinstance(payload, dict), f"unexpected tool result: {res!r}"
        assert "error" not in payload, f"tool error: {payload.get('error')}"

        png_path = pathlib.Path(payload["path"])
        assert png_path.exists(), f"PNG not written: {png_path}"

        with PIL_Image.open(png_path) as img:
            img.load()
            assert img.format == "PNG"
            assert img.size == (_WIDTH, _HEIGHT), (
                f"PNG dims {img.size} != encoded {(_WIDTH, _HEIGHT)}"
            )

        assert payload["width"]  == _WIDTH
        assert payload["height"] == _HEIGHT
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


async def test_get_latest_frame_via_live_hub(
    tmp_path:   pathlib.Path,
    hub_addrs:  tuple[str, str],
    hub,                                            # noqa: ARG001 — fixture starts the in-process HubEndpoint
    settle,
) -> None:
    """Round-trip a frame through the hub to ``get_latest_frame``.

    Exercises the realtime path the historical test bypasses: connector
    PUSH → hub PUB → video-mcp ``ProcessorEndpoint`` SUB → ``FRAME_REQUEST``
    → ``FRAME_DATA`` → PNG. ``recordings_dir`` is omitted so the server
    builds ``store=None`` and exposes ``get_latest_frame`` instead of the
    historical tools.
    """
    pid       = f"gpu_live_{uuid.uuid4().hex[:8]}"
    out_dir   = tmp_path / "queries"
    out_dir.mkdir()

    hub_pull_addr, hub_pub_addr = hub_addrs

    # The conftest's `make_connector` factory caps max_frame_bytes at 64 KiB;
    # 320×240 NV12 is 115 200 B, so we instantiate ConnectorEndpoint directly
    # with a 2 MiB slot — well above one frame, well below NV12 4K (≈12 MiB).
    conn = ConnectorEndpoint(
        push_addr       = hub_pull_addr,
        sub_addr        = hub_pub_addr,
        connector_id    = f"conn_{uuid.uuid4().hex[:8]}",
        shm_name        = f"xr_test_{uuid.uuid4().hex[:10]}",
        num_slots       = 4,
        max_frame_bytes = 2 * 1024 * 1024,
    )

    port     = _free_port()
    cfg_path = tmp_path / "video_mcp_server.yaml"
    ready    = tmp_path / "video_mcp.ready"
    # recordings_dir intentionally omitted → store=None → get_latest_frame is
    # the only frame-fetching tool registered.
    cfg_path.write_text(yaml.safe_dump({
        "out_dir":  str(out_dir),
        "hub_pub":  hub_pub_addr,
        "hub_push": hub_pull_addr,
        "host":     "127.0.0.1",
        "port":     port,
        "gpu_id":   0,
    }))

    proc = subprocess.Popen(
        [sys.executable, "-m", "video_mcp_server",
         "--config", str(cfg_path), "--ready-file", str(ready)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        await _wait_ready(ready, proc, timeout=30.0)
        # ready-file is touched before video-mcp's SUB has finished the ZMQ
        # subscription handshake against the hub PUB; give it one hop.
        await settle()

        await conn.register()
        await conn.notify_participant_joined(pid)
        # Let the hub finish processing the registration + join before the
        # first FRAME_SIGNAL — otherwise the hub may drop the frame because
        # the pid → connector mapping isn't installed yet.
        await settle()

        pts_us  = int(time.time() * 1_000_000)
        frame   = _synthetic_nv12(0)
        await conn.push_frame(
            data           = frame.tobytes(),
            width          = _WIDTH,
            height         = _HEIGHT,
            fmt            = PixelFormat.NV12,
            pts_us         = pts_us,
            participant_id = pid,
            track_id       = "cam",
        )
        # Two hops to drain: connector PUSH → hub, then hub PUB → video-mcp
        # SUB where the ProcessorEndpoint caches the FRAME_SIGNAL so the
        # subsequent fetch_latest can resolve it.
        await settle()
        await settle()

        url = f"http://127.0.0.1:{port}/mcp"
        async with McpClient(url) as client:
            res = await client.call_tool("get_latest_frame", {"participant_id": pid})

        payload = getattr(res, "data", None) or getattr(res, "structured_content", None)
        assert isinstance(payload, dict), f"unexpected tool result: {res!r}"
        assert "error" not in payload, f"tool error: {payload.get('error')}"

        png_path = pathlib.Path(payload["path"])
        assert png_path.exists(), f"PNG not written: {png_path}"
        with PIL_Image.open(png_path) as img:
            img.load()
            assert img.format == "PNG"
            assert img.size == (_WIDTH, _HEIGHT)

        assert payload["width"]        == _WIDTH
        assert payload["height"]       == _HEIGHT
        assert payload["timestamp_us"] == pts_us
    finally:
        conn.stop()
        try:
            conn.close()
        except FileNotFoundError:
            # Connector teardown can race with the hub closing the shm
            # segment; the OS-level unlink may already be gone.
            pass
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
