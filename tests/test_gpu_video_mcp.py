# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end GPU test for the video-mcp server.

Synthesises a short NVENC-encoded H.264 chunk on disk in the on-disk layout
the hub recorder writes (chunk + JSON sidecar + ``.identity`` sidecar),
boots ``video_mcp_server`` against that directory in a subprocess, and
asks it to retrieve a frame via the ``get_frame_from_time`` MCP tool —
exercising the NVDEC decode + Pillow PNG re-encode path end-to-end.

Skipped cleanly when PyNvVideoCodec or NVENC/NVDEC hardware is missing.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import socket
import subprocess
import sys
import tempfile
import time
import uuid

import numpy as np
import pytest
import yaml

nvc = pytest.importorskip("PyNvVideoCodec")
PIL_Image = pytest.importorskip("PIL.Image")
pytest.importorskip("fastmcp")

from fastmcp import Client as McpClient  # noqa: E402


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
