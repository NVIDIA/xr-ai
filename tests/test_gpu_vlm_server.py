# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU smoke test for ai-services/vlm-server.

Spawns the real ``vlm_server`` subprocess against a generated temp YAML, waits
for the HTTP port to open, issues one chat-completions request with a tiny
embedded PNG, and checks for a non-empty content string.

Skipped automatically when ``uv`` is missing or no Cosmos-Reason1-7B weights
are present on disk.
"""
from __future__ import annotations

import asyncio
import base64
import os
import shutil
import signal
import socket
import struct
import sys
import tempfile
import time
import zlib
from pathlib import Path

import pytest
import yaml

from _helpers import kill_orphan_vllm

pytestmark = [pytest.mark.asyncio, pytest.mark.gpu]


_REPO_ROOT       = Path(__file__).resolve().parent.parent
_VLM_SERVER_DIR  = _REPO_ROOT / "ai-services" / "vlm-server"
_DEFAULT_WEIGHTS = Path("~/.cache/huggingface/hub/models--nvidia--Cosmos-Reason1-7B").expanduser()
_CONFIGURED_WEIGHTS = _REPO_ROOT / "ai-services" / "models" / "hub" / "models--nvidia--Cosmos-Reason1-7B"

_STARTUP_TIMEOUT_S   = 240.0
_SHUTDOWN_TIMEOUT_S  = 30.0


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _tiny_png_bytes(size: int = 32) -> bytes:
    """Return a minimal valid 32x32 grayscale PNG (no external deps)."""
    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 0, 0, 0, 0)  # 8-bit grayscale
    # one filter byte (0) + `size` pixels per row
    raw = b"".join(b"\x00" + bytes([(i * 8) & 0xFF] * size) for i in range(size))
    idat = zlib.compress(raw, 9)
    return sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def _weights_available() -> bool:
    return _DEFAULT_WEIGHTS.exists() or _CONFIGURED_WEIGHTS.exists()


async def _tcp_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        fut = asyncio.open_connection(host, port)
        reader, writer = await asyncio.wait_for(fut, timeout=timeout)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return True
    except (OSError, asyncio.TimeoutError):
        return False


async def _wait_for_port(host: str, port: int, deadline: float, proc: asyncio.subprocess.Process) -> None:
    backoff = 0.5
    while time.monotonic() < deadline:
        if proc.returncode is not None:
            raise RuntimeError(f"vlm_server exited early with code {proc.returncode}")
        if await _tcp_open(host, port):
            return
        await asyncio.sleep(backoff)
        backoff = min(backoff * 1.5, 4.0)
    raise TimeoutError(f"vlm_server did not open {host}:{port} within {_STARTUP_TIMEOUT_S}s")


async def _terminate(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), timeout=_SHUTDOWN_TIMEOUT_S)
        return
    except asyncio.TimeoutError:
        pass
    proc.kill()
    try:
        await asyncio.wait_for(proc.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        pass


async def test_vlm_server_chat_completions_smoke():
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")
    if not _weights_available():
        pytest.skip(
            "Cosmos-Reason1-7B weights not on disk; pre-download to "
            f"{_DEFAULT_WEIGHTS} or {_CONFIGURED_WEIGHTS}",
        )
    try:
        import httpx
    except ImportError:
        pytest.skip("httpx not installed")

    port = _pick_free_port()

    # Prefer the cache that already has the weights so vLLM doesn't redownload.
    if _DEFAULT_WEIGHTS.exists():
        model_cache = _DEFAULT_WEIGHTS.parents[1]  # ~/.cache/huggingface
    else:
        model_cache = _CONFIGURED_WEIGHTS.parents[1]

    with tempfile.TemporaryDirectory(prefix="vlm_smoke_") as td:
        td_path = Path(td)
        cfg_yaml = td_path / "vlm_server.yaml"
        ready_file = td_path / "ready"
        cfg = {
            "model": "nvidia/Cosmos-Reason1-7B",
            "host": "127.0.0.1",
            "port": port,
            "served_model_name": "vlm",
            "model_cache": str(model_cache),
            "max_num_seqs": 1,
            "tensor_parallel_size": 1,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.80,
            "enforce_eager": True,  # skip CUDA graph capture for faster smoke
            "max_images_per_prompt": 1,
            "max_videos_per_prompt": 0,
            "vllm_backend": "pip",
        }
        cfg_yaml.write_text(yaml.safe_dump(cfg))

        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = await asyncio.create_subprocess_exec(
            "uv", "run", "--directory", str(_VLM_SERVER_DIR),
            "vlm_server",
            "--config", str(cfg_yaml),
            "--ready-file", str(ready_file),
            stdout=sys.stdout, stderr=sys.stderr,
            env=env,
        )

        try:
            deadline = time.monotonic() + _STARTUP_TIMEOUT_S
            await _wait_for_port("127.0.0.1", port, deadline, proc)

            png_b64 = base64.b64encode(_tiny_png_bytes(32)).decode("ascii")
            payload = {
                "model": "vlm",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{png_b64}"}},
                        {"type": "text", "text": "describe this image"},
                    ],
                }],
                "max_tokens": 20,
                "temperature": 0,
            }
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json=payload,
                )
            assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:500]}"
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            assert isinstance(content, str) and content.strip(), f"empty content: {data!r}"
        finally:
            await _terminate(proc)
            kill_orphan_vllm(port)
