# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU smoke test for services/vlm-server.

Spawns the real ``vlm_server`` subprocess against a generated temp YAML, waits
for the HTTP port to open, and requires the first 1280x720 image stream to
complete within the sample's 120-second inference budget. A fresh managed
container must not depend on a previous container's compilation cache.

Skipped automatically when ``uv`` is missing.
"""
from __future__ import annotations

import asyncio
import io
import os
import shutil
import signal
import socket
import sys
import tempfile
import time
from pathlib import Path

import pytest
import yaml
from PIL import Image
from xr_ai_models import load_models_config_from_dict, make_vlm

from _helpers import kill_orphan_vllm
from _helpers_vlm_stream import require_complete_vlm_stream

pytestmark = [pytest.mark.asyncio, pytest.mark.gpu]


_REPO_ROOT       = Path(__file__).resolve().parent.parent
_VLM_SERVER_DIR  = _REPO_ROOT / "services" / "vlm-server"
_DEFAULT_WEIGHTS = Path("~/.cache/huggingface/hub/models--nvidia--Cosmos3-Nano").expanduser()

# Allow checkpoint downloads and cold backend initialization independently of
# the bounded first-image inference check below.
_STARTUP_TIMEOUT_S   = 1800.0
_SHUTDOWN_TIMEOUT_S  = 30.0


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _tcp_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        fut = asyncio.open_connection(host, port)
        reader, writer = await asyncio.wait_for(fut, timeout=timeout)
        writer.close()
        try:
            await writer.wait_closed()
        # The probe only cares that the port accepted a connection; teardown errors are noise.
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
    # SIGTERM didn't reap within the grace window; escalate to SIGKILL below.
    except asyncio.TimeoutError:
        pass
    proc.kill()
    try:
        await asyncio.wait_for(proc.wait(), timeout=5.0)
    # Best-effort reap after SIGKILL — nothing left to do if the wait still times out.
    except asyncio.TimeoutError:
        pass


async def test_vlm_server_chat_completions_smoke(monkeypatch):
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")
    port = _pick_free_port()

    # Always point at the standard HF cache. vLLM uses cached weights if
    # present and otherwise downloads on first start; either way the test
    # passes without manual prereqs.
    model_cache = _DEFAULT_WEIGHTS.parents[1]  # ~/.cache/huggingface

    with tempfile.TemporaryDirectory(prefix="vlm_smoke_") as td:
        td_path = Path(td)
        cfg_yaml = td_path / "vlm_server.yaml"
        ready_file = td_path / "ready"
        cfg = {
            "model": "nvidia/Cosmos3-Nano",
            "host": "127.0.0.1",
            "port": port,
            "served_model_name": "vlm",
            "model_cache": str(model_cache),
            "max_num_seqs": 1,
            "tensor_parallel_size": 1,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.80,
            "enforce_eager": True,  # skip CUDA graph capture for faster smoke
            "async_scheduling": True,
            "hf_overrides": {
                "architectures": ["Cosmos3ForConditionalGeneration"],
            },
            "mm_encoder_tp_mode": "data",
            "max_images_per_prompt": 1,
            "max_videos_per_prompt": 0,
            # docker backend: nvcc + flashinfer are pre-built in the NGC image.
            "vllm_backend": "docker",
            "vllm_image": "nvcr.io/nvidia/vllm:26.09-py3",
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

            image = io.BytesIO()
            Image.new("RGB", (1280, 720), color=(128, 128, 128)).save(image, format="JPEG", quality=90)
            models = load_models_config_from_dict({"vlm": {
                "adapter": {"preset": "cosmos3_nano_reasoner"},
                "endpoint": {"base_url": f"http://127.0.0.1:{port}"},
            }})
            vlm = make_vlm(models, "vlm")
            content_chunks = []
            try:
                with require_complete_vlm_stream(monkeypatch):
                    async with asyncio.timeout(120):
                        async for chunk in vlm.stream_images(
                            [image.getvalue()], "What is the dominant color?",
                            system_prompt="Answer with one word.", max_tokens=4, timeout=120,
                        ):
                            content_chunks.append(chunk)
            finally:
                await vlm.close()
            content = "".join(content_chunks).strip()
            assert content, "stream returned no content"
            assert "gray" in content.casefold() or "grey" in content.casefold(), content
            assert "<think>" not in content.casefold(), (
                "Cosmos3 Reasoner leaked reasoning markup into streamed content"
            )
        finally:
            await _terminate(proc)
            kill_orphan_vllm(port)
