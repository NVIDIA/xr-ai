# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GPU smoke test for ai-services/tts/magpie.

Spawns the Magpie NeMo TTS server as a subprocess (out of its own venv —
the test harness must not pull NeMo / lightning) and round-trips a short
synthesis request through the OpenAI-compatible
``POST /v1/audio/speech`` endpoint.

Marked ``gpu``: NeMo MagpieTTSModel loads ~1 GiB onto CUDA and synthesis
takes seconds on CPU; CI skips it. Also skipped cleanly if the magpie
venv hasn't been ``uv sync``'d.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import urllib.request
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.asyncio, pytest.mark.gpu]


_REPO_ROOT      = Path(__file__).resolve().parents[1]
_MAGPIE_PROJECT = _REPO_ROOT / "ai-services" / "tts" / "magpie"
_MAGPIE_BIN     = _MAGPIE_PROJECT / ".venv" / "bin" / "magpie_tts_server"
_MAGPIE_YAML    = _MAGPIE_PROJECT / "magpie_tts_server.yaml"
_DEFAULT_PORT   = 8104


def _pick_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        # EADDRINUSE on the preferred port — fall through to an ephemeral bind.
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


async def _wait_for_port(port: int, *, proc: subprocess.Popen, timeout: float) -> None:
    """Poll the bind port until accept-able or proc dies."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"magpie_tts_server exited early with code {proc.returncode}"
            )
        if _port_open(port):
            return
        await asyncio.sleep(0.5)
    raise TimeoutError(f"magpie_tts_server did not open port {port} within {timeout}s")


def _post_speech(port: int, model: str, text: str) -> bytes:
    payload = json.dumps({
        "model":           model,
        "input":           text,
        "voice":           "default",
        "response_format": "wav",
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/audio/speech",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        assert resp.status == 200, f"unexpected status {resp.status}"
        return resp.read()


async def test_magpie_tts_smoke(tmp_path: Path) -> None:
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")
    if not _MAGPIE_BIN.exists():
        subprocess.run(
            ["uv", "sync", "--directory", str(_MAGPIE_PROJECT)],
            check=True,
        )
        if not _MAGPIE_BIN.exists():
            pytest.skip(f"uv sync did not produce {_MAGPIE_BIN}")

    ref_cfg     = yaml.safe_load(_MAGPIE_YAML.read_text())
    model_name  = ref_cfg["model"]
    sample_rate = int(ref_cfg.get("sample_rate", 22050))
    model_cache = (_MAGPIE_YAML.parent / ref_cfg.get("model_cache", "../models")).resolve()

    port = _pick_port(_DEFAULT_PORT)

    cfg_path = tmp_path / "magpie_tts_server.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "model":       model_name,
        "device":      "auto",
        "port":        port,
        "host":        "127.0.0.1",
        "sample_rate": sample_rate,
        "model_cache": str(model_cache),
    }))

    env = {**os.environ, "XR_AI_LOG_ROOT": str(tmp_path / "logs")}

    proc = subprocess.Popen(
        [str(_MAGPIE_BIN), "--config", str(cfg_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )

    try:
        # NeMo first-load + CUDA init + cold weight download can take a
        # while; budget for a full ~1 GB pull on a fresh machine.
        await _wait_for_port(port, proc=proc, timeout=900.0)
        body = await asyncio.get_running_loop().run_in_executor(
            None, _post_speech, port, model_name, "Hello, world.",
        )
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    assert body, "empty response body"
    assert body[:4] == b"RIFF", f"expected WAV RIFF header, got {body[:8]!r}"
