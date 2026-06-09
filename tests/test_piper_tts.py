# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Smoke test for ai-services/tts/piper.

Spawns the Piper TTS server as a subprocess (out of its own venv — the
test harness must not import the heavy piper/fastapi deps) and round-trips
a tiny synthesis request through the OpenAI-compatible
``POST /v1/audio/speech`` endpoint.

CPU-only: Piper runs ONNX on CPU at ~100 ms/sentence. No ``gpu`` marker,
so CI picks this up. Skipped cleanly when the environment can't support it:
``uv`` is missing, the piper venv hasn't been ``uv sync``'d, or the
configured voice can't be obtained (offline with an empty cache, or a
transient HuggingFace download failure — the server signals this with a
dedicated exit code). Any other early exit fails the test with the server's
captured output so the cause is visible in CI.
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

pytestmark = pytest.mark.asyncio


_REPO_ROOT     = Path(__file__).resolve().parents[1]
_PIPER_PROJECT = _REPO_ROOT / "ai-services" / "tts" / "piper"
_PIPER_BIN     = _PIPER_PROJECT / ".venv" / "bin" / "piper_tts_server"
_PIPER_YAML    = _PIPER_PROJECT / "piper_tts_server.yaml"
_DEFAULT_PORT  = 8105

# Must match _EXIT_VOICE_UNAVAILABLE in piper_tts_server/__main__.py: the
# server uses this exit code when the voice can't be obtained for
# environmental reasons (offline empty cache / transient HF download failure),
# which the smoke test treats as skip rather than fail.
_EXIT_VOICE_UNAVAILABLE = 3


class _ServerExited(Exception):
    """Raised when piper_tts_server exits before binding its port.

    Carries the process return code and the captured stdout+stderr so callers
    can decide whether to skip (environmental) or fail (real error) — and so
    the real cause is visible instead of a bare "exited with code N".
    """

    def __init__(self, returncode: int, output: str) -> None:
        self.returncode = returncode
        self.output = output
        super().__init__(f"piper_tts_server exited early with code {returncode}")


def _pick_port(preferred: int) -> int:
    """Return *preferred* if free, else ask the kernel for an ephemeral one."""
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


def _voice_cached(voice: str, hf_cache: Path) -> bool:
    """True iff both the .onnx and .onnx.json for *voice* are present in cache.

    Walks the HF cache layout directly so we don't pull huggingface_hub into
    the test venv just for an offline probe.
    """
    parts = voice.split("-")
    if len(parts) < 3:
        return False
    locale, speaker, quality = parts[0], parts[1], "-".join(parts[2:])
    lang = locale.split("_")[0]
    rel  = f"{lang}/{locale}/{speaker}/{quality}/{voice}"
    snapshots = hf_cache / "models--rhasspy--piper-voices" / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(
        (snap / f"{rel}.onnx").exists() and (snap / f"{rel}.onnx.json").exists()
        for snap in snapshots.iterdir()
    )


async def _wait_for_port(port: int, *, proc: subprocess.Popen, timeout: float) -> None:
    """Poll the bind port until it accepts a TCP connection.

    Raises ``_ServerExited`` (with the captured output) if the process dies
    before binding, or ``TimeoutError`` if the port never opens.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if proc.poll() is not None:
            # The process is dead, so its pipe won't block — read the captured
            # stdout+stderr and surface it instead of just the exit code.
            output = proc.stdout.read().decode("utf-8", "replace") if proc.stdout else ""
            raise _ServerExited(proc.returncode, output)
        if _port_open(port):
            return
        await asyncio.sleep(0.2)
    raise TimeoutError(f"piper_tts_server did not open port {port} within {timeout}s")


def _post_speech(port: int, voice: str, text: str) -> bytes:
    payload = json.dumps({
        "model":           voice,
        "input":           text,
        "voice":           voice,
        "response_format": "wav",
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/audio/speech",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        assert resp.status == 200, f"unexpected status {resp.status}"
        return resp.read()


async def test_piper_tts_smoke(tmp_path: Path) -> None:
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")
    if not _PIPER_BIN.exists():
        subprocess.run(
            ["uv", "sync", "--directory", str(_PIPER_PROJECT)],
            check=True,
        )
        if not _PIPER_BIN.exists():
            pytest.skip(f"uv sync did not produce {_PIPER_BIN}")

    ref_cfg     = yaml.safe_load(_PIPER_YAML.read_text())
    voice       = ref_cfg["voice"]
    model_cache = (_PIPER_YAML.parent / ref_cfg.get("model_cache", "../models")).resolve()
    # The piper server eagerly resolves + downloads the configured voice
    # on startup (see ai-services/tts/piper/piper_tts_server/__main__.py),
    # so we don't pre-check voice cache state here.

    port = _pick_port(_DEFAULT_PORT)

    cfg_path = tmp_path / "piper_tts_server.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "voice":       voice,
        "port":        port,
        "host":        "127.0.0.1",
        "use_cuda":    False,
        "model_cache": str(model_cache),
    }))

    env = {**os.environ, "XR_AI_LOG_ROOT": str(tmp_path / "logs")}

    proc = subprocess.Popen(
        [str(_PIPER_BIN), "--config", str(cfg_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,  # SIGTERM must reach the whole tree
    )

    try:
        # First-run voice download (~50–200 MB) plus ONNX init can take a
        # couple of minutes on a cold cache; reuse is sub-second.
        try:
            await _wait_for_port(port, proc=proc, timeout=300.0)
        except _ServerExited as exc:
            tail = exc.output.strip()[-2000:]
            # A voice-unavailable exit is environmental, not a code bug: an
            # offline empty cache or a transient HuggingFace download failure
            # (HTTP 429 rate-limit on the anonymous voice download). Skip
            # cleanly rather than fail the suite — the smoke test only asserts
            # the server path when the voice can actually be obtained.
            if exc.returncode == _EXIT_VOICE_UNAVAILABLE:
                pytest.skip(
                    "piper voice could not be obtained (offline cache or "
                    f"transient HuggingFace download failure):\n{tail}"
                )
            # Any other early exit is a real failure — surface the captured
            # server output so it's diagnosable from the CI log.
            pytest.fail(
                f"piper_tts_server exited with code {exc.returncode}:\n{tail}"
            )
        body = await asyncio.get_running_loop().run_in_executor(
            None, _post_speech, port, voice, "Hello, world.",
        )
        # Regression #194: whitespace-only input must return a valid (empty)
        # WAV, not an HTTP 500 (wave.Error: # channels not specified).
        # _post_speech asserts status 200, so a 500 here fails the test.
        empty_body = await asyncio.get_running_loop().run_in_executor(
            None, _post_speech, port, voice, "   ",
        )
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    assert body, "empty response body"
    assert body[:4] == b"RIFF", f"expected WAV RIFF header, got {body[:8]!r}"
    # Whitespace-only input → a valid WAV header (zero audio frames), not a 500.
    assert empty_body[:4] == b"RIFF", (
        f"expected WAV for whitespace input, got {empty_body[:8]!r}"
    )
