# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU smoke test for ai-services/stt-server.

Boots the real NeMo ASR server (parakeet-tdt-0.6b-v3) as a subprocess via
``uv run``, POSTs a short sine-wave WAV to ``/v1/audio/transcriptions``, and
asserts the response is well-formed JSON with a ``text`` field. The transcript
content is intentionally unchecked — a brief sine wave produces no semantic
output and the goal here is wiring (model loads, port serves, endpoint
returns 200), not recognition quality.
"""
from __future__ import annotations

import asyncio
import io
import json
import math
import os
import shutil
import signal
import socket
import struct
import subprocess
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

import pytest
from xr_ai_vllm import stop_persistent_servers

pytestmark = [pytest.mark.asyncio, pytest.mark.gpu]


_REPO_ROOT = Path(__file__).resolve().parents[1]
_STT_DIR   = _REPO_ROOT / "ai-services" / "stt-server"
_REPO_CACHE = _REPO_ROOT / "ai-services" / "models"
# Honor HF_HOME so callers with a non-default cache (e.g. a shared NAS or
# a CI bind-mount) don't trigger a cold redownload.
_HF_CACHE   = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
_CANDIDATE_CACHES = (_REPO_CACHE, _HF_CACHE)
_MODEL     = "nvidia/parakeet-tdt-0.6b-v3"

# Generous cold-start budget — first run may download the ~600 MB
# parakeet bundle from HF before model load.
_STARTUP_TIMEOUT_S   = 900.0
_SHUTDOWN_TIMEOUT_S  = 20.0


def _pick_free_port() -> int:
    # TOCTOU: another process can grab this port between bind() and the
    # server's own bind(). Acceptable for a single-host dev test; would
    # need a port-0 + parse-stdout pattern for hard determinism.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_sine_wav(seconds: float = 1.0, sample_rate: int = 16_000,
                   freq_hz: float = 440.0, amplitude: float = 0.2) -> bytes:
    """Return raw bytes of a mono 16-bit PCM WAV file (stdlib only)."""
    n_samples = int(seconds * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        peak = int(amplitude * 32767)
        frames = bytearray()
        for n in range(n_samples):
            v = int(peak * math.sin(2 * math.pi * freq_hz * n / sample_rate))
            frames += struct.pack("<h", v)
        wf.writeframes(bytes(frames))
    return buf.getvalue()


def _resolve_cached_cache_root() -> Path | None:
    """Return the first cache root containing parakeet weights, or None.

    Checks both the repo-local model cache (``ai-services/models``, the
    canonical location wired into the stt-server YAML) and the standard
    Hugging Face cache (``~/.cache/huggingface``) so weights downloaded by
    any other tool are reused without a redundant fetch.
    """
    for root in _CANDIDATE_CACHES:
        for sub in ("nemo", "huggingface", "hub"):
            base = root / sub
            if not base.is_dir():
                continue
            for pattern in ("*.nemo", "*.safetensors"):
                if next(base.rglob(pattern), None) is not None:
                    return root
    return None


def _build_multipart(field_name: str, filename: str, payload: bytes,
                     extra_fields: dict[str, str]) -> tuple[bytes, str]:
    """Hand-rolled multipart/form-data — saves pulling in requests/httpx."""
    boundary = f"----xrai-stt-{uuid.uuid4().hex}"
    crlf     = b"\r\n"
    body     = bytearray()
    for key, val in extra_fields.items():
        body += f"--{boundary}".encode() + crlf
        body += f'Content-Disposition: form-data; name="{key}"'.encode() + crlf + crlf
        body += val.encode() + crlf
    body += f"--{boundary}".encode() + crlf
    body += (
        f'Content-Disposition: form-data; name="{field_name}"; '
        f'filename="{filename}"'
    ).encode() + crlf
    body += b"Content-Type: audio/wav" + crlf + crlf
    body += payload + crlf
    body += f"--{boundary}--".encode() + crlf
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _health_ok(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=2,
        ) as r:
            return r.status == 200
    # Connection refused / DNS / timeout during the poll loop — caller retries.
    except Exception:
        return False


async def _wait_for_health(port: int, deadline_s: float) -> None:
    loop  = asyncio.get_running_loop()
    until = loop.time() + deadline_s
    while loop.time() < until:
        if await loop.run_in_executor(None, _health_ok, port):
            return
        await asyncio.sleep(1.0)
    raise TimeoutError(f"stt server did not become healthy on :{port}")


def _terminate(proc: subprocess.Popen) -> bytes:
    """Reap the launcher and return whatever it buffered on stdout.

    The launcher spawns a *detached* ``--_serve`` server in its own session
    (see ``stt_server/__main__.py``); that child outlives the launcher's
    SIGTERM and keeps the inherited write-end of this pipe open. A plain
    ``read()``-to-EOF would therefore block forever on a pipe the persistent
    server never closes — callers must stop that server first via
    ``stop_persistent_servers``. Drain non-blocking regardless so teardown can
    never wedge even if the server is somehow still up.
    """
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=_SHUTDOWN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    if proc.stdout is None:
        return b""
    try:
        os.set_blocking(proc.stdout.fileno(), False)
        return proc.stdout.read() or b""
    except (ValueError, OSError):
        return b""  # pipe already closed by the kernel during teardown


@pytest.fixture
def stt_yaml(tmp_path: Path) -> tuple[Path, int]:
    """Write a temp YAML pointing at whichever cache root holds the weights."""
    port = _pick_free_port()
    cache_root = _resolve_cached_cache_root() or _REPO_CACHE
    cfg  = (
        "model: nvidia/parakeet-tdt-0.6b-v3\n"
        "device: auto\n"
        f"port: {port}\n"
        'host: "127.0.0.1"\n'
        f"model_cache: {cache_root}\n"
    )
    path = tmp_path / "stt_server.yaml"
    path.write_text(cfg)
    return path, port


async def test_stt_server_transcribes_sine_wav(stt_yaml: tuple[Path, int]) -> None:
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")
    if not _STT_DIR.exists():
        pytest.skip(f"stt-server source tree missing: {_STT_DIR}")

    # The server downloads parakeet weights on first run into the resolved
    # cache root (see stt_yaml fixture); subsequent runs reuse them.

    cfg_path, port = stt_yaml

    proc = subprocess.Popen(
        ["uv", "run", "--directory", str(_STT_DIR),
         "stt_server", "--config", str(cfg_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    try:
        try:
            await _wait_for_health(port, _STARTUP_TIMEOUT_S)
        except TimeoutError:
            # Drain whatever the subprocess printed so the failure is debuggable.
            stop_persistent_servers([("stt", port)])
            tail = _terminate(proc).decode(errors="replace")
            pytest.fail(
                f"stt server failed to start on :{port}\n"
                f"--- subprocess output ---\n{tail[-4000:]}",
            )

        wav_bytes = _make_sine_wav()
        body, content_type = _build_multipart(
            "file", "sine.wav", wav_bytes,
            extra_fields={"model": _MODEL, "response_format": "json"},
        )

        loop = asyncio.get_running_loop()

        def _post() -> tuple[int, bytes]:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/audio/transcriptions",
                data=body,
                method="POST",
                headers={"Content-Type": content_type},
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.status, r.read()
            except urllib.error.HTTPError as e:
                return e.code, e.read()

        status, payload = await loop.run_in_executor(None, _post)

        assert status == 200, f"unexpected status {status}: {payload[:500]!r}"
        obj = json.loads(payload.decode())
        assert "text" in obj, f"missing 'text' key in response: {obj}"
        assert isinstance(obj["text"], str), f"'text' is not a string: {obj}"

    finally:
        # The launcher's detached `--_serve` server (own session) survives the
        # launcher's SIGTERM and holds ~3 GB of GPU plus the inherited stdout
        # pipe. Stop it by port first — that frees the GPU for the next gpu
        # test and lets `_terminate`'s drain reach EOF instead of blocking.
        stop_persistent_servers([("stt", port)])
        _terminate(proc)
