# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU smoke tests for the three vLLM-backed LLM servers.

Three tests, run serially by pytest:

* ``test_llama_nemotron_tool_call`` — spawns ``llama_nemotron_llm_server``
  (pip backend), POSTs a chat-completions request with a tool spec, and
  asserts the response includes ``tool_calls`` whose ``arguments`` is
  parseable JSON.

* ``test_nemotron3_nano_persistent`` — spawns
  ``nemotron3_nano_llm_server`` (persistent mode), makes one tiny request,
  SIGTERMs the *wrapper* (vLLM keeps running in its own session group),
  re-spawns the wrapper, and verifies the second start is fast — meaning
  the persistent inner vLLM was reused.  Cleans up the persistent server
  via ``stop_persistent_servers`` so the next test on a single 46 GB GPU
  does not OOM.

* ``test_nemotron_omni_multimodal`` — spawns ``nemotron_omni_llm_server``
  and sends a text + image chat-completion, asserting HTTP 200 + non-empty
  content.

Each test skips automatically when ``uv`` is missing or the model weights
are not present in the HuggingFace cache.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import signal
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

import pytest
import yaml

from xr_ai_vllm import stop_persistent_servers

pytestmark = [pytest.mark.asyncio, pytest.mark.gpu]


_REPO_ROOT = Path(__file__).resolve().parents[1]
_LLAMA_DIR = _REPO_ROOT / "ai-services" / "llm" / "llama_nemotron"
_N3_DIR    = _REPO_ROOT / "ai-services" / "llm" / "nemotron3_nano"
_OMNI_DIR  = _REPO_ROOT / "ai-services" / "llm" / "nemotron_omni"

_HF_HUB_DIRS = [
    Path("~/.cache/huggingface/hub").expanduser(),
    _REPO_ROOT / "models" / "hub",
    _REPO_ROOT / "ai-services" / "models" / "hub",
]

# 30B model cold-start (weight load + FlashInfer JIT) is multi-minute on
# pre-Blackwell; 8B is faster but vLLM startup itself eats ~60s.
_COLD_STARTUP_TIMEOUT_S = 900.0
_HOT_STARTUP_TIMEOUT_S  = 60.0
_SHUTDOWN_TIMEOUT_S     = 30.0


# ── helpers ─────────────────────────────────────────────────────────────────


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _hf_model_cached(repo_id: str) -> bool:
    """True iff the HF hub directory for *repo_id* exists with safetensors."""
    folder = "models--" + repo_id.replace("/", "--")
    for root in _HF_HUB_DIRS:
        candidate = root / folder
        if candidate.is_dir() and any(candidate.rglob("*.safetensors")):
            return True
    return False


def _hf_any_cached(repo_ids: list[str]) -> tuple[str, Path] | None:
    """Return (repo_id, hub_root) of the first cached repo, or None."""
    for repo_id in repo_ids:
        folder = "models--" + repo_id.replace("/", "--")
        for root in _HF_HUB_DIRS:
            candidate = root / folder
            if candidate.is_dir() and any(candidate.rglob("*.safetensors")):
                return repo_id, root.parent  # parent so HF_HOME has /hub child
    return None


def _tiny_png_bytes(size: int = 32) -> bytes:
    """Return bytes of a minimal valid grayscale PNG (stdlib only)."""
    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    sig  = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 0, 0, 0, 0)
    raw  = b"".join(b"\x00" + bytes([(i * 8) & 0xFF] * size) for i in range(size))
    idat = zlib.compress(raw, 9)
    return sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def _health_ok(port: int, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=timeout,
        ) as r:
            return r.status == 200
    except Exception:
        return False


async def _wait_for_health(
    port: int, deadline_s: float, proc: subprocess.Popen,
) -> None:
    loop  = asyncio.get_running_loop()
    until = loop.time() + deadline_s
    while loop.time() < until:
        if proc.poll() is not None:
            raise RuntimeError(
                f"wrapper exited early with code {proc.returncode} before "
                f":{port} became healthy",
            )
        if await loop.run_in_executor(None, _health_ok, port):
            return
        await asyncio.sleep(1.0)
    raise TimeoutError(f"server did not become healthy on :{port} within {deadline_s}s")


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=_SHUTDOWN_TIMEOUT_S)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _post_json(url: str, payload: dict, timeout: float = 120.0) -> tuple[int, dict]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {}


# ── test 1: llama_nemotron tool-call ─────────────────────────────────────────


_LLAMA_MODEL = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"


async def test_llama_nemotron_tool_call(tmp_path: Path) -> None:
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")
    if not _LLAMA_DIR.exists():
        pytest.skip(f"llama_nemotron source tree missing: {_LLAMA_DIR}")
    if not _hf_model_cached(_LLAMA_MODEL):
        pytest.skip(
            f"HF weights for {_LLAMA_MODEL} not cached under any of {_HF_HUB_DIRS}",
        )

    port = _pick_free_port()
    cfg = {
        "model":                  _LLAMA_MODEL,
        "host":                   "127.0.0.1",
        "port":                   port,
        "served_model_name":      "llm",
        "model_cache":            str(Path("~/.cache/huggingface").expanduser()),
        "max_num_seqs":           1,
        "tensor_parallel_size":   1,
        "max_model_len":          4096,
        "gpu_memory_utilization": 0.60,
        "enforce_eager":          True,
        "tool_call_parser":       "llama3_json",
        "enable_tool_choice":     True,
        "vllm_backend":           "pip",
    }
    cfg_yaml = tmp_path / "llama_nemotron_llm_server.yaml"
    cfg_yaml.write_text(yaml.safe_dump(cfg))

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        ["uv", "run", "--directory", str(_LLAMA_DIR),
         "llama_nemotron_llm_server", "--config", str(cfg_yaml)],
        env=env,
    )

    try:
        await _wait_for_health(port, _COLD_STARTUP_TIMEOUT_S, proc)

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name, e.g. 'Paris'.",
                        },
                    },
                    "required": ["location"],
                },
            },
        }]
        payload = {
            "model":       "llm",
            "messages":    [{"role": "user",
                             "content": "What's the weather in Paris right now?"}],
            "tools":       tools,
            "tool_choice": "auto",
            "max_tokens":  64,
            "temperature": 0,
        }
        loop = asyncio.get_running_loop()
        status, data = await loop.run_in_executor(
            None, _post_json, f"http://127.0.0.1:{port}/v1/chat/completions", payload,
        )
        assert status == 200, f"HTTP {status}: {data!r}"
        msg = data["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        assert tool_calls, f"no tool_calls in response: {msg!r}"
        args_str = tool_calls[0]["function"]["arguments"]
        args = json.loads(args_str)
        assert isinstance(args, dict), f"tool args not a dict: {args_str!r}"
    finally:
        _terminate(proc)


# ── test 2: nemotron3_nano persistence ──────────────────────────────────────


_N3_MODELS = [
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
]


async def test_nemotron3_nano_persistent(tmp_path: Path) -> None:
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")
    if not _N3_DIR.exists():
        pytest.skip(f"nemotron3_nano source tree missing: {_N3_DIR}")

    found = _hf_any_cached(_N3_MODELS)
    if not found:
        pytest.skip(
            f"No Nemotron-3-Nano-30B weights cached (checked {_N3_MODELS})",
        )
    _, hf_root = found

    port = _pick_free_port()
    cfg = {
        "host":                   "127.0.0.1",
        "port":                   port,
        "served_model_name":      "llm",
        "model_cache":            str(hf_root),
        "max_num_seqs":           1,
        "tensor_parallel_size":   1,
        "max_model_len":          4096,
        "gpu_memory_utilization": 0.85,
        "enforce_eager":          True,
        "vllm_backend":           "pip",
    }
    cfg_yaml = tmp_path / "nemotron3_nano_llm_server.yaml"
    cfg_yaml.write_text(yaml.safe_dump(cfg))

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    def _spawn() -> subprocess.Popen:
        return subprocess.Popen(
            ["uv", "run", "--directory", str(_N3_DIR),
             "nemotron3_nano_llm_server", "--config", str(cfg_yaml)],
            env=env,
        )

    proc1 = _spawn()
    cold_start = time.monotonic()
    try:
        try:
            await _wait_for_health(port, _COLD_STARTUP_TIMEOUT_S, proc1)
            cold_elapsed = time.monotonic() - cold_start

            # Tiny request — confirm the server actually serves.
            payload = {
                "model":       "llm",
                "messages":    [{"role": "user", "content": "Reply with the word OK."}],
                "max_tokens":  8,
                "temperature": 0,
            }
            loop = asyncio.get_running_loop()
            status, _ = await loop.run_in_executor(
                None, _post_json, f"http://127.0.0.1:{port}/v1/chat/completions", payload,
            )
            assert status == 200, f"first request returned HTTP {status}"
        finally:
            _terminate(proc1)

        # Inner vLLM is in its own session group; it should still be serving.
        assert _health_ok(port), (
            "persistent vLLM died with the wrapper — persistence broken"
        )

        proc2 = _spawn()
        hot_start = time.monotonic()
        try:
            await _wait_for_health(port, _HOT_STARTUP_TIMEOUT_S, proc2)
            hot_elapsed = time.monotonic() - hot_start
            assert hot_elapsed < cold_elapsed / 2, (
                f"hot restart ({hot_elapsed:.1f}s) was not meaningfully faster "
                f"than cold ({cold_elapsed:.1f}s) — vLLM was not reused"
            )
        finally:
            _terminate(proc2)
    finally:
        # vLLM lives in its own session past wrapper SIGTERM. Stop it
        # explicitly so the next test on a shared 46 GB GPU does not OOM.
        stop_persistent_servers([("nemotron3_nano", port)])


# ── test 3: nemotron_omni multimodal ────────────────────────────────────────


_OMNI_MODELS = [
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
]


async def test_nemotron_omni_multimodal(tmp_path: Path) -> None:
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")
    if not _OMNI_DIR.exists():
        pytest.skip(f"nemotron_omni source tree missing: {_OMNI_DIR}")

    found = _hf_any_cached(_OMNI_MODELS)
    if not found:
        pytest.skip(
            f"No Nemotron-3-Nano-Omni-30B weights cached (checked {_OMNI_MODELS})",
        )
    _, hf_root = found

    port = _pick_free_port()
    cfg = {
        "host":                   "127.0.0.1",
        "port":                   port,
        "served_model_name":      "llm",
        "model_cache":            str(hf_root),
        "max_num_seqs":           1,
        "tensor_parallel_size":   1,
        "max_model_len":          4096,
        "gpu_memory_utilization": 0.85,
        "enforce_eager":          True,
        "video_pruning_rate":     0.5,
        "video_fps":              2,
        "video_num_frames":       8,
        "vllm_backend":           "pip",
    }
    cfg_yaml = tmp_path / "nemotron_omni_llm_server.yaml"
    cfg_yaml.write_text(yaml.safe_dump(cfg))

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        ["uv", "run", "--directory", str(_OMNI_DIR),
         "nemotron_omni_llm_server", "--config", str(cfg_yaml)],
        env=env,
    )

    try:
        await _wait_for_health(port, _COLD_STARTUP_TIMEOUT_S, proc)

        png_b64 = base64.b64encode(_tiny_png_bytes(32)).decode("ascii")
        payload = {
            "model":    "llm",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{png_b64}"}},
                    {"type": "text", "text": "Describe what you see."},
                ],
            }],
            "max_tokens":  32,
            "temperature": 0,
        }
        loop = asyncio.get_running_loop()
        status, data = await loop.run_in_executor(
            None, _post_json, f"http://127.0.0.1:{port}/v1/chat/completions", payload,
        )
        assert status == 200, f"HTTP {status}: {data!r}"
        content = data["choices"][0]["message"]["content"]
        assert isinstance(content, str) and content.strip(), f"empty content: {data!r}"
    finally:
        _terminate(proc)
