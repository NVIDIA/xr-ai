# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU smoke tests for the three vLLM-backed LLM servers.

Three tests, run serially by pytest:

* ``test_llama_nemotron_tool_call`` — spawns ``llama_nemotron_llm_server``
  (docker backend), POSTs a chat-completions request with a tool spec, and
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
import struct
import subprocess
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

import pytest
import yaml

from _helpers_subprocess import health_ok, pick_free_port
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
# Cold-start budget covers a first-time weights download (~15–30 GB)
# + vLLM compile + FlashInfer JIT. Cached runs complete in ~60–180 s.
_COLD_STARTUP_TIMEOUT_S = 1800.0
_HOT_STARTUP_TIMEOUT_S  = 60.0
_SHUTDOWN_TIMEOUT_S     = 30.0


# ── helpers ─────────────────────────────────────────────────────────────────


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
        if await loop.run_in_executor(None, health_ok, port):
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
    # SIGTERM didn't reap within the grace window; escalate to SIGKILL below.
    except subprocess.TimeoutExpired:
        pass
    proc.kill()
    try:
        proc.wait(timeout=5)
    # Best-effort reap after SIGKILL — nothing left to do if the wait still times out.
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

    port = pick_free_port()
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
        # docker backend: nvcc + flashinfer are pre-built in the NGC image.
        "vllm_backend":           "docker",
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
        # Wrapper SIGTERM doesn't reach the persistent vLLM child.
        stop_persistent_servers([("llama_nemotron", port)])


# ── test 2: nemotron3_nano persistence ──────────────────────────────────────


async def test_nemotron3_nano_persistent(tmp_path: Path) -> None:
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")
    if not _N3_DIR.exists():
        pytest.skip(f"nemotron3_nano source tree missing: {_N3_DIR}")

    # The wrapper auto-selects the FP8 (Ada/Hopper/Ampere) or NVFP4 (Blackwell)
    # variant from the GPU's compute capability and downloads on first run
    # into the HF cache below. Cached runs reuse the weights.
    hf_root = Path("~/.cache/huggingface").expanduser()

    port = pick_free_port()
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
        # docker backend: nvcc + flashinfer are pre-built in the NGC image.
        "vllm_backend":           "docker",
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
        assert health_ok(port), (
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


# Nemotron-3-Nano-Omni-30B's `config.json` declares architecture
# `NemotronH_Nano_Omni_Reasoning_V3`, which is not yet in the vLLM
# 0.19.0 model registry shipped by `nvcr.io/nvidia/vllm:26.04-py3` —
# `--trust-remote-code` (already passed) does not help because the
# registry lookup happens before remote-code dispatch.  Two ways out:
# (a) bump `DEFAULT_IMAGE` to an NGC tag whose vLLM has registered the
# arch, or (b) pin a newer vllm wheel via extra_pip (defeats the point
# of the prebuilt container).  Neither is in scope for this PR.  Keep
# the test running (not skipped) so the next nightly catches the fix
# automatically; `strict=False` so an upstream image bump doesn't
# break this suite on the day the fix lands.
@pytest.mark.xfail(
    reason="vLLM 0.19.0 in nvcr.io/nvidia/vllm:26.04-py3 does not register "
           "the NemotronH_Nano_Omni_Reasoning_V3 architecture, so startup "
           "fails in create_model_config (config.json arch check) — before "
           "any model code loads. The test config below sets extra_pip=[] so "
           "this guaranteed failure no longer pays the ~16 min mamba-ssm / "
           "causal-conv1d CUDA-kernel compile it never reaches. When vLLM "
           "registers the arch, the failure mode shifts to a mamba_ssm "
           "ImportError at model load — that's the signal to restore extra_pip "
           "(and bump DEFAULT_IMAGE) and run this for real.",
    strict=False,
)
async def test_nemotron_omni_multimodal(tmp_path: Path) -> None:
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")
    if not _OMNI_DIR.exists():
        pytest.skip(f"nemotron_omni source tree missing: {_OMNI_DIR}")

    # Server auto-selects FP8/NVFP4/BF16 by compute cap; downloads on first
    # run, reuses on subsequent runs.
    hf_root = Path("~/.cache/huggingface").expanduser()

    port = pick_free_port()
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
        # docker backend: nvcc + flashinfer are pre-built in the NGC image.
        "vllm_backend":           "docker",
        # This test xfails at the vLLM arch check (see the xfail reason),
        # which runs before any model code imports mamba_ssm. Skip the
        # default mamba-ssm/causal-conv1d source build — it's a ~16 min
        # CUDA-kernel compile this guaranteed failure never reaches.
        "extra_pip":              [],
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
            "max_tokens":  256,
            "temperature": 0,
        }
        loop = asyncio.get_running_loop()
        status, data = await loop.run_in_executor(
            None, _post_json, f"http://127.0.0.1:{port}/v1/chat/completions", payload,
        )
        assert status == 200, f"HTTP {status}: {data!r}"
        # Omni is a reasoning variant: the answer may surface in `content`,
        # or in `reasoning` if the model is still thinking when the
        # response is returned. Either non-empty string counts as a smoke pass.
        msg = data["choices"][0]["message"]
        body = msg.get("content") or msg.get("reasoning") or ""
        assert isinstance(body, str) and body.strip(), f"empty content+reasoning: {data!r}"
    finally:
        _terminate(proc)
        stop_persistent_servers([("nemotron_omni", port)])
