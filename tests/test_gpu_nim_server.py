# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU smoke test for the self-hosted NIM path, end to end.

Spawns ``nim_server`` with a generated YAML (llama-3.1-8b NIM, the smallest
LLM NIM), waits for the container's ``/v1/health/ready``, chats with it
through a deployment profile + ``make_llm`` (the same path the sample
workers use), then tears it down via ``stop_persistent_servers`` and
asserts the container is gone.

First run on a fresh machine pulls the image and downloads the optimized
engine from NGC (multi-GB); the ``models/nim`` cache volume makes later
runs start in a couple of minutes. Skips when docker, uv, or
``NGC_API_KEY`` is unavailable.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest
from xr_ai_models import ChatMessage, load_models_config_from_dict, make_llm
from xr_ai_vllm import stop_persistent_servers

from _helpers_subprocess import pick_free_port

pytestmark = [pytest.mark.asyncio, pytest.mark.gpu]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_NIM_SERVER_DIR = _REPO_ROOT / "services" / "nim-server"

_IMAGE = "nvcr.io/nim/meta/llama-3.1-8b-instruct:2.0.10"
_MODEL_NAME = "meta/llama-3.1-8b-instruct"
_CONTAINER = "xr-ai-nim-llama-3.1-8b-instruct"

# Cold start = image pull + NGC engine download + load; must stay under the
# nightly workflow's per-test --timeout=1200. Warm (cached engine) is 1-3 min.
_READY_TIMEOUT_S = 1100.0
_CHAT_TIMEOUT_S = 120.0


def _ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/v1/health/ready", timeout=3
        ) as r:
            return r.status == 200
    except Exception:
        return False


def _tail(path: Path, n: int = 30) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "<no wrapper output captured>"


async def test_nim_container_serves_profile_llm(tmp_path) -> None:
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")
    if shutil.which("docker") is None:
        pytest.skip("docker not on PATH")
    if not os.environ.get("NGC_API_KEY"):
        pytest.skip("NGC_API_KEY not set (required for nvcr.io pull + engine download)")

    port = pick_free_port()
    config = tmp_path / "nim_llm_server.yaml"
    config.write_text(
        f"image:     {_IMAGE}\n"
        f"http_port: {port}\n"
        # Shared engine cache so nightly reruns skip the multi-GB download.
        f"nim_cache: {_REPO_ROOT / 'models' / 'nim'}\n"
        "cuda_visible_devices: \"0\"\n"
        "env:\n"
        "  NIM_MAX_MODEL_LEN: \"8192\"\n"
        "  NIM_KVCACHE_PERCENT: \"0.3\"\n"
    )

    wrapper_log = tmp_path / "nim_server.log"
    with wrapper_log.open("wb") as sink:
        wrapper = subprocess.Popen(
            ["uv", "run", "--directory", str(_NIM_SERVER_DIR),
             "nim_server", "--config", str(config)],
            stdout=sink,
            stderr=subprocess.STDOUT,
        )
    try:
        deadline = time.monotonic() + _READY_TIMEOUT_S
        while not _ready(port):
            if wrapper.poll() is not None:
                pytest.fail(
                    f"nim_server exited early (rc={wrapper.returncode}); "
                    f"wrapper output tail:\n{_tail(wrapper_log)}"
                )
            if time.monotonic() > deadline:
                pytest.fail(
                    f"NIM not ready after {_READY_TIMEOUT_S:.0f}s; "
                    f"wrapper output tail:\n{_tail(wrapper_log)}"
                )
            time.sleep(5)

        cfg = load_models_config_from_dict({
            "models": {
                "llm": {
                    "category": "llm",
                    "adapter": {
                        "kind": "openai_compat",
                        "model_name": _MODEL_NAME,
                    },
                    "endpoint": {
                        "base_url": f"http://localhost:{port}",
                        "readiness": "health",
                        "health_path": "/v1/health/ready",
                    },
                    "deployment": {
                        "ownership": "managed",
                        "service": "llm-nim",
                        "credentials": ["NGC_API_KEY"],
                    },
                },
            },
        })
        llm = make_llm(cfg, "llm")
        try:
            resp = await llm.chat(
                [ChatMessage(role="user", content="Reply with the single word: pong")],
                max_tokens=16,
                timeout=_CHAT_TIMEOUT_S,
            )
        finally:
            await llm.close()
        assert resp.content.strip()

        stop_persistent_servers([("llm-nim", port)])
        ps = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"name=^{_CONTAINER}$"],
            capture_output=True, text=True, timeout=20,
        )
        assert ps.stdout.strip() == "", "container survived stop_persistent_servers"
    finally:
        if wrapper.poll() is None:
            wrapper.terminate()
            try:
                wrapper.wait(timeout=60)
            except subprocess.TimeoutExpired:
                wrapper.kill()
        subprocess.run(
            ["docker", "rm", "-f", _CONTAINER],
            check=False, capture_output=True, timeout=30,
        )
