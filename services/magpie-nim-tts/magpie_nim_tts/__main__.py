# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve a persistent HTTP speech endpoint backed by Magpie Riva NIM."""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import httpx
import uvicorn
import yaml
from xr_ai_logging import setup_logging
from xr_ai_vllm._docker import has_xr_ai_ownership_marker, pid_on_port_checked

from .common import identity
from .speech import build_app


async def _reusable_listener(config: dict) -> int | None:
    port = int(config["port"])
    pid, checked, listening = pid_on_port_checked(port)
    if not checked or (listening and pid is None):
        raise RuntimeError(f"cannot inspect ownership of port {port}")
    if not listening:
        return None
    if not has_xr_ai_ownership_marker(pid, port):
        raise RuntimeError(f"port {port} belongs to an unmanaged listener; stop it before launching")
    # Health can take 3 s for gRPC plus 3 s for NIM HTTP readiness.
    async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
        try:
            response = await client.get(f"http://127.0.0.1:{port}/health")
            response.raise_for_status()
            observed_identity = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError(f"adapter on port {port} is unhealthy; stop it before launching") from exc
    expected = identity(config)
    # The extraction renamed the service identity without changing its TTS
    # contract. Accept the old name only for an otherwise identical TTS config.
    legacy = expected | {"service": "nim-model-adapter"}
    legacy_matches = expected["configuration"]["kind"] == "tts" and observed_identity == legacy
    if observed_identity != expected and not legacy_matches:
        raise RuntimeError(f"port {port} serves a different adapter configuration; stop it before launching")
    if pid_on_port_checked(port) != (pid, True, True):
        raise RuntimeError(f"listener on port {port} changed during inspection")
    return pid


async def _serve(config: dict, ready_file: Path | None) -> None:
    if pid := await _reusable_listener(config):
        print(f"[magpie-nim-tts] reusing managed listener {pid} on port {config['port']}", flush=True)
        if ready_file:
            ready_file.touch()
        if os.environ.get("_XR_AI_LAUNCHER_READY_PROCESS_MAY_EXIT") == "1":
            return
        # An inconclusive ownership probe is not evidence that the server died.
        while True:
            current, checked, listening = pid_on_port_checked(int(config["port"]))
            if checked and (not listening or current != pid):
                break
            await asyncio.sleep(0.25)
        return

    app = build_app(config)
    server = uvicorn.Server(uvicorn.Config(
        app, timeout_graceful_shutdown=5,
        host=config.get("host", "0.0.0.0"), port=int(config["port"]), log_level="warning",
    ))
    task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            if task.done():
                await task
                raise RuntimeError("model adapter stopped before listening")
            await asyncio.sleep(0.05)
        # Readiness means both the adapter and its configured NIM are healthy.
        await _reusable_listener(config)
        if ready_file:
            ready_file.touch()
        await task
    finally:
        server.should_exit = True
        if not task.done():
            await task


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    port = int(config["port"])
    markers = {"XR_AI_VLLM_MANAGED": "1", "XR_AI_VLLM_PORT": str(port)}
    if any(os.environ.get(key) != value for key, value in markers.items()):
        # /proc/<pid>/environ exposes the exec-time environment, so changing
        # os.environ in place cannot establish ownership for --stop.
        os.execvpe(sys.executable, [sys.executable, "-m", "magpie_nim_tts", *sys.argv[1:]],
                   os.environ | markers)
    setup_logging("magpie-nim-tts")
    asyncio.run(_serve(config, args.ready_file))


if __name__ == "__main__":
    run()
