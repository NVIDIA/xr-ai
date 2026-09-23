# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
xr-render-demo orchestrator. Runs the process stack for this sample.

Architecture (per AGENTS.md + the Agentic AI for XR design doc):

  Web client ── LiveKit ──► DeviceIOHub ──IPC──► worker (this sample's agent)
                                      └─IPC──► capture (with --capture)
  Web client ── WebRTC ──► cloudxr-runtime
                        worker ──native tool──► scene ──► LOVR (OpenXR)

The worker receives voice queries from the hub and routes them through a
supervisor plus five focused subagents (placement, appearance, object,
vision, memory). Each subagent calls sample-local scene tools to read and
mutate the XR scene. The scene process owns LOVR and scene state. CloudXR
runs alongside as its own stream; neither stack passes through the other.

Prerequisites
-------------
All model services must already be running before this demo starts. The sample
never starts or stops them. The shared model stack includes Pocket TTS:

    uv run --project model-server-samples/model-servers model_servers

How to run (from the repo root or any directory):
    uv run --project agent-samples/xr-render-demo xr_render_demo

Before launch, the hub and scene services prepare their owned artifacts. The
scene caches a verified LOVR v0.18.0 AppImage. For WebRTC device profiles, the
hub builds the web vendor bundle (requires npm + network). Native CloudXR
profiles skip the vendor build. Current artifacts are reused.

To use a custom LOVR build instead of the auto-downloaded one:
    export LOVR_BIN=/path/to/your/lovr      # or set lovr_bin: in scene/scene_service.yaml

Then open https://<host>:8080, click "Start Mic", click "Launch XR" (or the
WebXR DevUI on desktop). Speak a scene command; the agent interprets it and
mutates the XR scene (move, recolor, add, remove, etc.).

The CloudXR EULA is accepted via cloudxr_runtime.yaml (see ``accept_eula``).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from loguru import logger
from xr_ai_launcher import (
    Process,
    is_native_profile,
    read_device_profile,
    run_stack,
)
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent

_WORKER_CONFIG = "yaml/xr_render_demo_worker.yaml"
_CLOUDXR_CONFIG = "yaml/cloudxr_runtime.yaml"

# Must match _config_loader.NO_WEB_CLIENT_ENV.
_NO_WEB_CLIENT_ENV = "DEVICE_IO_HUB_NO_WEB_CLIENT"


# ── Process stack ─────────────────────────────────────────────────────────────
#
_CAPTURE_PROCESS = Process(
    "capture",
    "../../services/device-io-hub",
    "device_io_capture",
    config="yaml/media_capture.yaml",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Conversational agent for a live CloudXR scene.",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help=(
            "record participant video, bidirectional audio, and data-channel "
            "traffic"
        ),
    )
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _parser().parse_args(sys.argv[1:] if argv is None else argv)


def _build_processes(*, capture: bool = False) -> list[Process]:
    processes = [
        Process("hub",        "../../services/device-io-hub",                "device_io_hub",
                config="yaml/device_io_hub.yaml"),
        Process("cloudxr",    "../../services/cloudxr-runtime",               "cloudxr_runtime",
                config="yaml/cloudxr_runtime.yaml"),
        Process("video-memory", "../../services/video-memory-service", "video_memory_service",
                config="yaml/video_memory_service.yaml"),
        Process("scene",      "scene",                                "xr_render_scene",
                config="scene/scene_service.yaml"),
        Process("openxr-service", "../../services/openxr-service",  "openxr_service",
                config="yaml/openxr_service.yaml",
                quiet_native_output=True),
        Process("worker",     "worker",                              "xr_render_demo_worker",
                config=_WORKER_CONFIG),
    ]
    if capture:
        hub_index = next(
            index for index, process in enumerate(processes) if process.name == "hub"
        )
        processes.insert(hub_index + 1, _CAPTURE_PROCESS)
    return processes


def _prepare_process(process: Process) -> None:
    project = (_BASE / process.project).resolve()
    if shutil.which("uv"):
        command = ["uv", "run", "--quiet", "--project", str(project), process.command]
    else:
        command = [sys.executable, "-m", process.command]
    if process.config is not None:
        command.extend(["--config", str((_BASE / process.config).resolve())])
    command.append("--prepare")
    env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    result = subprocess.run(command, cwd=_BASE, env=env)
    if result.returncode:
        raise SystemExit(result.returncode)


# ── Entry point ───────────────────────────────────────────────────────────────

def run(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    setup_logging("orchestrator", namespace="xr-render-demo")
    if is_native_profile(read_device_profile(_BASE / _CLOUDXR_CONFIG)):
        os.environ[_NO_WEB_CLIENT_ENV] = "1"
        logger.info("native device profile: web client page disabled, skipping vendor build")
    processes = _build_processes(capture=args.capture)
    for process in processes:
        if process.name in {"hub", "scene"}:
            _prepare_process(process)
    run_stack(processes, _BASE)


if __name__ == "__main__":
    run()
