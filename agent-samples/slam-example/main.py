# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
slam-example orchestrator — SLAM-only worker for the xr-ai stack.

Brings up:
  * xr-media-hub        (LiveKit ingress + web client at http://<host>:8080)
  * kimera-mcp          (the SLAM backend wired on this branch)
  * slam-example-worker (frames + IMU + camera_meta → SLAM → pose.update)

The web client is identical to the one simple-vlm-example serves;
participants connect via LiveKit and the worker streams pose updates
back on data topic ``pose.update``.

How to run (from agent-samples/slam-example/):
    uv sync && uv run slam_example
"""
from pathlib import Path

from xr_ai_launcher import Process, run_stack
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent

_PROCESSES: list[Process] = [
    Process("hub",    "../../server-runtime",            "xr_media_hub",
            config="yaml/xr_media_hub.yaml"),
    # SLAM backend.  Branch-specific; on feat/kimera-vio this is kimera-mcp.
    # kimera-mcp auto-builds its `kimera_vio` docker image on first run
    # (~30 min, one-time) — see agent-mcp-servers/kimera-mcp/README.md.
    Process("slam",   "../../agent-mcp-servers/kimera-mcp", "kimera_mcp_server",
            config="yaml/slam_mcp_server.yaml"),
    Process("worker", "worker",                            "slam_example_worker",
            config="yaml/slam_example_worker.yaml"),
]


def run() -> None:
    setup_logging("orchestrator", namespace="slam-example")
    run_stack(_PROCESSES, _BASE)


if __name__ == "__main__":
    run()
