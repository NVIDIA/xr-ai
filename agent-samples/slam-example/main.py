# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
slam-example orchestrator — topological place-memory worker for the
xr-ai stack.

Brings up:
  * xr-media-hub        (LiveKit ingress + web client at http://<host>:8080)
  * space-mcp           (DINOv2-based topological place memory)
  * slam-example-worker (frames → space-mcp → place.update)

space-mcp does *not* produce a metric 6-DoF pose — it answers "which
region am I in" by matching each frame's DINOv2 embedding against the
centroid of every known region.  No IMU and no camera intrinsics are
required.  The worker echoes the inferred region info back to the
client on data topic ``place.update`` (compare ``pose.update`` on the
metric-SLAM branches).

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
    # SLAM backend.  Branch-specific; on feat/spatial-memory this is
    # space-mcp (DINOv2 ViT-S/14 topological place memory, no IMU,
    # no metric pose).  See agent-mcp-servers/space-mcp/README.md
    # for first-run install.
    Process("slam",   "../../agent-mcp-servers/space-mcp", "space_mcp_server",
            config="yaml/slam_mcp_server.yaml"),
    Process("worker", "worker",                            "slam_example_worker",
            config="yaml/slam_example_worker.yaml"),
]


def run() -> None:
    setup_logging("orchestrator", namespace="slam-example")
    run_stack(_PROCESSES, _BASE)


if __name__ == "__main__":
    run()
