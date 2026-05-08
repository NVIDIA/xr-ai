# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
mono-slam-example orchestrator — monocular visual odometry pose logger.

How to run (from agent-samples/mono-slam-example/):
    uv sync && uv run mono_slam_example
"""
from pathlib import Path

from xr_ai_launcher import Process, run_stack
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent

_PROCESSES: list[Process] = [
    Process("hub",    "../../server-runtime", "xr_media_hub",
            config="yaml/xr_media_hub.yaml"),
    Process("worker", "worker",               "mono_slam_example_worker",
            config="yaml/mono_slam_example_worker.yaml"),
]


def run() -> None:
    setup_logging("orchestrator", namespace="mono-slam-example")
    run_stack(_PROCESSES, _BASE)


if __name__ == "__main__":
    run()
