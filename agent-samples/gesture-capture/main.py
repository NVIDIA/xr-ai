# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start the gesture-capture hub and worker; reuse an existing VLM service."""

from pathlib import Path

from xr_ai_launcher import Process, run_stack
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent
PROCESSES = [
    Process("hub", "../../services/device-io-hub", "device_io_hub", config="yaml/device_io_hub.yaml"),
    Process("worker", "worker", "gesture_capture_worker", config="yaml/gesture_capture_worker.yaml"),
]


def run() -> None:
    setup_logging("orchestrator", namespace="gesture-capture")
    run_stack(PROCESSES, _BASE)


if __name__ == "__main__":
    run()
