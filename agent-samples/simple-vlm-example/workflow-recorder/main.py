# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the automatic workflow-recording prototype."""

from pathlib import Path

from xr_ai_launcher import Process, run_stack
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent

PROCESSES = [
    Process(
        "hub",
        "../../../services/device-io-hub",
        "device_io_hub",
        config="yaml/device_io_hub.yaml",
    ),
    Process(
        "stt",
        "../../../services/stt-server",
        "stt_server",
        launch_mode="reuse",
    ),
    Process(
        "omni",
        "../../../services/nemotron-omni-llm",
        "nemotron_omni_llm_server",
        launch_mode="reuse",
    ),
    Process(
        "vlm",
        "../../../services/vlm-server",
        "vlm_server",
        launch_mode="reuse",
    ),
    Process(
        "tts",
        "../../../services/piper-tts",
        "piper_tts_server",
        launch_mode="reuse",
    ),
    Process(
        "worker",
        "worker",
        "workflow_recorder_worker",
        config="yaml/workflow_recorder_worker.yaml",
    ),
]


def run() -> None:
    setup_logging("orchestrator", namespace="workflow-recorder")
    run_stack(PROCESSES, _BASE)


if __name__ == "__main__":
    run()
