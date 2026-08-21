# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
simple-vlm-example orchestrator — vision Q&A over voice or text.

Pipeline
--------
Audio in (mic)     → STT → text query ─┐
Text in (data ch.) ─────→ text query ──┴→ latest video frame → VLM stream
                                                      │
                             sentence-batched TTS  ←──┴──→ data channel reply

Model deployment
----------------
STT, VLM, and TTS are reused from services started outside this sample. The
sample launches only its hub and worker and never starts or stops model servers.

How to run (from agent-samples/simple-vlm-example/):
    uv sync && uv run simple_vlm_example
"""
from pathlib import Path

from xr_ai_launcher import Process, run_stack
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent

PROCESSES = [
    Process(
        "hub",
        "../../services/device-io-hub",
        "device_io_hub",
        config="yaml/device_io_hub.yaml",
    ),
    Process(
        "stt",
        "../../services/stt-server",
        "stt_server",
        launch_mode="reuse",
    ),
    Process(
        "vlm",
        "../../services/vlm-server",
        "vlm_server",
        launch_mode="reuse",
    ),
    Process(
        "tts",
        "../../services/piper-tts",
        "piper_tts_server",
        launch_mode="reuse",
    ),
    Process(
        "worker",
        "worker",
        "simple_vlm_example_worker",
        config="yaml/simple_vlm_example_worker.yaml",
    ),
]


def run() -> None:
    setup_logging("orchestrator", namespace="simple-vlm-example")
    run_stack(PROCESSES, _BASE)


if __name__ == "__main__":
    run()
