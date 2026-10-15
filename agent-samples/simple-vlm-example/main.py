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
sample launches its hub and worker, plus optional session recording, and never
starts or stops model servers.

How to run (from agent-samples/simple-vlm-example/):
    uv sync && uv run simple_vlm_example
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from xr_ai_launcher import Process, run_stack
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent

_SESSION_RECORDING_PROCESS = Process(
    "session-recording",
    "../../services/device-io-hub",
    "device_io_capture",
    config="yaml/media_capture.yaml",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vision question answering over voice or text.",
    )
    parser.add_argument(
        "--session-recording",
        action="store_true",
        help=(
            "enable session recording of participant video, bidirectional audio, "
            "and data-channel traffic"
        ),
    )
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _parser().parse_args(sys.argv[1:] if argv is None else argv)


def _build_processes(*, session_recording: bool = False) -> list[Process]:
    processes = [
        Process(
            "hub",
            "../../services/device-io-hub",
            "device_io_hub",
            config="yaml/device_io_hub.yaml",
        ),
        Process(
            "worker",
            "worker",
            "simple_vlm_example_worker",
            config="yaml/simple_vlm_example_worker.yaml",
        ),
    ]
    if session_recording:
        processes.insert(1, _SESSION_RECORDING_PROCESS)
    return processes


PROCESSES = _build_processes()


def run(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    setup_logging("orchestrator", namespace="simple-vlm-example")
    run_stack(_build_processes(session_recording=args.session_recording), _BASE)


if __name__ == "__main__":
    run()
