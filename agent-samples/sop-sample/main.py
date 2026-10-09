# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SOP demonstration capture stack."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from xr_ai_launcher import Process, run_stack
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent

def _build_processes(*, capture: bool = False) -> list[Process]:
    processes = [
        Process("hub", "../../services/device-io-hub", "device_io_hub", config="yaml/device_io_hub.yaml"),
        Process("worker", "worker", "sop_sample_worker", config="yaml/worker.yaml"),
    ]
    if capture:
        processes.insert(1, Process(
            "capture", "../../services/device-io-hub", "device_io_capture", config="yaml/media_capture.yaml",
        ))
    return processes


PROCESSES = _build_processes()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record SOP demonstrations with spoken start recording and stop recording commands."
    )
    parser.add_argument(
        "--capture", action="store_true",
        help="record participant video, bidirectional audio, and data-channel traffic",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    setup_logging("orchestrator", namespace="sop-sample")
    run_stack(_build_processes(capture=args.capture), _BASE)


if __name__ == "__main__":
    run()
