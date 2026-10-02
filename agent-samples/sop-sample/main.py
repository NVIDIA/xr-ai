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

PROCESSES = [
    Process("hub", "../../services/device-io-hub", "device_io_hub", config="yaml/device_io_hub.yaml"),
    Process("capture", "../../services/device-io-hub", "device_io_capture", config="yaml/media_capture.yaml"),
    Process("worker", "worker", "sop_sample_worker", config="yaml/worker.yaml"),
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record SOP demonstrations while the camera is on.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--capture", action="store_true", help="capture audio, video, frames, captions, and narration automatically"
    )
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    _parser().parse_args(argv)
    setup_logging("orchestrator", namespace="sop-sample")
    run_stack(PROCESSES, _BASE)


if __name__ == "__main__":
    run()
