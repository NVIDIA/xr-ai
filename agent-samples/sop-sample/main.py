# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flag-gated SOP capture and replay stacks."""

from __future__ import annotations

import argparse
import json
import tempfile
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
    parser = argparse.ArgumentParser(description="Capture a demonstration or replay one approved SOP guide.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--capture", action="store_true", help="capture audio, video, frames, captions, and narration automatically"
    )
    mode.add_argument("--replay", metavar="GUIDE_NAME", help="replay an approved local guide by exact name or ID")
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    setup_logging("orchestrator", namespace="sop-sample")
    if args.capture:
        run_stack(PROCESSES, _BASE)
        return
    if not args.replay.strip():
        _parser().error("--replay requires a non-empty guide name")
    # Keep CLI selection out of checked-in settings and out of the capture path.
    with tempfile.TemporaryDirectory(prefix="sop-replay-") as directory:
        request = Path(directory) / "replay.json"
        request.write_text(json.dumps({
            "settings": str(_BASE / "yaml/replay.yaml"), "guide_name": args.replay,
        }), encoding="utf-8")
        run_stack([
            PROCESSES[0],
            Process("worker", "worker", "sop_sample_replay", config=request),
        ], _BASE)


if __name__ == "__main__":
    run()
