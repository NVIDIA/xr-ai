# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SOP demonstration capture stack."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

from xr_ai_launcher import Process, run_stack
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent

def _build_processes(*, capture: bool = False, replay_config: Path | None = None) -> list[Process]:
    processes = [
        Process("hub", "../../services/device-io-hub", "device_io_hub", config="yaml/device_io_hub.yaml"),
        Process(
            "worker", "worker", "sop_sample_replay" if replay_config is not None else "sop_sample_worker",
            config=replay_config if replay_config is not None else "yaml/worker.yaml",
        ),
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
    parser.add_argument("--replay", metavar="GUIDE_NAME", help="replay an approved local guide by exact name or ID")
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    setup_logging("orchestrator", namespace="sop-sample")
    if args.replay is None:
        run_stack(_build_processes(capture=args.capture), _BASE)
        return
    if not args.replay.strip():
        _parser().error("--replay requires a non-empty guide name")
    with tempfile.TemporaryDirectory(prefix="sop-replay-") as directory:
        request = Path(directory) / "replay.json"
        request.write_text(json.dumps({
            "settings": str(_BASE / "yaml/replay.yaml"), "guide_name": args.replay,
        }), encoding="utf-8")
        run_stack(_build_processes(capture=args.capture, replay_config=request), _BASE)


if __name__ == "__main__":
    run()
