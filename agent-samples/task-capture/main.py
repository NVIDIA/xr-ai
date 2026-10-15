# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture a demonstrated procedure as images, captions, and spoken narration.

This sample saves evidence for creating a reusable procedure guide. Guided
replay is planned but is not implemented in this sample yet.

This is the user-facing process-stack entry point. The worker's __main__ module
runs inside its own process and launches task_worker/app.py. This launcher
supervises the hub and worker; model services must already be available.
After connecting, say "start capture" to begin and "stop capture" to finish.
Use --output-root to choose where evidence packets are saved.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

from xr_ai_launcher import Process, run_stack
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent

def _build_processes(*, worker_config: str | Path = "yaml/worker.yaml") -> list[Process]:
    return [
        Process("hub", "../../services/device-io-hub", "device_io_hub", config="yaml/device_io_hub.yaml"),
        Process("worker", "worker", "task_capture_worker", config=worker_config),
    ]


PROCESSES = _build_processes()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record task demonstrations with spoken start capture and stop capture commands."
    )
    parser.add_argument(
        "--output-root", type=Path, metavar="PATH",
        help="save task packets under PATH/sessions; overrides YAML artifacts_dir (relative to the launch directory)",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    setup_logging("orchestrator", namespace="task-capture")
    if args.output_root is None:
        run_stack(PROCESSES, _BASE)
        return
    # Pass the override without rewriting the user's YAML or relocating its
    # relative model and voice configuration paths. Keep it until workers exit.
    with tempfile.TemporaryDirectory(prefix="task-capture-") as directory:
        request = Path(directory) / "worker.json"
        request.write_text(json.dumps({
            "worker_config": str(_BASE / "yaml/worker.yaml"),
            "output_root": str(args.output_root.expanduser().resolve()),
        }), encoding="utf-8")
        run_stack(_build_processes(worker_config=request), _BASE)


if __name__ == "__main__":
    run()
