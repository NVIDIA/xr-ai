# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture a demonstrated procedure or guide a participant through approved steps.

Capture saves evidence for creating a reusable procedure guide. Replay loads an
approved local guide and verifies progress while providing spoken guidance.

This is the user-facing process-stack entry point. The worker's __main__ module
runs inside its own process and launches procedure_capture_replay_worker/app.py.
This launcher supervises the hub and worker; model services must already be available.
Choose --mode capture or --mode replay. In capture mode, say "start capture"
to begin and "stop capture" to finish; --output-root selects the packet directory.
Replay requires a guide name after --mode replay. Say "start replay" to begin
that guide and "stop replay" to cancel; connecting alone does not start it.
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
        Process("worker", "worker", "procedure_capture_replay_worker", config=worker_config),
    ]


PROCESSES = _build_processes()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture narrated procedures or replay an approved guide with spoken guidance."
    )
    parser.add_argument("--mode", choices=("capture", "replay"), default="capture", help="worker capability to run")
    parser.add_argument(
        "guide_name", nargs="?", metavar="GUIDE_NAME", help="exact approved guide name or ID (replay only)",
    )
    parser.add_argument(
        "--output-root", type=Path, metavar="PATH",
        help="save task packets under PATH/sessions; overrides YAML artifacts_dir (relative to the launch directory)",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.mode == "capture" and args.guide_name is not None:
        parser.error("GUIDE_NAME requires --mode replay")
    if args.mode == "replay" and args.output_root is not None:
        parser.error("--output-root requires --mode capture")
    if args.mode == "replay" and (args.guide_name is None or not args.guide_name.strip()):
        parser.error("--mode replay requires a non-empty GUIDE_NAME")
    setup_logging("orchestrator", namespace="procedure-capture-replay")
    if args.mode == "capture" and args.output_root is None:
        run_stack(PROCESSES, _BASE)
        return
    # Pass the override without rewriting the user's YAML or relocating its
    # relative model and voice configuration paths. Keep it until workers exit.
    with tempfile.TemporaryDirectory(prefix="procedure-capture-replay-") as directory:
        request = Path(directory) / "worker.json"
        request.write_text(json.dumps({
            "worker_config": str(_BASE / "yaml/worker.yaml"),
            "mode": args.mode,
            "guide_name": args.guide_name,
            "output_root": str(args.output_root.expanduser().resolve()) if args.output_root is not None else None,
        }), encoding="utf-8")
        run_stack(_build_processes(worker_config=request), _BASE)


if __name__ == "__main__":
    run()
