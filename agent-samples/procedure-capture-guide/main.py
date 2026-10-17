# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture a demonstrated procedure or guide a participant through approved steps.

Capture saves evidence for creating a reusable procedure guide. Guide mode loads an
approved local guide and verifies progress while providing spoken guidance.

This is the user-facing process-stack entry point. The worker's __main__ module
runs inside its own process and launches procedure_capture_guide_worker/app.py.
This launcher supervises the hub and worker; model services must already be available.
Choose --mode capture or --mode guide. In capture mode, say "start capture"
to begin and "stop capture" to finish; --output-root selects the packet directory.
Guide mode requires a guide name after --mode guide. Say "start guide" to begin
that guide and "stop guide" to cancel; connecting alone does not start it.
In either mode, --session-recording independently records the whole participant
connection; it does not start or stop procedure capture or guided execution.
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

_CAPTURE_PROCESS = Process(
    "capture",
    "../../services/device-io-hub",
    "device_io_capture",
    config="yaml/media_capture.yaml",
)


def _build_processes(
    *, worker_config: str | Path = "yaml/worker.yaml", capture: bool = False,
    hub_config: Path | None = None,
) -> list[Process]:
    processes = [
        Process("hub", "../../services/device-io-hub", "device_io_hub", config=hub_config or "yaml/device_io_hub.yaml"),
        Process("worker", "worker", "procedure_capture_guide_worker", config=worker_config),
    ]
    if capture:
        processes.insert(1, _CAPTURE_PROCESS)
    return processes


PROCESSES = _build_processes()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture narrated procedures or follow an approved procedure with spoken guidance."
    )
    parser.add_argument("--mode", choices=("capture", "guide"), default="capture", help="worker capability to run")
    parser.add_argument(
        "guide_name", nargs="?", metavar="GUIDE_NAME", help="exact approved guide name or ID (guide mode only)",
    )
    parser.add_argument(
        "--output-root", type=Path, metavar="PATH",
        help="save task packets under PATH/sessions; overrides YAML artifacts_dir (relative to the launch directory)",
    )
    parser.add_argument(
        "--session-recording",
        action="store_true",
        help=(
            "record participant video, bidirectional audio, and data-channel traffic "
            "for the whole connection in either mode, independently of procedure capture or guide commands"
        ),
    )
    parser.add_argument("--hub-config", type=Path, metavar="PATH",
                        help="use hub YAML instead of the sample default (relative to launch directory)")
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    hub_config = args.hub_config.expanduser().resolve() if args.hub_config is not None else None
    if args.mode == "capture" and args.guide_name is not None:
        parser.error("GUIDE_NAME requires --mode guide")
    if args.mode == "guide" and args.output_root is not None:
        parser.error("--output-root requires --mode capture")
    if args.mode == "guide" and (args.guide_name is None or not args.guide_name.strip()):
        parser.error("--mode guide requires a non-empty GUIDE_NAME")
    setup_logging("orchestrator", namespace="procedure-capture-guide")
    if args.mode == "capture" and args.output_root is None:
        run_stack(_build_processes(capture=args.session_recording, hub_config=hub_config), _BASE)
        return
    # Pass the override without rewriting the user's YAML or relocating its
    # relative model and voice configuration paths. Keep it until workers exit.
    with tempfile.TemporaryDirectory(prefix="procedure-capture-guide-") as directory:
        request = Path(directory) / "worker.json"
        request.write_text(json.dumps({
            "worker_config": str(_BASE / "yaml/worker.yaml"),
            "mode": args.mode,
            "guide_name": args.guide_name,
            "output_root": str(args.output_root.expanduser().resolve()) if args.output_root is not None else None,
        }), encoding="utf-8")
        run_stack(_build_processes(worker_config=request, capture=args.session_recording, hub_config=hub_config), _BASE)


if __name__ == "__main__":
    run()
