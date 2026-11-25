# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for native sample/model comparison runs."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path

from xr_ai_launcher import load_credentials

from ._report import _write_report
from ._scenario import _SAMPLES, _builtin


def _positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay native samples against caller-supplied model services.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        command = commands.add_parser(name, help="Prepare fixed WAVs." if name == "prepare" else "Run one replay.")
        command.add_argument("--sample", choices=_SAMPLES, required=True, help="Fixed sample speech/camera timeline.")
        command.add_argument(
            "--models-config", type=Path, required=True, help="Model profile for existing local or hosted services."
        )
        command.add_argument("--audio-dir", type=Path, required=True, help="Local hash-pinned WAV fixture directory.")
        if name == "run":
            command.add_argument(
                "--output-dir", type=Path, required=True, help="New local run directory; must not exist."
            )
            command.add_argument("--label", required=True, help="Operator's configuration label, for example baseline.")
            command.add_argument(
                "--image", action="append", default=[], help="Image asset NAME=PATH; Tea needs tea_package."
            )
            command.add_argument(
                "--metadata", type=Path, help="Operator-supplied JSON object describing serving/hardware."
            )
            command.add_argument("--voice-gate", type=Path, help="Optional sample voice-gate YAML override.")
            command.add_argument(
                "--worker-config", type=Path, help="Optional native worker YAML; required for scene replays."
            )
            command.add_argument("--rag-endpoint", help="Optional existing Tea RAG endpoint override.")
            command.add_argument(
                "--ready-timeout", type=_positive, default=600, help="Worker readiness timeout in seconds."
            )
    return parser


def _main() -> None:
    parser = _parser()
    args = parser.parse_args()
    load_credentials()
    if args.command == "prepare":
        from ._fixtures import _prepare

        asyncio.run(_prepare(_builtin(args.sample), args.audio_dir.resolve(), args.models_config.resolve()))
        print(args.audio_dir.resolve() / "audio.json")
        return
    else:
        from ._replay import _check_scene, _preflight, _run

        if not args.label.strip():
            parser.error("label must not be empty")
        scenario = _builtin(args.sample)
        if args.rag_endpoint and scenario.worker["package"] != "tea_making_worker":
            parser.error("rag-endpoint applies only to Tea")
        images = {}
        for binding in args.image:
            name, separator, path = binding.partition("=")
            if not separator or not name or not path or name in images:
                parser.error("image bindings must be unique NAME=PATH pairs")
            images[name] = Path(path).expanduser().resolve()
        metadata = json.loads(args.metadata.read_text()) if args.metadata else {}
        if not isinstance(metadata, dict):
            parser.error("metadata must be a JSON object")
        models = args.models_config.resolve()
        voice_gate = args.voice_gate.resolve() if args.voice_gate else None
        worker_config = args.worker_config.resolve() if args.worker_config else None
        inputs = _preflight(scenario, models, args.audio_dir.resolve(), images, voice_gate, worker_config)
        asyncio.run(_check_scene(inputs))
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
        (output / ".gitignore").write_text("*\n")
        try:
            asyncio.run(
                _run(
                    scenario,
                    inputs,
                    output,
                    args.label,
                    metadata,
                    args.rag_endpoint,
                    args.ready_timeout,
                )
            )
        finally:
            if (output / "events.jsonl").exists():
                _write_report(output)
        result = json.loads((output / "report.json").read_text())
    print(
        json.dumps(
            {
                "execution_status": result["execution_status"],
                "timing_comparable": result["timing_comparable"],
                "handler_coverage": result["handler_coverage"],
                "error_events": result["error_events"],
            }
        )
    )


if __name__ == "__main__":
    _main()
