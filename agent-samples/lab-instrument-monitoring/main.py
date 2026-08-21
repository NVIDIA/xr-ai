# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch lab instrument monitoring with reused model dependencies."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from xr_ai_launcher import Process, read_config_scalar, run_stack
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent
_WORKER_CONFIG = _BASE / "yaml" / "lab_instrument_monitoring_worker.yaml"

_MODEL_PROCESSES = [
    Process(
        "stt",
        "../../services/stt-server",
        "stt_server",
        launch_mode="reuse",
    ),
    Process(
        "omni",
        "../../services/nemotron-omni-llm",
        "nemotron_omni_llm_server",
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
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run marker-associated lab instrument monitoring.",
    )
    parser.add_argument(
        "--expose-web-events",
        action="store_true",
        help=(
            "bind the unauthenticated event viewer to all IPv4 interfaces "
            "instead of loopback"
        ),
    )
    return parser


def _write_config(
    source: Path, target: Path, overrides: Mapping[str, str | Path]
) -> None:
    pending = set(overrides)
    lines: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        key = line.partition(":")[0]
        if key in pending and line.startswith(f"{key}:"):
            lines.append(f"{key}: {json.dumps(str(overrides[key]))}")
            pending.remove(key)
        else:
            lines.append(line)
    if pending:
        raise ValueError(f"{source} has no top-level fields: {sorted(pending)}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_worker_path(key: str) -> Path:
    raw = read_config_scalar(_WORKER_CONFIG, key)
    if not raw:
        raise ValueError(f"{_WORKER_CONFIG} has no non-empty {key!r}")
    path = Path(raw)
    if not path.is_absolute():
        path = _WORKER_CONFIG.parent / path
    return path.resolve()


def _materialize_worker_config(
    runtime_dir: Path,
    *,
    expose_web_events: bool = False,
) -> Path:
    shutil.copyfile(_BASE / "yaml" / "models.json", runtime_dir / "models.json")
    worker_config = runtime_dir / "lab_instrument_monitoring_worker.yaml"
    _write_config(
        _WORKER_CONFIG,
        worker_config,
        {
            "voice_gate_yaml": _resolve_worker_path("voice_gate_yaml"),
            "device_map_yaml": _resolve_worker_path("device_map_yaml"),
            "artifacts_dir": _resolve_worker_path("artifacts_dir"),
            "web_events_host": "0.0.0.0" if expose_web_events else "127.0.0.1",
        },
    )
    return worker_config


def _build_processes(worker_config: Path = _WORKER_CONFIG) -> list[Process]:
    return [
        Process(
            "hub",
            "../../services/device-io-hub",
            "device_io_hub",
            config="yaml/device_io_hub.yaml",
        ),
        *_MODEL_PROCESSES,
        Process(
            "worker",
            "worker",
            "lab_instrument_monitoring_worker",
            config=worker_config,
        ),
    ]


def run(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    setup_logging("orchestrator", namespace="lab-instrument-monitoring")
    with tempfile.TemporaryDirectory(prefix="lab-instrument-monitoring-config-") as directory:
        worker_config = _materialize_worker_config(
            Path(directory),
            expose_web_events=args.expose_web_events,
        )
        run_stack(_build_processes(worker_config), _BASE)


if __name__ == "__main__":
    run()
