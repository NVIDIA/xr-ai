# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch lab instrument monitoring, voice, hub, and model dependencies."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from xr_ai_launcher import (
    Process,
    ensure_credentials,
    load_model_deployment,
    run_stack,
)
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent
_WORKER_CONFIG = _BASE / "yaml" / "lab_instrument_monitoring_worker.yaml"
_VLM_CONFIGS = {
    "cosmos": _BASE / "yaml" / "models.local.json",
    "omni": _BASE / "yaml" / "models.omni.json",
}

_MODEL_PROCESSES = {
    "stt": Process(
        "stt",
        "../../services/stt-server",
        "stt_server",
    ),
    "omni": Process(
        "omni",
        "../../services/nemotron-omni-llm",
        "nemotron_omni_llm_server",
    ),
    "vlm": Process(
        "vlm",
        "../../services/vlm-server",
        "vlm_server",
    ),
    "tts": Process(
        "tts",
        "../../services/piper-tts",
        "piper_tts_server",
        config="yaml/piper_tts_server.yaml",
    ),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run marker-associated lab instrument monitoring.",
    )
    parser.add_argument(
        "--vlm-mode",
        choices=("cosmos", "omni"),
        default="cosmos",
        help=(
            "cosmos (default): use Cosmos for vision; omni: use the existing "
            "Nemotron Omni service for both language and vision"
        ),
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


def _materialize_worker_config(
    runtime_dir: Path,
    vlm_mode: str,
    *,
    expose_web_events: bool = False,
) -> Path:
    worker_config = runtime_dir / "lab_instrument_monitoring_worker.yaml"
    _write_config(
        _WORKER_CONFIG,
        worker_config,
        {
            "models_config": _VLM_CONFIGS[vlm_mode].resolve(),
            "voice_gate_yaml": (_BASE / "yaml" / "voice_gate.yaml").resolve(),
            "device_map_yaml": (_BASE / "yaml" / "device_map.yaml").resolve(),
            "artifacts_dir": (_BASE / "artifacts").resolve(),
            "web_events_host": "0.0.0.0" if expose_web_events else "127.0.0.1",
        },
    )
    return worker_config


def _build_processes(worker_config: Path = _WORKER_CONFIG) -> tuple[list[Process], tuple[str, ...]]:
    deployment = load_model_deployment(worker_config)
    unknown_services = deployment.services.keys() - _MODEL_PROCESSES.keys()
    if unknown_services:
        raise ValueError(f"model profile declares unknown services: {sorted(unknown_services)}")

    processes = [
        Process(
            "hub",
            "../../services/device-io-hub",
            "device_io_hub",
            config="yaml/device_io_hub.yaml",
        )
    ]
    for service, process in _MODEL_PROCESSES.items():
        launch_mode = deployment.launch_mode(service)
        if launch_mode is not None:
            processes.append(replace(process, launch_mode=launch_mode))
    processes.append(
        Process(
            "worker",
            "worker",
            "lab_instrument_monitoring_worker",
            config=worker_config,
        )
    )
    return processes, deployment.required_credentials


def run(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    setup_logging("orchestrator", namespace="lab-instrument-monitoring")
    with tempfile.TemporaryDirectory(prefix="lab-instrument-monitoring-config-") as directory:
        worker_config = _materialize_worker_config(
            Path(directory),
            args.vlm_mode,
            expose_web_events=args.expose_web_events,
        )
        processes, credentials = _build_processes(worker_config)
        for credential in credentials:
            ensure_credentials(credential)
        run_stack(processes, _BASE)


if __name__ == "__main__":
    run()
