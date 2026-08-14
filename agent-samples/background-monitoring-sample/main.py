# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the background monitor, voice worker, hub, and model dependencies."""

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
_WORKER_CONFIG = "yaml/background_monitoring_worker.yaml"

_MODEL_PROCESSES = {
    "stt": Process(
        "stt",
        "../../services/stt-server",
        "stt_server",
    ),
    "agent-llm": Process(
        "agent-llm",
        "../../services/nemotron3-nano-llm",
        "nemotron3_nano_llm_server",
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


def _build_processes() -> tuple[list[Process], tuple[str, ...]]:
    deployment = load_model_deployment(_BASE / _WORKER_CONFIG)
    unknown_services = deployment.services.keys() - _MODEL_PROCESSES.keys()
    if unknown_services:
        raise ValueError(
            f"model profile declares unknown services: {sorted(unknown_services)}"
        )

    processes = [
        Process(
            "hub",
            "../../services/xr-media-hub",
            "xr_media_hub",
            config="yaml/xr_media_hub.yaml",
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
            "background_monitoring_worker",
            config=_WORKER_CONFIG,
        )
    )
    return processes, deployment.required_credentials


def run() -> None:
    setup_logging("orchestrator", namespace="background-monitoring-sample")
    processes, credentials = _build_processes()
    for credential in credentials:
        ensure_credentials(credential)
    run_stack(processes, _BASE)


if __name__ == "__main__":
    run()
