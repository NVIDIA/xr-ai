# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the hub, selected model services, and simple VLM worker."""
from pathlib import Path

from xr_ai_launcher import (
    Process,
    ensure_credentials,
    load_model_deployment,
    run_stack,
    warn_if_missing,
)
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent
_WORKER_CONFIG = "yaml/simple_vlm_example_worker.yaml"

_MODEL_SERVICES = {
    "vlm": (
        "../../ai-services/vlm-server",
        "vlm_server",
        "yaml/vlm_server.yaml",
        8100,
    ),
    "omni": (
        "../../ai-services/llm/nemotron_omni",
        "nemotron_omni_llm_server",
        None,
        8108,
    ),
    "stt": (
        "../../ai-services/stt-server",
        "stt_server",
        "yaml/stt_server.yaml",
        8103,
    ),
    "tts": (
        "../../ai-services/tts/piper",
        "piper_tts_server",
        "yaml/piper_tts_server.yaml",
        8105,
    ),
}


def _build_processes() -> list[Process]:
    deployment = load_model_deployment(_BASE / _WORKER_CONFIG)
    unknown = deployment.services.keys() - _MODEL_SERVICES.keys()
    if unknown:
        raise ValueError(f"model profile declares unknown services: {sorted(unknown)}")

    processes = [
        Process(
            "hub",
            "../../server-runtime",
            "xr_media_hub",
            config="yaml/xr_media_hub.yaml",
        ),
    ]
    for service, definition in _MODEL_SERVICES.items():
        launch_mode = deployment.launch_mode(service)
        if launch_mode is None:
            continue
        project, command, config, port = definition
        processes.append(
            Process(
                service,
                project,
                command,
                config=config,
                launch_mode=launch_mode,
                port=port,
            )
        )
    processes.append(
        Process(
            "worker",
            "worker",
            "simple_vlm_example_worker",
            config=_WORKER_CONFIG,
        )
    )
    return processes


def run() -> None:
    setup_logging("orchestrator", namespace="simple-vlm-example")
    deployment = load_model_deployment(_BASE / _WORKER_CONFIG)
    warn_if_missing("HF_TOKEN")
    for credential in deployment.required_credentials:
        ensure_credentials(credential)
    run_stack(_build_processes(), _BASE)


if __name__ == "__main__":
    run()
