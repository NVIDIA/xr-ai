# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
simple-vlm-example orchestrator — vision Q&A over voice or text.

Pipeline
--------
Audio in (mic)     → STT → text query ─┐
Text in (data ch.) ─────→ text query ──┴→ latest video frame → VLM stream
                                                      │
                             sentence-batched TTS  ←──┴──→ data channel reply

Model deployment
----------------
``models_config`` in yaml/simple_vlm_example_worker.yaml selects a deployment
profile. The default profile owns local STT, VLM, and TTS services; the hosted
profile replaces only the VLM with NVIDIA NIM; models.vlm_llm_nim.json and
models.vlm_speech_nim.json reuse self-hosted NIM containers from the
model-servers nim / vlm_speech_nim stacks (start the matching stack first).
On hardware supported by the shared model-server layouts, the local VLM uses
the same launch config so a compatible persistent container is reused. Smaller
standalone systems keep the sample-local VLM and STT resource settings.

How to run (from agent-samples/simple-vlm-example/):
    uv sync && uv run simple_vlm_example
"""
import argparse
from dataclasses import replace
from pathlib import Path

from xr_ai_launcher import (
    GPUInventoryError,
    Process,
    detect_gpu_config,
    ensure_credentials,
    load_model_deployment,
    require_credentials,
    run_stack,
)
from xr_ai_launcher._config import _resolve_config_variant
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent
_MODEL_SERVERS_YAML = _BASE.parent / "model-servers" / "yaml"

_WORKER_CONFIG = "yaml/simple_vlm_example_worker.yaml"

_MODEL_PROCESSES = {
    # NIM services are reused from the model-servers vlm_llm_nim / vlm_speech_nim
    # stacks; start the matching stack first.
    "stt-nim": Process(
        "stt-nim", "../../services/nim-server", "nim_server",
    ),
    "tts-nim": Process(
        "tts-nim", "../../services/nim-server", "nim_server",
    ),
    "vlm-nim": Process(
        "vlm-nim", "../../services/nim-server", "nim_server",
    ),
    "vlm": Process(
        "vlm", "../../services/vlm-server", "vlm_server",
        config="yaml/vlm_server.yaml",
    ),
    "vlm-omni": Process(
        "vlm-omni",
        "../../services/nemotron-omni-llm",
        "nemotron_omni_llm_server",
    ),
    "stt": Process(
        "stt", "../../services/stt-server", "stt_server",
        config="yaml/stt_server.yaml",
    ),
    "tts": Process(
        "tts", "../../services/piper-tts", "piper_tts_server",
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
    procs = [
        Process("hub", "../../services/device-io-hub", "device_io_hub",
                config="yaml/device_io_hub.yaml"),
    ]
    for service, process in _MODEL_PROCESSES.items():
        launch_mode = deployment.launch_mode(service)
        if launch_mode is not None:
            if launch_mode == "own" and service == "vlm":
                # Share the complete launch contract with model-servers so a
                # compatible persistent server has the same fingerprint, but
                # retain the standalone config on unsupported hardware.
                try:
                    gpu_config = detect_gpu_config()
                except GPUInventoryError:
                    gpu_config = None
                if gpu_config is not None:
                    shared_config = _resolve_config_variant(
                        _MODEL_SERVERS_YAML / gpu_config,
                        "vlm_server",
                        "default",
                    )
                    if not shared_config.is_file():
                        raise FileNotFoundError(
                            f"shared VLM config does not exist: {shared_config}"
                        )
                    process = replace(process, config=shared_config)
            procs.append(replace(process, launch_mode=launch_mode))
    procs.append(
        Process(
            "worker", "worker", "simple_vlm_example_worker",
            config=_WORKER_CONFIG,
        )
    )
    return procs, deployment.required_credentials


def run() -> None:
    setup_logging("orchestrator", namespace="simple-vlm-example")

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--allow-anonymous", action="store_true",
                   help="Start without HF_TOKEN (unauthenticated checkpoint "
                        "downloads may stall indefinitely).")
    ns, _ = p.parse_known_args()

    processes, credentials = _build_processes()
    # A missing HF_TOKEN silently stalls the multi-GB first-run download; see
    # docs/source/getting_started/credentials.md.
    require_credentials("HF_TOKEN", allow_missing=ns.allow_anonymous)
    for credential in credentials:
        ensure_credentials(credential)
    run_stack(processes, _BASE)


if __name__ == "__main__":
    run()
