# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
simple-vlm-example orchestrator — vision Q&A over voice or text.

Pipeline
--------
Audio in (mic)        → STT → text query
Text in (data ch.)    → text query
"ping" data message   → default prompt ("Describe what you see.")
                                                │
                                                ▼
                  latest video frame + query → VLM stream
                                                │
                       sentence-batched TTS  ←──┴──→  data channel reply

Model deployment
----------------
``models_config`` in yaml/simple_vlm_example_worker.yaml selects a deployment
profile. The default profile owns local STT, VLM, and TTS services; the hosted
profile replaces only the VLM with NVIDIA NIM; models.nim_local.json serves
every role from self-hosted NIM containers (yaml/nim_*_server.yaml).

How to run (from agent-samples/simple-vlm-example/):
    uv sync && uv run simple_vlm_example
"""
from dataclasses import replace
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

_MODEL_PROCESSES = {
    # NIM containers precede the local servers: speech NIMs allocate fixed
    # VRAM, while the VLM NIM grabs most of the free VRAM on its GPU at
    # startup for KV cache.
    "stt-nim": Process(
        "stt-nim", "../../ai-services/nim-server", "nim_server",
        config="yaml/nim_stt_server.yaml",
    ),
    "tts-nim": Process(
        "tts-nim", "../../ai-services/nim-server", "nim_server",
        config="yaml/nim_tts_server.yaml",
    ),
    "vlm-nim": Process(
        "vlm-nim", "../../ai-services/nim-server", "nim_server",
        config="yaml/nim_vlm_server.yaml",
    ),
    "vlm": Process(
        "vlm", "../../ai-services/vlm-server", "vlm_server",
        config="yaml/vlm_server.yaml",
    ),
    "vlm-omni": Process(
        "vlm-omni",
        "../../ai-services/llm/nemotron_omni",
        "nemotron_omni_llm_server",
    ),
    "stt": Process(
        "stt", "../../ai-services/stt-server", "stt_server",
        config="yaml/stt_server.yaml",
    ),
    "tts": Process(
        "tts", "../../ai-services/tts/piper", "piper_tts_server",
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
        Process("hub", "../../server-runtime", "xr_media_hub",
                config="yaml/xr_media_hub.yaml"),
    ]
    for service, process in _MODEL_PROCESSES.items():
        launch_mode = deployment.launch_mode(service)
        if launch_mode is not None:
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
    processes, credentials = _build_processes()
    # HF_TOKEN is optional for the default (public) model — it only raises HF
    # rate limits / download speed and is required only for gated models.
    # Warn instead of prompting; see docs/credentials.md.
    warn_if_missing("HF_TOKEN")
    for credential in credentials:
        ensure_credentials(credential)
    run_stack(processes, _BASE)


if __name__ == "__main__":
    run()
