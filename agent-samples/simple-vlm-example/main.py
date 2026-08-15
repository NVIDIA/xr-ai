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
profile replaces only the VLM with NVIDIA NIM.

How to run (from agent-samples/simple-vlm-example/):
    uv sync && uv run simple_vlm_example
"""
import argparse
from dataclasses import replace
from pathlib import Path

from xr_ai_launcher import (
    EndpointProbe,
    Process,
    ensure_credentials,
    load_model_deployment,
    require_credentials,
    run_stack,
)
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent

_WORKER_CONFIG = "yaml/simple_vlm_example_worker.yaml"

_MODEL_PROCESSES = {
    "vlm": Process(
        "vlm", "../../services/vlm-server", "vlm_server",
        config="yaml/vlm_server.yaml",
        port=8100,
    ),
    "vlm-omni": Process(
        "vlm-omni",
        "../../services/nemotron-omni-llm",
        "nemotron_omni_llm_server",
        port=8108,
    ),
    "stt": Process(
        "stt", "../../services/stt-server", "stt_server",
        config="yaml/stt_server.yaml",
        port=8103,
    ),
    "tts": Process(
        "tts", "../../services/piper-tts", "piper_tts_server",
        config="yaml/piper_tts_server.yaml",
        port=8105,
    ),
}


def _build_processes() -> tuple[
    list[Process],
    tuple[str, ...],
    tuple[EndpointProbe, ...],
]:
    deployment = load_model_deployment(_BASE / _WORKER_CONFIG)
    unknown_services = deployment.services.keys() - _MODEL_PROCESSES.keys()
    if unknown_services:
        raise ValueError(
            f"model profile declares unknown services: {sorted(unknown_services)}"
        )
    procs = [
        Process("hub", "../../services/xr-media-hub", "xr_media_hub",
                config="yaml/xr_media_hub.yaml"),
    ]
    for service, process in _MODEL_PROCESSES.items():
        launch_mode = deployment.launch_mode(service)
        if launch_mode is not None:
            if launch_mode == "reuse":
                endpoint = deployment.reused_endpoints[service]
                procs.append(
                    replace(
                        process,
                        launch_mode=launch_mode,
                        health_url=endpoint.health_url,
                        readiness=endpoint.readiness,
                        api_key_env=endpoint.api_key_env,
                    )
                )
            else:
                procs.append(replace(process, launch_mode=launch_mode))
    procs.append(
        Process(
            "worker", "worker", "simple_vlm_example_worker",
            config=_WORKER_CONFIG,
        )
    )
    endpoint_probes = tuple(
        EndpointProbe(
            name=role,
            health_url=endpoint.health_url,
            readiness=endpoint.readiness,
            api_key_env=endpoint.api_key_env,
        )
        for role, endpoint in deployment.external_endpoints.items()
    )
    return procs, deployment.required_credentials, endpoint_probes


def run() -> None:
    setup_logging("orchestrator", namespace="simple-vlm-example")

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--allow-anonymous", action="store_true",
                   help="Start without HF_TOKEN (unauthenticated checkpoint "
                        "downloads may stall indefinitely).")
    ns, _ = p.parse_known_args()

    processes, credentials, endpoint_probes = _build_processes()
    # A missing HF_TOKEN silently stalls the multi-GB first-run download; see
    # docs/source/getting_started/credentials.md.
    require_credentials("HF_TOKEN", allow_missing=ns.allow_anonymous)
    for credential in credentials:
        ensure_credentials(credential)
    run_stack(processes, _BASE, endpoint_probes=endpoint_probes)


if __name__ == "__main__":
    run()
