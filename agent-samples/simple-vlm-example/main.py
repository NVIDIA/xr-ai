# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
simple-vlm-example orchestrator — vision Q&A over voice or text.

Pipeline
--------
Audio in (mic)     → STT → text query ─┐
Text in (data ch.) ─────→ text query ──┴→ latest video frame → VLM stream
                                                      │
                    sentence-batched TTS audio  ←──┴──→ data channel reply

Model deployment
----------------
``--piper`` keeps the CPU Piper path from main. ``--magpie`` selects the
streaming Magpie TTS NIM path and its lower-memory local VLM profile.

How to run (from agent-samples/simple-vlm-example/):
    uv sync && uv run main.py --piper
    uv run main.py --magpie
"""
import argparse
from dataclasses import replace
from pathlib import Path

from xr_ai_launcher import (
    Process,
    ensure_credentials,
    load_model_deployment,
    require_credentials,
    run_stack,
)
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent

_WORKER_CONFIGS = {
    "piper": "yaml/simple_vlm_example_worker.yaml",
    "magpie": "yaml/simple_vlm_example_worker.magpie.yaml",
}

_MODEL_PROCESSES = {
    "vlm": Process(
        "vlm", "../../services/vlm-server", "vlm_server",
        config="yaml/vlm_server.yaml",
    ),
    "vlm-magpie": Process(
        "vlm", "../../services/vlm-server", "vlm_server",
        config="yaml/vlm_server.magpie.yaml",
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
    "tts-magpie": Process(
        "tts", "../../services/magpie-tts-nim", "magpie_tts_nim_server",
        config="yaml/magpie_tts_nim_server.yaml",
        quiet_native_output=True,
    ),
}


def _build_processes(
    tts_backend: str = "piper",
) -> tuple[list[Process], tuple[str, ...]]:
    try:
        worker_config = _WORKER_CONFIGS[tts_backend]
    except KeyError as exc:
        raise ValueError(f"unsupported TTS backend: {tts_backend!r}") from exc
    deployment = load_model_deployment(_BASE / worker_config)
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
            procs.append(replace(process, launch_mode=launch_mode))
    procs.append(
        Process(
            "worker", "worker", "simple_vlm_example_worker",
            config=worker_config,
        )
    )
    credentials = set(deployment.required_credentials)
    if tts_backend == "magpie":
        credentials.add("NGC_API_KEY")
    return procs, tuple(sorted(credentials))


def run() -> None:
    setup_logging("orchestrator", namespace="simple-vlm-example")

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--allow-anonymous", action="store_true",
                   help="Start without HF_TOKEN (unauthenticated checkpoint "
                        "downloads may stall indefinitely).")
    tts = p.add_mutually_exclusive_group()
    tts.add_argument(
        "--piper",
        dest="tts_backend",
        action="store_const",
        const="piper",
        help="Use the existing CPU Piper TTS path (default).",
    )
    tts.add_argument(
        "--magpie",
        dest="tts_backend",
        action="store_const",
        const="magpie",
        help="Use streaming Magpie TTS NIM on the GPU.",
    )
    p.set_defaults(tts_backend="piper")
    ns, _ = p.parse_known_args()

    processes, credentials = _build_processes(ns.tts_backend)
    # A missing HF_TOKEN silently stalls the multi-GB first-run download; see
    # docs/source/getting_started/credentials.md.
    require_credentials("HF_TOKEN", allow_missing=ns.allow_anonymous)
    for credential in credentials:
        ensure_credentials(credential)
    run_stack(processes, _BASE)


if __name__ == "__main__":
    run()
