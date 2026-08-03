# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
tea-making-sample orchestrator - YAML-driven guided workflow over voice.

How to run:

    uv run --project agent-samples/tea-making-sample tea_making_sample
"""

import socket
from pathlib import Path

from xr_ai_launcher import Process, detect_gpu_config, run_stack, warn_if_missing
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent

_WORKER_CONFIG = "yaml/tea_making_worker.yaml"
_OMNI_PORT = 8108
_EMBEDDING_PORT = 8109
_PROFILE_ALIASES = {"spark": "96G_blackwell"}
_SUPPORTED_PROFILES = {"96G_blackwell", "dual_48G_ada"}
_LEGACY_MODEL_PORTS = {
    8100: "VLM",
    8106: "Llama-Nemotron",
    8107: "Nemotron-Nano",
}


def _build_processes(profile: str | None = None) -> list[Process]:
    detected_profile = profile or detect_gpu_config()
    selected_profile = _PROFILE_ALIASES.get(detected_profile, detected_profile)
    if selected_profile not in _SUPPORTED_PROFILES:
        supported = ", ".join(sorted(_SUPPORTED_PROFILES))
        raise RuntimeError(
            f"Unsupported GPU profile {detected_profile!r}; expected one of: {supported}"
        )
    ai = f"yaml/{selected_profile}"

    return [
        Process(
            "omni",
            "../../ai-services/llm/nemotron_omni",
            "nemotron_omni_llm_server",
            config=f"{ai}/nemotron_omni_llm_server.yaml",
            port=_OMNI_PORT,
        ),
        Process(
            "stt",
            "../../ai-services/stt-server",
            "stt_server",
            config=f"{ai}/stt_server.yaml",
            port=8103,
        ),
        Process(
            "embedding",
            "../../ai-services/embedding-server",
            "embedding_server",
            config=f"{ai}/embedding_server.yaml",
            port=_EMBEDDING_PORT,
        ),
        Process(
            "rag",
            "../../services/rag-service",
            "rag_service",
            config="yaml/rag_service.yaml",
        ),
        Process(
            "hub",
            "../../server-runtime",
            "xr_media_hub",
            config="yaml/xr_media_hub.yaml",
        ),
        Process(
            "tts",
            "../../ai-services/tts/piper",
            "piper_tts_server",
            config="yaml/piper_tts_server.yaml",
            port=8105,
        ),
        Process(
            "worker",
            "worker",
            "tea_making_worker",
            config=_WORKER_CONFIG,
        ),
    ]


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _check_model_ports() -> None:
    if _port_open(_OMNI_PORT):
        return
    busy = [
        f"{name} on {port}"
        for port, name in _LEGACY_MODEL_PORTS.items()
        if _port_open(port)
    ]
    if not busy:
        return
    raise SystemExit(
        "tea-making-sample uses Nemotron-Omni on port 8108, but these local "
        f"model services are already running and likely occupy GPU memory: {', '.join(busy)}.\n"
        "Stop the shared model-server stack first:\n"
        "  uv run --project agent-samples/model-servers model_servers --stop\n"
        "Then rerun:\n"
        "  uv run --project agent-samples/tea-making-sample tea_making_sample"
    )


def run() -> None:
    setup_logging("orchestrator", namespace="tea-making-sample")
    warn_if_missing("HF_TOKEN")
    _check_model_ports()
    run_stack(_build_processes(), _BASE)


if __name__ == "__main__":
    run()
