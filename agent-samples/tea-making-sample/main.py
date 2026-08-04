# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
tea-making-sample orchestrator - YAML-driven guided workflow over voice.

How to run:

    uv run --project agent-samples/model-servers model_servers
    uv run --project agent-samples/tea-making-sample tea_making_sample
"""

import socket
from pathlib import Path

from xr_ai_launcher import Process, run_stack
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent

_WORKER_CONFIG = "yaml/tea_making_worker.yaml"
_STT_PORT = 8103
_AGENT_LLM_PORT = 8107
_VLM_PORT = 8100
_EMBEDDING_PORT = 8109
_REQUIRED_MODEL_PORTS = {
    _STT_PORT: "Parakeet STT",
    _AGENT_LLM_PORT: "Nemotron-3 Nano",
    _VLM_PORT: "Cosmos",
    _EMBEDDING_PORT: "Nemotron Embed",
}


def _build_processes() -> list[Process]:
    return [
        Process(
            "stt", "../../ai-services/stt-server", "stt_server",
            launch_mode="reuse", port=_STT_PORT,
        ),
        Process(
            "agent-llm", "../../ai-services/llm/nemotron3_nano",
            "nemotron3_nano_llm_server",
            launch_mode="reuse", port=_AGENT_LLM_PORT,
        ),
        Process(
            "vlm", "../../ai-services/vlm-server", "vlm_server",
            launch_mode="reuse", port=_VLM_PORT,
        ),
        Process(
            "embedding", "../../ai-services/embedding-server", "embedding_server",
            launch_mode="reuse", port=_EMBEDDING_PORT,
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
    missing = [
        f"{name} on {port}"
        for port, name in _REQUIRED_MODEL_PORTS.items()
        if not _port_open(port)
    ]
    if not missing:
        return

    raise SystemExit(
        "tea-making-sample reuses the shared model-server stack, but these "
        f"required services are missing: {', '.join(missing)}.\n"
        "Start the model servers first:\n"
        "  uv run --project agent-samples/model-servers model_servers\n"
        "Then start the sample:\n"
        "  uv run --project agent-samples/tea-making-sample tea_making_sample"
    )


def run() -> None:
    setup_logging("orchestrator", namespace="tea-making-sample")
    _check_model_ports()
    run_stack(_build_processes(), _BASE)


if __name__ == "__main__":
    run()
