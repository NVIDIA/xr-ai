# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the YAML-driven tea guidance sample."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from xr_ai_launcher import Process, detect_gpu_config, load_model_deployment, run_stack, warn_if_missing
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent
_WORKER_CONFIG = "yaml/tea_making_worker.yaml"


def _model_processes() -> dict[str, Process]:
    detected = detect_gpu_config()
    profile = {"spark": "96G_blackwell"}.get(detected, detected)
    if profile not in {"96G_blackwell", "dual_48G_ada"}:
        raise RuntimeError(f"unsupported local model profile: {profile}")
    return {
        "agent-llm": Process(
            "agent-llm",
            "../../ai-services/llm/nemotron3_nano",
            "nemotron3_nano_llm_server",
        ),
        "vlm": Process("vlm", "../../ai-services/vlm-server", "vlm_server"),
        "embedding": Process(
            "embedding",
            "../../ai-services/embedding-server",
            "embedding_server",
            config=f"yaml/{profile}/embedding_server.yaml",
        ),
        "stt": Process(
            "stt",
            "../../ai-services/stt-server",
            "stt_server",
            config=f"yaml/{profile}/stt_server.yaml",
        ),
        "omni": Process(
            "omni",
            "../../ai-services/llm/nemotron_omni",
            "nemotron_omni_llm_server",
            config=f"yaml/{profile}/nemotron_omni_llm_server.yaml",
        ),
        "tts": Process(
            "tts",
            "../../ai-services/tts/piper",
            "piper_tts_server",
            config="yaml/piper_tts_server.yaml",
        ),
    }


def _build_processes() -> list[Process]:
    deployment = load_model_deployment(_BASE / _WORKER_CONFIG)
    available = _model_processes()
    unknown = deployment.services.keys() - available.keys()
    if unknown:
        raise ValueError(f"model profile declares unknown services: {sorted(unknown)}")
    processes = [
        replace(available[name], launch_mode=mode)
        for name, mode in deployment.services.items()
    ]
    processes.extend(
        [
            Process("rag", "../../services/rag-service", "rag_service", config="yaml/rag_service.yaml"),
            Process("hub", "../../server-runtime", "xr_media_hub", config="yaml/xr_media_hub.yaml"),
            Process("worker", "worker", "tea_making_worker", config=_WORKER_CONFIG),
        ]
    )
    return processes


def run() -> None:
    setup_logging("orchestrator", namespace="tea-making-sample")
    warn_if_missing("HF_TOKEN")
    run_stack(_build_processes(), _BASE)


if __name__ == "__main__":
    run()
