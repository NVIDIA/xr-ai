# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Require shared models, then launch the hub and visual task worker."""

from pathlib import Path

from xr_ai_launcher import Process, ensure_credentials, load_model_deployment, run_stack, warn_if_missing

_BASE = Path(__file__).resolve().parent
_WORKER_CONFIG = "yaml/visual_task_guide_worker.yaml"


def _build_processes() -> list[Process]:
    deployment = load_model_deployment(_BASE / _WORKER_CONFIG)
    definitions = {
        "vlm": ("../../ai-services/vlm-server", "vlm_server", None, 8100),
        "llm": (
            "../../ai-services/llm/llama_nemotron",
            "llama_nemotron_llm_server",
            None,
            8106,
        ),
        "stt": ("../../ai-services/stt-server", "stt_server", None, 8103),
        "tts": ("../../ai-services/tts/piper", "piper_tts_server", "yaml/piper_tts_server.yaml", 8105),
        "embedding": ("../../ai-services/embedding-server", "embedding_server", None, 8109),
    }
    unknown = deployment.services.keys() - definitions.keys()
    if unknown:
        raise ValueError(f"model profile declares unknown services: {sorted(unknown)}")
    processes = []
    for role in ("vlm", "llm", "stt", "tts", "embedding"):
        mode = deployment.launch_mode(role)
        if mode:
            project, command, config, port = definitions[role]
            processes.append(Process(role, project, command, config=config, launch_mode=mode, port=port))
    processes.extend(
        [
            Process(
                "rag",
                "../../services/rag-service",
                "rag_service",
                config="yaml/rag_service.yaml",
            ),
            Process("hub", "../../server-runtime", "xr_media_hub", config="yaml/xr_media_hub.yaml"),
            Process("worker", "worker", "visual_task_guide_worker", config=_WORKER_CONFIG),
        ]
    )
    return processes


def run() -> None:
    deployment = load_model_deployment(_BASE / _WORKER_CONFIG)
    warn_if_missing("HF_TOKEN")
    for credential in deployment.required_credentials:
        ensure_credentials(credential)
    run_stack(_build_processes(), _BASE)


if __name__ == "__main__":
    run()
