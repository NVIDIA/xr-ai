# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
model-servers orchestrator — starts one shared AI inference stack and exits.

All servers are launch_mode="persist" so they keep running after this
process exits.  Model weights stay hot across stack restarts.

Servers started
---------------
  stt        — nvidia/parakeet-tdt-0.6b-v3        port 8103  (NeMo ASR)
  omni       — Nemotron-3-Nano-Omni-30B-A3B       port 8108  (vLLM)
  vlm        — nvidia/Cosmos-Reason1-7B           port 8100  (vLLM)
  embedding  — nvidia/llama-nemotron-embed-1b-v2  port 8109  (vLLM)

How to run:
    uv run --project agent-samples/model-servers model_servers

To stop all model servers:
    uv run --project agent-samples/model-servers model_servers --stop
"""
import argparse
from pathlib import Path

from xr_ai_launcher import Process, detect_gpu_config, run_stack, warn_if_missing
from xr_ai_logging import setup_logging
from xr_ai_vllm import stop_persistent_servers

_BASE = Path(__file__).resolve().parent
_STOP_SERVICES = [
    ("stt", 8103),
    ("agent-llm", 8107),
    ("vlm", 8100),
    ("omni", 8108),
    ("embedding", 8109),
]
_LEGACY_SERVICES = [("agent-llm", 8107)]

# Omni loads before Cosmos so its MoE kernels compile before other vLLM
# services reserve memory on single-GPU profiles.
def _build_processes() -> list[Process]:
    """Return the shared multimodal model stack for the detected GPU profile."""
    ai = f"yaml/{detect_gpu_config()}"
    stt = Process(
        "stt", "../../ai-services/stt-server", "stt_server",
        config=f"{ai}/stt_server.yaml",
        launch_mode="persist", port=8103,
    )
    embedding = Process(
        "embedding", "../../ai-services/embedding-server", "embedding_server",
        config=f"{ai}/embedding_server.yaml",
        launch_mode="persist", port=8109,
    )
    return [
        stt,
        Process(
            "omni", "../../ai-services/llm/nemotron_omni",
            "nemotron_omni_llm_server",
            config=f"{ai}/nemotron_omni_llm_server.yaml",
            launch_mode="persist", port=8108,
        ),
        Process("vlm",       "../../ai-services/vlm-server",         "vlm_server",
                config=f"{ai}/vlm_server.yaml",
                launch_mode="persist", port=8100),
        embedding,
    ]


def _stop_models() -> None:
    # Surface docker/ss/lsof failures so operators see why --stop aborted
    # instead of a silent traceback exit.
    try:
        stop_persistent_servers(_STOP_SERVICES)
    except Exception as exc:
        print(f"model-servers: failed to stop persistent servers: {exc}", flush=True)


def _stop_legacy_models() -> None:
    """Free GPU capacity held by models removed from the shared stack."""
    if not stop_persistent_servers(_LEGACY_SERVICES):
        raise RuntimeError("could not stop legacy persistent model servers")


def run() -> None:
    setup_logging("orchestrator", namespace="model-servers")

    p = argparse.ArgumentParser(description="Start the shared Omni + Cosmos model stack.")
    p.add_argument(
        "--stop", action="store_true",
        help="Stop every persisted shared model and exit.",
    )
    ns = p.parse_args()

    if ns.stop:
        _stop_models()
        return

    # HF_TOKEN is optional for the default (public) models — it only raises HF
    # rate limits / download speed and is required only for gated models.
    # Warn instead of prompting; see docs/credentials.md.
    warn_if_missing("HF_TOKEN")
    _stop_legacy_models()
    run_stack(_build_processes(), _BASE, exit_after_ready=True)


if __name__ == "__main__":
    run()
