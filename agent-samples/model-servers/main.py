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
  vlm        — nvidia/Cosmos3-Nano Reasoner       port 8100  (vLLM)
  embedding  — nvidia/llama-nemotron-embed-1b-v2  port 8109  (vLLM)

How to run:
    uv run --project agent-samples/model-servers model_servers

To stop all model servers:
    uv run --project agent-samples/model-servers model_servers --stop
"""
import argparse
import os
from pathlib import Path

from xr_ai_launcher import Process, detect_gpu_config, require_credentials, run_stack
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
_REPLACED_SERVICES = [("agent-llm", 8107)]


def _build_processes() -> list[Process]:
    """Return the shared Omni, Cosmos, speech, and embedding services."""
    ai = f"yaml/{detect_gpu_config()}"
    stt = Process(
        "stt", "../../services/stt-server", "stt_server",
        config=f"{ai}/stt_server.yaml",
        launch_mode="persist", port=8103,
    )
    embedding = Process(
        "embedding", "../../services/embedding-server", "embedding_server",
        config=f"{ai}/embedding_server.yaml",
        launch_mode="persist", port=8109,
    )
    return [
        stt,
        Process("omni",      "../../services/nemotron-omni-llm", "nemotron_omni_llm_server",
                config=f"{ai}/nemotron_omni_llm_server.yaml",
                launch_mode="persist", port=8108),
        Process("vlm",       "../../services/vlm-server",          "vlm_server",
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


def _stop_replaced_models() -> None:
    """Free GPU capacity held by the superseded Nano text model."""
    if not stop_persistent_servers(_REPLACED_SERVICES):
        raise RuntimeError("could not stop replaced persistent model")


def run() -> None:
    setup_logging("orchestrator", namespace="model-servers")

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--stop", action="store_true",
        help="Stop every persisted model service and exit.",
    )
    p.add_argument("--allow-anonymous", action="store_true",
                   help="Start without HF_TOKEN (unauthenticated downloads "
                        "of the multi-GB checkpoints may stall indefinitely).")
    ns = p.parse_args()

    if ns.stop:
        _stop_models()
        return

    # A missing HF_TOKEN silently stalls the multi-GB first-run download; see
    # docs/source/getting_started/credentials.md.
    require_credentials("HF_TOKEN", allow_missing=ns.allow_anonymous)
    _stop_replaced_models()
    run_stack(_build_processes(), _BASE, exit_after_ready=True)


if __name__ == "__main__":
    run()
