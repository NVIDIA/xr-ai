# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
model-servers orchestrator — starts the four AI inference servers and exits.

All four servers are launch_mode="persist" so they keep running after this
process exits.  Model weights stay hot across stack restarts.

Servers started
---------------
  stt        — nvidia/parakeet-tdt-0.6b-v3        port 8103  (NeMo ASR)
  agent-llm  — NVIDIA-Nemotron-3-Nano-30B-A3B      port 8107  (vLLM)
  vlm        — nvidia/Cosmos-Reason1-7B            port 8100  (vLLM)
  llm        — nvidia/Llama-3.1-Nemotron-Nano-8B   port 8106  (vLLM)

How to run:
    uv run --project agent-samples/model-servers model_servers

To stop all model servers:
    uv run --project agent-samples/model-servers model_servers --stop
"""
import argparse
import os
from pathlib import Path

from xr_ai_launcher import Process, detect_gpu_config, run_stack, warn_if_missing
from xr_ai_logging import setup_logging
from xr_ai_vllm import stop_persistent_servers

_BASE = Path(__file__).resolve().parent

# agent-llm (Nemotron-30B) loads first on single-GPU profiles so its
# FlashInfer MoE JIT compilation runs with the full GPU free.  The compiled
# kernels are cached after the first run (~3-8 min).
def _build_processes() -> list[Process]:
    """Detect the GPU profile and return the per-profile process list."""
    ai = f"yaml/{detect_gpu_config()}"
    return [
        Process("stt",       "../../ai-services/stt-server",         "stt_server",
                config=f"{ai}/stt_server.yaml",
                launch_mode="persist", port=8103),
        Process("agent-llm", "../../ai-services/llm/nemotron3_nano", "nemotron3_nano_llm_server",
                config=f"{ai}/nemotron3_nano_llm_server.yaml",
                launch_mode="persist", port=8107),
        Process("vlm",       "../../ai-services/vlm-server",         "vlm_server",
                config=f"{ai}/vlm_server.yaml",
                launch_mode="persist", port=8100),
        Process("llm",       "../../ai-services/llm/llama_nemotron", "llama_nemotron_llm_server",
                config=f"{ai}/llama_nemotron_llm_server.yaml",
                launch_mode="persist", port=8106),
    ]


def _stop_models() -> None:
    # Surface docker/ss/lsof failures so operators see why --stop aborted
    # instead of a silent traceback exit.
    try:
        stop_persistent_servers([
            (p.name, p.port)
            for p in _build_processes()
            if p.launch_mode == "persist" and p.port is not None
        ])
    except Exception as exc:
        print(f"model-servers: failed to stop persistent servers: {exc}", flush=True)


def run() -> None:
    setup_logging("orchestrator", namespace="model-servers")

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--stop", action="store_true",
                   help="Stop any persisted vLLM model servers and exit.")
    ns, _ = p.parse_known_args()

    if ns.stop:
        _stop_models()
        return

    # HF_TOKEN is optional for the default (public) models — it only raises HF
    # rate limits / download speed and is required only for gated models.
    # Warn instead of prompting; see docs/credentials.md.
    warn_if_missing("HF_TOKEN")
    run_stack(_build_processes(), _BASE, exit_after_ready=True)


if __name__ == "__main__":
    run()
