# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
model-servers orchestrator — starts one shared AI inference stack and exits.

All servers are launch_mode="persist" so they keep running after this
process exits.  Model weights stay hot across stack restarts.

Servers started
---------------
  default / --vlm-llm-stack
    stt        — nvidia/parakeet-tdt-0.6b-v3        port 8103  (NeMo ASR)
    agent-llm  — NVIDIA-Nemotron-3-Nano-30B-A3B     port 8107  (vLLM)
    vlm        — nvidia/Cosmos-Reason1-7B           port 8100  (vLLM)
    embedding  — nvidia/llama-nemotron-embed-1b-v2  port 8109  (vLLM)

  --omni-stack
    stt        — nvidia/parakeet-tdt-0.6b-v3        port 8103  (NeMo ASR)
    omni       — Nemotron-3-Nano-Omni-30B-A3B       port 8108  (vLLM)
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
_INCOMPATIBLE_STACK_SERVICES = {
    "vlm-llm": [("agent-llm", 8107), ("vlm", 8100)],
    "omni": [("omni", 8108)],
}

# agent-llm (Nemotron-30B) loads first on single-GPU profiles so its
# FlashInfer MoE JIT compilation runs with the full GPU free.  The compiled
# kernels are cached after the first run (~3-8 min).
def _build_processes(stack: str = "vlm-llm") -> list[Process]:
    """Return the selected shared model stack for the detected GPU profile."""
    ai = f"yaml/{detect_gpu_config()}"
    embedding_config = f"{ai}/embedding_server_{stack.replace('-', '_')}.yaml"
    if not (_BASE / embedding_config).exists():
        embedding_config = f"{ai}/embedding_server.yaml"
    stt = Process(
        "stt", "../../ai-services/stt-server", "stt_server",
        config=f"{ai}/stt_server.yaml",
        launch_mode="persist", port=8103,
    )
    embedding = Process(
        "embedding", "../../ai-services/embedding-server", "embedding_server",
        config=embedding_config,
        launch_mode="persist", port=8109,
    )
    if stack == "omni":
        return [
            stt,
            Process(
                "omni", "../../ai-services/llm/nemotron_omni",
                "nemotron_omni_llm_server",
                config=f"{ai}/nemotron_omni_llm_server.yaml",
                launch_mode="persist", port=8108,
            ),
            embedding,
        ]
    if stack != "vlm-llm":
        raise ValueError(f"unknown model stack: {stack}")
    return [
        stt,
        Process("agent-llm", "../../ai-services/llm/nemotron3_nano", "nemotron3_nano_llm_server",
                config=f"{ai}/nemotron3_nano_llm_server.yaml",
                launch_mode="persist", port=8107),
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


def _stop_incompatible_stack(stack: str) -> None:
    """Free GPU capacity held by the other mutually exclusive model stack."""
    incompatible = _INCOMPATIBLE_STACK_SERVICES["vlm-llm" if stack == "omni" else "omni"]
    if not stop_persistent_servers(incompatible):
        raise RuntimeError("could not stop incompatible persistent model stack")


def run() -> None:
    setup_logging("orchestrator", namespace="model-servers")

    p = argparse.ArgumentParser(add_help=False)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--stop", action="store_true",
        help="Stop every persisted model-server stack and exit.",
    )
    mode.add_argument(
        "--omni-stack", action="store_const", const="omni", dest="stack",
        help="Start Nemotron-3-Nano-Omni with STT instead of the VLM + LLM stack.",
    )
    mode.add_argument(
        "--vlm-llm-stack", action="store_const", const="vlm-llm", dest="stack",
        help="Start the default Nemotron-3-Nano + Cosmos VLM stack.",
    )
    p.set_defaults(stack="vlm-llm")
    p.add_argument("--allow-anonymous", action="store_true",
                   help="Start without HF_TOKEN (unauthenticated downloads "
                        "of the multi-GB checkpoints may stall indefinitely).")
    ns, _ = p.parse_known_args()

    if ns.stop:
        _stop_models()
        return

    # A missing HF_TOKEN silently stalls the multi-GB first-run download; see
    # docs/credentials.md.
    require_credentials("HF_TOKEN", allow_missing=ns.allow_anonymous)
    _stop_incompatible_stack(ns.stack)
    run_stack(_build_processes(ns.stack), _BASE, exit_after_ready=True)


if __name__ == "__main__":
    run()
