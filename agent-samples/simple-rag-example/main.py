# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
simple-rag-example orchestrator — vision Q&A with optional doc grounding.

Pipeline
--------
Audio in (mic)        → STT → text query
Text in (data ch.)    → text query
"ping" data message   → default prompt ("Describe what you see.")
                                               │
       retrieve top-k doc chunks ← rag-mcp (dense retrieval via embedding-server)
       (always attempted; ~10–50 ms)           │
                                               │
            latest video frame + query + context (if any) → VLM stream
                                               │
                    sentence-batched TTS ←─────┴──→  data channel reply

How to run (from agent-samples/simple-rag-example/):
    uv sync && uv run simple_rag_example
"""
from pathlib import Path

from xr_ai_launcher import Process, ensure_credentials, run_stack
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent

_PROCESSES: list[Process] = [
    Process("hub",     "../../server-runtime",                  "xr_media_hub",
            config="yaml/xr_media_hub.yaml"),
    Process("vlm",     "../../ai-services/vlm-server",          "vlm_server",
            config="yaml/vlm_server.yaml"),
    Process("stt",     "../../ai-services/stt-server",          "stt_server",
            config="yaml/stt_server.yaml"),
    Process("tts",     "../../ai-services/tts/piper",           "piper_tts_server",
            config="yaml/piper_tts_server.yaml"),
    # embedding-server must start before rag-mcp: rag-mcp polls /health at startup.
    Process("embed",   "../../ai-services/embedding-server",    "embedding_server",
            config="yaml/embedding_server.yaml"),
    Process("rag-mcp", "../../agent-mcp-servers/rag-mcp",       "rag_mcp_server",
            config="yaml/rag_mcp_server.yaml"),
    Process("worker",  "worker",                                "simple_rag_example_worker",
            config="yaml/simple_rag_example_worker.yaml"),
]


def run() -> None:
    setup_logging("orchestrator", namespace="simple-rag-example")
    ensure_credentials("HF_TOKEN")
    run_stack(_PROCESSES, _BASE)


if __name__ == "__main__":
    run()
