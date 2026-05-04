# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
simple-vlm-example orchestrator — vision Q&A over voice or text.

Pipeline
--------
Audio in (mic)        → STT → text query
Text in (data ch.)    → text query
"ping" data message   → default prompt ("Describe what you see.")
                                                │
                                                ▼
                  latest video frame + query → VLM stream
                                                │
                       sentence-batched TTS  ←──┴──→  data channel reply

How to run (from agent-samples/simple-vlm-example/):
    uv sync && uv run simple_vlm_example
"""
import asyncio
from pathlib import Path

import yaml

from xr_ai_launcher import Process, ensure_credentials, run_stack

_BASE = Path(__file__).resolve().parent


def _build_processes() -> list[Process]:
    # Read vlm_backend from the worker YAML so the orchestrator starts the
    # matching AI service automatically.
    worker_cfg: dict = {}
    worker_yaml = _BASE / "simple_vlm_example_worker.yaml"
    if worker_yaml.exists():
        with open(worker_yaml) as f:
            worker_cfg = yaml.safe_load(f) or {}

    backend = worker_cfg.get("vlm_backend", "cosmos")
    if backend == "omni":
        vlm = Process("vlm", "../../ai-services/llm/nemotron_omni",
                      "nemotron_omni_llm_server")
    else:  # cosmos (default)
        vlm = Process("vlm", "../../ai-services/vlm-server", "vlm_server")

    return [
        Process("hub",    "../../server-runtime",        "xr_media_hub"),
        vlm,
        Process("stt",    "../../ai-services/stt-server", "stt_server"),
        Process("tts",    "../../ai-services/tts/piper",  "piper_tts_server"),
        Process("worker", "worker",                       "simple_vlm_example_worker"),
    ]


def run() -> None:
    ensure_credentials("HF_TOKEN")
    asyncio.run(run_stack(_build_processes(), _BASE))


if __name__ == "__main__":
    run()
