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

Model backend
-------------
``model_backend`` in yaml/simple_vlm_example_worker.yaml selects where the
VLM runs: ``local`` (default) uses the local vlm-server; ``nim`` uses hosted
NVIDIA NIM (the worker loads models.nim.yaml and the local vlm-server is not
started). STT and TTS always run locally. See the README section
"Hosting models on NVIDIA NIM".

How to run (from agent-samples/simple-vlm-example/):
    uv sync && uv run simple_vlm_example
"""
import re
from pathlib import Path

from xr_ai_launcher import Process, ensure_credentials, run_stack, warn_if_missing
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent

_WORKER_CONFIG = "yaml/simple_vlm_example_worker.yaml"

# Read the model_backend scalar from the worker YAML without pyyaml — the
# orchestrator is stdlib-only. Mirrors the regex-read precedent used for
# lovr_bin in xr-render-demo.
_BACKEND_RE = re.compile(r"^\s*model_backend\s*:\s*[\"']?(\w+)[\"']?", re.MULTILINE)


def _model_backend() -> str:
    try:
        m = _BACKEND_RE.search((_BASE / _WORKER_CONFIG).read_text())
    except OSError:
        return "local"
    return m.group(1).lower() if m else "local"


def _build_processes(backend: str) -> list[Process]:
    procs = [
        Process("hub", "../../server-runtime", "xr_media_hub",
                config="yaml/xr_media_hub.yaml"),
    ]
    # The local vlm-server is only needed when the VLM runs locally; with
    # model_backend: nim the worker calls hosted NIM instead, so starting it
    # would waste GPU (and a load failure would abort the fail-fast stack).
    if backend != "nim":
        procs.append(
            Process("vlm", "../../ai-services/vlm-server", "vlm_server",
                    config="yaml/vlm_server.yaml"),
        )
    procs += [
        Process("stt", "../../ai-services/stt-server", "stt_server",
                config="yaml/stt_server.yaml"),
        Process("tts", "../../ai-services/tts/piper", "piper_tts_server",
                config="yaml/piper_tts_server.yaml"),
        Process("worker", "worker", "simple_vlm_example_worker",
                config=_WORKER_CONFIG),
    ]
    return procs


def run() -> None:
    setup_logging("orchestrator", namespace="simple-vlm-example")
    backend = _model_backend()
    # HF_TOKEN is optional for the default (public) model — it only raises HF
    # rate limits / download speed and is required only for gated models.
    # Warn instead of prompting; see docs/credentials.md.
    warn_if_missing("HF_TOKEN")
    if backend == "nim":
        ensure_credentials("NGC_API_KEY")
    run_stack(_build_processes(backend), _BASE)


if __name__ == "__main__":
    run()
