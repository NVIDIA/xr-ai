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

from xr_ai_launcher import Process, ensure_credentials, run_stack

_BASE = Path(__file__).resolve().parent

PROCESSES = [
    Process("hub",    "../../server-runtime",         "xr_media_hub"),
    Process("vlm",    "../../ai-services/vlm-server", "vlm_server"),
    Process("stt",    "../../ai-services/stt-server", "stt_server"),
    Process("tts",    "../../ai-services/tts/piper",  "piper_tts_server"),
    Process("worker", "worker",                       "simple_vlm_example_worker"),
]


def run() -> None:
    ensure_credentials("HF_TOKEN")
    asyncio.run(run_stack(PROCESSES, _BASE))


if __name__ == "__main__":
    run()
