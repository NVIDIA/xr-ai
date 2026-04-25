"""
VLM agent orchestrator.

Declares the process sequence and delegates to run_stack.
Each process owns its own venv and YAML config (<command>.yaml in this dir).

How to run (from agent-samples/vlm-agent/):
    uv sync && uv run vlm_agent
"""
import asyncio
from pathlib import Path

from xr_ai_launcher import Process, ensure_credentials, run_stack

_BASE = Path(__file__).resolve().parents[1]  # agent-samples/vlm-agent/

PROCESSES = [
    Process("hub",    "../../server-runtime",          "xr_media_hub"),
    Process("vlm",    "../../ai-services/vlm-server",  "vlm_server"),
    Process("worker", "worker",                        "vlm_agent_worker"),
]


def run() -> None:
    ensure_credentials("HF_TOKEN")
    asyncio.run(run_stack(PROCESSES, _BASE))


if __name__ == "__main__":
    run()
