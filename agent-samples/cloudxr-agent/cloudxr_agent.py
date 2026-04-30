"""
cloudxr-agent orchestrator.

Launches the hub, CloudXR runtime, and agent worker as isolated subprocesses.
Each process owns its own venv and YAML config (<command>.yaml in this dir).

How to run (from agent-samples/cloudxr-agent/):
    uv sync && uv run cloudxr_agent

Accept the CloudXR EULA non-interactively:
    Set accept_eula: true in cloudxr_runtime.yaml  (written once to ~/.cloudxr/run/eula_accepted)
"""
import asyncio
from pathlib import Path

from xr_ai_launcher import Process, run_stack

_BASE = Path(__file__).resolve().parent

PROCESSES = [
    Process("hub",     "../../server-runtime",  "xr_media_hub"),
    Process("cloudxr", "../../cloudxr-runtime", "cloudxr_runtime"),
    Process("worker",  "worker",                "cloudxr_agent_worker"),
]


def run() -> None:
    asyncio.run(run_stack(PROCESSES, _BASE))


if __name__ == "__main__":
    run()
