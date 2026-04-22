"""
Echo agent orchestrator.

Declares the process sequence and delegates to run_stack.
Each process owns its own venv and YAML config (<command>.yaml in this dir).

How to run (from agent-samples/echo-agent/):
    uv sync && uv run echo_agent
"""
import asyncio
from pathlib import Path

from xr_ai_launcher import Process, run_stack

_BASE = Path(__file__).resolve().parents[1]  # agent-samples/echo-agent/

PROCESSES = [
    Process("hub",    "../../server-runtime", "xr_media_hub"),
    Process("worker", "worker",               "echo_agent_worker"),
]


def run() -> None:
    asyncio.run(run_stack(PROCESSES, _BASE))


if __name__ == "__main__":
    run()
