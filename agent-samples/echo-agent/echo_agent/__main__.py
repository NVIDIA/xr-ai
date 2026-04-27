"""
Echo agent orchestrator — STT → TTS echo pipeline.

Audio path:   mic audio → STT → TTS → speaker audio
Text path:    typed text → TTS → speaker audio

How to run (from agent-samples/echo-agent/):
    uv sync && uv run echo_agent
"""
import asyncio
from pathlib import Path

from xr_ai_launcher import Process, run_stack

_BASE = Path(__file__).resolve().parents[1]  # agent-samples/echo-agent/

PROCESSES = [
    Process("hub",    "../../server-runtime",         "xr_media_hub"),
    Process("stt",    "../../ai-services/stt-server", "stt_server"),
    Process("tts",    "../../ai-services/tts/magpie",  "magpie_tts_server"),
    Process("worker", "worker",                       "echo_agent_worker"),
]


def run() -> None:
    asyncio.run(run_stack(PROCESSES, _BASE))


if __name__ == "__main__":
    run()
