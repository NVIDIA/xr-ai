"""
MCP agent orchestrator — continuous STT + MCP-accessible transcript and video.

What this starts
----------------
  hub     — XR-Media-Hub (video recording enabled via xr_media_hub.yaml)
  stt     — STT server (parakeet-tdt-0.6b-v3)
  mcp     — Composed MCP server: mounts transcript + video skills into one
            FastMCP instance (skills selected via mcp_server.yaml)
  worker  — mcp_agent_worker: VAD → STT → POST /ingest → mcp server

MCP endpoint: http://localhost:8200/mcp
  Tools: transcript_*, video_* (whichever skills are enabled in mcp_server.yaml)

How to run (from agent-samples/mcp-agent/):
    uv sync && uv run mcp_agent
"""
import asyncio
from pathlib import Path

from xr_ai_launcher import Process, ensure_credentials, run_stack

_BASE = Path(__file__).resolve().parents[1]  # agent-samples/mcp-agent/

PROCESSES = [
    Process("hub",    "../../server-runtime",        "xr_media_hub"),
    Process("stt",    "../../ai-services/stt-server", "stt_server"),
    Process("mcp",    "mcp_server",                   "mcp_server"),
    Process("worker", "worker",                        "mcp_agent_worker"),
]


def run() -> None:
    ensure_credentials("HF_TOKEN")
    asyncio.run(run_stack(PROCESSES, _BASE))


if __name__ == "__main__":
    run()
