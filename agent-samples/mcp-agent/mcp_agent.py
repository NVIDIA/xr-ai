# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MCP agent orchestrator — continuous STT + MCP-accessible transcript and video.

What this starts
----------------
  hub     — XR-Media-Hub (video recording enabled via xr_media_hub.yaml)
  stt     — STT server (parakeet-tdt-0.6b-v3)
  mcp     — Composed pure-FastMCP server: mounts transcript + video sub-servers
            into one FastMCP instance at /mcp (no REST endpoints)
  worker  — mcp_agent_worker: VAD → STT → MCP transcript_add_transcript

MCP endpoint: http://localhost:8200/mcp
  Tools: transcript_*, video_*

How to run (from agent-samples/mcp-agent/):
    uv sync && uv run mcp_agent
"""
import asyncio
from pathlib import Path

from xr_ai_launcher import Process, ensure_credentials, run_stack

_BASE = Path(__file__).resolve().parent

PROCESSES = [
    Process("hub",    "../../server-runtime",         "xr_media_hub"),
    Process("stt",    "../../ai-services/stt-server", "stt_server"),
    Process("mcp",    "mcp_server",                   "mcp_server"),
    Process("worker", "worker",                       "mcp_agent_worker"),
]


def run() -> None:
    ensure_credentials("HF_TOKEN")
    asyncio.run(run_stack(PROCESSES, _BASE))


if __name__ == "__main__":
    run()
