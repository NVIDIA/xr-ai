# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
nat-agent orchestrator. Runs the process stack for this sample.

Voice + vision agent driven by NeMo Agent Toolkit's tool_calling_agent.
The local nemotron3_nano LLM (port 8107) chooses video-mcp + vlm-mcp tool
calls per turn; no NAT-side wrappers, every MCP tool is exposed verbatim.

How to run (from agent-samples/nat-agent/):
    uv sync && (cd worker && uv sync)
    uv run nat_agent
"""
import asyncio
from pathlib import Path

from xr_ai_launcher import Process, ensure_credentials, run_stack

_BASE = Path(__file__).resolve().parent

PROCESSES = [
    Process("hub",       "../../server-runtime",                  "xr_media_hub"),
    Process("stt",       "../../ai-services/stt-server",          "stt_server",                        gpu="1"),
    Process("tts",       "../../ai-services/tts/piper",           "piper_tts_server"),
    Process("vlm",       "../../ai-services/vlm-server",          "vlm_server",                        gpu="1"),
    Process("vlm-mcp",   "../../agent-mcp-servers/vlm-mcp",       "vlm_mcp_server"),
    Process("video-mcp", "../../agent-mcp-servers/video-mcp",     "video_mcp_server"),
    Process("llm",       "../../ai-services/llm/nemotron3_nano",  "nemotron3_nano_llm_server",          gpu="0"),
    Process("worker",    "worker",                                "nat_agent_worker"),
]


def run() -> None:
    ensure_credentials("HF_TOKEN")
    asyncio.run(run_stack(PROCESSES, _BASE))


if __name__ == "__main__":
    run()
