"""
MCP agent worker — entry point.

Launched as a subprocess by ``uv run mcp_agent`` (the orchestrator).
Do not run this directly.

What it does
------------
1. Listens for audio from XR clients via the hub IPC.
2. Runs VAD to detect speech boundaries (same logic as echo-agent).
3. At end of each utterance, runs STT on the full audio buffer.
4. Calls the ``transcript_add_transcript`` MCP tool with
   ``source_id=<participant_id>`` to record the utterance.  (The
   transcript store is keyed by an arbitrary ``source_id`` string —
   live participant identities here, but agents can also write under
   internal source names like ``"agent-vlm"``.)
5. On any data-channel message, calls ``transcript_get_transcript_stats``
   and ``video_get_video_stats`` and sends a summary back on topic
   ``mcp.stats``.

The composed mcp-server is a pure-FastMCP process at /mcp (no REST). It
mounts the transcript and video sub-servers under their respective
namespaces. External LLM agents can connect to the same /mcp to call
tools directly.

Config (mcp_agent_worker.yaml — auto-passed by the launcher)
-------------------------------------------------------------
    stt_server:        http://localhost:8103
    mcp_server:        http://localhost:8200
    silence_threshold: 0.01
    silence_duration:  0.8
    min_speech:        0.3
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import pathlib
import signal

import yaml

from xr_ai_agent import ProcessorEndpoint

from agent import McpAgent
from services import SttClient, wait_for_services

log = logging.getLogger("mcp_agent")

_HUB_PUB  = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH = "ipc:///tmp/xr_hub_in"


async def main(cfg: dict) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    stt     = SttClient(cfg.get("stt_server", "http://localhost:8103"))
    mcp_url = cfg.get("mcp_server", "http://localhost:8200").rstrip("/") + "/mcp"
    await wait_for_services(stt.health_url, mcp_url)

    ep    = ProcessorEndpoint(sub_addr=_HUB_PUB, push_addr=_HUB_PUSH)
    agent = McpAgent(
        ep, stt, mcp_url,
        silence_threshold=float(cfg.get("silence_threshold", 0.01)),
        silence_duration =float(cfg.get("silence_duration",  0.8)),
        min_speech       =float(cfg.get("min_speech",        0.3)),
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, agent.shutdown)

    log.info("mcp-agent connecting  sub=%s  push=%s", _HUB_PUB, _HUB_PUSH)
    try:
        await agent.run()
    finally:
        agent.shutdown()
    log.info("mcp-agent stopped")


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=pathlib.Path, default=None)
    ns, _ = p.parse_known_args()

    cfg: dict = {}
    if ns.config and ns.config.exists():
        with open(ns.config) as f:
            cfg = yaml.safe_load(f) or {}

    asyncio.run(main(cfg))


if __name__ == "__main__":
    run()
