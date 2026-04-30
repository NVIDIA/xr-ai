# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
nat-agent worker — voice + vision conversational agent driven by NeMo Agent
Toolkit's tool_calling_agent.

Launched as a subprocess by ``uv run nat_agent`` (the orchestrator).
Do not run this directly.

Pipeline
--------
Audio path: client mic -> hub -> StT -> NAT (LLM + MCP tools) -> TTS -> hub -> client speaker
Data path:  client data channel -> NAT (LLM + MCP tools) -> {data reply, TTS}

The LLM is the planner. It chooses which video-mcp tool to call (latest
frame / past frame / stats / roster / video clip), composes its own
question for vlm-mcp's ``ask_image`` tool, and decides when no tool call
is needed at all (math, conversation, recall).

Echo cancellation (TTS audio looping into the mic) is handled at the client
audio input device, not at the worker. The worker has no STTMuteFrame
plumbing or playback-tail timing.

Protocol
--------
Client -> agent  (data channel, any topic except agent-outbound):
    Raw UTF-8 text OR JSON {"query": "...", ...}

    The string ``"ping"`` (case-insensitive) and any payload starting with
    ``ping:`` is rewritten to "Describe what you see." so the LLM picks the
    visual path.

Agent -> client:
    topic ``stt.transcript``     — final STT transcript (audio path only)
    topic ``agent.response``     — agent's text answer (whether or not the
                                   LLM chose to call vlm-mcp / video-mcp)
    track ``xr-hub-return-{pid}`` — sentence-batched Piper TTS

Config (nat_agent_worker.yaml — auto-passed by the launcher)
-------------------------------------------------------------
    stt_server, tts_server, llm_server (HTTP URLs)
    vlm_mcp_url, video_mcp_url (MCP StreamableHTTP endpoints)
    llm_system_prompt, llm_max_tokens, llm_temperature, llm_top_p, llm_request_timeout_s
    silence_threshold, silence_duration, min_speech, stream_interval (VAD)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import pathlib
import signal

from agent import NatAgent
from config import WorkerConfig, load_config
from nat_backend import NatBackend
from services import (
    SttClient, TtsClient,
    http_probe, mcp_probe, wait_for_services,
)

log = logging.getLogger("nat_agent")

_HUB_PUB  = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH = "ipc:///tmp/xr_hub_in"


async def main(cfg: WorkerConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log.info("nat-agent worker")
    log.info("  stt=%s  tts=%s  llm=%s", cfg.stt_server, cfg.tts_server, cfg.llm_server)
    log.info("  vlm-mcp=%s  video-mcp=%s", cfg.vlm_mcp_url, cfg.video_mcp_url)

    log.info("Waiting for services to become healthy…")
    await wait_for_services({
        "STT":       http_probe(cfg.stt_server.rstrip("/") + "/health"),
        "TTS":       http_probe(cfg.tts_server.rstrip("/") + "/health"),
        "LLM":       http_probe(cfg.llm_server.rstrip("/") + "/health"),
        "vlm-mcp":   mcp_probe(cfg.vlm_mcp_url),
        "video-mcp": mcp_probe(cfg.video_mcp_url),
    })

    stt = SttClient(cfg.stt_server)
    tts = TtsClient(cfg.tts_server)
    nat = NatBackend(cfg)

    agent = NatAgent(cfg, stt, tts, nat)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, agent.shutdown)

    log.info("nat-agent connecting  sub=%s  push=%s", _HUB_PUB, _HUB_PUSH)
    try:
        await agent.run()
    finally:
        agent.shutdown()
    log.info("nat-agent stopped")


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=pathlib.Path, default=None)
    ns, _ = p.parse_known_args()

    cfg = load_config(ns.config)
    asyncio.run(main(cfg))


if __name__ == "__main__":
    run()
