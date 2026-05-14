# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
simple-rag-example worker — entry point.

Launched as a subprocess by ``uv run simple_rag_example`` (the orchestrator).
Do not run this directly.

Protocol
--------
Client → agent  (LiveKit data channel, any topic):
    "ping"      — case-insensitive trigger for the configured default prompt
    Any other UTF-8 text — used verbatim as the query

Audio in (mic) → VAD → STT → text → query (same path as a data message).

Agent → client:
    Topic "vlm.response"        — assembled UTF-8 text reply
    `xr-hub-return-{pid}` track — sentence-by-sentence Piper TTS audio

RAG context is optional — the agent works fine without doc-relevant queries.
rag-mcp retrieve is always called for every query using dense vector search
(embedding-server at port 8109); when it returns no chunks the agent answers
from the camera feed and general knowledge, exactly as simple-vlm-example would.

Config (simple_rag_example_worker.yaml — auto-passed by the launcher)
----------------------------------------------------------------------
    stt_server:          http://localhost:8103
    vlm_server:          http://localhost:8100
    tts_server:          http://localhost:8105   # piper_tts_server
    rag_mcp_server:      http://localhost:8240/mcp
    top_k:               5       # doc chunks returned per retrieval query
    default_prompt:      "Describe what you see."
    system_prompt:            <multiline string>
    frame_max_age_s:         5.0
    camera_on_timeout_s:    30.0
    camera_grace_s:          5.0
    silence_threshold:       0.01
    silence_duration:        0.8
    min_speech:              0.3
"""
from __future__ import annotations

import argparse
import asyncio
import pathlib
import signal

import yaml
from loguru import logger
from xr_ai_agent import ProcessorEndpoint
from xr_ai_logging import setup_logging

from agent import DEFAULT_SYSTEM_PROMPT, SimpleRagAgent
from services import RagClient, SttClient, TtsClient, VlmClient, wait_for_health

_HUB_PUB  = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH = "ipc:///tmp/xr_hub_in"


async def main(cfg: dict, ready_file: pathlib.Path | None = None) -> None:
    setup_logging("worker")

    stt = SttClient(cfg.get("stt_server", "http://localhost:8103"))
    vlm = VlmClient(cfg.get("vlm_server", "http://localhost:8100"),
                    model_name=cfg.get("vlm_model_name", "vlm"))
    tts = TtsClient(cfg.get("tts_server", "http://localhost:8105"))
    rag = RagClient(cfg.get("rag_mcp_server", "http://localhost:8240/mcp"))

    await wait_for_health({
        "STT":     stt.health_url,
        "VLM":     vlm.health_url,
        "TTS":     tts.health_url,
        "RAG-MCP": rag.health_url,
    })

    if ready_file:
        ready_file.touch()

    ep    = ProcessorEndpoint(sub_addr=_HUB_PUB, push_addr=_HUB_PUSH)
    agent = SimpleRagAgent(
        ep, stt, vlm, tts, rag,
        top_k                  = int(cfg.get("top_k",                       4)),
        default_prompt         = cfg.get("default_prompt",     "Describe what you see."),
        system_prompt          = cfg.get("system_prompt",      DEFAULT_SYSTEM_PROMPT),
        frame_max_age_s        = float(cfg.get("frame_max_age_s",         5.0)),
        camera_on_timeout_s    = float(cfg.get("camera_on_timeout_s",     5.0)),
        camera_grace_s         = float(cfg.get("camera_grace_s",          5.0)),
        doc_skip_camera_score  = float(cfg.get("doc_skip_camera_score",  0.35)),
        silence_threshold      = float(cfg.get("silence_threshold",      0.01)),
        silence_duration       = float(cfg.get("silence_duration",        0.8)),
        min_speech             = float(cfg.get("min_speech",              0.3)),
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, agent.shutdown)

    logger.info("simple-rag-example connecting  sub={}  push={}", _HUB_PUB, _HUB_PUSH)
    try:
        await agent.run()
    finally:
        agent.shutdown()
    logger.info("simple-rag-example stopped")


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config",     type=pathlib.Path, default=None)
    p.add_argument("--ready-file", type=pathlib.Path, default=None)
    ns, _ = p.parse_known_args()

    cfg: dict = {}
    if ns.config and ns.config.exists():
        with open(ns.config) as f:
            cfg = yaml.safe_load(f) or {}

    asyncio.run(main(cfg, ready_file=ns.ready_file))


if __name__ == "__main__":
    run()
