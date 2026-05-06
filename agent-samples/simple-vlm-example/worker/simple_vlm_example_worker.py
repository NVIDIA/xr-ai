# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
simple-vlm-example worker — entry point.

Launched as a subprocess by ``uv run simple_vlm_example`` (the orchestrator).
Do not run this directly.

Protocol
--------
Client → agent  (LiveKit data channel, any topic):
    "ping"      — case-insensitive trigger for the configured default prompt
    Any other UTF-8 text — used verbatim as the query

Audio in (mic) → STT → text → query (same path as a data message).

Agent → client:
    Topic "vlm.response"        — assembled UTF-8 text reply
    `xr-hub-return-{pid}` track — sentence-by-sentence Piper TTS audio

Config (simple_vlm_example_worker.yaml — auto-passed by the launcher)
----------------------------------------------------------------------
    stt_server:        http://localhost:8103
    vlm_server:        http://localhost:8100
    tts_server:        http://localhost:8105   # piper_tts_server
    default_prompt:    "Describe what you see."
    system_prompt:          <multiline string>   # role/style guidance for the VLM
    frame_max_age_s:       2.0   # frames older than this trigger a camera-on request
    camera_on_timeout_s:  15.0   # how long to wait for a fresh frame after startCamera
    camera_grace_s:        5.0   # keep camera on this long after a query (avoids restart on follow-ups)
    silence_threshold:      0.01  # float32 RMS below which audio is silence
    silence_duration:       0.8   # seconds of silence that ends an utterance
    min_speech:             0.3   # minimum seconds of speech before STT fires
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

from agent import DEFAULT_SYSTEM_PROMPT, SimpleVlmAgent
from services import SttClient, TtsClient, VlmClient, wait_for_health

_HUB_PUB  = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH = "ipc:///tmp/xr_hub_in"


async def main(cfg: dict, ready_file: pathlib.Path | None = None) -> None:
    setup_logging("worker")

    # VLM backend selection.
    # "cosmos" → vlm-server on port 8100 (Cosmos-Reason1-7B, model="vlm")
    # "omni"   → nemotron_omni on port 8108 (Nemotron-Omni-30B, model="llm")
    backend = cfg.get("vlm_backend", "cosmos")
    if backend == "omni":
        vlm_default_url   = "http://localhost:8108"
        vlm_default_model = "llm"
    else:
        vlm_default_url   = "http://localhost:8100"
        vlm_default_model = "vlm"

    stt = SttClient(cfg.get("stt_server", "http://localhost:8103"))
    vlm = VlmClient(cfg.get("vlm_server", vlm_default_url),
                    model_name=cfg.get("vlm_model_name", vlm_default_model))
    tts = TtsClient(cfg.get("tts_server", "http://localhost:8105"))
    await wait_for_health({
        "STT": stt.health_url,
        "VLM": vlm.health_url,
        "TTS": tts.health_url,
    })

    if ready_file:
        ready_file.touch()

    ep    = ProcessorEndpoint(sub_addr=_HUB_PUB, push_addr=_HUB_PUSH)
    agent = SimpleVlmAgent(
        ep, stt, vlm, tts,
        default_prompt        =cfg.get("default_prompt",        "Describe what you see."),
        system_prompt         =cfg.get("system_prompt",         DEFAULT_SYSTEM_PROMPT),
        frame_max_age_s       =float(cfg.get("frame_max_age_s",       2.0)),
        camera_on_timeout_s   =float(cfg.get("camera_on_timeout_s",  10.0)),
        camera_grace_s        =float(cfg.get("camera_grace_s",         5.0)),
        silence_threshold     =float(cfg.get("silence_threshold",     0.01)),
        silence_duration      =float(cfg.get("silence_duration",      0.8)),
        min_speech            =float(cfg.get("min_speech",            0.3)),
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, agent.shutdown)

    logger.info("simple-vlm-example connecting  sub={}  push={}", _HUB_PUB, _HUB_PUSH)
    try:
        await agent.run()
    finally:
        agent.shutdown()
    logger.info("simple-vlm-example stopped")


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
