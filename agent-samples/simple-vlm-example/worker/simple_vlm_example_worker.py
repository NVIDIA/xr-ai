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
    models_yaml:           yaml/models.yaml   # path to models config (relative to yaml dir)
    default_prompt:        "Describe what you see."
    system_prompt:              <multiline string>   # role/style guidance for the VLM
    frame_max_age_s:           2.0   # frames older than this trigger a camera-on request
    camera_on_timeout_s:      15.0   # how long to wait for a fresh frame after startCamera
    camera_grace_s:            5.0   # keep camera on this long after a query (avoids restart on follow-ups)
    silence_threshold:          0.01  # float32 RMS below which audio is silence
    silence_duration:           0.8   # seconds of silence that ends an utterance
    min_speech:                 0.3   # minimum seconds of speech before STT fires
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
from xr_ai_models import load_models_config, make_stt, make_tts, make_vlm

from agent import DEFAULT_SYSTEM_PROMPT, SimpleVlmAgent

_HUB_PUB  = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH = "ipc:///tmp/xr_hub_in"


async def main(cfg: dict, ready_file: pathlib.Path | None = None) -> None:
    setup_logging("worker")

    # models_yaml is resolved relative to cwd (the sample root, where the launcher runs).
    models_yaml_path = cfg.get("models_yaml", "yaml/models.yaml")
    models_cfg = load_models_config(models_yaml_path)

    stt = make_stt(models_cfg, "stt")
    vlm = make_vlm(models_cfg, "vlm")
    tts = make_tts(models_cfg, "tts")

    await _wait_for_health(stt=stt, vlm=vlm, tts=tts)

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
        for svc in (stt, vlm, tts):
            await svc.close()  # type: ignore[attr-defined]
    logger.info("simple-vlm-example stopped")


async def _wait_for_health(**services: object) -> None:
    """Block until every service's health endpoint reports healthy."""
    pending: dict[str, object] = dict(services)
    while pending:
        results = await asyncio.gather(
            *(svc.health() for svc in pending.values()),
            return_exceptions=True,
        )
        still_waiting = {
            name: svc
            for (name, svc), ok in zip(pending.items(), results)
            if not (isinstance(ok, bool) and ok)
        }
        for name in pending:
            if name not in still_waiting:
                logger.info("{} ready", name)
        pending = still_waiting
        if pending:
            logger.info("still waiting for: {}", ", ".join(sorted(pending)))
            await asyncio.sleep(5.0)


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
