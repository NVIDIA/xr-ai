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
If any ``magic_phrases`` are configured, STT transcripts must begin
with one of them (case-insensitive, strict prefix) or the utterance is
dropped; the matched phrase is stripped before the query is dispatched.
Multiple phrases enable several wordings ("agent", "hey agent", …)
without falling back to fuzzy matching. After a match, the next
utterance from the same participant within ``followup_grace_s`` seconds
bypasses the gate so the conversation flows naturally. The text data
channel is not gated; a *spoken* "ping" is gated, but the data-channel
"ping" shortcut is unaffected.

Agent → client:
    Topic "vlm.response"        — assembled UTF-8 text reply
    `xr-hub-return-{pid}` track — sentence-by-sentence Piper TTS audio

Config (simple_vlm_example_worker.yaml — auto-passed by the launcher)
----------------------------------------------------------------------
    models_yaml:           yaml/models.yaml   # path to models config (relative to yaml dir)
    default_prompt:        "Describe what you see."
    system_prompt:              <multiline string>   # role/style guidance for the VLM
    magic_phrases:              []    # list of speech-only opt-in prefixes; empty = always-on
    listening_chime:           false  # play a short bell when a magic phrase matches
    followup_grace_s:          5.0    # after a match, next utterance within Xs bypasses gate
    frame_max_age_s:           2.0   # frames older than this trigger a camera-on request
    camera_on_timeout_s:      15.0   # how long to wait for a fresh frame after startCamera
    camera_grace_s:            5.0   # keep camera on this long after a query (avoids restart on follow-ups)
    silero_threshold:           0.5   # Silero speech probability gate (0..1)
    silence_duration:           0.8   # seconds of silence that ends an utterance
    min_speech:                 0.1   # minimum seconds of speech before STT fires
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


async def main(
    cfg: dict,
    config_path: pathlib.Path | None = None,
    ready_file: pathlib.Path | None = None,
) -> None:
    setup_logging("worker")

    # Resolve `models_yaml` relative to the worker YAML's parent directory.
    # Matches the convention used by xr-render-demo (and any future sample)
    # so all samples behave the same regardless of CWD. The bare default
    # `"models.yaml"` sits next to the worker yaml in `yaml/`.
    models_yaml_raw = cfg.get("models_yaml", "models.yaml")
    models_yaml_path = pathlib.Path(models_yaml_raw)
    if config_path and not models_yaml_path.is_absolute():
        models_yaml_path = config_path.parent / models_yaml_path
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
        # Accept `magic_phrases:` as a YAML list of strict-prefix
        # phrases. `or []` handles the empty-YAML-value (None) case.
        magic_phrases         =cfg.get("magic_phrases") or [],
        listening_chime       =bool(cfg.get("listening_chime", False)),
        followup_grace_s      =float(cfg.get("followup_grace_s",      5.0)),
        frame_max_age_s       =float(cfg.get("frame_max_age_s",       2.0)),
        camera_on_timeout_s   =float(cfg.get("camera_on_timeout_s",  10.0)),
        camera_grace_s        =float(cfg.get("camera_grace_s",         5.0)),
        silence_duration      =float(cfg.get("silence_duration",      0.8)),
        min_speech            =float(cfg.get("min_speech",            0.1)),
        silero_threshold      =float(cfg.get("silero_threshold",      0.5)),
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

    asyncio.run(main(cfg, config_path=ns.config, ready_file=ns.ready_file))


if __name__ == "__main__":
    run()
