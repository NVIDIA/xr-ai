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

Audio in (mic) → STT → text → ``xr-ai-voicegate`` → query.
The voice gate owns the magic-phrase, STOP, and follow-up ladder; the
worker only wires handlers and feeds transcripts (both via the shared
``xr_ai_conversation.ConversationLoop``). Configure phrases via
``yaml/voice_gate.yaml`` (referenced by ``voice_gate_yaml`` in the
worker YAML; an empty ``magic_phrases`` list disables the gate). See
``utils/xr-ai-voicegate`` for the full event ladder. The text data
channel is never gated.

Agent → client:
    Topic "vlm.response"        — assembled UTF-8 text reply
    `xr-hub-return-{pid}` track — sentence-by-sentence Piper TTS audio

Config (simple_vlm_example_worker.yaml — auto-passed by the launcher)
----------------------------------------------------------------------
    models_yaml:           yaml/models.yaml   # path to models config (relative to yaml dir)
    default_prompt:        "Describe what you see."
    system_prompt:              <multiline string>   # role/style guidance for the VLM
    voice_gate_yaml:       voice_gate.yaml   # path to standalone voice_gate config (relative to yaml dir)
    frame_max_age_s:           2.0      # frames older than this trigger a camera-on request
    camera_on_timeout_s:      15.0      # how long to wait for a fresh frame after startCamera
    camera_grace_s:            5.0      # keep camera on this long after a query (avoids restart on follow-ups)
    silero_threshold:           0.5     # Silero speech probability gate (0..1)
    silence_duration:           0.8     # seconds of silence that ends an utterance
    min_speech:                 0.1     # minimum seconds of speech before STT fires
"""
from __future__ import annotations

import argparse
import asyncio
import pathlib
import signal

import yaml
from loguru import logger
from xr_ai_agent import ProcessorEndpoint
from xr_ai_conversation import ConversationLoop, VadConfig
from xr_ai_logging import setup_logging
from xr_ai_models import load_models_config, make_stt, make_tts, make_vlm
from xr_ai_voicegate import VoiceGate, load_voice_gate_config

from agent import DEFAULT_SYSTEM_PROMPT, SimpleVlmBrain

_HUB_PUB  = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH = "ipc:///tmp/xr_hub_in"


def _greeting(gate: VoiceGate) -> str:
    """Sample-specific greeting that preserves "about what you see"."""
    help_text = gate.format_phrase_help()
    if help_text is None:
        return "Hi, I'm listening. Ask me anything about what you see."
    return f"Hi, I'm listening. {help_text}"


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

    # Resolve `voice_gate_yaml` the same way as `models_yaml`. A missing
    # file degrades to gate defaults (always-on); see load_voice_gate_config.
    voice_gate_yaml_raw  = cfg.get("voice_gate_yaml", "voice_gate.yaml")
    voice_gate_yaml_path = pathlib.Path(voice_gate_yaml_raw)
    if config_path and not voice_gate_yaml_path.is_absolute():
        voice_gate_yaml_path = config_path.parent / voice_gate_yaml_path
    voice_gate_cfg = load_voice_gate_config(voice_gate_yaml_path)

    stt = make_stt(models_cfg, "stt")
    vlm = make_vlm(models_cfg, "vlm")
    tts = make_tts(models_cfg, "tts")

    await _wait_for_health(stt=stt, vlm=vlm, tts=tts)

    if ready_file:
        ready_file.touch()

    ep = ProcessorEndpoint(sub_addr=_HUB_PUB, push_addr=_HUB_PUSH)

    brain = SimpleVlmBrain(
        ep, vlm,
        default_prompt      = cfg.get("default_prompt", "Describe what you see."),
        system_prompt       = cfg.get("system_prompt",  DEFAULT_SYSTEM_PROMPT),
        frame_max_age_s     = float(cfg.get("frame_max_age_s",     2.0)),
        camera_on_timeout_s = float(cfg.get("camera_on_timeout_s", 10.0)),
        camera_grace_s      = float(cfg.get("camera_grace_s",       5.0)),
    )

    # Brain-owned endpoint hooks (data + frame). The loop owns audio and
    # participant events.
    ep.on_data(brain.on_data)
    ep.on_frame(brain.on_frame)

    conv = ConversationLoop(
        ep              = ep,
        stt             = stt,
        tts             = tts,
        voice_gate_cfg  = voice_gate_cfg,
        vad_cfg         = VadConfig(
            silence_duration = float(cfg.get("silence_duration", 0.8)),
            min_speech       = float(cfg.get("min_speech",       0.1)),
            silero_threshold = float(cfg.get("silero_threshold", 0.5)),
        ),
        on_query              = brain.handle_query,
        on_speech_start       = brain.on_speech_start,
        on_participant_left   = brain.on_participant_left,
        on_stop_extra         = brain.on_stop_extra,
        on_phrase_only_extra  = brain.on_phrase_only,
        on_drop_extra         = brain.on_drop,
        text_topic            = "vlm.response",
        greeting              = _greeting,
    )

    # Wire the dispatch reference now that both sides exist (chicken-and-egg
    # between brain.on_data and loop.dispatch).
    brain.dispatch = conv.dispatch

    asyncio_loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio_loop.add_signal_handler(sig, conv.shutdown)

    logger.info("simple-vlm-example connecting  sub={}  push={}", _HUB_PUB, _HUB_PUSH)
    try:
        await conv.run()
    finally:
        conv.shutdown()
        brain.shutdown()
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
