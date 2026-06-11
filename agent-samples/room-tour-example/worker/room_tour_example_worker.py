# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
room-tour-example worker — entry point.

Launched as a subprocess by ``uv run room_tour_example`` (the orchestrator).
Do not run this directly.

Pipeline
--------
The voice path runs through the shared pipecat pipeline assembled by
``xr_ai_pipecat.make_voice_pipeline`` — the same STT / VAD / TTS the other
agent-samples use:

    XRMediaHubInputTransport → VadSttProcessor → VoiceGateProcessor →
    RoomTourBrain → StreamingTtsProcessor → XRMediaHubOutputTransport

The voice gate is configured with NO magic phrases (``voice_gate.yaml`` →
empty ``magic_phrases``), i.e. always-on: every finalized utterance reaches the
brain, which routes tour commands and spatial questions itself. STT (NeMo
Parakeet), VAD (Silero via xr-ai-vad), and TTS (Piper) come from xr-ai-models
/ xr-ai-pipecat; the VLM (Cosmos) comes from xr-ai-models.

Config (room_tour_example_worker.yaml — auto-passed by the launcher)
--------------------------------------------------------------------
    models_yaml:               yaml/models.yaml      # stt / vlm / tts endpoints
    voice_gate_yaml:           yaml/voice_gate.yaml  # always-on (empty phrases)
    system_prompt:             <multiline string>    # spoken-answer style
    frame_max_age_s:           5.0    # a frame older than this is re-fetched
    tour_capture_interval_s:   3.0    # how often to ingest a frame while touring
    match_threshold:           0.62   # textslam place data-association threshold
    silero_threshold:          0.5    # Silero speech-probability gate (0..1)
    silence_duration:          0.4    # seconds of silence ending an utterance
    min_speech:                0.1    # min seconds of speech before STT fires
    idle_timeout_secs:         0      # 0 = stay connected (no idle auto-cancel)
"""
from __future__ import annotations

import argparse
import asyncio
import pathlib
import signal

import yaml
from loguru import logger
from pipecat.pipeline.runner import PipelineRunner
from xr_ai_logging import setup_logging
from xr_ai_models import load_models_config, make_stt, make_tts, make_vlm
from xr_ai_pipecat import VadConfig, make_voice_pipeline
from xr_ai_pipecat.transport import XRMediaHubTransport
from xr_ai_voicegate import load_voice_gate_config

from agent import DEFAULT_SYSTEM_PROMPT, RoomTourBrain


def _resolve(cfg_path: pathlib.Path | None, raw: str) -> pathlib.Path:
    """Resolve a YAML-referenced path relative to the worker YAML's directory."""
    p = pathlib.Path(raw)
    if cfg_path and not p.is_absolute():
        p = cfg_path.parent / p
    return p


async def _wait_for_health(**services: object) -> None:
    """Block until every service's health endpoint reports healthy."""
    pending: dict[str, object] = dict(services)
    while pending:
        results = await asyncio.gather(
            *(svc.health() for svc in pending.values()),  # type: ignore[attr-defined]
            return_exceptions=True,
        )
        still = {
            name: svc
            for (name, svc), ok in zip(pending.items(), results)
            if not (isinstance(ok, bool) and ok)
        }
        for name in pending:
            if name not in still:
                logger.info("{} ready", name)
        pending = still
        if pending:
            logger.info("still waiting for: {}", ", ".join(sorted(pending)))
            await asyncio.sleep(5.0)


async def main(
    cfg: dict,
    config_path: pathlib.Path | None = None,
    ready_file: pathlib.Path | None = None,
) -> None:
    setup_logging("worker")

    models_cfg = load_models_config(
        _resolve(config_path, cfg.get("models_yaml", "models.yaml")),
    )
    stt = make_stt(models_cfg, "stt")
    vlm = make_vlm(models_cfg, "vlm")
    tts = make_tts(models_cfg, "tts")

    await _wait_for_health(stt=stt, vlm=vlm, tts=tts)

    if ready_file:
        ready_file.touch()

    voice_gate_cfg = load_voice_gate_config(
        _resolve(config_path, cfg.get("voice_gate_yaml", "voice_gate.yaml")),
    )

    transport = XRMediaHubTransport()
    brain = RoomTourBrain(
        transport               = transport,
        vlm                     = vlm,
        system_prompt           = cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        frame_max_age_s         = float(cfg.get("frame_max_age_s",         5.0)),
        tour_capture_interval_s = float(cfg.get("tour_capture_interval_s", 3.0)),
        match_threshold         = float(cfg.get("match_threshold",         0.62)),
        nav_monitor_interval_s  = float(cfg.get("nav_monitor_interval_s",  4.0)),
        nav_max_secs            = float(cfg.get("nav_max_secs",          120.0)),
    )

    _, task = make_voice_pipeline(
        transport      = transport,
        stt            = stt,
        tts            = tts,
        brain          = brain,
        vad_cfg        = VadConfig(
            silence_duration = float(cfg.get("silence_duration", 0.4)),
            min_speech       = float(cfg.get("min_speech",       0.1)),
            silero_threshold = float(cfg.get("silero_threshold", 0.5)),
        ),
        voice_gate_cfg = voice_gate_cfg,
        text_topic     = "agent.response",
        idle_timeout_secs = (float(cfg["idle_timeout_secs"])
                             if cfg.get("idle_timeout_secs") else None),
    )

    loop = asyncio.get_running_loop()
    cancel_requested = False

    def _request_cancel() -> None:
        nonlocal cancel_requested
        if cancel_requested:
            return
        cancel_requested = True
        asyncio.create_task(task.cancel())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _request_cancel)

    logger.info("room-tour-example starting pipecat pipeline")
    try:
        await PipelineRunner().run(task)
    finally:
        transport.shutdown()
        for svc in (stt, vlm, tts):
            try:
                await svc.close()  # type: ignore[attr-defined]
            except Exception:
                logger.opt(exception=True).warning("service close failed")
    logger.info("room-tour-example stopped")


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
