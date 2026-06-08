# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
simple-vlm-example worker — entry point.

Launched as a subprocess by ``uv run simple_vlm_example`` (the orchestrator).
Do not run this directly.

Pipeline
--------
Audio + voice path runs through the unified pipecat pipeline assembled by
``xr_ai_pipecat.make_voice_pipeline``:

    XRMediaHubInputTransport → VadSttProcessor → VoiceGateProcessor →
    SimpleVlmBrain → StreamingTtsProcessor → XRMediaHubOutputTransport

The voice gate (magic phrases, follow-up grace, listening chime, stop
ack) is owned by :class:`xr_ai_voicegate.VoiceGate` inside the
``VoiceGateProcessor``. Wake-word config moves from this worker's YAML
to ``yaml/voice_gate.yaml`` so every pipecat sample shares the schema.

Text data channel + frame tracking + camera-on-demand are owned by
``SimpleVlmBrain`` and continue to use the ``ProcessorEndpoint`` API
directly.

Config (simple_vlm_example_worker.yaml — auto-passed by the launcher)
---------------------------------------------------------------------
    model_backend:         local   # "local" (default) or "nim" (hosted VLM; uses models.nim.yaml)
    models_yaml:           yaml/models.yaml      # local-backend models config
    voice_gate_yaml:       yaml/voice_gate.yaml  # path to voice-gate config
    default_prompt:        "Describe what you see."
    system_prompt:               <multiline string>   # role/style guidance
    frame_max_age_s:            2.0   # frames older than this trigger startCamera
    camera_on_timeout_s:       15.0   # wait for a fresh frame after startCamera
    camera_grace_s:             5.0   # keep camera on after a query
    silero_threshold:           0.5   # Silero speech probability gate (0..1)
    silence_duration:           0.4   # seconds of silence ending an utterance
    min_speech:                 0.1   # min seconds of speech before STT fires
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

from agent import DEFAULT_SYSTEM_PROMPT, SimpleVlmBrain


def _resolve(cfg_path: pathlib.Path | None, raw: str) -> pathlib.Path:
    """Resolve a YAML-referenced path relative to the worker YAML's
    parent directory so sample CWD doesn't change which files load."""
    p = pathlib.Path(raw)
    if cfg_path and not p.is_absolute():
        p = cfg_path.parent / p
    return p


async def main(
    cfg: dict,
    config_path: pathlib.Path | None = None,
    ready_file: pathlib.Path | None = None,
) -> None:
    setup_logging("worker")

    # `model_backend: nim` selects the shipped NIM overlay (hosted VLM); the
    # orchestrator reads the same key to skip the local vlm-server. Otherwise
    # `models_yaml` picks the local config (default models.yaml).
    backend = str(cfg.get("model_backend", "local")).lower()
    models_yaml_raw = (
        "models.nim.yaml" if backend == "nim"
        else cfg.get("models_yaml", "models.yaml")
    )
    models_cfg = load_models_config(_resolve(config_path, models_yaml_raw))

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
    brain = SimpleVlmBrain(
        transport           = transport,
        vlm                 = vlm,
        default_prompt      = cfg.get("default_prompt", "Describe what you see."),
        system_prompt       = cfg.get("system_prompt",  DEFAULT_SYSTEM_PROMPT),
        frame_max_age_s     = float(cfg.get("frame_max_age_s",      2.0)),
        camera_on_timeout_s = float(cfg.get("camera_on_timeout_s", 15.0)),
        camera_grace_s      = float(cfg.get("camera_grace_s",       5.0)),
    )

    # make_voice_pipeline returns (pipeline, task); only the task is run.
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
        text_topic     = "vlm.response",
    )

    loop = asyncio.get_running_loop()
    cancel_requested = False

    def _request_cancel() -> None:
        # PipelineTask.cancel is a coroutine; add_signal_handler needs a
        # sync callable. Guard against a second signal (e.g. double
        # ctrl-c) spawning a redundant cancel task while the first is
        # still draining the pipeline.
        nonlocal cancel_requested
        if cancel_requested:
            return
        cancel_requested = True
        asyncio.create_task(task.cancel())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _request_cancel)

    logger.info("simple-vlm-example starting pipecat pipeline")
    try:
        await PipelineRunner().run(task)
    finally:
        transport.shutdown()
        for svc in (stt, vlm, tts):
            try:
                await svc.close()  # type: ignore[attr-defined]
            except Exception:
                logger.opt(exception=True).warning("service close failed")
    logger.info("simple-vlm-example stopped")


async def _wait_for_health(**services: object) -> None:
    """Block until every service's health endpoint reports healthy."""
    pending: dict[str, object] = dict(services)
    while pending:
        results = await asyncio.gather(
            *(svc.health() for svc in pending.values()),  # type: ignore[attr-defined]
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
