# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose voice, frame tools, and the agent that captures task demonstrations.

Infrastructure is constructed here and injected into CaptureAgent's recorder.
The agent owns commands and narration ordering; the recorder owns sampling,
captioning, and durable files. This module owns startup and resource cleanup.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from xr_ai_hub import ProcessorEndpoint
from xr_ai_logging import setup_logging
from xr_ai_models import load_models_config, make_stt, make_tts, make_vlm
from xr_ai_runtime import AgentRuntime
from xr_ai_tools.current_frame import CurrentFrameTool
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import MultiImageQueryTool
from xr_ai_voice import VadConfig, VoiceAgent
from xr_ai_voicegate import load_voice_gate_config

from .capture.agent import CAPTURE_INPUT_TOPIC, CaptureAgent
from .capture.recorder import _Recorder
from .config import WorkerConfig

_HUB_PUB = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH = "ipc:///tmp/xr_hub_in"


async def run_app(config: WorkerConfig, *, ready_file: Path | None = None) -> None:
    setup_logging("worker")
    models = load_models_config(config.models_config)
    stt, tts, vlm = make_stt(models, "stt"), make_tts(models, "tts"), make_vlm(models, "vlm")
    # Recording must finish draining after voice stops. Its frame and participant
    # endpoint therefore has a lifetime controlled by this composition layer.
    endpoint = ProcessorEndpoint(sub_addr=_HUB_PUB, push_addr=_HUB_PUSH)
    images = ImageRegistry()
    frames = CurrentFrameTool(
        endpoint=endpoint, images=images, frame_max_age_s=config.frame_max_age_s, frame_timeout_s=config.frame_timeout_s
    )
    recorder = _Recorder(
        sessions_dir=config.artifacts_dir / "sessions",
        current_frame=frames,
        images=images,
        query_images=MultiImageQueryTool(images=images, vlm=vlm, system_prompt=config.caption_prompt),
        capture_fps=config.capture_fps,
        caption_interval_s=config.caption_interval_s,
    )
    # One owner orders command boundaries and admitted narration, including
    # repeated recordings on the same participant connection.
    capture = CaptureAgent(recorder)
    endpoint.on_participant(capture.receive)
    # Accepted voice input is delivered after prior speech is cancelled, so
    # each recording command cannot interrupt its own acknowledgement.
    voice = VoiceAgent(
        query_topic=CAPTURE_INPUT_TOPIC,
        stt=stt,
        tts=tts,
        vad=VadConfig(
            silence_duration=config.silence_duration,
            min_speech=config.min_speech,
            silero_threshold=config.silero_threshold,
            # Wait for the final utterance before interpreting recording commands;
            # a partial "stop" must not interrupt narration or acknowledgements.
            stop_probe_after_s=0,
        ),
        voice_gate=load_voice_gate_config(config.voice_gate_yaml),
        probes={"stt": stt.health, "tts": tts.health, "vlm": vlm.health},
        ready_file=ready_file,
        text_input=False,
    )
    runtime = AgentRuntime()
    runtime.register("voice", voice)
    runtime.register("capture", capture)
    receiver = asyncio.create_task(endpoint.run())
    try:
        await endpoint.wait_until_running()
        async with runtime:
            await voice.run(runtime)
    finally:
        try:
            # Runtime shutdown drains message delivery, not agent-owned tasks
            # or filesystem threads. Explicitly finalize before closing resources.
            await capture.close()
        finally:
            # stop() alone cannot wake an idle ZMQ receive. Drain recordings
            # first, then cancel the receiver before closing its sockets.
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
            endpoint.stop()
            endpoint.close()
            await vlm.close()
