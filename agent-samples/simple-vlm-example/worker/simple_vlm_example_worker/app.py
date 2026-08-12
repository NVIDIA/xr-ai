# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose the simple VLM assistant from shared SDK primitives."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from loguru import logger
from xr_ai_logging import setup_logging
from xr_ai_models import load_models_config, make_stt, make_tts, make_vlm
from xr_ai_tools.streaming_vision import StreamingVisionTool, VisionRequest
from xr_ai_voice import TextMessageInput, VadConfig, VoiceHandler, VoiceSession
from xr_ai_voicegate import load_voice_gate_config

from .config import WorkerConfig


def _make_vision_handler(vision: StreamingVisionTool) -> VoiceHandler:
    async def handle(turn):
        async def response():
            request = VisionRequest(
                participant_id=turn.participant_id,
                query=turn.text,
            )
            stream = vision.stream(request)
            try:
                async for chunk in stream:
                    yield chunk.text
            finally:
                close = getattr(stream, "aclose", None)
                if close is not None:
                    await close()

        return response()

    return handle


def _text_transform(default_prompt: str) -> Callable[[str], str]:
    return lambda text: default_prompt if text.lower() == "ping" else text


async def run_app(
    config: WorkerConfig,
    *,
    ready_file: Path | None = None,
) -> None:
    """Run the worker until the voice session shuts down."""

    setup_logging("worker")
    models = load_models_config(config.models_config)
    voice_gate = load_voice_gate_config(config.voice_gate_yaml)
    stt = make_stt(models, "stt")
    vlm = make_vlm(models, "vlm")
    tts = make_tts(models, "tts")

    session = VoiceSession(
        stt=stt,
        tts=tts,
        vad=VadConfig(
            silence_duration=config.silence_duration,
            min_speech=config.min_speech,
            silero_threshold=config.silero_threshold,
        ),
        voice_gate=voice_gate,
        probes={"vlm": vlm.health},
        ready_file=ready_file,
        closeables=(vlm,),
        text_topic="vlm.response",
        idle_timeout_secs=config.idle_timeout_secs,
    )

    async with session:
        vision = StreamingVisionTool(
            endpoint=session.transport.endpoint,
            vlm=vlm,
            system_prompt=config.system_prompt,
            frame_max_age_s=config.frame_max_age_s,
            frame_timeout_s=config.frame_timeout_s,
        )
        TextMessageInput(
            session=session,
            transform=_text_transform(config.default_prompt),
            fresh_match=True,
        )

        logger.info("simple-vlm-example starting")
        await session.run(
            _make_vision_handler(vision),
            on_participant_left=vision.release,
            interrupt_on_supersede=True,
        )
        logger.info("simple-vlm-example stopped")
