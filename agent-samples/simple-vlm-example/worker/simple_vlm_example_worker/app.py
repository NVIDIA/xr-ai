# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose the simple VLM assistant from shared SDK primitives."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import nemo_relay
from loguru import logger
from xr_ai_logging import setup_logging
from xr_ai_models import load_models_config, make_stt, make_tts, make_vlm
from xr_ai_runtime import AgentRuntime
from xr_ai_tools.current_frame import CurrentFrameTool
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import StreamingImageQueryTool
from xr_ai_voice import HubVoiceTransport, VadConfig, VoiceAgent
from xr_ai_voicegate import load_voice_gate_config

from .agent import (
    INTERRUPTED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    USER_QUERY_TOPIC,
    SimpleVlmAgent,
)
from .config import WorkerConfig


@asynccontextmanager
async def _relay_event_log(log_file: Path) -> AsyncIterator[Path]:
    event_path = log_file.parent / "relay-events.jsonl"
    sink = event_path.open("w", encoding="utf-8")
    lock = Lock()
    subscriber = "simple-vlm-compact-event-log"

    def write_event(event: nemo_relay.Event) -> None:
        if event.kind == "mark" and event.name == "llm.chunk":
            return
        with lock:
            sink.write(event.to_json())
            sink.write("\n")
            sink.flush()

    try:
        nemo_relay.subscribers.register(subscriber, write_event)
    except Exception:
        sink.close()
        raise
    try:
        yield event_path
    finally:
        await nemo_relay.subscribers.flush_async()
        nemo_relay.subscribers.deregister(subscriber)
        sink.close()


async def run_app(
    config: WorkerConfig,
    *,
    ready_file: Path | None = None,
) -> None:
    """Run the worker until the voice session shuts down."""

    log_file = setup_logging("worker")
    models = load_models_config(config.models_config)
    voice_gate = load_voice_gate_config(config.voice_gate_yaml)
    stt = make_stt(models, "stt")
    vlm = make_vlm(models, "vlm")
    tts = make_tts(models, "tts")

    transport = HubVoiceTransport()
    voice = VoiceAgent(
        query_topic=USER_QUERY_TOPIC,
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
        transport=transport,
        participant_left_topic=PARTICIPANT_LEFT_TOPIC,
        interrupted_topic=INTERRUPTED_TOPIC,
        interrupt_on_supersede=True,
    )

    runtime = AgentRuntime()
    images = ImageRegistry()
    simple_vlm = runtime.register(
        "simple-vlm",
        SimpleVlmAgent(
            lambda: (
                CurrentFrameTool(
                    endpoint=transport.endpoint,
                    images=images,
                    frame_max_age_s=config.frame_max_age_s,
                    frame_timeout_s=config.frame_timeout_s,
                ),
                StreamingImageQueryTool(
                    images=images,
                    vlm=vlm,
                    system_prompt=config.system_prompt,
                ),
            ),
            transport.endpoint.set_status,
        ),
    )
    runtime.register("voice", voice)

    logger.info("Relay events → {}", log_file.parent / "relay-events.jsonl")
    logger.info("simple-vlm-example starting")
    async with _relay_event_log(log_file):
        async with runtime:
            try:
                await voice.run(runtime)
            finally:
                await simple_vlm.stop()
    logger.info("simple-vlm-example stopped")
