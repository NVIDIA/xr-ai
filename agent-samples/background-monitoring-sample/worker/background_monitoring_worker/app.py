# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose background monitoring, foreground queries, voice, and file output."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import nemo_relay
from loguru import logger
from xr_ai_hub import ParticipantEvent
from xr_ai_logging import setup_logging
from xr_ai_models import load_models_config, make_llm, make_stt, make_tts, make_vlm
from xr_ai_runtime import AgentRuntime, RuntimeClosedError
from xr_ai_voice import VadConfig, VoiceAgent
from xr_ai_voicegate import load_voice_gate_config

from .config import WorkerConfig
from .events import (
    INTERRUPTED_TOPIC,
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    USER_QUERY_TOPIC,
    ParticipantJoined,
)
from .file_output import FileOutputAgent
from .foreground import ForegroundAgent
from .images import ParticipantImageAgent
from .monitor import MonitorAgent
from .qr_instruments import QRInstrumentAgent
from .transcript import TranscriptAgent


@asynccontextmanager
async def _relay_event_log(output_dir: Path) -> AsyncIterator[Path]:
    path = output_dir / "relay-events.jsonl"
    sink = path.open("w", encoding="utf-8")
    lock = Lock()
    subscriber = "background-monitoring-event-log"

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
        yield path
    finally:
        await nemo_relay.subscribers.flush_async()
        nemo_relay.subscribers.deregister(subscriber)
        sink.close()


async def run_app(config: WorkerConfig, *, ready_file: Path | None = None) -> None:
    """Run the native multi-agent worker until its voice session stops."""

    setup_logging("worker")
    models = load_models_config(config.models_config)
    llm = make_llm(models, "llm")
    vlm = make_vlm(models, "vlm")
    stt = make_stt(models, "stt")
    tts = make_tts(models, "tts")
    voice = VoiceAgent(
        query_topic=USER_QUERY_TOPIC,
        stt=stt,
        tts=tts,
        vad=VadConfig(
            silence_duration=config.silence_duration,
            min_speech=config.min_speech,
            silero_threshold=config.silero_threshold,
        ),
        voice_gate=load_voice_gate_config(config.voice_gate_yaml),
        probes={"llm": llm.health, "vlm": vlm.health},
        ready_file=ready_file,
        closeables=(llm, vlm),
        text_topic="",
        idle_timeout_secs=config.idle_timeout_secs,
        participant_left_topic=PARTICIPANT_LEFT_TOPIC,
        interrupted_topic=INTERRUPTED_TOPIC,
        interrupt_on_supersede=True,
    )

    runtime = AgentRuntime()
    files = runtime.register(
        "file-output",
        FileOutputAgent(
            config.artifacts_dir,
            history_size=config.monitor_history_size,
        ),
    )
    images = runtime.register(
        "images",
        ParticipantImageAgent(
            endpoint=voice.endpoint,
            frame_max_age_s=config.frame_max_age_s,
            frame_timeout_s=config.frame_timeout_s,
        ),
    )
    monitor = runtime.register(
        "monitor",
        MonitorAgent(
            images=images,
            vlm=vlm,
            prompt=config.monitor_prompt,
            interval_s=config.monitor_interval_s,
        ),
    )
    qr_instruments = runtime.register(
        "qr-instruments",
        QRInstrumentAgent(
            images=images,
            vlm=vlm,
            interval_s=config.instrument_monitor_interval_s,
            debug_dir=config.artifacts_dir / "qr-scans",
        ),
    )
    foreground = runtime.register(
        "foreground",
        ForegroundAgent(
            llm=llm,
            images=images,
            vlm=vlm,
            files=files,
            monitor=monitor,
            qr_instruments=qr_instruments,
            prompt=config.foreground_prompt,
        ),
    )
    runtime.register("transcript", TranscriptAgent())
    runtime.register("voice", voice)

    async def participant_event(event: ParticipantEvent) -> None:
        if not event.joined:
            return
        try:
            await runtime.publish(
                PARTICIPANT_JOINED_TOPIC,
                ParticipantJoined(timestamp_us=event.pts_us),
                participant_id=event.participant_id,
                source="hub",
            )
        except RuntimeClosedError:
            return

    voice.endpoint.on_participant(participant_event)

    logger.info("file outputs → {}", config.artifacts_dir)
    logger.info("background-monitoring-sample starting")
    async with _relay_event_log(config.artifacts_dir):
        async with runtime:
            monitor.bind_runtime(runtime)
            qr_instruments.bind_runtime(runtime)
            try:
                await voice.run(runtime)
            finally:
                await foreground.stop()
                await qr_instruments.stop()
                await monitor.stop()
                await images.stop()
    logger.info("background-monitoring-sample stopped")


__all__ = ["run_app"]
