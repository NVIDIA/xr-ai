# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose lab monitoring, foreground queries, voice, and file output."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import nemo_relay
from loguru import logger
from xr_ai_logging import setup_logging
from xr_ai_models import load_models_config, make_llm, make_stt, make_tts, make_vlm
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, subscribe
from xr_ai_voice import (
    HubVoiceTransport,
    VadConfig,
    VoiceAgent,
    VoiceAggregationAgent,
    VoiceParticipantLeft,
)
from xr_ai_voicegate import load_voice_gate_config
from xr_ai_web_events import WebEventsAgent

from .config import WorkerConfig
from .events import (
    INTERRUPTED_TOPIC,
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    USER_QUERY_TOPIC,
)
from .file_output import FileOutputAgent
from .foreground import ForegroundAgent
from .images import ParticipantImageAgent
from .instrument_alerts import InstrumentAlertAgent
from .instrument_monitor import InstrumentMonitorAgent
from .instruments import LabInstrumentAgent
from .monitor import MonitorAgent
from .web_events import WebEventsAdapterAgent


class _VoiceAggregationLifecycleAgent(Agent):
    def __init__(self, voice_aggregation: VoiceAggregationAgent) -> None:
        super().__init__()
        self._voice_aggregation = voice_aggregation

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self,
        _event: VoiceParticipantLeft,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = ctx.metadata.participant_id
        if participant_id is not None:
            await self._voice_aggregation.release(participant_id)


@asynccontextmanager
async def _relay_event_log(output_dir: Path) -> AsyncIterator[Path]:
    path = output_dir / "relay-events.jsonl"
    sink = path.open("w", encoding="utf-8")
    lock = Lock()
    subscriber = "lab-instrument-monitoring-event-log"

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
        voice_gate=load_voice_gate_config(config.voice_gate_yaml),
        probes={"llm": llm.health, "vlm": vlm.health},
        ready_file=ready_file,
        closeables=(llm, vlm),
        text_topic="agent.response",
        idle_timeout_secs=config.idle_timeout_secs,
        transport=transport,
        participant_joined_topic=PARTICIPANT_JOINED_TOPIC,
        participant_left_topic=PARTICIPANT_LEFT_TOPIC,
        interrupted_topic=INTERRUPTED_TOPIC,
        interrupt_on_supersede=True,
    )

    runtime = AgentRuntime()
    web_events = runtime.register(
        "web-events",
        WebEventsAgent(
            host=config.web_events_host,
            port=config.web_events_port,
            max_events=config.web_events_max_events,
            title="Lab instrument monitoring events",
        ),
    )
    runtime.register("web-events-adapter", WebEventsAdapterAgent())
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
            endpoint=transport.endpoint,
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
    lab_instruments = runtime.register(
        "lab-instruments",
        LabInstrumentAgent(
            images=images,
            vlm=vlm,
            device_map=config.device_map,
            debug_dir=(config.artifacts_dir / "marker-scans" if config.capture_marker_scans else None),
        ),
    )
    instrument_monitor = runtime.register(
        "instrument-monitor",
        InstrumentMonitorAgent(
            reader=lab_instruments,
            interval_s=config.instrument_monitor_interval_s,
            snapshot_interval_s=config.instrument_state_interval_s,
            lost_after_s=config.instrument_lost_after_s,
        ),
    )
    runtime.register("instrument-alerts", InstrumentAlertAgent())
    foreground = runtime.register(
        "foreground",
        ForegroundAgent(
            llm=llm,
            images=images,
            vlm=vlm,
            files=files,
            monitor=monitor,
            lab_instruments=lab_instruments,
            instrument_monitor=instrument_monitor,
            prompt=config.foreground_prompt,
        ),
    )
    voice_aggregation = runtime.register(
        "voice-aggregation",
        VoiceAggregationAgent(llm=llm),
    )
    runtime.register(
        "voice-aggregation-lifecycle",
        _VoiceAggregationLifecycleAgent(voice_aggregation),
    )
    runtime.register("voice", voice)

    logger.info("file outputs → {}", config.artifacts_dir)
    logger.info("lab-instrument-monitoring starting")
    async with _relay_event_log(config.artifacts_dir):
        async with web_events:
            logger.info("live events → {}", web_events.url)
            async with runtime:
                monitor.bind_runtime(runtime)
                instrument_monitor.bind_runtime(runtime)
                try:
                    await voice.run(runtime)
                finally:
                    await foreground.stop()
                    await instrument_monitor.stop()
                    await monitor.stop()
                    await images.stop()
                    await voice_aggregation.stop()
    logger.info("lab-instrument-monitoring stopped")


__all__ = ["run_app"]
