# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose native tea guidance, background applications, voice, and files."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import nemo_relay
from loguru import logger
from xr_ai_logging import setup_logging
from xr_ai_models import load_models_config, make_llm, make_stt, make_tts, make_vlm
from xr_ai_runtime import AgentRuntime
from xr_ai_tools.rag import RAGTools
from xr_ai_tools.vision import ImageQueryTool
from xr_ai_voice import HubVoiceTransport, VadConfig, VoiceAgent, VoiceAggregationAgent
from xr_ai_voicegate import load_voice_gate_config

from .background_context import BackgroundContextAgent
from .change_watch import ChangeWatchAgent
from .config import WorkerConfig
from .events import (
    INTERRUPTED_TOPIC,
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    USER_QUERY_TOPIC,
)
from .file_output import FileOutputAgent
from .foreground import ForegroundAgent
from .guidance_voice import GuidanceVoiceAgent
from .images import ParticipantImageAgent
from .spec import load_workflow
from .transcript import TranscriptAgent
from .video_log import VideoLogAgent
from .workflow import GuidanceAgent


@asynccontextmanager
async def _relay_event_log(output_dir: Path) -> AsyncIterator[Path]:
    path = output_dir / "relay-events.jsonl"
    sink = path.open("w", encoding="utf-8")
    lock = Lock()
    subscriber = "tea-making-event-log"

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
    rag = RAGTools(config.rag_endpoint)
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
        probes={"llm": llm.health, "vlm": vlm.health, "rag": rag.health},
        ready_file=ready_file,
        closeables=(llm, vlm, rag),
        text_topic="",
        idle_timeout_secs=config.idle_timeout_secs,
        transport=transport,
        participant_joined_topic=PARTICIPANT_JOINED_TOPIC,
        participant_left_topic=PARTICIPANT_LEFT_TOPIC,
        interrupted_topic=INTERRUPTED_TOPIC,
        interrupt_on_supersede=True,
    )

    runtime = AgentRuntime()
    files = runtime.register(
        "file-output",
        FileOutputAgent(
            config.artifacts_dir,
            history_size=config.background_history_size,
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
    background_context = runtime.register(
        "background-context",
        BackgroundContextAgent(capacity=config.background_history_size),
    )
    guidance = runtime.register(
        "guidance",
        GuidanceAgent(
            workflow=load_workflow(config.workflow_config),
            llm=llm,
            current_frame=images.get_current_frame,
            image_query=ImageQueryTool(images=images.images, vlm=vlm),
            rag=rag,
            vlm_timeout_s=config.vlm_timeout_s,
        ),
    )
    change_watch = runtime.register(
        "change-watch",
        ChangeWatchAgent(
            images=images,
            vlm=vlm,
            llm=llm,
            caption_prompt=config.change_watch_caption_prompt,
            event_prompt=config.change_watch_event_prompt,
            default_instruction=config.change_watch_default_instruction,
            interval_s=config.change_watch_interval_s,
        ),
    )
    transcript = runtime.register(
        "transcript",
        TranscriptAgent(
            llm=llm,
            summary_prompt=config.transcript_summary_prompt,
            summary_interval_s=config.transcript_summary_interval_s,
        ),
    )
    video_log = runtime.register(
        "video-log",
        VideoLogAgent(
            images=images,
            vlm=vlm,
            llm=llm,
            caption_prompt=config.video_caption_prompt,
            delta_prompt=config.video_delta_prompt,
            interval_s=config.video_log_interval_s,
            history_size=5,
        ),
    )
    foreground = runtime.register(
        "foreground",
        ForegroundAgent(
            llm=llm,
            images=images,
            vlm=vlm,
            rag=rag,
            guidance=guidance,
            background_context=background_context,
            change_watch=change_watch,
            transcript=transcript,
            video_log=video_log,
            prompt=config.foreground_prompt,
            vlm_timeout_s=config.vlm_timeout_s,
        ),
    )
    runtime.register("guidance-voice", GuidanceVoiceAgent())
    voice_aggregation = runtime.register(
        "voice-aggregation",
        VoiceAggregationAgent(llm=llm),
    )
    runtime.register("voice", voice)

    logger.info("file outputs → {}", config.artifacts_dir)
    logger.info("tea-making starting with Omni for language and vision")
    async with _relay_event_log(config.artifacts_dir):
        async with runtime:
            guidance.bind_runtime(runtime)
            change_watch.bind_runtime(runtime)
            transcript.bind_runtime(runtime)
            video_log.bind_runtime(runtime)
            try:
                await voice.run(runtime)
            finally:
                await foreground.stop()
                await guidance.stop()
                await change_watch.stop()
                await transcript.stop()
                await video_log.stop()
                await images.stop()
                await files.stop()
                await voice_aggregation.stop()
    logger.info("tea-making stopped")


__all__ = ["run_app"]
