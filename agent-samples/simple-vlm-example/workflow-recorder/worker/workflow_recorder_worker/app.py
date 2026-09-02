# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose automatic capture, transcription, captioning, and guide discovery."""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from xr_ai_logging import setup_logging
from xr_ai_models import load_models_config, make_llm, make_stt, make_tts, make_vlm
from xr_ai_runtime import AgentRuntime
from xr_ai_tools.current_frame import CurrentFrameTool
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import ImageQueryTool
from xr_ai_voice import HubVoiceTransport, VadConfig, VoiceAgent
from xr_ai_voicegate import load_voice_gate_config

from ._workflow_engine import SopEngineAgent
from .catalog import GuideCatalog
from .config import WorkerConfig
from .events import PARTICIPANT_JOINED_TOPIC, PARTICIPANT_LEFT_TOPIC, USER_QUERY_TOPIC
from .recorder import RecorderAgent

_PACKAGE = Path(__file__).resolve().parent


async def run_app(config: WorkerConfig, *, ready_file: Path | None = None) -> None:
    setup_logging("worker")
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    models = load_models_config(config.models_config)
    llm = make_llm(models, "llm")
    stt = make_stt(models, "stt")
    vlm = make_vlm(models, "vlm")
    tts = make_tts(models, "tts")
    transport = HubVoiceTransport()
    images = ImageRegistry(capacity=max(256, int(config.capture_fps * config.caption_interval_s * 8)))
    current_frame = CurrentFrameTool(
        endpoint=transport.endpoint,
        images=images,
        frame_max_age_s=config.frame_max_age_s,
        frame_timeout_s=config.frame_timeout_s,
    )
    query_image = ImageQueryTool(images=images, vlm=vlm, system_prompt=config.caption_prompt)
    sop_image_query = ImageQueryTool(
        images=images,
        vlm=vlm,
        system_prompt=(_PACKAGE / "prompts" / "sop_vision.txt").read_text(encoding="utf-8"),
    )
    catalog = GuideCatalog(
        config.guides_dir,
        config.artifacts_dir / "guide-index.json",
        interval_s=config.guide_scan_interval_s,
    )
    recorder = RecorderAgent(
        sessions_dir=config.artifacts_dir / "sessions",
        current_frame=current_frame,
        images=images,
        query_image=query_image,
        guide_catalog=catalog,
        capture_fps=config.capture_fps,
        caption_interval_s=config.caption_interval_s,
    )
    sop_engine = SopEngineAgent(
        catalog=catalog,
        llm=llm,
        current_frame=current_frame,
        image_query=sop_image_query,
        vision_timeout_s=config.frame_timeout_s,
    )
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
        ready_file=ready_file,
        closeables=(llm, vlm),
        text_topic="workflow-recorder.status",
        idle_timeout_secs=config.idle_timeout_secs,
        transport=transport,
        participant_joined_topic=PARTICIPANT_JOINED_TOPIC,
        participant_left_topic=PARTICIPANT_LEFT_TOPIC,
    )

    runtime = AgentRuntime()
    runtime.register("recorder", recorder)
    runtime.register("sop-engine", sop_engine)
    runtime.register("voice", voice)
    sop_engine.bind_runtime(runtime)

    await catalog.start()
    logger.info("recording packets → {}", config.artifacts_dir / "sessions")
    logger.info("workflow guides → {}", config.guides_dir)
    async with runtime:
        try:
            await voice.run(runtime)
        finally:
            await sop_engine.stop()
            await recorder.stop()
            await catalog.stop()
            images.clear()
