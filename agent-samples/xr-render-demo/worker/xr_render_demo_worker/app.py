# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire model services, native tools, the supervisor, and the agent runtime."""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from xr_ai_logging import setup_logging
from xr_ai_models import load_models_config, make_llm, make_stt, make_tts, make_vlm
from xr_ai_runtime import AgentRuntime
from xr_ai_tools.current_frame import CurrentFrameTool
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.text_memory import TextMemoryTools
from xr_ai_tools.tracking import TrackingTools
from xr_ai_tools.video_memory import VideoMemoryTools
from xr_ai_voice import HubVoiceTransport, VadConfig, VoiceAgent
from xr_ai_voicegate import load_voice_gate_config
from xr_render_scene import SceneTools

from .agent import (
    INTERRUPTED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    USER_QUERY_TOPIC,
    RenderAgent,
)
from .config import WorkerConfig
from .supervisor import SceneSupervisor
from .xr_session import XRSessionController


async def run_app(
    config: WorkerConfig,
    *,
    ready_file: Path | None = None,
) -> None:
    """Run the render sample until the runtime exits."""
    setup_logging("worker")
    models = load_models_config(config.models_config)
    llm = make_llm(models, "agent_llm")
    stt = make_stt(models, "stt")
    tts = make_tts(models, "tts")
    vlm = make_vlm(models, "vlm")

    transport = HubVoiceTransport()
    scene = SceneTools(config.scene_endpoint)
    tracking = TrackingTools(config.openxr_endpoint)
    text_memory = TextMemoryTools(config.text_memory_dir)
    video = VideoMemoryTools(config.video_memory_endpoint)
    images = ImageRegistry(allow_external=True)
    current_frame = CurrentFrameTool(
        endpoint=transport.endpoint,
        images=images,
    )

    try:
        supervisor = SceneSupervisor(
            llm=llm,
            scene=scene,
            tracking=tracking,
            text_memory=text_memory,
            vlm=vlm,
            images=images,
            current_frame=current_frame,
        )

        render = RenderAgent(
            supervisor,
            on_participant_left=current_frame.release,
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
            probes={"agent-llm": llm.health, "vlm": vlm.health},
            ready_file=ready_file,
            closeables=(llm, vlm),
            idle_timeout_secs=config.idle_timeout_secs,
            transport=transport,
            participant_left_topic=PARTICIPANT_LEFT_TOPIC,
            interrupted_topic=INTERRUPTED_TOPIC,
        )
        runtime = AgentRuntime()
        runtime.register("voice", voice)
        runtime.register("xr-render", render)

        XRSessionController(
            transport=transport,
            start_xr=scene.start_xr,
            get_render_health=scene.get_health,
        ).attach()

        logger.info("xr-render-demo worker starting")
        async with runtime:
            try:
                await voice.run(runtime)
            finally:
                await render.stop()
        logger.info("xr-render-demo worker stopped")
    finally:
        await scene.client.close()
        await tracking.close()
        await video.close()
