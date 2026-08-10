# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire model services, native NAT function groups, subagents, and the voice session."""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from nat.builder.workflow_builder import WorkflowBuilder
from xr_ai_logging import setup_logging
from xr_ai_models import load_models_config, make_llm, make_stt, make_tts, make_vlm
from xr_ai_nat.adapters import as_voice_handler, record_voice_transcripts
from xr_ai_nat.functions.spatial_math import SpatialMathFunctionsConfig
from xr_ai_nat.functions.text_memory import ConversationMemoryFunctionsConfig, TextMemoryFunctionsConfig
from xr_ai_nat.functions.video_memory import VideoMemoryFunctionsConfig
from xr_ai_nat.functions.vision import VisionToolsConfig
from xr_ai_nat.functions.xr_tracking import XRTrackingFunctionsConfig
from xr_ai_voice import VadConfig, VoiceSession
from xr_ai_voicegate import load_voice_gate_config
from xr_render_scene import (
    SceneControlFunctionsConfig,
    SceneObjectFunctionsConfig,
    SceneStateFunctionsConfig,
    SceneUpdateFunctionsConfig,
)

from .config import WorkerConfig
from .models import SceneRequest
from .supervisor import scene_supervisor
from .xr_session import XRSessionController


async def run_app(config: WorkerConfig, *, ready_file: Path | None = None) -> None:
    """Run the render sample until the shared voice session exits."""
    setup_logging("worker")
    models = load_models_config(config.models_yaml)
    llm = make_llm(models, "agent_llm")
    stt = make_stt(models, "stt")
    tts = make_tts(models, "tts")
    vlm = make_vlm(models, "vlm")

    session = VoiceSession(
        stt=stt,
        tts=tts,
        vad=VadConfig(
            silence_duration=config.silence_duration,
            min_speech=config.min_speech,
            silero_threshold=config.silero_threshold,
        ),
        voice_gate=load_voice_gate_config(Path(config.voice_gate_yaml)),
        probes={"agent-llm": llm.health, "vlm": vlm.health},
        ready_file=ready_file,
        closeables=(llm, vlm),
        idle_timeout_secs=config.idle_timeout_secs,
    )

    async with session, WorkflowBuilder() as builder:
        await builder.add_function_group("scene_state", SceneStateFunctionsConfig(endpoint=config.scene_endpoint))
        await builder.add_function_group("scene_updates", SceneUpdateFunctionsConfig(endpoint=config.scene_endpoint))
        await builder.add_function_group("scene_objects", SceneObjectFunctionsConfig(endpoint=config.scene_endpoint))
        await builder.add_function_group("scene_control", SceneControlFunctionsConfig(endpoint=config.scene_endpoint))
        await builder.add_function_group("tracking", XRTrackingFunctionsConfig(endpoint=config.openxr_endpoint))
        await builder.add_function_group("spatial", SpatialMathFunctionsConfig())
        await builder.add_function_group("text_memory", TextMemoryFunctionsConfig(directory=config.text_memory_dir))
        await builder.add_function_group("conversations", ConversationMemoryFunctionsConfig())
        await builder.add_function_group(
            "video_memory",
            VideoMemoryFunctionsConfig(endpoint=config.video_memory_endpoint),
        )
        vision = VisionToolsConfig(endpoint=session.transport.endpoint, vlm=vlm)
        await builder.add_function_group("vision", vision)

        supervisor = await scene_supervisor(builder=builder, llm=llm)
        handler = as_voice_handler(
            supervisor,
            request=lambda turn: SceneRequest(
                transcript=turn.text,
                participant_id=turn.participant_id,
                timestamp_us=turn.timestamp_us,
            ),
            response=lambda reply: reply.response,
        )
        text_memory = await builder.get_function_group("text_memory")
        text_memory_functions = await text_memory.get_all_functions()
        scene_control = await builder.get_function_group("scene_control")
        scene_control_functions = await scene_control.get_all_functions()
        xr_session = XRSessionController(
            session=session,
            start_xr=scene_control_functions["scene_control__start_xr"],
            get_render_health=scene_control_functions["scene_control__get_health"],
        )
        xr_session.attach()

        logger.info("xr-render-demo worker starting")
        await session.run(
            handler,
            observer=record_voice_transcripts(text_memory_functions["text_memory__add_transcript"]),
            on_participant_left=vision.release,
        )
        logger.info("xr-render-demo worker stopped")
