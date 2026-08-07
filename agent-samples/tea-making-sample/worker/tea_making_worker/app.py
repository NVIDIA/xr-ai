# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose the YAML workflow entirely from native NAT functions and agents."""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger
from nat.builder.workflow_builder import WorkflowBuilder
from xr_ai_logging import setup_logging
from xr_ai_models import load_models_config, make_llm, make_stt, make_tts, make_vlm
from xr_ai_nat.functions.rag import RAGFunctionsConfig
from xr_ai_nat.functions.vision import VisionToolsConfig
from xr_ai_voice import TextMessageInput, VadConfig, VoiceQuery, VoiceSession
from xr_ai_voicegate import load_voice_gate_config

from .agents import AgentRegistry
from .config import WorkerConfig
from .engine import Coordinator, NoticeBridge, TriggerRegistry
from .functions import (
    CurrentViewConfig,
    RAGLookupConfig,
    add_clock_functions,
    add_temperature_functions,
    add_workflow_functions,
)
from .runtime.state import SessionStore
from .spec import load_workflow


async def run_app(config: WorkerConfig, *, ready_file: Path | None = None) -> None:
    setup_logging("worker", namespace="tea-making-sample")
    models = load_models_config(config.models_config)
    llm = make_llm(models, "agent_llm")
    vlm = make_vlm(models, "vlm")
    workflow = load_workflow(config.workflow_config)
    voice_gate = load_voice_gate_config(config.voice_gate_config)
    session = VoiceSession(
        stt=make_stt(models, "stt"),
        tts=make_tts(models, "tts"),
        vad=VadConfig(
            silence_duration=config.silence_duration,
            min_speech=config.min_speech,
            silero_threshold=config.silero_threshold,
        ),
        voice_gate=voice_gate,
        probes={"agent-llm": llm.health, "vlm": vlm.health},
        ready_file=ready_file,
        closeables=(llm, vlm),
        text_topic="guide.response",
        idle_timeout_secs=config.idle_timeout_secs,
    )

    async with session, WorkflowBuilder() as builder:
        store = SessionStore(workflow)
        agents = AgentRegistry(workflow)
        notices = NoticeBridge(session)
        vision_config = VisionToolsConfig(
            endpoint=session.transport.endpoint,
            vlm=vlm,
            frame_max_age_s=config.frame_max_age_s,
            frame_timeout_s=config.frame_timeout_s,
        )
        vision_group = await builder.add_function_group("vision", vision_config)
        vision_functions = await vision_group.get_all_functions()
        await builder.add_function(
            "current_view",
            CurrentViewConfig(
                source=vision_functions["vision__look_at_current_frame"],
                timeout_s=config.vlm_timeout_s,
            ),
        )
        rag_group = await builder.add_function_group(
            "rag",
            RAGFunctionsConfig(endpoint=config.rag_endpoint),
        )
        rag_functions = await rag_group.get_all_functions()
        await builder.add_function(
            "rag_lookup",
            RAGLookupConfig(source=rag_functions["rag__retrieve"]),
        )
        await add_clock_functions(builder)
        await add_temperature_functions(builder)
        await add_workflow_functions(builder, store=store)
        await agents.build(builder, llm)
        triggers = TriggerRegistry(workflow)
        await triggers.build(builder)
        coordinator = Coordinator(store=store, agents=agents, triggers=triggers, notice=notices.send)
        TextMessageInput(session=session, fresh_match=True)

        async def handler(turn: VoiceQuery) -> str:
            notice = notices.take(turn.text)
            if notice is not None:
                return notice
            return await coordinator.handle_query(turn.participant_id, turn.text)

        async def participant_left(participant_id: str) -> None:
            vision_config.release(participant_id)
            await coordinator.participant_left(participant_id)

        async def run_voice() -> None:
            try:
                await session.run(
                    handler,
                    on_participant_joined=coordinator.participant_joined,
                    on_participant_left=participant_left,
                    interrupt_on_supersede=True,
                    queue_queries=True,
                )
            finally:
                coordinator.stop()

        logger.info("tea-making-sample starting")
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(run_voice(), name="tea-guide-voice")
            tasks.create_task(coordinator.monitor(), name="tea-guide-monitor")
        logger.info("tea-making-sample stopped")


__all__ = ["run_app"]
