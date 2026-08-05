# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose models, voice, live vision, and the YAML workflow guide."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from loguru import logger
from nat.builder.workflow_builder import WorkflowBuilder
from xr_ai_logging import setup_logging
from xr_ai_models import load_models_config, make_llm, make_stt, make_tts, make_vlm
from xr_ai_nat.functions.rag import RAGFunctionsConfig
from xr_ai_voice import TextMessageInput, VadConfig, VoiceQuery, VoiceSession
from xr_ai_voicegate import load_voice_gate_config

from .agent import WorkflowAgent
from .config import WorkerConfig
from .guide import WorkflowGuide
from .tools import GuideTools
from .vision import StepVision
from .workflow import WorkflowDefinition


class NoticeBridge:
    """Turns background guidance notices into normal voice-session queries."""

    def __init__(self, session: VoiceSession) -> None:
        self._session = session
        self._pending: dict[str, str] = {}

    async def speak(self, participant_id: str, text: str) -> None:
        if not self._session.is_running:
            return
        token = f"__tea_notice_{uuid.uuid4().hex}"
        self._pending[token] = text
        await self._session.enqueue_query(participant_id, token, fresh_match=True)

    def pop(self, token: str) -> str | None:
        return self._pending.pop(token, None)


async def run_app(config: WorkerConfig, *, ready_file: Path | None = None) -> None:
    """Run the tea-making guided workflow worker until shutdown."""

    setup_logging("worker", namespace="tea-making-sample")
    models = load_models_config(config.models_yaml)
    agent_llm = make_llm(models, "agent_llm")
    stt = make_stt(models, "stt")
    tts = make_tts(models, "tts")
    vlm = make_vlm(models, "vlm")

    workflow = WorkflowDefinition.load(config.workflow_yaml)
    voice_gate = load_voice_gate_config(config.voice_gate_yaml)

    session = VoiceSession(
        stt=stt,
        tts=tts,
        vad=VadConfig(
            silence_duration=config.silence_duration,
            min_speech=config.min_speech,
            silero_threshold=config.silero_threshold,
        ),
        voice_gate=voice_gate,
        probes={"agent-llm": agent_llm.health, "vlm": vlm.health},
        ready_file=ready_file,
        closeables=(agent_llm, vlm),
        text_topic="agent.response",
        idle_timeout_secs=config.idle_timeout_secs,
    )
    async with WorkflowBuilder() as builder:
        await builder.add_function_group(
            "rag",
            RAGFunctionsConfig(endpoint=config.rag_endpoint),
        )
        rag_group = await builder.get_function_group("rag")
        tools = GuideTools(await rag_group.get_all_functions())
        notice_bridge = NoticeBridge(session)
        vision = StepVision(
            endpoint=session.transport.endpoint,
            vlm=vlm,
            frame_max_age_s=config.frame_max_age_s,
            frame_timeout_s=config.frame_timeout_s,
            vlm_timeout_s=config.vlm_timeout_s,
            system_prompt=str(workflow.runtime.get("vlm_system_prompt", "")),
        )
        agent = WorkflowAgent(
            llm=agent_llm,
            tools=tools,
            workflow=workflow,
            answer_prompt=config.answer_prompt,
        )
        guide = WorkflowGuide(
            workflow=workflow,
            vision=vision,
            agent=agent,
            notice=notice_bridge.speak,
        )

        async def handler(turn: VoiceQuery) -> str:
            notice = notice_bridge.pop(turn.text)
            if notice is not None:
                return notice
            return await guide.handle_query(
                participant_id=turn.participant_id,
                text=turn.text,
            )

        async def participant_left(participant_id: str) -> None:
            await guide.release(participant_id)

        async def participant_joined(participant_id: str) -> None:
            await guide.reset(participant_id)

        monitor_task: asyncio.Task[None] | None = None
        async with session:
            TextMessageInput(session=session, fresh_match=True)
            monitor_task = asyncio.create_task(
                guide.monitor_forever(),
                name="tea-guide-monitor",
            )
            logger.info("tea-making-sample worker starting")
            try:
                await session.run(
                    handler,
                    on_participant_joined=participant_joined,
                    on_participant_left=participant_left,
                    interrupt_on_supersede=True,
                    queue_queries=True,
                )
            finally:
                logger.info("tea-making-sample worker stopping")
                await guide.close()
                if monitor_task is not None:
                    monitor_task.cancel()
                    try:
                        await monitor_task
                    except asyncio.CancelledError:
                        pass


__all__ = ["run_app"]
