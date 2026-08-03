# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose live monitoring, explicit task controls, guidance, and voice I/O."""

from __future__ import annotations

import time
from pathlib import Path

from loguru import logger
from nat.builder.workflow_builder import WorkflowBuilder
from xr_ai_hub import DataMessage
from xr_ai_logging import setup_logging
from xr_ai_models import load_models_config, make_llm, make_stt, make_tts, make_vlm
from xr_ai_nat.adapters import as_voice_handler
from xr_ai_nat.functions.vision import StreamingVisionConfig
from xr_ai_nat.llm import ModelsLLMConfig
from xr_ai_voice import TextMessageInput, VadConfig, VoiceSession
from xr_ai_voicegate import load_voice_gate_config

from .agent import TaskGuideAgentConfig
from .config import WorkerConfig
from .models import TaskGuideRequest, TaskStatusResult
from .task_functions import (
    TaskControlFunctionsConfig,
    TaskKnowledgeFunctionsConfig,
    TaskStateFunctionsConfig,
)
from .task_store import TaskStore
from .workflow import TaskGuideWorkflowConfig, format_task_status

_OUTPUT_TOPIC = "agent.response"


async def run_app(config: WorkerConfig, *, ready_file: Path | None = None) -> None:
    setup_logging("worker")
    models = load_models_config(config.models_config)
    llm = make_llm(models, "guide_llm")
    vlm = make_vlm(models, "vlm")
    stt = make_stt(models, "stt")
    tts = make_tts(models, "tts")
    store = TaskStore(config.task_directory)
    logger.info(
        "task guide loaded task={} title={!r} steps={}",
        store.definition.id,
        store.definition.title,
        len(store.steps),
    )
    voice = VoiceSession(
        stt=stt,
        tts=tts,
        vad=VadConfig(
            silence_duration=config.silence_duration,
            min_speech=config.min_speech,
            silero_threshold=config.silero_threshold,
        ),
        voice_gate=load_voice_gate_config(config.voice_gate_yaml),
        probes={"guide-llm": llm.health, "vlm": vlm.health},
        ready_file=ready_file,
        closeables=(llm, vlm),
        text_topic=_OUTPUT_TOPIC,
        idle_timeout_secs=config.idle_timeout_secs,
    )

    async with WorkflowBuilder() as builder, voice:
        vision_config = StreamingVisionConfig(
            endpoint=voice.transport.endpoint,
            vlm=vlm,
            system_prompt=config.caption_prompt,
            frame_max_age_s=config.frame_max_age_s,
            frame_timeout_s=config.frame_timeout_s,
        )
        await builder.add_function("streaming_vision", vision_config)
        await builder.add_function_group("task_state", TaskStateFunctionsConfig(store=store))
        await builder.add_function_group("task_control", TaskControlFunctionsConfig(store=store))
        await builder.add_function_group("task_knowledge", TaskKnowledgeFunctionsConfig(store=store))
        await builder.add_llm(
            "guide_llm",
            ModelsLLMConfig(
                service=llm,
                model_name="visual-task-guide",
                temperature=0.0,
                max_tokens=128,
            ),
        )
        await builder.add_function("task_guide_agent", TaskGuideAgentConfig())
        workflow = await builder.add_function(
            "task_guide_workflow",
            TaskGuideWorkflowConfig(),
        )
        control_group = await builder.get_function_group("task_control")
        control_functions = await control_group.get_all_functions()
        reset_task = control_functions["task_control__reset_task"]

        async def participant_joined(participant_id: str) -> None:
            status = TaskStatusResult.model_validate(
                await reset_task.ainvoke({"participant_id": participant_id})
            )
            await voice.transport.endpoint.send_return_data(
                DataMessage(
                    participant_id=participant_id,
                    topic=_OUTPUT_TOPIC,
                    pts_us=time.time_ns() // 1_000,
                    data=format_task_status(status).encode("utf-8"),
                )
            )

        async def participant_left(participant_id: str) -> None:
            vision_config.release(participant_id)

        handler = as_voice_handler(
            workflow,
            request=lambda query: TaskGuideRequest(
                participant_id=query.participant_id,
                text=query.text,
            ),
            response=lambda result: result.response,
        )
        TextMessageInput(session=voice, fresh_match=True)
        logger.info("visual task guide ready; say 'start task'")
        try:
            await voice.run(
                handler,
                on_participant_joined=participant_joined,
                on_participant_left=participant_left,
                interrupt_on_supersede=True,
                queue_queries=True,
            )
        finally:
            for participant_id in voice.transport.endpoint.connected_participants:
                vision_config.release(participant_id)


__all__ = ["run_app"]
