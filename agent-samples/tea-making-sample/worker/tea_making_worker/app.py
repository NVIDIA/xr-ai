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
from xr_ai_nat.adapters import as_voice_event_handler
from xr_ai_nat.events import EventDispatcher, EventEnvelope, add_event_handler
from xr_ai_nat.functions.rag import RAGFunctionsConfig
from xr_ai_nat.functions.vision import VisionToolsConfig
from xr_ai_voice import TextMessageInput, VadConfig, VoiceSession
from xr_ai_voicegate import load_voice_gate_config

from .agents import AgentRegistry
from .agents.factory import add_guidance_llm
from .applications.compose import build_applications
from .applications.context import ApplicationContextFunctionsConfig, add_context_query
from .applications.context.functions import ContextClearRequest
from .applications.events import (
    APPLICATION_REQUEST,
    APPLICATION_RESET,
    BACKGROUND_FACT,
    RAW_TRANSCRIPT,
    ApplicationRequest,
)
from .applications.manager.runtime import ApplicationOwnership
from .applications.manager.spec import load_application_catalog
from .applications.output import UserOutputDelivery
from .config import WorkerConfig
from .engine import Coordinator, TriggerRegistry
from .functions import (
    CurrentViewConfig,
    RAGLookupConfig,
    add_clock_functions,
    add_temperature_functions,
    add_workflow_functions,
)
from .runtime.events import emit
from .runtime.state import SessionStore
from .spec import load_workflow


async def run_app(config: WorkerConfig, *, ready_file: Path | None = None) -> None:
    setup_logging("worker", namespace="tea-making-sample")
    models = load_models_config(config.models_config)
    llm = make_llm(models, "agent_llm")
    vlm = make_vlm(models, "vlm")
    workflow = load_workflow(config.workflow_config)
    application_spec = load_application_catalog(config.applications_config)
    application_ownership = ApplicationOwnership(application_spec)
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

        def observe_event(event, subscribers) -> None:
            payload = event.payload
            if event.topic == RAW_TRANSCRIPT.name:
                payload = {"characters": len(str(payload["text"]))}
            emit(
                "application.event",
                participant_id=event.participant_id,
                topic=event.topic,
                producer=event.producer,
                event_id=event.event_id,
                correlation_id=event.correlation_id,
                parent_event_id=event.parent_event_id,
                subscribers=subscribers,
                payload=payload,
            )

        events = EventDispatcher(observer=observe_event)
        output = UserOutputDelivery(events, session)
        await output.build(builder)
        vision_config = VisionToolsConfig(
            endpoint=session.transport.endpoint,
            vlm=vlm,
            frame_max_age_s=config.frame_max_age_s,
            frame_timeout_s=config.frame_timeout_s,
        )
        vision_group = await builder.add_function_group("vision", vision_config)
        vision_functions = await vision_group.get_all_functions()
        current_view = await builder.add_function(
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
        await add_workflow_functions(
            builder,
            store=store,
            application_ownership=application_ownership,
        )
        context_group = await builder.add_function_group(
            "context_store",
            ApplicationContextFunctionsConfig(),
        )
        context_functions = await context_group.get_all_functions()
        events.subscribe(
            BACKGROUND_FACT,
            subscriber_id="context.recorder",
            function=context_functions["context_store__record"],
        )

        async def reset_context(event: EventEnvelope) -> None:
            APPLICATION_RESET.payload_from(event)
            await context_functions["context_store__clear"].ainvoke(ContextClearRequest())

        context_reset = await add_event_handler(
            builder,
            name="application__reset_context",
            handler=reset_context,
            description="Clear participant context when applications reset.",
        )
        events.subscribe(
            APPLICATION_RESET,
            subscriber_id="context.recorder",
            function=context_reset,
        )
        await add_context_query(builder, context_functions["context_store__query"])
        llm_ref = await add_guidance_llm(builder, llm)
        await agents.build(builder, llm_ref)
        applications = await build_applications(
            builder,
            llm_ref=llm_ref,
            spec=application_spec,
            ownership=application_ownership,
            tea=agents,
            current_view=current_view,
            events=events,
            output=output,
            store=store,
        )
        triggers = TriggerRegistry(workflow)
        await triggers.build(builder)
        coordinator = Coordinator(
            store=store,
            agents=agents,
            manager=applications.manager,
            events=events,
            output=output,
            reset_subscriber_ids=applications.background_ids | {"context.recorder"},
            triggers=triggers,
        )
        TextMessageInput(session=session, fresh_match=True)

        handler = as_voice_event_handler(
            events,
            APPLICATION_REQUEST,
            payload=lambda query: ApplicationRequest(text=query.text),
            subscribers={"application.manager"},
        )

        async def participant_left(participant_id: str) -> None:
            vision_config.release(participant_id)
            await coordinator.participant_left(participant_id)

        async def run_voice() -> None:
            try:
                await session.run(
                    handler,
                    transcription_observer=coordinator.handle_transcription,
                    on_participant_joined=coordinator.participant_joined,
                    on_participant_left=participant_left,
                    interrupt_on_supersede=True,
                    queue_queries=True,
                )
            finally:
                coordinator.stop()
                await applications.close_periodic_sources()

        logger.info("tea-making-sample starting")
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(run_voice(), name="tea-guide-voice")
            tasks.create_task(coordinator.monitor(), name="tea-guide-monitor")
            tasks.create_task(
                applications.run_periodic_sources(),
                name="tea-guide-background-triggers",
            )
        logger.info("tea-making-sample stopped")


__all__ = ["run_app"]
