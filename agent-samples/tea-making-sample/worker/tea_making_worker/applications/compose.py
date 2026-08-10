# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose the sample applications from generic routed NAT functions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from nat.builder.function import Function
from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugin_api import LLMRef
from xr_ai_nat.events import (
    EventDispatcher,
    EventEnvelope,
    PeriodicEventSource,
    add_event_handler,
)

from ..agents import AgentRegistry
from ..runtime.events import emit
from ..runtime.scope import current_invocation, invocation_scope
from ..runtime.state import SessionStore
from .change_watch import ChangeWatchApplication
from .controls import background_function_specs
from .events import (
    APPLICATION_REQUEST,
    APPLICATION_RESET,
    CLOCK_TICK,
    RAW_TRANSCRIPT,
    ClockTick,
    OutputTiming,
    UserOutput,
)
from .manager.functions import add_application_manager_functions, application_manager_status_function
from .manager.registry import ApplicationManager
from .manager.runtime import ApplicationOwnership
from .manager.spec import ApplicationCatalog
from .manager.turn import ApplicationTurn, add_application_turn
from .manager.types import InvocationEffect, RoutedFunction
from .output import UserOutputDelivery
from .transcript import TranscriptApplication
from .video_log import VideoLogApplication


@dataclass(frozen=True, slots=True)
class ComposedApplications:
    manager: ApplicationManager
    events: EventDispatcher
    output: UserOutputDelivery
    background_ids: frozenset[str]
    change_watch: ChangeWatchApplication
    transcript: TranscriptApplication
    video_log: VideoLogApplication
    periodic_sources: tuple[PeriodicEventSource[ClockTick], ...]

    async def run_periodic_sources(self) -> None:
        try:
            await asyncio.gather(*(source.run() for source in self.periodic_sources))
        finally:
            await self.close_periodic_sources()

    async def close_periodic_sources(self) -> None:
        await asyncio.gather(*(source.close() for source in self.periodic_sources))


async def build_applications(
    builder: WorkflowBuilder,
    *,
    llm_ref: LLMRef,
    spec: ApplicationCatalog,
    ownership: ApplicationOwnership,
    tea: AgentRegistry,
    current_view: Function,
    events: EventDispatcher,
    output: UserOutputDelivery,
    store: SessionStore,
) -> ComposedApplications:
    manager = ApplicationManager(spec, ownership)
    change_spec = spec.application("change_watch")
    transcript_spec = spec.application("transcript")
    video_log_spec = spec.application("video_log")
    change_clock = _periodic_source(
        events,
        change_spec.id,
        float(change_spec.settings.get("interval_s", 2)),
    )
    transcript_clock = _periodic_source(
        events,
        transcript_spec.id,
        float(transcript_spec.settings.get("summary_interval_s", 120)),
        immediate=False,
    )
    video_log_clock = _periodic_source(
        events,
        video_log_spec.id,
        float(video_log_spec.settings.get("interval_s", 2)),
    )
    change_watch = ChangeWatchApplication(
        change_spec,
        ownership,
        events,
        periodic=change_clock,
    )
    transcript = TranscriptApplication(
        transcript_spec,
        ownership,
        events,
        periodic=transcript_clock,
    )
    video_log = VideoLogApplication(
        video_log_spec,
        ownership,
        events,
        periodic=video_log_clock,
    )
    await change_watch.build(builder, llm_ref, current_view)
    await transcript.build(builder, llm_ref)
    await video_log.build(builder, llm_ref, current_view)
    await add_application_manager_functions(builder, ownership)
    root_functions = root_function_specs(spec)
    tea_turn = await add_application_turn(
        builder,
        name="application__tea_turn",
        description=spec.application("tea").route,
        handler=tea.route,
    )
    manager.register_foreground("tea", tea_turn)
    await manager.build(builder, llm_ref, root_functions)
    manager_turn = await add_event_handler(
        builder,
        name="application__request",
        handler=lambda event: _handle_application_request(manager, output, store, event),
        description="Deliver a participant request to the current foreground NAT application.",
    )
    events.subscribe(APPLICATION_REQUEST, subscriber_id="application.manager", function=manager_turn)
    backgrounds = (change_watch, transcript, video_log)
    for application in backgrounds:
        await _subscribe_background(builder, events, store, application)
    return ComposedApplications(
        manager,
        events,
        output,
        frozenset(application.app_id for application in backgrounds),
        change_watch,
        transcript,
        video_log,
        (change_clock, transcript_clock, video_log_clock),
    )


def _periodic_source(
    events: EventDispatcher,
    application_id: str,
    interval_s: float,
    *,
    immediate: bool = True,
) -> PeriodicEventSource[ClockTick]:
    return PeriodicEventSource(
        events,
        CLOCK_TICK,
        payload=lambda _participant_id: ClockTick(),
        producer=f"{application_id}.clock",
        subscriber_id=application_id,
        interval_s=interval_s,
        immediate=immediate,
    )


async def _handle_application_request(
    manager: ApplicationManager,
    output: UserOutputDelivery,
    store: SessionStore,
    event: EventEnvelope,
) -> str:
    request = APPLICATION_REQUEST.payload_from(event)
    session = store.get(event.participant_id)
    async with session.lock:
        with invocation_scope(session, event.correlation_id):
            result = await manager.function.ainvoke(
                ApplicationTurn(request=request.text),
                to_type=str,
            )
            result = await output.publish(
                event.participant_id,
                "application.manager",
                UserOutput(text=result, timing=OutputTiming.REPLY),
                correlation_id=event.correlation_id,
                parent_event_id=event.event_id,
            )
    emit(
        "application.request.complete",
        participant_id=event.participant_id,
        trace_id=event.correlation_id,
        producer=event.producer,
        response=result,
    )
    return result


async def _subscribe_background(
    builder: WorkflowBuilder,
    events: EventDispatcher,
    store: SessionStore,
    application: ChangeWatchApplication | TranscriptApplication | VideoLogApplication,
) -> None:
    async def transcript(event: EventEnvelope) -> None:
        payload = RAW_TRANSCRIPT.payload_from(event)
        call = current_invocation()
        await application.on_transcription(call.session, payload.text, call.trace_id)

    async def tick(event: EventEnvelope) -> None:
        CLOCK_TICK.payload_from(event)
        await application.tick(store.get(event.participant_id), event.correlation_id)

    async def reset(event: EventEnvelope) -> None:
        APPLICATION_RESET.payload_from(event)
        await application.release(current_invocation().session)

    handlers = (
        (RAW_TRANSCRIPT, "transcript", transcript),
        (CLOCK_TICK, "tick", tick),
        (APPLICATION_RESET, "reset", reset),
    )
    for topic, suffix, handler in handlers:
        function = await add_event_handler(
            builder,
            name=f"application__{application.app_id}_{suffix}",
            handler=handler,
            description=f"Deliver {topic.name} to {application.spec.title}.",
        )
        events.subscribe(topic, subscriber_id=application.app_id, function=function)


def root_function_specs(spec: ApplicationCatalog) -> tuple[RoutedFunction, ...]:
    background = tuple(
        function
        for app in spec.applications.values()
        if app.mode == "background"
        for function in background_function_specs(app)
    )
    return (
        RoutedFunction("current_view", spec.capabilities["current_view"]),
        RoutedFunction("rag_lookup", spec.capabilities["rag_lookup"]),
        RoutedFunction(
            "application_context__query",
            spec.capabilities["application_context__query"],
        ),
        RoutedFunction(
            "workflow__start",
            spec.application("tea").route,
            effect=InvocationEffect.FOREGROUND,
            return_direct=True,
        ),
        application_manager_status_function(),
        *background,
    )


__all__ = ["ComposedApplications", "build_applications", "root_function_specs"]
