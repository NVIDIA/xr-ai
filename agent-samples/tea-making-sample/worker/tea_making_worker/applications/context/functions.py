# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NAT function group for participant-local application context."""

from nat.plugin_api import Builder, FunctionGroup, FunctionGroupBaseConfig, register_function_group
from pydantic import BaseModel, ConfigDict, Field
from xr_ai_nat.events import EventEnvelope

from ...runtime.events import emit
from ...runtime.scope import current_invocation
from ..events import BACKGROUND_FACT
from .models import ContextItem, ContextPublishRequest, ContextQueryRequest, ContextQueryResult
from .store import ApplicationContextStore


class ContextClearRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ContextClearResult(BaseModel):
    removed: int


class ApplicationContextFunctionsConfig(
    FunctionGroupBaseConfig,
    name="voice_application_context",
):
    capacity_per_participant: int = Field(default=64, ge=1, le=1_000)


@register_function_group(config_type=ApplicationContextFunctionsConfig)
async def application_context_functions(config: ApplicationContextFunctionsConfig, _builder: Builder):
    store = ApplicationContextStore(config.capacity_per_participant)
    group = FunctionGroup(config=config)

    async def record(event: EventEnvelope) -> ContextItem:
        fact = BACKGROUND_FACT.payload_from(event)
        item = store.publish(
            event.participant_id,
            ContextPublishRequest(
                producer=event.producer,
                topic=fact.topic,
                summary=fact.summary,
                source_ref=fact.source_ref,
            ),
        )
        emit(
            "context.published",
            participant_id=event.participant_id,
            trace_id=event.correlation_id,
            sequence=item.sequence,
            producer=item.producer,
            topic=item.topic,
            summary=item.summary,
            source_ref=item.source_ref,
        )
        return item

    async def query(request: ContextQueryRequest) -> ContextQueryResult:
        call = current_invocation()
        call.route_operation = "application_context.query"
        items = store.query(call.session.participant_id, request)
        emit(
            "context.queried",
            participant_id=call.session.participant_id,
            trace_id=call.trace_id,
            topics=request.topics,
            sequences=[item.sequence for item in items],
        )
        return ContextQueryResult(items=items)

    async def clear(request: ContextClearRequest) -> ContextClearResult:
        del request
        call = current_invocation()
        removed = store.clear(call.session.participant_id)
        emit(
            "context.cleared",
            participant_id=call.session.participant_id,
            trace_id=call.trace_id,
            removed=removed,
        )
        return ContextClearResult(removed=removed)

    group.add_function("record", record, description="Record one concise background application fact.")
    group.add_function(
        "query",
        query,
        description=(
            "Read recent outputs from background applications only when they are needed to answer "
            "the current request. Select only the needed topics, age window, and item count."
        ),
    )
    group.add_function("clear", clear, description="Clear one participant's ephemeral application context.")
    yield group


__all__ = ["ApplicationContextFunctionsConfig", "ContextClearRequest", "ContextClearResult"]
