# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Individually selectable state and workflow-management functions."""

import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from typing import Any, Literal

from nat.plugin_api import Builder, FunctionBaseConfig, FunctionInfo, register_function
from pydantic import BaseModel, ConfigDict, Field

from ..runtime.events import emit
from ..runtime.scope import current_invocation
from ..runtime.state import Session, SessionStore

VoiceAnswer = Callable[[Session, str, str], Awaitable[str]]
Operation = Literal["commit", "start", "advance", "reset", "status", "ask_step", "ask_general"]


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyRequest(_Request):
    pass


class AdvanceRequest(_Request):
    skip: bool = Field(default=False, description="True only when the user explicitly asks to skip.")


class AskStepRequest(_Request):
    question: str = Field(min_length=1, max_length=500)


class CommitRequest(_Request):
    updates: dict[str, Any] = Field(
        default_factory=dict,
        description="Supported active-step fields only; omit unchanged fields.",
    )
    message: str = Field(
        default="",
        max_length=240,
        description=(
            "Brief natural spoken update for a real non-completing state change; empty otherwise."
        ),
    )


class WorkflowFunctionConfig(FunctionBaseConfig, name="tea_guidance_workflow_function"):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    operation: Operation
    store: Any = Field(exclude=True, repr=False)
    answer_step: Any = Field(exclude=True, repr=False)
    answer_general: Any = Field(exclude=True, repr=False)


@register_function(config_type=WorkflowFunctionConfig)
async def workflow_function(config: WorkflowFunctionConfig, _builder: Builder):
    store: SessionStore = config.store
    answer_step: VoiceAnswer = config.answer_step
    answer_general: VoiceAnswer = config.answer_general

    async def commit(request: CommitRequest) -> str:
        call = current_invocation()
        return json.dumps(asdict(store.commit(call.session, request.updates, request.message)), separators=(",", ":"))

    async def start(request: EmptyRequest) -> str:
        return store.start(current_invocation().session)

    async def advance(request: AdvanceRequest) -> str:
        return store.advance(current_invocation().session, skip=request.skip)

    async def reset(request: EmptyRequest) -> str:
        return store.reset(current_invocation().session)

    async def status(request: EmptyRequest) -> str:
        return store.status(current_invocation().session)

    async def ask_step(request: AskStepRequest) -> str:
        call = current_invocation()
        emit(
            "voice.delegate",
            participant_id=call.session.participant_id,
            step=call.session.step_id,
            trace_id=call.trace_id,
            target="step",
            question=request.question,
        )
        return await answer_step(call.session, request.question, call.trace_id)

    async def ask_general(request: AskStepRequest) -> str:
        call = current_invocation()
        emit(
            "voice.delegate",
            participant_id=call.session.participant_id,
            step=call.session.step_id,
            trace_id=call.trace_id,
            target="general",
            question=request.question,
        )
        return await answer_general(call.session, request.question, call.trace_id)

    handlers: dict[str, tuple[Any, str]] = {
        "commit": (
            commit,
            "Atomically update supported active-step state according to the supplied state contract.",
        ),
        "start": (start, "Use only for an explicit request to start guidance."),
        "advance": (
            advance,
            "Handle explicit step changes: skip false for next/continue/advance; true only for skip.",
        ),
        "reset": (reset, "Use only for explicit stop, cancel, or reset commands."),
        "status": (status, "Use only when asked which workflow step is active or whether guidance is running."),
        "ask_step": (
            ask_step,
            "Active workflow only: current-step questions, readings, timers, help, or action reports.",
        ),
        "ask_general": (
            ask_general,
            "General tea knowledge or visual questions; works while idle or active and never manages workflow.",
        ),
    }
    handler, description = handlers[config.operation]
    yield FunctionInfo.from_fn(handler, description=description)


async def add_workflow_functions(
    builder: Builder,
    *,
    store: SessionStore,
    answer_step: VoiceAnswer,
    answer_general: VoiceAnswer,
) -> None:
    for operation in ("commit", "start", "advance", "reset", "status", "ask_step", "ask_general"):
        await builder.add_function(
            f"workflow__{operation}",
            WorkflowFunctionConfig(
                operation=operation,
                store=store,
                answer_step=answer_step,
                answer_general=answer_general,
            ),
        )


__all__ = ["WorkflowFunctionConfig", "add_workflow_functions"]
