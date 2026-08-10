# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Individually selectable state and workflow-management functions."""

import json
from dataclasses import asdict
from typing import Any, Literal

from nat.plugin_api import Builder, FunctionBaseConfig, FunctionInfo, register_function
from pydantic import BaseModel, ConfigDict, Field

from ..applications.manager.runtime import ApplicationOwnership
from ..runtime.scope import current_invocation
from ..runtime.state import SessionStore

Operation = Literal["commit", "start", "advance", "reset", "restart", "status"]


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyRequest(_Request):
    pass


class GuideRequest(_Request):
    scope: Literal["tea_guide"] = Field(description="The interactive tea guide itself.")


class AdvanceRequest(_Request):
    skip: bool = Field(default=False, description="True only when the user explicitly asks to skip.")


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
    application_ownership: Any = Field(exclude=True, repr=False)


@register_function(config_type=WorkflowFunctionConfig)
async def workflow_function(config: WorkflowFunctionConfig, _builder: Builder):
    store: SessionStore = config.store
    application_ownership: ApplicationOwnership = config.application_ownership

    async def commit(request: CommitRequest) -> str:
        call = current_invocation()
        return json.dumps(asdict(store.commit(call.session, request.updates, request.message)), separators=(",", ":"))

    async def start(request: GuideRequest) -> str:
        call = current_invocation()
        call.route_operation = "start"
        result = store.start(call.session)
        application_ownership.capture(call.session, "tea")
        return result

    async def advance(request: AdvanceRequest) -> str:
        call = current_invocation()
        call.route_operation = "advance"
        result = store.advance(call.session, skip=request.skip)
        if not call.session.active:
            application_ownership.release(call.session, "tea")
        return result

    async def reset(request: GuideRequest) -> str:
        call = current_invocation()
        call.route_operation = "reset"
        result = store.reset(call.session)
        application_ownership.release(call.session, "tea")
        return result

    async def restart(request: GuideRequest) -> str:
        call = current_invocation()
        call.route_operation = "restart"
        return store.restart(call.session)

    async def status(request: EmptyRequest) -> str:
        call = current_invocation()
        call.route_operation = "status"
        return store.status(call.session)

    handlers: dict[str, tuple[Any, str]] = {
        "commit": (
            commit,
            "Atomically update supported active-step state according to the supplied state contract.",
        ),
        "start": (start, "Start the idle tea guide. While active this cannot reset or change its step."),
        "advance": (
            advance,
            "Explicitly change this guide's step: false for next/continue/advance; true only for skip.",
        ),
        "reset": (reset, "Explicitly stop or reset this tea guide; not an appliance or timer."),
        "restart": (
            restart,
            "Explicitly start this tea guide over from its first step; not an appliance or timer.",
        ),
        "status": (status, "Report whether this guide is active and its current step."),
    }
    handler, description = handlers[config.operation]
    yield FunctionInfo.from_fn(handler, description=description)


async def add_workflow_functions(
    builder: Builder,
    *,
    store: SessionStore,
    application_ownership: ApplicationOwnership,
) -> None:
    for operation in ("commit", "start", "advance", "reset", "restart", "status"):
        await builder.add_function(
            f"workflow__{operation}",
            WorkflowFunctionConfig(
                operation=operation,
                store=store,
                application_ownership=application_ownership,
            ),
        )


__all__ = ["WorkflowFunctionConfig", "add_workflow_functions"]
