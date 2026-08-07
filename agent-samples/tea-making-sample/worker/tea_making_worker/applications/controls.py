# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate root-routable NAT controls for background applications."""

from typing import Any, Literal, Protocol

from nat.plugin_api import Builder, FunctionBaseConfig, FunctionInfo, register_function
from pydantic import BaseModel, ConfigDict, Field

from ..desktop.spec import ApplicationSpec
from ..desktop.types import FunctionEffect, RoutedFunction
from ..runtime.scope import current_invocation
from ..runtime.state import Session


class BackgroundController(Protocol):
    spec: ApplicationSpec

    async def start(self, session: Session, instruction: str = "") -> str: ...

    async def stop(self, session: Session) -> str: ...

    async def status(self, session: Session) -> str: ...


class BackgroundControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BackgroundStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(
        default="",
        max_length=240,
        description="Optional concise instruction describing what the background application should do.",
    )


class BackgroundControlConfig(FunctionBaseConfig, name="voice_desktop_background_control"):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    operation: Literal["start", "stop", "status"]
    controller: Any = Field(exclude=True, repr=False)


@register_function(config_type=BackgroundControlConfig)
async def background_control(config: BackgroundControlConfig, _builder: Builder):
    controller: BackgroundController = config.controller

    async def start(request: BackgroundStartRequest) -> str:
        call = current_invocation()
        call.route_operation = f"{controller.spec.id}.start"
        return await controller.start(call.session, request.instruction.strip())

    async def invoke(request: BackgroundControlRequest) -> str:
        call = current_invocation()
        call.route_operation = f"{controller.spec.id}.{config.operation}"
        handler = getattr(controller, config.operation)
        return await handler(call.session)

    descriptions = {
        "start": (
            f"Start {controller.spec.title} in the background without changing the foreground. "
            f"Route: {controller.spec.route}. Pass the user's requested focus as instruction."
        ),
        "stop": f"Stop the background {controller.spec.title}.",
        "status": f"Report whether the background {controller.spec.title} is running.",
    }
    handler = start if config.operation == "start" else invoke
    yield FunctionInfo.from_fn(handler, description=descriptions[config.operation])


async def add_background_controls(
    builder: Builder,
    controller: BackgroundController,
) -> tuple[RoutedFunction, ...]:
    functions = background_function_specs(controller.spec)
    for operation, function in zip(("start", "stop", "status"), functions, strict=True):
        await builder.add_function(
            function.name,
            BackgroundControlConfig(operation=operation, controller=controller),
        )
    return functions


def background_function_specs(spec: ApplicationSpec) -> tuple[RoutedFunction, ...]:
    routes = {
        "start": spec.route,
        "stop": f"stop {spec.id}",
        "status": f"{spec.id} status",
    }
    return tuple(
        RoutedFunction(
            name=f"{spec.id}__{operation}",
            route=routes[operation],
            effect=FunctionEffect.BACKGROUND,
            return_direct=True,
        )
        for operation in ("start", "stop", "status")
    )


__all__ = [
    "BackgroundControlRequest",
    "BackgroundStartRequest",
    "BackgroundController",
    "add_background_controls",
    "background_function_specs",
]
