# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NAT function adapter for typed event subscribers."""

from collections.abc import Awaitable, Callable
from typing import Any

from nat.builder.function import Function
from nat.plugin_api import Builder, FunctionBaseConfig, FunctionInfo, register_function
from pydantic import ConfigDict, Field

from .models import EventEnvelope

EventHandler = Callable[[EventEnvelope], Awaitable[Any]]


class EventHandlerConfig(FunctionBaseConfig, name="xr_event_handler"):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    handler: EventHandler = Field(exclude=True, repr=False)
    function_description: str = "Handle a typed application event."


@register_function(config_type=EventHandlerConfig)
async def event_handler(config: EventHandlerConfig, _builder: Builder):
    async def invoke(event: EventEnvelope) -> Any:
        return await config.handler(event)

    yield FunctionInfo.from_fn(invoke, description=config.function_description)


async def add_event_handler(
    builder: Builder,
    *,
    name: str,
    handler: EventHandler,
    description: str = "Handle a typed application event.",
) -> Function:
    """Register an event consumer as a native NAT function."""
    return await builder.add_function(
        name,
        EventHandlerConfig(handler=handler, function_description=description),
    )


__all__ = ["EventHandler", "EventHandlerConfig", "add_event_handler"]
