# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed NAT boundary shared by foreground applications."""

from collections.abc import Awaitable, Callable

from nat.builder.function import Function
from nat.plugin_api import Builder, FunctionBaseConfig, FunctionInfo, register_function
from pydantic import BaseModel, ConfigDict, Field

from ...runtime.scope import current_invocation
from ...runtime.state import Session

TurnHandler = Callable[[Session, str, str], Awaitable[str]]


class ApplicationTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: str


class ApplicationTurnConfig(FunctionBaseConfig, name="voice_application_turn"):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    handler: TurnHandler = Field(exclude=True, repr=False)
    function_description: str


@register_function(config_type=ApplicationTurnConfig)
async def application_turn(config: ApplicationTurnConfig, _builder: Builder):
    async def invoke(turn: ApplicationTurn) -> str:
        call = current_invocation()
        return await config.handler(call.session, turn.request, call.trace_id)

    yield FunctionInfo.from_fn(invoke, description=config.function_description)


async def add_application_turn(
    builder: Builder,
    *,
    name: str,
    description: str,
    handler: TurnHandler,
) -> Function:
    return await builder.add_function(
        name,
        ApplicationTurnConfig(handler=handler, function_description=description),
    )


__all__ = ["ApplicationTurn", "add_application_turn"]
