# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Root-visible desktop management NAT functions."""

from typing import Any

from nat.plugin_api import Builder, FunctionBaseConfig, FunctionInfo, register_function
from pydantic import BaseModel, ConfigDict, Field

from ..runtime.scope import current_invocation
from .runtime import DesktopRuntime
from .types import RoutedFunction


class DesktopStatusRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DesktopStatusConfig(FunctionBaseConfig, name="voice_desktop_status"):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    runtime: Any = Field(exclude=True, repr=False)


@register_function(config_type=DesktopStatusConfig)
async def desktop_status(config: DesktopStatusConfig, _builder: Builder):
    runtime: DesktopRuntime = config.runtime

    async def status(request: DesktopStatusRequest) -> str:
        call = current_invocation()
        call.route_operation = "desktop.status"
        return runtime.status(call.session)

    yield FunctionInfo.from_fn(
        status,
        description=(
            "Report the active foreground and running background applications only when the user "
            "explicitly asks which applications are active or running."
        ),
    )


async def add_desktop_functions(builder: Builder, runtime: DesktopRuntime) -> tuple[RoutedFunction, ...]:
    function = desktop_status_function()
    await builder.add_function(function.name, DesktopStatusConfig(runtime=runtime))
    return (function,)


def desktop_status_function() -> RoutedFunction:
    return RoutedFunction(
        name="desktop__status",
        route="running applications status",
        return_direct=True,
    )


__all__ = ["add_desktop_functions", "desktop_status_function"]
