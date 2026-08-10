# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Root-visible application-manager status NAT function."""

from typing import Any

from nat.plugin_api import Builder, FunctionBaseConfig, FunctionInfo, register_function
from pydantic import BaseModel, ConfigDict, Field

from ...runtime.scope import current_invocation
from .runtime import ApplicationOwnership
from .types import RoutedFunction


class ApplicationManagerStatusRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ApplicationManagerStatusConfig(FunctionBaseConfig, name="voice_application_status"):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    runtime: Any = Field(exclude=True, repr=False)


@register_function(config_type=ApplicationManagerStatusConfig)
async def application_manager_status(config: ApplicationManagerStatusConfig, _builder: Builder):
    ownership: ApplicationOwnership = config.runtime

    async def status(request: ApplicationManagerStatusRequest) -> str:
        call = current_invocation()
        call.route_operation = "application_manager.status"
        return ownership.status(call.session)

    yield FunctionInfo.from_fn(
        status,
        description=(
            "Report the active foreground and running background applications only when the user "
            "explicitly asks which applications are active or running."
        ),
    )


async def add_application_manager_functions(
    builder: Builder,
    ownership: ApplicationOwnership,
) -> tuple[RoutedFunction, ...]:
    function = application_manager_status_function()
    await builder.add_function(function.name, ApplicationManagerStatusConfig(runtime=ownership))
    return (function,)


def application_manager_status_function() -> RoutedFunction:
    return RoutedFunction(
        name="application_manager__status",
        route="running applications status",
        return_direct=True,
    )


__all__ = ["add_application_manager_functions", "application_manager_status_function"]
