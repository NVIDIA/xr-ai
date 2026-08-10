# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-selectable alias for the context group's read function."""

from typing import Any

from nat.builder.function import Function
from nat.plugin_api import Builder, FunctionBaseConfig, FunctionInfo, register_function
from pydantic import ConfigDict, Field

from .models import ContextQueryRequest, ContextQueryResult


class ContextQueryConfig(FunctionBaseConfig, name="voice_application_context_query"):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: Any = Field(exclude=True, repr=False)


@register_function(config_type=ContextQueryConfig)
async def context_query(config: ContextQueryConfig, _builder: Builder):
    source: Function = config.source

    async def query(request: ContextQueryRequest) -> ContextQueryResult:
        return await source.ainvoke(request, to_type=ContextQueryResult)

    yield FunctionInfo.from_fn(
        query,
        description=(
            "Read recent outputs from background applications only when they are needed to answer "
            "the current request. Select only the needed topics, age window, and item count."
        ),
    )


async def add_context_query(builder: Builder, source: Function) -> Function:
    return await builder.add_function(
        "application_context__query",
        ContextQueryConfig(source=source),
    )


__all__ = ["ContextQueryConfig", "add_context_query"]
