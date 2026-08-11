# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-neutral agent runners exposed through native tools."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

from pydantic import BaseModel

from .tools import Tool

RunnerRequestT = TypeVar("RunnerRequestT", contravariant=True)
RunnerResultT = TypeVar("RunnerResultT", covariant=True)
ToolRequestT = TypeVar("ToolRequestT", bound=BaseModel)
ToolResultT = TypeVar("ToolResultT", bound=BaseModel)


class AgentRunner(Protocol[RunnerRequestT, RunnerResultT]):
    """An application-owned agent implementation that completes one asynchronous turn."""

    async def run(self, request: RunnerRequestT) -> RunnerResultT:
        """Run one turn and return the implementation-specific result."""
        raise NotImplementedError


def as_agent_tool(
    *,
    name: str,
    description: str,
    agent: AgentRunner[RunnerRequestT, RunnerResultT],
    request_model: type[ToolRequestT],
    result_model: type[ToolResultT],
    request: Callable[[ToolRequestT], RunnerRequestT],
    response: Callable[[RunnerResultT], ToolResultT],
    return_direct: bool = False,
) -> Tool[ToolRequestT, ToolResultT]:
    """Expose any ``AgentRunner`` through the same ``Tool`` interface as capabilities."""

    async def invoke(value: ToolRequestT) -> ToolResultT:
        return response(await agent.run(request(value)))

    return Tool(
        name,
        description,
        request_model,
        result_model,
        invoke,
        return_direct=return_direct,
    )


__all__ = ["AgentRunner", "as_agent_tool"]
