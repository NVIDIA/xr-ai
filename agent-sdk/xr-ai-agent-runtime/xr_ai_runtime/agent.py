# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent resource scopes and their exposed native tools."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from xr_ai_tools import AsyncTool, Tool

if TYPE_CHECKING:
    from .runtime import RuntimeContext


class Agent:
    """A runtime-managed resource scope that exposes ordinary native tools."""

    def __init__(
        self,
        tools: Iterable[Tool[Any, Any] | AsyncTool[Any, Any]] = (),
    ) -> None:
        owned = tuple(tools)
        names: set[str] = set()
        for tool in owned:
            if not isinstance(tool, (Tool, AsyncTool)):
                raise TypeError("agents may expose only Tool or AsyncTool instances")
            if tool.name in names:
                raise ValueError(f"duplicate agent tool name: {tool.name}")
            names.add(tool.name)
        self._tools = owned

    @property
    def tools(self) -> tuple[Tool[Any, Any] | AsyncTool[Any, Any], ...]:
        """Return the existing native tools exposed by this agent."""

        return self._tools

    @asynccontextmanager
    async def lifespan(self, _ctx: RuntimeContext) -> AsyncIterator[None]:
        """Optionally acquire resources for this agent's runtime lifetime."""

        yield


__all__ = ["Agent"]
