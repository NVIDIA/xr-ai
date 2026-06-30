# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-agnostic agent capability interface.

An :class:`AgentCapability` is a self-contained feature that an agent brain can
compose.  It exposes one or more *brain-local* tools (tools that the brain
executes inline rather than by calling an external MCP server) and implements
their execution logic.

The canonical example is :class:`~xr_ai_capabilities.VisionModule`: it exposes
a ``look_at_current_frame`` tool that turns the camera on, grabs a live frame,
and runs the VLM — without going through any MCP server.

Capability vs. :class:`~xr_ai_capabilities.MCPToolset`:

* ``MCPToolset`` — a remote MCP server; tool calls cross a network boundary.
* ``AgentCapability`` — local code in the brain process; no network hop.

A brain collects tool defs from both and merges them before sending the tool
list to the LLM.  When a tool call arrives, the brain checks capabilities first
(by name) and falls back to the MCP toolset router for everything else.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from xr_ai_models import ToolDef


class AgentCapability(ABC):
    """Base class for brain-local agent capabilities.

    Subclass and implement :meth:`as_tool_defs` and :meth:`execute`.
    """

    @abstractmethod
    def as_tool_defs(self) -> list[ToolDef]:
        """Return the tool definitions this capability contributes to the LLM's
        tool list.  Called once per session setup; the result is merged with the
        definitions from all registered MCP toolsets.
        """

    @abstractmethod
    async def execute(
        self,
        name: str,
        args: dict,
        pid: str,
        *,
        onset_pts_us: int = 0,
        end_pts_us: int = 0,
    ) -> dict:
        """Execute a tool call and return the result as a plain dict.

        Parameters
        ----------
        name:
            The tool name (must be one this capability owns).
        args:
            The arguments from the LLM tool call.
        pid:
            The participant ID for the current turn.
        onset_pts_us / end_pts_us:
            Speech-window timestamps (Unix µs) from the utterance that triggered
            this turn.  Zero when not available (e.g. text input).  Capabilities
            that do time-anchored lookups (live-frame VLM, head-pose) use these
            to pick the frame / pose from when the user was speaking rather than
            from when the LLM started processing.
        """
