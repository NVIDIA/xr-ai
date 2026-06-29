# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP tool-set routing for agent brains.

``MCPToolset`` pairs an MCP client with the set of tool names it owns.  A list
of toolsets replaces the per-sample hardcoded ``frozenset`` + ``if tool in
_X_TOOLS`` dispatch pattern: a brain accepts a ``list[MCPToolset]`` and routes
tool calls automatically with :func:`route_tool`.

Usage::

    brain = MyBrain(
        toolsets=[
            MCPToolset(oxr,    _OXR_TOOLS),    # explicit claim
            MCPToolset(vec,    _VEC_TOOLS),
            MCPToolset(render),                # catch-all (tools=None)
        ],
    )

    # in the agentic loop:
    toolset = route_tool(brain.toolsets, tool_name)
    result  = await toolset.call_tool(tool_name, args)

``collect_tool_defs`` queries each client once and converts the results to the
``ToolDef`` objects that :class:`~xr_ai_models.LLMService` understands.

Neither class imports FastMCP — the client is typed by the
:class:`McpClientProtocol` structural protocol, so any object that has
``list_tools`` and ``call_tool`` methods satisfies it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from loguru import logger
from xr_ai_models import ToolDef


@runtime_checkable
class McpClientProtocol(Protocol):
    """Structural protocol for any MCP client.

    Satisfied by ``fastmcp.Client`` (and any compatible implementation)
    without importing FastMCP as a hard dependency.
    """

    async def list_tools(self) -> list[Any]: ...

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None,
    ) -> Any: ...


@dataclass
class MCPToolset:
    """An MCP client paired with the tool names it owns.

    Parameters
    ----------
    client:
        Any MCP client satisfying :class:`McpClientProtocol`.
    tools:
        The set of tool names this client serves.  ``None`` means
        *catch-all*: this toolset answers any tool not claimed by an
        earlier entry in the list.  Put the catch-all last.
    """

    client: McpClientProtocol
    tools: frozenset[str] | None = field(default=None)

    async def call_tool(
        self, name: str, args: dict[str, Any],
    ) -> dict[str, Any] | list[Any] | None:
        """Call ``name`` on the underlying MCP client and unwrap the result.

        FastMCP wraps results in an object with a ``data`` or
        ``structured_content`` attribute.  This unwraps either form so callers
        get a plain Python object without depending on FastMCP directly.
        """
        res = await self.client.call_tool(name, args)
        if hasattr(res, "data") and res.data is not None:
            return res.data
        return getattr(res, "structured_content", None)


def route_tool(
    toolsets: list[MCPToolset], name: str,
) -> MCPToolset | None:
    """Return the toolset that owns ``name``.

    Searches ``toolsets`` in order.  The first entry whose ``tools`` set
    contains ``name`` wins.  A catch-all entry (``tools=None``) is
    returned if no explicit owner is found.  Returns ``None`` when
    ``toolsets`` is empty or no entry matches.
    """
    catch_all: MCPToolset | None = None
    for ts in toolsets:
        if ts.tools is None:
            catch_all = ts
        elif name in ts.tools:
            return ts
    return catch_all


async def collect_tool_defs(
    toolsets: list[MCPToolset],
    *,
    exclude: frozenset[str] = frozenset(),
) -> list[ToolDef]:
    """Query every toolset client and return the union as :class:`ToolDef` objects.

    Parameters
    ----------
    toolsets:
        Toolsets to query.  Discovery errors on any single client are
        logged as warnings and skipped (the rest still succeed).
    exclude:
        Tool names to omit — typically worker-managed tools that should
        not appear in the model's tool list (e.g. ``{"start_xr"}``).

    Tool ordering follows the toolset order, then the server's own order
    within each toolset.  Duplicates (same name from multiple clients)
    are silently deduplicated: the first occurrence wins.
    """
    result: list[ToolDef] = []
    seen: set[str] = set()

    for ts in toolsets:
        try:
            mcp_tools = await ts.client.list_tools()
        except Exception as exc:
            logger.warning("tool discovery failed: {}", exc)
            continue

        for t in mcp_tools:
            name: str = t.name
            if name in exclude or name in seen:
                continue
            # Respect explicit ownership: skip tools not in this toolset's set.
            if ts.tools is not None and name not in ts.tools:
                continue
            seen.add(name)
            schema = (
                getattr(t, "inputSchema", None)
                or {"type": "object", "properties": {}}
            )
            result.append(ToolDef(
                name=name,
                description=(t.description or "").strip(),
                parameters=schema,
            ))

    return result
