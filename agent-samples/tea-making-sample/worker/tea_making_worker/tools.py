# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool definitions available to workflow step and answer agents."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from nat.builder.function import Function
from pydantic import BaseModel
from xr_ai_models import ToolDef

ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class AgentTool:
    """A tool definition paired with its handler and effect capability."""

    definition: ToolDef
    handler: ToolHandler
    read_only: bool = False

    @property
    def name(self) -> str:
        return self.definition.name


class ToolCatalog:
    """Select and invoke tools without embedding tool names in an agent loop."""

    def __init__(self, tools: Iterable[AgentTool] = ()) -> None:
        self._tools: dict[str, AgentTool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate agent tool: {tool.name}")
            self._tools[tool.name] = tool

    def select(
        self,
        names: Iterable[str] | None = None,
        *,
        read_only_only: bool = False,
    ) -> list[AgentTool]:
        selected = (
            list(self._tools.values())
            if names is None
            else [self._tools[name] for name in names if name in self._tools]
        )
        if read_only_only:
            selected = [tool for tool in selected if tool.read_only]
        return selected

    def missing(self, names: Iterable[str]) -> set[str]:
        return set(names) - self._tools.keys()

    async def invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"unknown tool: {name}"}
        return await tool.handler(arguments)


class GuideTools:
    """General tools that keep the workflow reusable across tasks."""

    def __init__(
        self,
        rag_functions: dict[str, Function],
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        matches = [function for function in rag_functions.values() if function.instance_name.endswith("__retrieve")]
        if len(matches) != 1:
            raise ValueError("native RAG group must expose exactly one retrieve function")
        self._rag_retrieve = matches[0]
        self._clock = clock or (lambda: datetime.now().astimezone())

    def agent_tools(self) -> list[AgentTool]:
        """Return reusable tools with explicit side-effect capabilities."""

        definitions = {tool.name: tool for tool in self.definitions()}
        return [
            AgentTool(definitions["rag_lookup"], self._rag_lookup, read_only=True),
            AgentTool(definitions["get_current_time"], self._get_current_time, read_only=True),
            AgentTool(definitions["get_timer_status"], self._get_timer_status, read_only=True),
        ]

    def definitions(self) -> list[ToolDef]:
        return [
            ToolDef(
                name="rag_lookup",
                description=(self._rag_retrieve.description or "").strip(),
                parameters=self._rag_retrieve.input_schema.model_json_schema(),
            ),
            ToolDef(
                name="get_current_time",
                description=(
                    "Return the current wall-clock time. Use this before writing a timestamp into workflow context."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            ),
            ToolDef(
                name="get_timer_status",
                description=(
                    "Calculate elapsed and remaining wall-clock time from a recorded "
                    "start timestamp and duration. Call this for every timer completion "
                    "check and every user question about elapsed or remaining time, "
                    "including repeated questions. Never reuse an earlier result."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "started_at_us": {
                            "type": "integer",
                            "description": "Timer start as epoch microseconds.",
                        },
                        "duration_seconds": {
                            "type": "integer",
                            "description": "Positive timer duration in seconds.",
                        },
                        "label": {
                            "type": "string",
                            "description": "Optional human-readable timer label.",
                        },
                    },
                    "required": ["started_at_us", "duration_seconds"],
                    "additionalProperties": False,
                },
            ),
        ]

    async def invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handlers: dict[str, ToolHandler] = {
            "rag_lookup": self._rag_lookup,
            "get_current_time": self._get_current_time,
            "get_timer_status": self._get_timer_status,
        }
        handler = handlers.get(name)
        if handler is None:
            return {"error": f"unknown tool: {name}"}
        return await handler(arguments)

    async def _rag_lookup(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return _plain(await self._rag_retrieve.ainvoke(arguments))

    async def _get_current_time(self, _arguments: dict[str, Any]) -> dict[str, Any]:
        now = self._clock()
        return {
            "epoch_us": int(now.timestamp() * 1_000_000),
            "iso": now.isoformat(timespec="seconds"),
            "timezone": now.tzname(),
        }

    async def _get_timer_status(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return _timer_status(arguments, now=self._clock())


def _timer_status(arguments: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    try:
        started_at_us = int(arguments.get("started_at_us") or 0)
        duration_seconds = int(arguments.get("duration_seconds") or 0)
    except (TypeError, ValueError):
        started_at_us = 0
        duration_seconds = 0
    label = str(arguments.get("label") or "timer").strip() or "timer"
    if started_at_us <= 0 or duration_seconds <= 0:
        return {
            "label": label,
            "started": False,
            "expired": False,
            "error": "A positive start timestamp and duration are required.",
        }

    now_us = int(now.timestamp() * 1_000_000)
    elapsed_us = max(0, now_us - started_at_us)
    duration_us = duration_seconds * 1_000_000
    return {
        "label": label,
        "started": True,
        "started_at_us": started_at_us,
        "started_at_iso": datetime.fromtimestamp(
            started_at_us / 1_000_000,
            tz=now.tzinfo,
        ).isoformat(timespec="seconds"),
        "now_us": now_us,
        "duration_seconds": duration_seconds,
        "elapsed_seconds": elapsed_us // 1_000_000,
        "remaining_seconds": max(
            0,
            math.ceil((duration_us - elapsed_us) / 1_000_000),
        ),
        "expired": elapsed_us >= duration_us,
    }


def _plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


__all__ = ["AgentTool", "GuideTools", "ToolCatalog"]
