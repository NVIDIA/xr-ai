# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool definitions available to workflow step and answer agents."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime
from typing import Any

from nat.builder.function import Function
from pydantic import BaseModel
from xr_ai_models import ToolDef


class GuideTools:
    """General tools that keep the workflow reusable across tasks."""

    def __init__(
        self,
        rag_functions: dict[str, Function],
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        matches = [
            function
            for function in rag_functions.values()
            if function.instance_name.endswith("__retrieve")
        ]
        if len(matches) != 1:
            raise ValueError("native RAG group must expose exactly one retrieve function")
        self._rag_retrieve = matches[0]
        self._clock = clock or (lambda: datetime.now().astimezone())

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
                    "Return the current wall-clock time. Use this before writing a "
                    "timestamp into workflow context."
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
                    "start timestamp and duration. Use this for timer completion and "
                    "for user questions about elapsed or remaining time."
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
        if name == "rag_lookup":
            return _plain(await self._rag_retrieve.ainvoke(arguments))
        if name == "get_current_time":
            now = self._clock()
            return {
                "epoch_us": int(now.timestamp() * 1_000_000),
                "iso": now.isoformat(timespec="seconds"),
                "timezone": now.tzname(),
            }
        if name == "get_timer_status":
            return _timer_status(arguments, now=self._clock())
        return {"error": f"unknown tool: {name}"}


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


__all__ = ["GuideTools"]
