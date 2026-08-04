# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool definitions available to workflow step and answer agents."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from nat.builder.function import Function
from pydantic import BaseModel
from xr_ai_models import ToolDef


class GuideTools:
    """General tools that keep the workflow reusable across tasks."""

    def __init__(self, rag_functions: dict[str, Function]) -> None:
        matches = [
            function
            for function in rag_functions.values()
            if function.instance_name.endswith("__retrieve")
        ]
        if len(matches) != 1:
            raise ValueError("native RAG group must expose exactly one retrieve function")
        self._rag_retrieve = matches[0]

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
        ]

    async def invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "rag_lookup":
            return _plain(await self._rag_retrieve.ainvoke(arguments))
        if name == "get_current_time":
            now = datetime.now().astimezone()
            return {
                "epoch_us": int(now.timestamp() * 1_000_000),
                "iso": now.isoformat(timespec="seconds"),
                "timezone": now.tzname(),
            }
        return {"error": f"unknown tool: {name}"}


def _plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


__all__ = ["GuideTools"]
