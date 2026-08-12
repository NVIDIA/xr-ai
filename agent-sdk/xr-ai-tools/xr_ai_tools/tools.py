# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed native tools with a Relay-managed execution boundary."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any, Generic, TypeVar, cast

from nemo_relay import typed
from pydantic import BaseModel, ValidationError

RequestT = TypeVar("RequestT", bound=BaseModel)
ResultT = TypeVar("ResultT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ToolInvocationResult:
    """A model-visible result from one local tool invocation."""

    content: str
    return_direct: bool


class Tool(Generic[RequestT, ResultT]):
    """A Pydantic-validated tool shared by agents, voice, and background triggers."""

    def __init__(
        self,
        name: str,
        description: str,
        request_model: type[RequestT],
        result_model: type[ResultT],
        handler: Callable[[RequestT], Awaitable[ResultT] | ResultT],
        *,
        return_direct: bool = False,
        render_result: Callable[[ResultT], str] | None = None,
    ) -> None:
        if not name:
            raise ValueError("tool name must not be empty")
        if not description:
            raise ValueError(f"tool {name!r} needs a description")
        self.name = name
        self.description = description
        self.request_model = request_model
        self.result_model = result_model
        self.handler = handler
        self.return_direct = return_direct
        self._request_codec = typed.PydanticCodec(request_model)
        self._result_codec = typed.PydanticCodec(result_model)
        self._render_result = render_result or _json_result

    async def execute(self, request: RequestT) -> ResultT:
        """Run one validated request through the shared Relay tool lifecycle."""

        return await typed.tool_execute(
            self.name,
            request,
            self._execute_handler,
            self._request_codec,
            self._result_codec,
        )

    async def invoke(self, arguments: str) -> ToolInvocationResult:
        """Validate and run one model-supplied JSON argument payload."""

        try:
            raw_arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            return self._validation_error(f"arguments must be valid JSON: {exc.msg}")
        if not isinstance(raw_arguments, dict):
            return self._validation_error("arguments must be a JSON object")
        try:
            request = self.request_model.model_validate(raw_arguments)
        except ValidationError as exc:
            return self._validation_error(exc.json(include_url=False))
        return ToolInvocationResult(
            self._render_result(await self.execute(request)),
            self.return_direct,
        )

    def _validation_error(self, detail: str) -> ToolInvocationResult:
        return ToolInvocationResult(
            content=json.dumps({"error": "invalid_tool_arguments", "detail": detail}),
            return_direct=False,
        )

    async def _execute_handler(self, request: RequestT) -> ResultT:
        result = self.handler(request)
        if isawaitable(result):
            return await cast(Awaitable[ResultT], result)
        return cast(ResultT, result)


class ToolSet:
    """A non-overlapping native tool catalog."""

    def __init__(self, tools: Iterable[Tool[Any, Any]]) -> None:
        by_name: dict[str, Tool[Any, Any]] = {}
        for tool in tools:
            if tool.name in by_name:
                raise ValueError(f"duplicate tool name: {tool.name}")
            by_name[tool.name] = tool
        self._by_name = by_name

    def get(self, name: str) -> Tool[Any, Any] | None:
        """Return the named tool when this catalog owns it."""

        return self._by_name.get(name)


def _json_result(result: BaseModel) -> str:
    return result.model_dump_json()


__all__ = ["Tool", "ToolInvocationResult", "ToolSet"]
