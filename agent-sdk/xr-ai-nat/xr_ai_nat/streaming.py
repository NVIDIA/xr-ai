# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed streaming tools for trigger-driven native application paths."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from inspect import isawaitable
from typing import Any, Generic, TypeVar, cast

import nemo_relay
from pydantic import BaseModel

RequestT = TypeVar("RequestT", bound=BaseModel)
ChunkT = TypeVar("ChunkT", bound=BaseModel)
ValueT = TypeVar("ValueT")


async def _resolve(value: ValueT | Awaitable[ValueT]) -> ValueT:
    """Normalize Relay's sync-or-async hook contract for an async tool path."""

    if isawaitable(value):
        return await cast(Awaitable[ValueT], value)
    return value


class StreamingTool(Generic[RequestT, ChunkT]):
    """A typed tool that yields Pydantic chunks while keeping one Relay tool span open."""

    def __init__(
        self,
        name: str,
        description: str,
        request_model: type[RequestT],
        chunk_model: type[ChunkT],
        handler: Callable[[RequestT], AsyncIterator[ChunkT]],
    ) -> None:
        if not name:
            raise ValueError("tool name must not be empty")
        if not description:
            raise ValueError(f"tool {name!r} needs a description")
        self.name = name
        self.description = description
        self.request_model = request_model
        self.chunk_model = chunk_model
        self.handler = handler

    async def stream(self, request: RequestT) -> AsyncIterator[ChunkT]:
        """Yield one validated tool stream under a Relay lifecycle span."""

        raw_request = request.model_dump(mode="json")
        await _resolve(nemo_relay.tools.conditional_execution(self.name, raw_request))
        intercepted = await _resolve(nemo_relay.tools.request_intercepts(self.name, raw_request))
        if not isinstance(intercepted, dict):
            raise TypeError("Relay tool request intercepts must return an object")
        request = self.request_model.model_validate(intercepted)
        handle = nemo_relay.tools.call(self.name, intercepted)
        chunks: list[dict[str, Any]] = []
        try:
            async for chunk in self.handler(request):
                value = self.chunk_model.model_validate(chunk)
                chunks.append(value.model_dump(mode="json"))
                yield value
        except BaseException:
            nemo_relay.tools.call_end(handle, {"status": "interrupted", "chunks": chunks})
            raise
        else:
            nemo_relay.tools.call_end(handle, {"status": "ok", "chunks": chunks})


__all__ = ["StreamingTool"]
