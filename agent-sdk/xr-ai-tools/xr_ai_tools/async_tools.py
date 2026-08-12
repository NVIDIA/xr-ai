# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed tools that yield asynchronous result chunks."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

import nemo_relay
from pydantic import BaseModel

RequestT = TypeVar("RequestT", bound=BaseModel)
ChunkT = TypeVar("ChunkT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class _StreamFailure:
    error: Exception


class _StreamEnd:
    pass


_STREAM_END = _StreamEnd()


class AsyncTool(Generic[RequestT, ChunkT]):
    """A validated asynchronous tool independent of any output transport."""

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

    async def stream(
        self,
        request: RequestT | Mapping[str, object],
    ) -> AsyncIterator[ChunkT]:
        """Validate and yield one typed result stream under a tool scope."""

        value = self.request_model.model_validate(request)
        queue: asyncio.Queue[ChunkT | _StreamFailure | _StreamEnd] = (
            asyncio.Queue(maxsize=1)
        )
        # The producer owns the task-local Relay scope and handler cleanup.
        producer = asyncio.create_task(
            self._produce(value, queue),
            name=f"xr-ai-tool:{self.name}",
            context=nemo_relay.fork_asyncio_context(),
        )
        try:
            while True:
                item = await queue.get()
                if isinstance(item, _StreamFailure):
                    raise item.error
                if isinstance(item, _StreamEnd):
                    return
                yield item
        finally:
            if not producer.done():
                producer.cancel()
            try:
                await producer
            except asyncio.CancelledError:
                pass

    async def _produce(
        self,
        value: RequestT,
        queue: asyncio.Queue[ChunkT | _StreamFailure | _StreamEnd],
    ) -> None:
        try:
            with nemo_relay.scope.scope(
                self.name,
                nemo_relay.ScopeType.Tool,
                input=value.model_dump(mode="json"),
            ):
                stream = self.handler(value)
                try:
                    async for chunk in stream:
                        await queue.put(self.chunk_model.model_validate(chunk))
                finally:
                    close = getattr(stream, "aclose", None)
                    if close is not None:
                        await close()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await queue.put(_StreamFailure(exc))
        else:
            await queue.put(_STREAM_END)


__all__ = ["AsyncTool"]
