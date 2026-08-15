# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed tools that yield asynchronous result chunks."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import suppress
from typing import Generic, TypeVar

import nemo_relay
from pydantic import BaseModel

RequestT = TypeVar("RequestT", bound=BaseModel)
ChunkT = TypeVar("ChunkT", bound=BaseModel)


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
        """Validate and yield chunks produced in an isolated Relay tool scope.

        The producer task owns the scope and handler cleanup; chunks are yielded
        in the consumer's context. Parent scope-local registrations are not
        transferred into the isolated context.
        """

        value = self.request_model.model_validate(request)
        queue: asyncio.Queue[ChunkT] = asyncio.Queue(maxsize=1)
        producer = asyncio.create_task(
            self._produce(value, queue),
            name=f"xr-ai-tool:{self.name}",
            context=nemo_relay.fork_asyncio_context(),
        )
        try:
            while True:
                next_chunk = asyncio.create_task(queue.get())
                try:
                    done, _ = await asyncio.wait(
                        (next_chunk, producer),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    if not next_chunk.done():
                        next_chunk.cancel()
                        with suppress(asyncio.CancelledError):
                            _ = await next_chunk

                if next_chunk in done:
                    yield next_chunk.result()
                    continue

                if not queue.empty():
                    # Producer completion can win with its final chunk buffered.
                    yield queue.get_nowait()
                    continue

                producer.result()
                return
        finally:
            if not producer.done():
                producer.cancel()
            with suppress(asyncio.CancelledError):
                _ = await producer

    async def _produce(
        self,
        value: RequestT,
        queue: asyncio.Queue[ChunkT],
    ) -> None:
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


__all__ = ["AsyncTool"]
