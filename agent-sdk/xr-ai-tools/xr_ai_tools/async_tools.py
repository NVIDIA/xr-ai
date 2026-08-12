# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed tools that yield asynchronous result chunks."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
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
        """Validate and yield one typed result stream under a tool scope."""

        value = self.request_model.model_validate(request)
        with nemo_relay.scope.scope(
            self.name,
            nemo_relay.ScopeType.Tool,
            input=value.model_dump(mode="json"),
        ):
            async for chunk in self.handler(value):
                yield self.chunk_model.model_validate(chunk)


__all__ = ["AsyncTool"]
