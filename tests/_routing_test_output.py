# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct destination output capture for pure routing contract tests."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextvars import ContextVar

from xr_ai_routing import AgentRouter, AgentTurn, DestinationResult

_sink: ContextVar[Callable[[str, str], Awaitable[None]]] = ContextVar("test_destination_output")


async def emit_control(_turn: AgentTurn, source: str, text: str) -> None:
    await _sink.get()(source, text)


def destination_output(started: Callable[[], Awaitable[bool]]) -> Callable[[str, str], Awaitable[None]]:
    async def emit(source: str, text: str) -> None:
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise asyncio.CancelledError
        if text.strip():
            await started()
        await _sink.get()(source, text)

    return emit


async def standalone_started() -> bool:
    return True


async def publish_output(source: str, text: str) -> None:
    await _sink.get()(source, text)


async def route(
    router: AgentRouter, turn: AgentTurn, *, emit: Callable[[str, str], Awaitable[None]]
) -> DestinationResult:
    token = _sink.set(emit)
    try:
        return await router.handle(turn)
    finally:
        _sink.reset(token)
