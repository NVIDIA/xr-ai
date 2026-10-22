# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct destination output capture for pure routing contract tests."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextvars import ContextVar

from xr_ai_routing import AgentRouter, AgentTurn, DestinationResult, notify_response_started

_sink: ContextVar[Callable[[str, str], Awaitable[None]]] = ContextVar("test_destination_output")


async def emit_destination(source: str, text: str) -> None:
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        raise asyncio.CancelledError
    if text.strip():
        notify_response_started()
    await _sink.get()(source, text)


async def route(
    router: AgentRouter, turn: AgentTurn, *, emit: Callable[[str, str], Awaitable[None]]
) -> DestinationResult:
    token = _sink.set(emit)
    try:
        return await router.handle(turn, emit=emit)
    finally:
        _sink.reset(token)
