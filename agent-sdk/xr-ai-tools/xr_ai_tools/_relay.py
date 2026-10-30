# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Handler lifetime and request validation at the tools package's Relay boundary."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")


class RelayHandlers:
    """Let a Relay caller wait for the handlers Relay runs in its own tasks.

    Relay returns a cancelled call to its caller before the handler task it started has
    finished cancelling, so model work and its cleanup can outlive the caller. Use it as
    ``async with RelayHandlers() as handlers:`` around the Relay call and pass each handler
    through :meth:`track` or :meth:`track_stream`; leaving the block, including by
    cancellation, waits until those handlers have finished.

    Tracks https://github.com/NVIDIA/xr-ai/issues/697. Remove the lifetime
    wrapper once the supported Relay version awaits handler cleanup itself.
    """

    def __init__(self) -> None:
        self._running = 0
        self._idle = asyncio.Event()
        self._idle.set()

    async def __aenter__(self) -> RelayHandlers:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.settled()

    def track(self, handler: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def tracked(*args: Any) -> T:
            self._enter()
            try:
                return await handler(*args)
            finally:
                self._exit()

        return tracked

    def track_stream(self, handler: Callable[..., AsyncIterator[T]]) -> Callable[..., AsyncIterator[T]]:
        async def tracked(*args: Any) -> AsyncIterator[T]:
            self._enter()
            try:
                stream = handler(*args)
                try:
                    async for item in stream:
                        yield item
                finally:
                    close = getattr(stream, "aclose", None)
                    if close is not None:
                        await close()
            finally:
                self._exit()

        return tracked

    async def settled(self) -> None:
        """Wait until no tracked handler is running; a cancellation received meanwhile is re-raised after."""

        cancelled = None
        while not self._idle.is_set():
            try:
                await self._idle.wait()
            except asyncio.CancelledError as error:
                if cancelled is None:
                    cancelled = error
        if cancelled is not None:
            raise cancelled

    def _enter(self) -> None:
        self._running += 1
        self._idle.clear()

    def _exit(self) -> None:
        self._running -= 1
        if not self._running:
            self._idle.set()


def headers_from_relay(raw: object) -> dict[str, str]:
    """Return string-only headers supplied by a Relay request intercept."""

    if not isinstance(raw, dict):
        raise TypeError("Relay LLM request headers must be an object")
    headers: dict[str, str] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise TypeError("Relay LLM request headers must be strings")
        headers[name] = value
    return headers


__all__ = ["RelayHandlers", "headers_from_relay"]
