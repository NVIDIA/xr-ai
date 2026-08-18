# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent and lifecycle for the live web-events viewer."""

from __future__ import annotations

import asyncio
import threading
from types import TracebackType

from xr_ai_runtime import Agent, RuntimeContext, subscribe

from ._models import WEB_EVENT_TOPIC, WebEvent
from ._server import _WebEventsServer
from ._store import _EventStore


class WebEventsAgent(Agent):
    """Serve selected runtime events in a bounded participant-aware web view."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8092,
        max_events: int = 5_000,
        title: str = "XR AI Agent Events",
    ) -> None:
        if not host.strip():
            raise ValueError("web-events host must not be empty")
        if not 0 <= port <= 65_535:
            raise ValueError("web-events port must be between 0 and 65535")
        if max_events <= 0:
            raise ValueError("web-events max_events must be positive")
        if not title.strip():
            raise ValueError("web-events title must not be empty")
        super().__init__()
        self.host = host.strip()
        self.port = port
        self.max_events = max_events
        self.title = title.strip()
        self._store = _EventStore(max_events)
        self._server: _WebEventsServer | None = None
        self._thread: threading.Thread | None = None
        self._lifecycle_lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        """Whether the HTTP server is currently running."""

        thread = self._thread
        return self._server is not None and thread is not None and thread.is_alive()

    @property
    def url(self) -> str:
        """Return the configured or currently bound HTTP URL."""

        server = self._server
        if server is None:
            host, port = self.host, self.port
        else:
            host, port = server.server_address[:2]
        return f"http://{host}:{port}"

    async def start(self) -> None:
        """Bind the HTTP listener and start serving the viewer."""

        async with self._lifecycle_lock:
            if self.running:
                return
            server = _WebEventsServer(
                (self.host, self.port),
                store=self._store,
                title=self.title,
            )
            thread = threading.Thread(
                target=server.serve_forever,
                kwargs={"poll_interval": 0.1},
                name="xr-ai-web-events",
                daemon=True,
            )
            try:
                thread.start()
            except Exception:
                server.server_close()
                raise
            self._server = server
            self._thread = thread

    async def stop(self) -> None:
        """Stop the HTTP listener without discarding retained events."""

        async with self._lifecycle_lock:
            server = self._server
            thread = self._thread
            if server is None:
                return
            cleanup = asyncio.create_task(
                self._close_server(server, thread),
                name="xr-ai-web-events-stop",
            )
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                # The listener owns potentially sensitive in-memory data. Do
                # not abandon its socket when the surrounding worker is being
                # cancelled; finish cleanup before propagating cancellation.
                await cleanup
                raise
            finally:
                if cleanup.done():
                    self._server = None
                    self._thread = None

    @staticmethod
    async def _close_server(
        server: _WebEventsServer,
        thread: threading.Thread | None,
    ) -> None:
        try:
            if thread is not None and thread.is_alive():
                await asyncio.to_thread(server.shutdown)
        finally:
            server.server_close()
            if thread is not None:
                await asyncio.to_thread(thread.join)

    async def __aenter__(self) -> WebEventsAgent:
        """Start the viewer and return this agent."""

        await self.start()
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Stop the viewer when its application scope exits."""

        await self.stop()

    @subscribe(WEB_EVENT_TOPIC)
    async def _capture(self, event: WebEvent, ctx: RuntimeContext) -> None:
        self._store.append(event, ctx.metadata)


__all__ = ["WebEventsAgent"]
