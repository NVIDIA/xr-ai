# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute a guide step through a common flow lifecycle.

The runner serializes observations, delivers progress, and owns cancellation.
The guide backend accepts results within its run/step scope and decides when
progress satisfies the guide. This component has no YAML, voice, or model policy.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from typing import Generic, Literal, TypeVar

import nemo_relay

from .flow import Flow, FlowUpdate

T = TypeVar("T")


class StepRunner(Generic[T]):
    def __init__(
        self, flow: Flow[T], *, apply: Callable[[FlowUpdate[T]], Awaitable[bool]],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.flow = flow
        self._apply = apply
        self._clock = clock
        self.next_poll_at = 0.0
        self.status: Literal["idle", "running", "completed", "cancelled", "failed"] = "idle"
        self.error: Exception | None = None
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def poll(self) -> None:
        """Run one due observation, also used by deterministic packet evaluation."""
        async with self._lock:
            if self.status in {"completed", "cancelled", "failed"} or self._clock() < self.next_poll_at:
                return
            self.status = "running"
            update = await self.flow.poll()
            if asyncio.current_task().cancelling() or self.status == "cancelled":
                raise asyncio.CancelledError
            if not math.isfinite(update.delay_s) or update.delay_s <= 0:
                raise ValueError("flow delay_s must be finite and positive")
            self.next_poll_at = self._clock() + update.delay_s
            complete = await self._apply(update)
            if asyncio.current_task().cancelling() or self.status == "cancelled":
                raise asyncio.CancelledError
            if complete:
                self.status = "completed"

    def start(self, *, name: str, poll: Callable[[], Awaitable[None]] | None = None) -> asyncio.Task[None]:
        """Start once; an optional scoped adapter attributes evaluation events."""
        if self._task is not None:
            return self._task
        self._task = asyncio.create_task(self._run(poll or self.poll), name=name,
                                         context=nemo_relay.fork_asyncio_context())
        return self._task

    async def _run(self, poll: Callable[[], Awaitable[None]]) -> None:
        try:
            while self.status not in {"completed", "cancelled", "failed"}:
                await poll()
                if self.status == "completed":
                    return
                await asyncio.sleep(max(0.001, self.next_poll_at - self._clock()))
        except asyncio.CancelledError:
            self.status = "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001 — retain task failure; never turn it into successful completion.
            self.status, self.error = "failed", exc
            raise

    async def stop(self) -> None:
        """Invalidate first, then drain the owned observation before replacement."""
        if self.status not in {"completed", "failed"}:
            self.status = "cancelled"
        if self._task is not None and self._task is not asyncio.current_task():
            if not self._task.done() and not self._task.cancelling():
                self._task.cancel()
            # Caller cancellation cannot leave the old flow cleaning up while
            # its replacement acquires the same camera/tool resources.
            drain = asyncio.gather(self._task, return_exceptions=True)
            interrupted = False
            while not drain.done():
                try:
                    await asyncio.shield(drain)
                except asyncio.CancelledError:
                    interrupted = True
            if interrupted:
                raise asyncio.CancelledError
