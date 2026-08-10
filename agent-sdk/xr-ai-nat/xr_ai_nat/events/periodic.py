# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-scoped periodic event sources with explicit lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from .dispatcher import EventDispatcher
from .models import EventTopic

PayloadT = TypeVar("PayloadT", bound=BaseModel)


class PeriodicEventSource(Generic[PayloadT]):
    """Publish one typed event on an independent schedule for each participant."""

    def __init__(
        self,
        dispatcher: EventDispatcher,
        topic: EventTopic[PayloadT],
        *,
        payload: Callable[[str], PayloadT | dict[str, Any]],
        producer: str,
        subscriber_id: str,
        interval_s: float,
        immediate: bool = True,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("periodic event interval must be positive")
        self._dispatcher = dispatcher
        self._topic = topic
        self._payload = payload
        self._producer = producer
        self._subscriber_id = subscriber_id
        self._interval_s = interval_s
        self._immediate = immediate
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._errors: asyncio.Queue[BaseException] = asyncio.Queue()
        self._closed = asyncio.Event()

    def start(self, participant_id: str) -> bool:
        """Start this source for a participant; return false when already active."""
        if self._closed.is_set():
            raise RuntimeError("periodic event source is closed")
        if participant_id in self._tasks:
            return False
        task = asyncio.create_task(
            self._publish_periodically(participant_id),
            name=f"{self._producer}:{participant_id}",
        )
        self._tasks[participant_id] = task
        task.add_done_callback(lambda completed: self._completed(participant_id, completed))
        return True

    async def stop(self, participant_id: str) -> bool:
        """Stop this source for one participant and await its cancellation."""
        task = self._tasks.pop(participant_id, None)
        if task is None:
            return False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return True

    async def run(self) -> None:
        """Supervise source tasks so an unexpected publisher failure propagates."""
        failure = asyncio.create_task(self._errors.get())
        closed = asyncio.create_task(self._closed.wait())
        try:
            done, _ = await asyncio.wait(
                (failure, closed),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if failure in done:
                raise failure.result()
        finally:
            failure.cancel()
            closed.cancel()
            await asyncio.gather(failure, closed, return_exceptions=True)
            await self.close()

    async def close(self) -> None:
        """Stop every participant schedule and release the supervisor."""
        if self._closed.is_set():
            return
        tasks = tuple(self._tasks.values())
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._closed.set()

    async def _publish_periodically(self, participant_id: str) -> None:
        loop = asyncio.get_running_loop()
        next_tick = loop.time() if self._immediate else loop.time() + self._interval_s
        while True:
            await asyncio.sleep(max(0.0, next_tick - loop.time()))
            next_tick += self._interval_s
            await self._dispatcher.publish(
                self._topic,
                participant_id=participant_id,
                producer=self._producer,
                payload=self._payload(participant_id),
                subscribers={self._subscriber_id},
            )

    def _completed(self, participant_id: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(participant_id) is task:
            self._tasks.pop(participant_id, None)
        if task.cancelled():
            return
        if error := task.exception():
            self._errors.put_nowait(error)


__all__ = ["PeriodicEventSource"]
