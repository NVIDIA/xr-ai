# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinate voice turns and identical observation iterations per participant."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable

from xr_ai_hub import FrameUnavailable

from ..agents import AgentRegistry
from ..runtime.events import emit
from ..runtime.scope import invocation_scope
from ..runtime.state import Session, SessionStore
from .triggers import TriggerRegistry

Notice = Callable[[str, str], Awaitable[None]]


class Coordinator:
    def __init__(
        self,
        *,
        store: SessionStore,
        agents: AgentRegistry,
        triggers: TriggerRegistry,
        notice: Notice,
    ) -> None:
        self.store = store
        self.agents = agents
        self.triggers = triggers
        self.notice = notice
        self._stopped = asyncio.Event()
        self._connected: set[str] = set()

    async def handle_query(self, participant_id: str, text: str) -> str:
        session = self.store.get(participant_id)
        trace_id = _trace_id()
        async with session.lock:
            with invocation_scope(session, trace_id):
                result = await self.agents.route(session, text, trace_id)
        emit(
            "voice.complete",
            participant_id=participant_id,
            trace_id=trace_id,
            response=result,
        )
        return result

    async def participant_joined(self, participant_id: str) -> None:
        if participant_id in self._connected:
            emit("participant.replayed", participant_id=participant_id)
            return
        self._connected.add(participant_id)
        session = self.store.get(participant_id)
        async with session.lock:
            self.store.reset(session)
        emit("participant.joined", participant_id=participant_id)

    async def participant_left(self, participant_id: str) -> None:
        self._connected.discard(participant_id)
        session = self.store.get(participant_id)
        async with session.lock:
            self.store.release(participant_id)
        emit("participant.left", participant_id=participant_id)

    async def monitor(self) -> None:
        while not self._stopped.is_set():
            await asyncio.gather(*(self._tick(session) for session in self.store.active()))
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=0.25)
            except TimeoutError:
                pass

    def stop(self) -> None:
        self._stopped.set()

    async def _tick(self, session: Session) -> None:
        notices: tuple[str, ...] = ()
        async with session.lock:
            if not session.active or session.step_id is None:
                return
            now = time.monotonic()
            if session.next_tick > now:
                return
            step = self.store.workflow.step(session.step_id)
            session.next_tick = now + step.trigger.interval_s
            trace_id = _trace_id()
            with invocation_scope(session, trace_id):
                try:
                    observation = await self.triggers.invoke(session, step, trace_id)
                except FrameUnavailable as exc:
                    emit(
                        "trigger.unavailable",
                        participant_id=session.participant_id,
                        step=step.id,
                        trace_id=trace_id,
                        reason=str(exc),
                    )
                    return
                self.store.observe(session, observation, trace_id)
                await self.agents.observe(session, observation, trace_id)
            notices = self.store.drain_notices(session)
        for message in notices:
            await self.notice(session.participant_id, message)


def _trace_id() -> str:
    return uuid.uuid4().hex[:12]


__all__ = ["Coordinator"]
