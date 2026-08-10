# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinate voice turns and identical observation iterations per participant."""

from __future__ import annotations

import asyncio
import time
import uuid

from xr_ai_hub import FrameUnavailable
from xr_ai_nat.events import EventDispatcher
from xr_ai_voice import VoiceTurn

from ..agents import AgentRegistry
from ..applications.events import (
    APPLICATION_RESET,
    RAW_TRANSCRIPT,
    ApplicationReset,
    RawTranscript,
    UserOutput,
)
from ..applications.manager.registry import ApplicationManager
from ..applications.output import UserOutputDelivery
from ..runtime.events import emit
from ..runtime.scope import invocation_scope
from ..runtime.state import Session, SessionStore
from .triggers import TriggerRegistry


class Coordinator:
    def __init__(
        self,
        *,
        store: SessionStore,
        agents: AgentRegistry,
        manager: ApplicationManager,
        events: EventDispatcher,
        output: UserOutputDelivery,
        reset_subscriber_ids: frozenset[str],
        triggers: TriggerRegistry,
    ) -> None:
        self.store = store
        self.agents = agents
        self.manager = manager
        self.events = events
        self.output = output
        self.reset_subscriber_ids = reset_subscriber_ids
        self.triggers = triggers
        self._stopped = asyncio.Event()
        self._connected: set[str] = set()

    async def handle_transcription(self, turn: VoiceTurn) -> None:
        """Forward final STT output before wake-word command gating."""
        trace_id = _trace_id()
        session = self.store.get(turn.participant_id)
        async with session.lock:
            with invocation_scope(session, trace_id):
                await self.events.publish(
                    RAW_TRANSCRIPT,
                    participant_id=turn.participant_id,
                    producer="voice.input",
                    payload=RawTranscript(text=turn.text),
                    subscribers=session.applications.background,
                    correlation_id=trace_id,
                    timestamp_us=turn.timestamp_us,
                )
        emit(
            "transcription.observed",
            participant_id=turn.participant_id,
            trace_id=trace_id,
            timestamp_us=turn.timestamp_us,
            characters=len(turn.text),
        )

    async def participant_joined(self, participant_id: str) -> None:
        if participant_id in self._connected:
            emit("participant.replayed", participant_id=participant_id)
            return
        self._connected.add(participant_id)
        session = self.store.get(participant_id)
        trace_id = _trace_id()
        async with session.lock:
            with invocation_scope(session, trace_id):
                await self.events.publish(
                    APPLICATION_RESET,
                    participant_id=participant_id,
                    producer="application.manager",
                    payload=ApplicationReset(),
                    subscribers=self.reset_subscriber_ids,
                    correlation_id=trace_id,
                )
                self.manager.ownership.reset(session)
                self.store.reset(session)
        emit("participant.joined", participant_id=participant_id)

    async def participant_left(self, participant_id: str) -> None:
        self._connected.discard(participant_id)
        session = self.store.get(participant_id)
        trace_id = _trace_id()
        async with session.lock:
            with invocation_scope(session, trace_id):
                await self.events.publish(
                    APPLICATION_RESET,
                    participant_id=participant_id,
                    producer="application.manager",
                    payload=ApplicationReset(),
                    subscribers=self.reset_subscriber_ids,
                    correlation_id=trace_id,
                )
                self.manager.ownership.reset(session)
                self.store.release(participant_id)
        emit("participant.left", participant_id=participant_id)

    async def monitor(self) -> None:
        while not self._stopped.is_set():
            sessions = self.store.sessions()
            await asyncio.gather(*(self._tick(session) for session in sessions if session.active))
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=0.25)
            except TimeoutError:
                pass

    def stop(self) -> None:
        self._stopped.set()

    async def _tick(self, session: Session) -> None:
        notices: tuple[str, ...] = ()
        trace_id = ""
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
            await self.output.publish(
                session.participant_id,
                "tea",
                UserOutput(text=message),
                correlation_id=trace_id,
            )


def _trace_id() -> str:
    return uuid.uuid4().hex[:12]


__all__ = ["Coordinator"]
