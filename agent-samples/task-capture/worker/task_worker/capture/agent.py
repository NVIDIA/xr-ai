# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinate voice commands and narration for task capture.

Commands and narration must stay ordered so speech is saved to the intended
capture session. Participant disconnects must also finalize active capture
and prevent queued commands from starting work after the participant leaves.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from loguru import logger
from xr_ai_hub import ParticipantEvent
from xr_ai_runtime import Agent, RuntimeContext, Topic, subscribe
from xr_ai_voice import VOICE_OUTPUT_TOPIC, UserQuery, VoiceOutput

from .capturing import _Capture

_COMMAND = re.compile(r"\s*(start|stop)\s+capture[.!?]*\s*", re.IGNORECASE)
CAPTURE_INPUT_TOPIC = Topic("task.capture.input", UserQuery)


@dataclass
class _Connection:
    session_id: str
    joined_at_us: int
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    recording: _Recording | None = None


@dataclass
class _Recording:
    connection: _Connection


class CaptureAgent(Agent):
    """Admit input per connection and serialize each participant's capture work.

    Pending command identity and the capture backend's physical session are distinct:
    a new start can be queued while the previous packet is still finalizing.
    Connection invalidation is immediate so departure cancels queued starts.
    """
    def __init__(self, capture_backend: _Capture) -> None:
        super().__init__()
        self._capture_backend = capture_backend
        self._sessions: dict[str, _Recording] = {}
        self._tails: dict[str, asyncio.Task[None]] = {}
        self._connections: dict[str, _Connection] = {}
        self._closed = False

    def _enqueue(self, pid: str, action: Callable[[], Awaitable[None]]) -> asyncio.Task[None] | None:
        previous = self._tails.get(pid)

        async def apply() -> None:
            if previous is not None:
                await previous
            try:
                await action()
            except Exception:
                logger.exception("task capture action failed pid={!r}", pid)

        task = None if self._closed else asyncio.create_task(apply())
        if task is not None:
            self._tails[pid] = task
        return task

    async def _say(self, pid: str, connection: _Connection, ctx: RuntimeContext, text: str) -> None:
        if self._closed or connection.cancelled.is_set() or self._connections.get(pid) is not connection:
            return
        try:
            # Preserve the order of start and stop acknowledgements.
            await ctx.publish(VOICE_OUTPUT_TOPIC, VoiceOutput(text=text), participant_id=pid)
        except (Exception, BaseExceptionGroup):
            # Runtime shutdown wraps cancelled output deliveries in a group.
            # Announcement delivery is best effort, not a prerequisite for saving.
            logger.exception("task capture announcement failed pid={!r}", pid)

    def _forget(self, pid: str, recording: _Recording) -> None:
        # A later start may already be queued on this connection.
        if recording.connection.recording is recording:
            recording.connection.recording = None
        if self._sessions.get(pid) is recording:
            self._sessions.pop(pid)

    def receive(self, event: ParticipantEvent) -> Awaitable[None]:
        pid = event.participant_id
        connection = self._connections.get(pid)
        duplicate = connection is not None and connection.session_id == event.participant_session_id
        # Invalidate queued starts immediately, not behind pending finalization.
        if not self._closed:
            if event.joined and not duplicate:
                if connection is not None:
                    connection.cancelled.set()
                self._connections[pid] = _Connection(event.participant_session_id, event.pts_us)
            elif not event.joined and duplicate:
                connection.cancelled.set()
                self._connections.pop(pid, None)

        async def apply() -> None:
            if not event.joined and not duplicate:
                return
            recording = self._sessions.get(pid)
            try:
                if recording is not None and recording.connection is connection and not (event.joined and duplicate):
                    try:
                        await self._capture_backend.finish_capture(pid)
                    finally:
                        self._forget(pid, recording)
            finally:
                if not event.joined and pid not in self._connections:
                    self._capture_backend.release(pid)

        task = self._enqueue(pid, apply)

        async def wait() -> None:
            if task is not None:
                await asyncio.shield(task)

        return wait()

    @subscribe(CAPTURE_INPUT_TOPIC)
    async def on_input(self, transcript: UserQuery, ctx: RuntimeContext) -> None:
        # VoiceAgent dispatches accepted input after cancelling prior output.
        # Responding to the earlier raw-transcript event races that cancellation.
        # Input ordering relies on this handler enqueueing without awaiting and
        # acknowledgements remaining finite text. Streamed replies can delay
        # cancellation cleanup, letting a later utterance overtake an earlier one.
        pid = ctx.metadata.participant_id
        connection = self._connections.get(pid)
        if (
            self._closed or connection is None or connection.cancelled.is_set()
            or transcript.timestamp_us < connection.joined_at_us
        ):
            return
        command = _COMMAND.fullmatch(transcript.text)
        recording = connection.recording
        if command and command[1].lower() == "start":
            if recording is not None:
                return
            recording = _Recording(connection)
            connection.recording = recording

            async def start() -> None:
                if self._closed or connection.cancelled.is_set():
                    self._forget(pid, recording)
                    return
                self._sessions[pid] = recording
                try:
                    await self._capture_backend.start_capture(
                        pid, cancelled=connection.cancelled, timestamp_us=transcript.timestamp_us,
                    )
                    if self._capture_backend.is_capturing(pid):
                        await self._say(pid, connection, ctx, "Capture started.")
                finally:
                    if not self._capture_backend.is_capturing(pid):
                        self._forget(pid, recording)

            self._enqueue(pid, start)
        elif command:
            connection.recording = None

            async def finish() -> None:
                if recording is not None and self._sessions.get(pid) is recording:
                    await self._say(pid, connection, ctx, "Capture stopped.")
                    try:
                        await self._capture_backend.finish_capture(
                            pid,
                            stop_command={"text": transcript.text.strip(), "timestamp_us": transcript.timestamp_us},
                        )
                    finally:
                        self._forget(pid, recording)

            self._enqueue(pid, finish)
        elif recording is not None:
            async def narrate() -> None:
                if self._sessions.get(pid) is recording:
                    await self._capture_backend.capture_transcript(pid, transcript.text, transcript.timestamp_us)

            self._enqueue(pid, narrate)
        # Return promptly; this owner drains per-participant writes and boundaries.

    async def close(self) -> None:
        self._closed = True
        for connection in self._connections.values():
            connection.cancelled.set()
        try:
            pending = tuple(self._tails.items())
            results = await asyncio.gather(*(task for _, task in pending), return_exceptions=True)
            for (pid, _), result in zip(pending, results):
                if isinstance(result, BaseException):
                    logger.opt(exception=result).error("task capture action failed during shutdown pid={!r}", pid)
        finally:
            try:
                await self._capture_backend.stop()
            finally:
                for pid in self._connections.keys() | self._sessions.keys():
                    self._capture_backend.release(pid)
                self._connections.clear()
                self._sessions.clear()
