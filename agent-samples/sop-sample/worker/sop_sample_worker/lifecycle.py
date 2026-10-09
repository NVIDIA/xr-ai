# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Voice-controlled SOP boundaries, independent of camera and media capture."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from loguru import logger
from xr_ai_hub import ParticipantEvent
from xr_ai_runtime import Agent, RuntimeContext, subscribe
from xr_ai_voice import VOICE_TRANSCRIPT_TOPIC, VoiceTranscript

from .recorder import RecorderAgent

_COMMAND = re.compile(r"\s*(start|stop)\s+recording[.!?]*\s*", re.IGNORECASE)


@dataclass
class _Connection:
    session_id: str
    joined_at_us: int
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    recording: _Recording | None = None


@dataclass
class _Recording:
    connection: _Connection


class CaptureRecording(Agent):
    def __init__(self, recorder: RecorderAgent) -> None:
        super().__init__()
        self._recorder = recorder
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
                logger.exception("SOP recording action failed pid={!r}", pid)

        task = None if self._closed else asyncio.create_task(apply())
        if task is not None:
            self._tails[pid] = task
        return task

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
            if recording is not None and recording.connection is connection and not (event.joined and duplicate):
                await self._recorder.finish_recording(pid)
                self._sessions.pop(pid, None)

        task = self._enqueue(pid, apply)

        async def wait() -> None:
            if task is not None:
                await asyncio.shield(task)

        return wait()

    @subscribe(VOICE_TRANSCRIPT_TOPIC)
    async def transcript(self, transcript: VoiceTranscript, ctx: RuntimeContext) -> None:
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
                    return
                self._sessions[pid] = recording
                await self._recorder.start_recording(
                    pid, cancelled=connection.cancelled, timestamp_us=transcript.timestamp_us,
                )

            self._enqueue(pid, start)
        elif command:
            connection.recording = None

            async def finish() -> None:
                if recording is not None and self._sessions.get(pid) is recording:
                    await self._recorder.finish_recording(
                        pid, stop_command={"text": transcript.text.strip(), "timestamp_us": transcript.timestamp_us},
                    )
                    self._sessions.pop(pid, None)

            self._enqueue(pid, finish)
        elif recording is not None:
            async def narrate() -> None:
                if self._sessions.get(pid) is recording:
                    await self._recorder.record_transcript(pid, transcript.text, transcript.timestamp_us)

            self._enqueue(pid, narrate)
        # Return promptly: VoiceAgent shares one transcript delivery queue across
        # participants. This owner drains per-participant writes and boundaries.

    async def close(self) -> None:
        self._closed = True
        for connection in self._connections.values():
            connection.cancelled.set()
        try:
            await asyncio.gather(*self._tails.values())
        finally:
            await self._recorder.stop()
