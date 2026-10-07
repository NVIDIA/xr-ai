# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serialize camera boundaries, including rapid toggles and departure."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable

from xr_ai_hub import ParticipantEvent, VideoTrackEvent

from .recorder import RecorderAgent


class CameraRecording:
    def __init__(self, recorder: RecorderAgent) -> None:
        self._recorder = recorder
        self._sessions: dict[str, str] = {}
        self._tracks: dict[str, set[str]] = {}
        self._tails: dict[str, asyncio.Task[None]] = {}
        self._connections: dict[str, tuple[str, asyncio.Event]] = {}
        self._closed = False

    def receive(self, event: ParticipantEvent | VideoTrackEvent) -> Awaitable[None]:
        # Enqueue synchronously: ProcessorEndpoint detaches the returned
        # awaitable, but event order must survive slow manifest finalization.
        pid = event.participant_id
        connection = self._connections.get(pid)
        if not self._closed and isinstance(event, ParticipantEvent):
            if event.joined:
                if connection is None or connection[0] != event.participant_session_id:
                    if connection is not None:
                        connection[1].set()
                    connection = (event.participant_session_id, asyncio.Event())
                    self._connections[pid] = connection
            elif connection is not None and connection[0] == event.participant_session_id:
                # Departure must wake a pending start before joining its queue.
                connection[1].set()
                self._connections.pop(pid, None)
        cancelled = connection[1] if connection and connection[0] == event.participant_session_id else None
        previous = self._tails.get(pid)

        async def apply() -> None:
            if previous is not None:
                await previous
            if isinstance(event, VideoTrackEvent) and (
                self._closed or cancelled is None or cancelled.is_set()
            ):
                return
            await self._apply(event, cancelled)

        async def wait() -> None:
            if task is not None:
                await asyncio.shield(task)

        task = None if self._closed else asyncio.create_task(apply())
        if task is not None:
            self._tails[event.participant_id] = task
        return wait()

    async def _apply(self, event: ParticipantEvent | VideoTrackEvent, cancelled: asyncio.Event | None) -> None:
        pid = event.participant_id
        if isinstance(event, ParticipantEvent):
            if event.joined:
                if pid in self._sessions and self._sessions[pid] != event.participant_session_id:
                    await self._recorder.finish_recording(pid)
                    self._tracks.pop(pid, None)
                self._sessions[pid] = event.participant_session_id
            elif self._sessions.get(pid) == event.participant_session_id:
                self._sessions.pop(pid, None)
                self._tracks.pop(pid, None)
                await self._recorder.finish_recording(pid)
            return
        if self._sessions.get(pid) != event.participant_session_id:
            return
        tracks = self._tracks.setdefault(pid, set())
        if event.active:
            if event.track_id not in tracks:
                tracks.add(event.track_id)
                await self._recorder.start_recording(pid, cancelled=cancelled)
        else:
            tracks.discard(event.track_id)
            if not tracks:
                await self._recorder.finish_recording(pid)

    async def close(self) -> None:
        self._closed = True
        for _, cancelled in self._connections.values():
            cancelled.set()
        try:
            await asyncio.gather(*self._tails.values())
        finally:
            await self._recorder.stop()
