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
        self._closed = False

    def receive(self, event: ParticipantEvent | VideoTrackEvent) -> Awaitable[None]:
        # Enqueue synchronously: ProcessorEndpoint detaches the returned
        # awaitable, but event order must survive slow manifest finalization.
        previous = self._tails.get(event.participant_id)

        async def apply() -> None:
            if previous is not None:
                await previous
            await self._apply(event)

        async def wait() -> None:
            if task is not None:
                await asyncio.shield(task)

        task = None if self._closed else asyncio.create_task(apply())
        if task is not None:
            self._tails[event.participant_id] = task
        return wait()

    async def _apply(self, event: ParticipantEvent | VideoTrackEvent) -> None:
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
                await self._recorder.start_recording(pid)
        else:
            tracks.discard(event.track_id)
            if not tracks:
                await self._recorder.finish_recording(pid)

    async def close(self) -> None:
        self._closed = True
        try:
            await asyncio.gather(*self._tails.values())
        finally:
            await self._recorder.stop()
