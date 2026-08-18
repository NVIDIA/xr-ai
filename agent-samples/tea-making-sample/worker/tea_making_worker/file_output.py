# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-scoped JSONL output for the native tea-making agents."""

from __future__ import annotations

import asyncio
import json
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from xr_ai_runtime import Agent, RuntimeContext, subscribe
from xr_ai_voice import VoiceParticipantJoined, VoiceParticipantLeft

from .events import (
    BACKGROUND_FACT_TOPIC,
    CHANGE_WATCH_RECORD_TOPIC,
    FOREGROUND_RECORD_TOPIC,
    GUIDANCE_NOTICE_TOPIC,
    GUIDANCE_RECORD_TOPIC,
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    TRANSCRIPT_RECORD_TOPIC,
    VIDEO_LOG_RECORD_TOPIC,
    BackgroundFact,
    ChangeWatchRecord,
    ForegroundRecord,
    GuidanceNotice,
    GuidanceRecord,
    TranscriptRecord,
    VideoLogRecord,
)

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(slots=True)
class _SessionFiles:
    participant_id: str
    directory: Path
    recent: deque[dict[str, Any]]
    initialized: set[str] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active: bool = True


def _session_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _safe_participant(participant_id: str) -> str:
    return _SAFE.sub("-", participant_id).strip("-._") or "participant"


def _append_lines(path: Path, records: tuple[dict[str, Any], ...]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


class FileOutputAgent(Agent):
    """Own all durable outputs produced by one participant's session."""

    def __init__(self, output_dir: Path, *, history_size: int = 64) -> None:
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._history_size = history_size
        self._sessions: dict[str, _SessionFiles] = {}
        self._sessions_lock = asyncio.Lock()
        self._active: set[str] = set()
        self._closed: set[str] = set()
        super().__init__()

    @subscribe(PARTICIPANT_JOINED_TOPIC)
    async def participant_joined(
        self,
        _event: VoiceParticipantJoined,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = self._participant(ctx)
        async with self._sessions_lock:
            self._closed.discard(participant_id)
            self._active.add(participant_id)
        await self._state(participant_id)

    @subscribe(FOREGROUND_RECORD_TOPIC)
    async def write_foreground(
        self,
        record: ForegroundRecord,
        ctx: RuntimeContext,
    ) -> None:
        await self._write("foreground.jsonl", record, ctx)

    @subscribe(GUIDANCE_RECORD_TOPIC)
    async def write_guidance(
        self,
        record: GuidanceRecord,
        ctx: RuntimeContext,
    ) -> None:
        await self._write("guidance.jsonl", record, ctx)

    @subscribe(GUIDANCE_NOTICE_TOPIC)
    async def write_guidance_notice(
        self,
        record: GuidanceNotice,
        ctx: RuntimeContext,
    ) -> None:
        await self._write("guidance-notices.jsonl", record, ctx)

    @subscribe(BACKGROUND_FACT_TOPIC)
    async def write_background_fact(
        self,
        record: BackgroundFact,
        ctx: RuntimeContext,
    ) -> None:
        await self._write("background-facts.jsonl", record, ctx)

    @subscribe(CHANGE_WATCH_RECORD_TOPIC)
    async def write_change_watch(
        self,
        record: ChangeWatchRecord,
        ctx: RuntimeContext,
    ) -> None:
        await self._write("change-watch.jsonl", record, ctx)

    @subscribe(TRANSCRIPT_RECORD_TOPIC)
    async def write_transcript(
        self,
        record: TranscriptRecord,
        ctx: RuntimeContext,
    ) -> None:
        await self._write("transcript.jsonl", record, ctx)

    @subscribe(VIDEO_LOG_RECORD_TOPIC)
    async def write_video_log(
        self,
        record: VideoLogRecord,
        ctx: RuntimeContext,
    ) -> None:
        await self._write("video-log.jsonl", record, ctx)

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self,
        _event: VoiceParticipantLeft,
        ctx: RuntimeContext,
    ) -> None:
        await self._close_participant(self._participant(ctx))

    async def stop(self) -> None:
        """Close every active session before the runtime shuts down."""

        async with self._sessions_lock:
            participant_ids = tuple(self._sessions)
        for participant_id in participant_ids:
            await self._close_participant(participant_id)

    async def recent_records(
        self,
        participant_id: str,
    ) -> tuple[dict[str, Any], ...]:
        """Return a snapshot of the bounded in-memory output index."""

        async with self._sessions_lock:
            state = self._sessions.get(participant_id)
        if state is None:
            return ()
        async with state.lock:
            return tuple(dict(record) for record in state.recent)

    async def _write(
        self,
        filename: str,
        record: BaseModel,
        ctx: RuntimeContext,
    ) -> None:
        state = await self._state(self._participant(ctx))
        if state is None:
            return
        payload = record.model_dump(mode="json")
        async with state.lock:
            if not state.active:
                return
            await self._append(state, filename, payload)
            state.recent.append({"file": filename, **payload})

    async def _state(self, participant_id: str) -> _SessionFiles | None:
        async with self._sessions_lock:
            existing = self._sessions.get(participant_id)
            if existing is not None:
                return existing
            if participant_id in self._closed:
                return None
            # Runtime subscribers fan out concurrently. Another agent can
            # publish its join record before this agent receives the same join.
            self._active.add(participant_id)
            directory = self.output_dir / (
                f"{_safe_participant(participant_id)}-{_session_stamp()}"
            )
            await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=False)
            state = _SessionFiles(
                participant_id=participant_id,
                directory=directory,
                recent=deque(maxlen=self._history_size),
            )
            self._sessions[participant_id] = state
            return state

    async def _append(
        self,
        state: _SessionFiles,
        filename: str,
        record: dict[str, Any],
    ) -> None:
        records: list[dict[str, Any]] = []
        if filename not in state.initialized:
            state.initialized.add(filename)
            records.append(
                {
                    "type": "session",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "participant_id": state.participant_id,
                }
            )
        records.append(record)
        await asyncio.to_thread(
            _append_lines,
            state.directory / filename,
            tuple(records),
        )

    async def _close_participant(self, participant_id: str) -> None:
        async with self._sessions_lock:
            state = self._sessions.pop(participant_id, None)
            self._active.discard(participant_id)
            self._closed.add(participant_id)
        if state is None:
            return
        end = {
            "type": "session_end",
            "timestamp": datetime.now(UTC).isoformat(),
            "participant_id": participant_id,
        }
        async with state.lock:
            state.active = False
            for filename in sorted(state.initialized):
                await asyncio.to_thread(
                    _append_lines,
                    state.directory / filename,
                    (end,),
                )

    @staticmethod
    def _participant(ctx: RuntimeContext) -> str:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("file output requires a participant")
        return participant_id


__all__ = ["FileOutputAgent"]
