# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-scoped JSONL persistence and recent monitoring lookup."""

from __future__ import annotations

import asyncio
import json
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from xr_ai_runtime import Agent, RuntimeContext, subscribe
from xr_ai_tools import Tool
from xr_ai_voice import (
    VOICE_TRANSCRIPT_TOPIC,
    VoiceParticipantLeft,
    VoiceTranscript,
)

from .events import (
    FOREGROUND_RECORD_TOPIC,
    INSTRUMENT_READING_TOPIC,
    MONITOR_RECORD_TOPIC,
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    ForegroundRecord,
    InstrumentReading,
    MonitorRecord,
    ParticipantJoined,
)

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


class MonitoringHistoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    participant_id: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)


class MonitoringHistoryResult(BaseModel):
    observations: list[MonitorRecord]


@dataclass(slots=True)
class _SessionFiles:
    participant_id: str
    directory: Path
    monitor: deque[MonitorRecord]
    initialized: set[str] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active: bool = True


def _session_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _safe_participant(participant_id: str) -> str:
    return _SAFE.sub("-", participant_id).strip("-._") or "participant"


def _append_lines(path: Path, records: tuple[dict, ...]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


class FileOutputAgent(Agent):
    """Own every durable sample output and its bounded in-memory index."""

    def __init__(self, output_dir: Path, *, history_size: int) -> None:
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._history_size = history_size
        self._sessions: dict[str, _SessionFiles] = {}
        self._sessions_lock = asyncio.Lock()
        self._active: set[str] = set()
        self.read_monitoring_history = Tool(
            "read_monitoring_history",
            "Return recent persisted visual-monitor observations for one participant.",
            MonitoringHistoryRequest,
            MonitoringHistoryResult,
            self._read_monitoring_history,
        )
        super().__init__((self.read_monitoring_history,))

    @subscribe(PARTICIPANT_JOINED_TOPIC)
    async def participant_joined(
        self,
        _event: ParticipantJoined,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = self._participant(ctx)
        async with self._sessions_lock:
            self._active.add(participant_id)
        await self._state(participant_id)

    @subscribe(MONITOR_RECORD_TOPIC)
    async def write_monitor(self, record: MonitorRecord, ctx: RuntimeContext) -> None:
        participant_id = self._participant(ctx)
        state = await self._state(participant_id)
        if state is None:
            return
        async with state.lock:
            if not state.active:
                return
            await self._append(state, "monitor.jsonl", record.model_dump(mode="json"))
            state.monitor.append(record.model_copy(deep=True))

    @subscribe(INSTRUMENT_READING_TOPIC)
    async def write_instrument_reading(
        self,
        record: InstrumentReading,
        ctx: RuntimeContext,
    ) -> None:
        state = await self._state(self._participant(ctx))
        if state is None:
            return
        async with state.lock:
            if state.active:
                await self._append(
                    state,
                    "instrument-readings.jsonl",
                    record.model_dump(mode="json"),
                )

    @subscribe(VOICE_TRANSCRIPT_TOPIC)
    async def write_transcript(
        self,
        record: VoiceTranscript,
        ctx: RuntimeContext,
    ) -> None:
        state = await self._state(self._participant(ctx))
        if state is None:
            return
        async with state.lock:
            if not state.active:
                return
            await self._append(
                state,
                "transcript.jsonl",
                record.model_dump(mode="json"),
            )

    @subscribe(FOREGROUND_RECORD_TOPIC)
    async def write_foreground(
        self,
        record: ForegroundRecord,
        ctx: RuntimeContext,
    ) -> None:
        state = await self._state(self._participant(ctx))
        if state is None:
            return
        async with state.lock:
            if not state.active:
                return
            await self._append(
                state,
                "foreground.jsonl",
                record.model_dump(mode="json"),
            )

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self,
        _event: VoiceParticipantLeft,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = self._participant(ctx)
        async with self._sessions_lock:
            state = self._sessions.pop(participant_id, None)
            self._active.discard(participant_id)
        if state is None:
            return
        end = {
            "type": "session_end",
            "timestamp": datetime.now(UTC).isoformat(),
            "participant_id": participant_id,
        }
        async with state.lock:
            state.active = False
            for name in sorted(state.initialized):
                await asyncio.to_thread(_append_lines, state.directory / name, (end,))

    async def _read_monitoring_history(
        self,
        request: MonitoringHistoryRequest,
    ) -> MonitoringHistoryResult:
        async with self._sessions_lock:
            state = self._sessions.get(request.participant_id)
        if state is None:
            return MonitoringHistoryResult(observations=[])
        async with state.lock:
            return MonitoringHistoryResult(
                observations=[item.model_copy(deep=True) for item in tuple(state.monitor)[-request.limit :]]
            )

    async def _state(self, participant_id: str) -> _SessionFiles | None:
        async with self._sessions_lock:
            existing = self._sessions.get(participant_id)
            if existing is not None:
                return existing
            if participant_id not in self._active:
                return None
            directory = self.output_dir / (f"{_safe_participant(participant_id)}-{_session_stamp()}")
            await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=False)
            state = _SessionFiles(
                participant_id=participant_id,
                directory=directory,
                monitor=deque(maxlen=self._history_size),
            )
            self._sessions[participant_id] = state
            return state

    async def _append(
        self,
        state: _SessionFiles,
        name: str,
        record: dict,
    ) -> None:
        records: list[dict] = []
        if name not in state.initialized:
            state.initialized.add(name)
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
            state.directory / name,
            tuple(records),
        )

    @staticmethod
    def _participant(ctx: RuntimeContext) -> str:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("file output requires a participant")
        return participant_id


__all__ = [
    "FileOutputAgent",
    "MonitoringHistoryRequest",
    "MonitoringHistoryResult",
]
