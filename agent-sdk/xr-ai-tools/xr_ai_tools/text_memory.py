# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-scoped persistent text-memory tools."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from threading import Lock
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .tools import Tool
from .types import EmptyRequest, StrictRequest

_LOGGER = logging.getLogger(__name__)
_MAX_TIMESTAMP_US = 9_223_372_036_854_775_807


class AddTranscriptRequest(StrictRequest):
    """One timestamped transcript segment to persist."""

    source_id: str = Field(description="Participant or internal source identifier.")
    """Participant or internal source identifier."""

    timestamp_us: int = Field(description="Unix timestamp in microseconds.")
    """Unix timestamp in microseconds."""

    text: str = Field(min_length=1, description="Text segment to persist.")
    """Text segment to persist."""

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        """Require visible text in a persisted segment."""

        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class AddTranscriptResult(BaseModel):
    """Confirmation that a transcript segment was persisted."""

    ok: bool = True
    """Whether the append operation succeeded."""


class TranscriptSegment(BaseModel):
    """Persisted transcript text at a point in time."""

    timestamp_us: int
    """Unix timestamp in microseconds."""

    text: str
    """Persisted transcript text."""


class QueryTranscriptsRequest(StrictRequest):
    """Source and inclusive time window for a transcript query."""

    source_id: str = Field(description="Participant or internal source identifier.")
    """Participant or internal source identifier."""

    start_us: int = Field(description="Inclusive window start in Unix microseconds.")
    """Inclusive window start in Unix microseconds."""

    end_us: int = Field(description="Inclusive window end in Unix microseconds.")
    """Inclusive window end in Unix microseconds."""


class QueryTranscriptsResult(BaseModel):
    """Transcript segments selected from one source and time window."""

    segments: list[TranscriptSegment]
    """Matching segments ordered by timestamp."""


class ListTranscriptSourcesResult(BaseModel):
    """Persistent source identifiers known to text memory."""

    sources: list[str]
    """Source identifiers in lexical order."""


class TranscriptStatsRequest(StrictRequest):
    """Select one transcript source for statistics."""

    source_id: str = Field(description="Participant or internal source identifier.")
    """Participant or internal source identifier."""


class TranscriptStatsResult(BaseModel):
    """Aggregate storage and time-range statistics for one source."""

    source_id: str
    """Participant or internal source identifier."""

    count: int
    """Number of stored transcript segments."""

    total_chars: int
    """Total number of characters across stored segments."""

    earliest_us: int | None
    """Earliest stored Unix timestamp in microseconds, if present."""

    latest_us: int | None
    """Latest stored Unix timestamp in microseconds, if present."""


class RecallConversationRequest(StrictRequest):
    """Participant and inclusive time window for conversation recall."""

    participant_id: str = Field(description="Participant whose conversation to recall.")
    """Participant whose conversation to recall."""

    start_us: int = Field(default=0, description="Inclusive window start in Unix microseconds.")
    """Inclusive window start in Unix microseconds."""

    end_us: int = Field(
        default=_MAX_TIMESTAMP_US,
        description="Inclusive window end in Unix microseconds.",
    )
    """Inclusive window end in Unix microseconds."""


class ConversationEntry(BaseModel):
    """One timestamped user or agent turn in a recalled conversation."""

    timestamp_us: int
    """Unix timestamp in microseconds."""

    role: Literal["user", "agent"]
    """Speaker responsible for the text."""

    text: str
    """Conversation text."""


class RecallConversationResult(BaseModel):
    """Chronological user and agent turns recalled for a participant."""

    entries: list[ConversationEntry]
    """Conversation turns ordered by timestamp."""


class _TranscriptStore:
    """Append-only JSONL transcript storage keyed by arbitrary source IDs."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._root = self.directory.resolve()
        self._lock = Lock()

    @staticmethod
    def _safe(source_id: str) -> str:
        return "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in source_id
        )

    def _check(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self._root):
            raise ValueError(f"Path escapes transcript directory: {path}")
        return resolved

    def _path(self, source_id: str, *, create: bool) -> Path | None:
        stem = self._safe(source_id)
        if not create:
            canonical_data = self._check(self.directory / f"{stem}.jsonl")
            canonical_identity = self._check(self.directory / f"{stem}.identity")
            if canonical_data.exists() and (
                (
                    canonical_identity.exists()
                    and canonical_identity.read_text(encoding="utf-8") == source_id
                )
                or (not canonical_identity.exists() and source_id == stem)
            ):
                return canonical_data
            for identity_path in sorted(self.directory.glob("*.identity")):
                identity = self._check(identity_path)
                if identity.read_text(encoding="utf-8") == source_id:
                    data = self._check(identity.with_suffix(".jsonl"))
                    return data if data.exists() else None
            return None

        suffix = 1
        while True:
            candidate = stem if suffix == 1 else f"{stem}_{suffix}"
            identity = self._check(self.directory / f"{candidate}.identity")
            data = self._check(self.directory / f"{candidate}.jsonl")
            if identity.exists() and identity.read_text(encoding="utf-8") == source_id:
                return data
            if suffix == 1 and data.exists() and not identity.exists() and source_id == stem:
                identity.write_text(source_id, encoding="utf-8")
                return data
            if not identity.exists() and not data.exists():
                identity.write_text(source_id, encoding="utf-8")
                return data
            suffix += 1

    def append(self, source_id: str, timestamp_us: int, text: str) -> None:
        with self._lock:
            path = self._path(source_id, create=True)
            assert path is not None
            with path.open("a", encoding="utf-8") as output:
                output.write(json.dumps({"timestamp_us": timestamp_us, "text": text}) + "\n")

    def query(self, source_id: str, start_us: int, end_us: int) -> list[TranscriptSegment]:
        with self._lock:
            path = self._path(source_id, create=False)
            if path is None or not path.exists():
                return []
            lines = path.read_text(encoding="utf-8").splitlines()
        segments: list[TranscriptSegment] = []
        skipped = 0
        for line in lines:
            if not line.strip():
                continue
            try:
                segments.append(TranscriptSegment.model_validate_json(line))
            except ValueError:
                skipped += 1
        if skipped:
            _LOGGER.warning("Skipped %d corrupt transcript lines in %s", skipped, path)
        return sorted(
            (
                segment
                for segment in segments
                if start_us <= segment.timestamp_us <= end_us
            ),
            key=lambda segment: segment.timestamp_us,
        )

    def list_sources(self) -> list[str]:
        with self._lock:
            sources: set[str] = set()
            identified: set[str] = set()
            for identity_path in self.directory.glob("*.identity"):
                identity = self._check(identity_path)
                identified.add(identity.stem)
                if identity.with_suffix(".jsonl").exists():
                    sources.add(identity.read_text(encoding="utf-8"))
            for data_path in self.directory.glob("*.jsonl"):
                data = self._check(data_path)
                if data.stem not in identified:
                    sources.add(data.stem)
        return sorted(sources)

    def stats(self, source_id: str) -> TranscriptStatsResult:
        segments = self.query(source_id, 0, _MAX_TIMESTAMP_US)
        return TranscriptStatsResult(
            source_id=source_id,
            count=len(segments),
            total_chars=sum(len(segment.text) for segment in segments),
            earliest_us=segments[0].timestamp_us if segments else None,
            latest_us=segments[-1].timestamp_us if segments else None,
        )


class TextMemoryTool(Tool[AddTranscriptRequest, AddTranscriptResult]):
    """Append timestamped text to participant-scoped JSONL storage."""

    def __init__(self, directory: str | Path | _TranscriptStore) -> None:
        self._store = (
            directory if isinstance(directory, _TranscriptStore) else _TranscriptStore(directory)
        )
        super().__init__(
            "add_transcript",
            "Append one timestamped text segment to persistent memory.",
            AddTranscriptRequest,
            AddTranscriptResult,
            self._append,
        )

    async def _append(self, request: AddTranscriptRequest) -> AddTranscriptResult:
        await asyncio.to_thread(
            self._store.append,
            request.source_id,
            request.timestamp_us,
            request.text,
        )
        return AddTranscriptResult()


class TextMemoryTools:
    """Own transcript persistence, query, statistics, and conversation recall tools."""

    def __init__(self, directory: str | Path) -> None:
        self._store = _TranscriptStore(directory)
        self.add_transcript = TextMemoryTool(self._store)
        self.query_transcripts = Tool(
            "query_transcripts",
            "Return ordered text segments for one source and inclusive time window.",
            QueryTranscriptsRequest,
            QueryTranscriptsResult,
            self._query,
        )
        self.list_sources = Tool(
            "list_sources",
            "List source identifiers that have persistent text memory.",
            EmptyRequest,
            ListTranscriptSourcesResult,
            self._list_sources,
        )
        self.get_transcript_stats = Tool(
            "get_transcript_stats",
            "Return count, character, and time-range statistics for one source.",
            TranscriptStatsRequest,
            TranscriptStatsResult,
            self._stats,
        )
        self.recall_conversation = Tool(
            "recall_conversation",
            "Recall timestamped user and agent turns for one participant and time window.",
            RecallConversationRequest,
            RecallConversationResult,
            self._recall,
        )
        self.tools = (
            self.add_transcript,
            self.query_transcripts,
            self.list_sources,
            self.get_transcript_stats,
            self.recall_conversation,
        )

    async def _query(self, request: QueryTranscriptsRequest) -> QueryTranscriptsResult:
        segments = await asyncio.to_thread(
            self._store.query,
            request.source_id,
            request.start_us,
            request.end_us,
        )
        return QueryTranscriptsResult(segments=segments)

    async def _list_sources(self, _request: EmptyRequest) -> ListTranscriptSourcesResult:
        return ListTranscriptSourcesResult(
            sources=await asyncio.to_thread(self._store.list_sources)
        )

    async def _stats(self, request: TranscriptStatsRequest) -> TranscriptStatsResult:
        return await asyncio.to_thread(self._store.stats, request.source_id)

    async def _recall(self, request: RecallConversationRequest) -> RecallConversationResult:
        entries: list[ConversationEntry] = []
        for role in ("user", "agent"):
            segments = await asyncio.to_thread(
                self._store.query,
                f"{request.participant_id}:{role}",
                request.start_us,
                request.end_us,
            )
            entries.extend(
                ConversationEntry(
                    timestamp_us=segment.timestamp_us,
                    role=role,
                    text=segment.text,
                )
                for segment in segments
            )
        entries.sort(key=lambda entry: entry.timestamp_us)
        return RecallConversationResult(entries=entries)


__all__ = [
    "AddTranscriptRequest",
    "AddTranscriptResult",
    "ConversationEntry",
    "ListTranscriptSourcesResult",
    "QueryTranscriptsRequest",
    "QueryTranscriptsResult",
    "RecallConversationRequest",
    "RecallConversationResult",
    "TextMemoryTool",
    "TextMemoryTools",
    "TranscriptSegment",
    "TranscriptStatsRequest",
    "TranscriptStatsResult",
]
