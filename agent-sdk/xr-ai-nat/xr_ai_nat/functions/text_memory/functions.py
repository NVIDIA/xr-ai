# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public native NAT functions for persistent transcript storage."""

import asyncio
import json
import logging
from pathlib import Path
from threading import Lock
from typing import Literal

from nat.plugin_api import (
    Builder,
    FunctionGroup,
    FunctionGroupBaseConfig,
    FunctionGroupRef,
    register_function_group,
)
from pydantic import BaseModel, Field, field_validator

from .._models import _StrictRequest

_LOGGER = logging.getLogger(__name__)


class AddTranscriptRequest(_StrictRequest):
    source_id: str = Field(description="Participant or internal source identifier.")
    timestamp_us: int = Field(description="Unix timestamp in microseconds.")
    text: str = Field(min_length=1, description="Text segment to persist.")

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class AddTranscriptResult(BaseModel):
    ok: bool = True


class TranscriptSegment(BaseModel):
    timestamp_us: int
    text: str


class QueryTranscriptsRequest(_StrictRequest):
    source_id: str = Field(description="Participant or internal source identifier.")
    start_us: int = Field(description="Inclusive window start in Unix microseconds.")
    end_us: int = Field(description="Inclusive window end in Unix microseconds.")


class QueryTranscriptsResult(BaseModel):
    segments: list[TranscriptSegment]


class ListTranscriptSourcesRequest(_StrictRequest):
    pass


class ListTranscriptSourcesResult(BaseModel):
    sources: list[str]


class TranscriptStatsRequest(_StrictRequest):
    source_id: str = Field(description="Participant or internal source identifier.")


class TranscriptStatsResult(BaseModel):
    source_id: str
    count: int
    total_chars: int
    earliest_us: int | None
    latest_us: int | None


class _TranscriptStore:
    """Append-only JSONL transcript storage keyed by arbitrary source IDs."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._root = self.directory.resolve()
        self._lock = Lock()

    @staticmethod
    def _safe(source_id: str) -> str:
        return "".join(character if character.isalnum() or character in "-_." else "_" for character in source_id)

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
                (canonical_identity.exists() and canonical_identity.read_text(encoding="utf-8") == source_id)
                or (not canonical_identity.exists() and source_id == stem)
            ):
                return canonical_data
            for identity in sorted(self.directory.glob("*.identity")):
                identity = self._check(identity)
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
            (segment for segment in segments if start_us <= segment.timestamp_us <= end_us),
            key=lambda segment: segment.timestamp_us,
        )

    def list_sources(self) -> list[str]:
        with self._lock:
            sources: set[str] = set()
            identified: set[str] = set()
            for identity in self.directory.glob("*.identity"):
                identity = self._check(identity)
                identified.add(identity.stem)
                if identity.with_suffix(".jsonl").exists():
                    sources.add(identity.read_text(encoding="utf-8"))
            for data in self.directory.glob("*.jsonl"):
                data = self._check(data)
                if data.stem not in identified:
                    sources.add(data.stem)
        return sorted(sources)

    def stats(self, source_id: str) -> TranscriptStatsResult:
        segments = self.query(source_id, 0, 9_223_372_036_854_775_807)
        return TranscriptStatsResult(
            source_id=source_id,
            count=len(segments),
            total_chars=sum(len(segment.text) for segment in segments),
            earliest_us=segments[0].timestamp_us if segments else None,
            latest_us=segments[-1].timestamp_us if segments else None,
        )


class RecallConversationRequest(_StrictRequest):
    participant_id: str = Field(description="Participant whose conversation to recall.")
    start_us: int = Field(default=0, description="Inclusive window start in Unix microseconds.")
    end_us: int = Field(
        default=9_223_372_036_854_775_807,
        description="Inclusive window end in Unix microseconds.",
    )


class ConversationEntry(BaseModel):
    """One recalled turn of a participant's conversation."""

    timestamp_us: int = Field(
        description="Unix-epoch microseconds when the turn was spoken."
    )
    role: Literal["user", "agent"] = Field(
        description="Who produced the turn: the participant ('user') or the agent ('agent')."
    )
    text: str = Field(description="Verbatim text of the turn.")


class RecallConversationResult(BaseModel):
    """A participant's recalled turns."""

    entries: list[ConversationEntry] = Field(
        description=(
            "Recalled turns in ascending time order. A user turn and the agent turn "
            "answering it share one timestamp, and the user turn is ordered first. "
            "Empty when the participant has no stored turns in the window."
        )
    )


class TextMemoryFunctionsConfig(FunctionGroupBaseConfig, name="xr_text_memory"):
    """Configure transcript functions over one persistent directory."""

    directory: str | Path


class ConversationMemoryFunctionsConfig(FunctionGroupBaseConfig, name="xr_conversation_memory"):
    """Configure participant-oriented conversation recall over the transcript store."""

    text_memory: FunctionGroupRef = Field(
        default=FunctionGroupRef("text_memory"),
        description=(
            "Instance name of the xr_text_memory function group holding the transcripts. "
            "Recall reads the role-scoped sources '{participant_id}:user' and "
            "'{participant_id}:agent' from it; it never touches storage directly."
        ),
    )


@register_function_group(config_type=TextMemoryFunctionsConfig)
async def text_memory_functions(config: TextMemoryFunctionsConfig, _builder: Builder):
    """Build the native transcript group over one persistent store."""

    store = _TranscriptStore(config.directory)

    async def add(request: AddTranscriptRequest) -> AddTranscriptResult:
        await asyncio.to_thread(store.append, request.source_id, request.timestamp_us, request.text)
        return AddTranscriptResult()

    async def query(request: QueryTranscriptsRequest) -> QueryTranscriptsResult:
        segments = await asyncio.to_thread(store.query, request.source_id, request.start_us, request.end_us)
        return QueryTranscriptsResult(segments=segments)

    async def list_sources(request: ListTranscriptSourcesRequest) -> ListTranscriptSourcesResult:
        del request
        return ListTranscriptSourcesResult(sources=await asyncio.to_thread(store.list_sources))

    async def stats(request: TranscriptStatsRequest) -> TranscriptStatsResult:
        return await asyncio.to_thread(store.stats, request.source_id)

    group = FunctionGroup(config=config)
    group.add_function(
        "add_transcript",
        add,
        description="Persist one timestamped text segment for a source.",
    )
    group.add_function(
        "query_transcripts",
        query,
        description="Return ordered text segments for one source and inclusive time window.",
    )
    group.add_function(
        "list_sources",
        list_sources,
        description="List source identifiers that have persistent text memory.",
    )
    group.add_function(
        "get_transcript_stats",
        stats,
        description="Return count, character, and time-range statistics for one source.",
    )
    yield group


@register_function_group(config_type=ConversationMemoryFunctionsConfig)
async def conversation_memory_functions(config: ConversationMemoryFunctionsConfig, builder: Builder):
    """Build participant conversation recall over the transcript store.

    Reads the ``{participant_id}:user`` and ``{participant_id}:agent`` transcript
    sources produced by ``xr_ai_nat.adapters.voice.record_voice_transcripts``.
    """

    text_memory = await builder.get_function_group(config.text_memory)
    text_memory_functions = await text_memory.get_all_functions()
    query = text_memory_functions[f"{text_memory.instance_name}__query_transcripts"]

    async def recall(request: RecallConversationRequest) -> RecallConversationResult:
        entries: list[ConversationEntry] = []
        for role in ("user", "agent"):
            result = await query.ainvoke(
                QueryTranscriptsRequest(
                    source_id=f"{request.participant_id}:{role}",
                    start_us=request.start_us,
                    end_us=request.end_us,
                )
            )
            entries.extend(
                ConversationEntry(timestamp_us=segment.timestamp_us, role=role, text=segment.text)
                for segment in result.segments
            )
        entries.sort(key=lambda entry: entry.timestamp_us)
        return RecallConversationResult(entries=entries)

    group = FunctionGroup(config=config)
    group.add_function(
        "recall_conversation",
        recall,
        description="Recall timestamped user and agent turns for one participant and time window.",
    )
    yield group


__all__ = [
    "AddTranscriptRequest",
    "AddTranscriptResult",
    "ConversationEntry",
    "ConversationMemoryFunctionsConfig",
    "ListTranscriptSourcesRequest",
    "ListTranscriptSourcesResult",
    "QueryTranscriptsRequest",
    "QueryTranscriptsResult",
    "RecallConversationRequest",
    "RecallConversationResult",
    "TextMemoryFunctionsConfig",
    "TranscriptSegment",
    "TranscriptStatsRequest",
    "TranscriptStatsResult",
]
