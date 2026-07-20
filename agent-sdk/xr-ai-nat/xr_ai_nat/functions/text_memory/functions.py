# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native NAT functions for persistent timestamped text."""

import asyncio
from pathlib import Path
from typing import Annotated

from nat.plugin_api import Builder, FunctionGroup, FunctionGroupBaseConfig, register_function_group
from pydantic import BaseModel, Field

from ._store import TextMemoryStore
from .schemas import OperationResult, TextMemoryError, TranscriptSegment, TranscriptStats


class _EmptyInput(BaseModel):
    """Explicit empty request keeps the zero-argument function's schema object-shaped."""

    pass


class TextMemoryFunctionsConfig(FunctionGroupBaseConfig, name="xr_text_memory"):
    """Configure text-memory functions over one persistent directory."""

    directory: str | Path


def _function_group(config: TextMemoryFunctionsConfig, store: TextMemoryStore) -> FunctionGroup:
    group = FunctionGroup(config=config)

    async def add_transcript(
        source_id: Annotated[str, Field(description="Participant or internal source identifier.")],
        timestamp_us: Annotated[int, Field(description="Unix timestamp in microseconds.")],
        text: str,
    ) -> OperationResult | TextMemoryError:
        if not text.strip():
            return TextMemoryError(error="text must not be empty")
        await asyncio.to_thread(store.append, source_id, timestamp_us, text)
        return OperationResult()

    group.add_function(
        "add_transcript",
        add_transcript,
        description="Persist one timestamped text segment for a source.",
    )

    async def query_transcripts(
        source_id: Annotated[str, Field(description="Participant or internal source identifier.")],
        start_us: Annotated[int, Field(description="Inclusive window start in Unix microseconds.")],
        end_us: Annotated[int, Field(description="Inclusive window end in Unix microseconds.")],
    ) -> list[TranscriptSegment]:
        return await asyncio.to_thread(store.query, source_id, start_us, end_us)

    group.add_function(
        "query_transcripts",
        query_transcripts,
        description="Return ordered text segments for one source and inclusive time window.",
    )

    async def list_sources(request: _EmptyInput) -> list[str]:
        del request
        return await asyncio.to_thread(store.list_sources)

    group.add_function(
        "list_sources",
        list_sources,
        description="List source identifiers that have persistent text memory.",
    )

    async def get_transcript_stats(source_id: str) -> TranscriptStats | TextMemoryError:
        result = await asyncio.to_thread(store.stats, source_id)
        if result is None:
            return TextMemoryError(error=f"No transcripts for {source_id!r}")
        return result

    group.add_function(
        "get_transcript_stats",
        get_transcript_stats,
        description="Return count, character, and time-range statistics for one source.",
    )

    return group


@register_function_group(config_type=TextMemoryFunctionsConfig)
async def text_memory_functions(config: TextMemoryFunctionsConfig, _builder: Builder):
    yield _function_group(config, TextMemoryStore(config.directory))


__all__ = ["TextMemoryFunctionsConfig"]
