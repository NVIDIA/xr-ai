# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for native text-memory tools."""

from pathlib import Path

import pytest
from xr_ai_tools.text_memory import (
    AddTranscriptRequest,
    QueryTranscriptsRequest,
    RecallConversationRequest,
    TextMemoryTools,
    TranscriptStatsRequest,
    _TranscriptStore,
)
from xr_ai_tools.types import EmptyRequest


async def test_text_memory_tools_persist_query_and_summarize(tmp_path: Path) -> None:
    memory = TextMemoryTools(tmp_path)

    await memory.add_transcript.execute(
        AddTranscriptRequest(source_id="alice@home", timestamp_us=20, text="later")
    )
    await memory.add_transcript.execute(
        AddTranscriptRequest(source_id="alice@home", timestamp_us=10, text="first")
    )

    segments = await memory.query_transcripts.execute(
        QueryTranscriptsRequest(source_id="alice@home", start_us=0, end_us=20)
    )
    source_ids = await memory.list_sources.execute(EmptyRequest())
    summary = await memory.get_transcript_stats.execute(
        TranscriptStatsRequest(source_id="alice@home")
    )
    missing = await memory.get_transcript_stats.execute(
        TranscriptStatsRequest(source_id="missing")
    )

    assert [segment.text for segment in segments.segments] == ["first", "later"]
    assert source_ids.sources == ["alice@home"]
    assert summary.model_dump() == {
        "source_id": "alice@home",
        "count": 2,
        "total_chars": 10,
        "earliest_us": 10,
        "latest_us": 20,
    }
    assert missing.model_dump() == {
        "source_id": "missing",
        "count": 0,
        "total_chars": 0,
        "earliest_us": None,
        "latest_us": None,
    }
    assert (tmp_path / "alice_home.identity").read_text() == "alice@home"


async def test_text_memory_recalls_role_scoped_conversation(tmp_path: Path) -> None:
    memory = TextMemoryTools(tmp_path)
    for source, timestamp, text in (
        ("alice:user", 10, "hello"),
        ("alice:agent", 10, "hi"),
        ("alice:user", 20, "remember this"),
    ):
        await memory.add_transcript.execute(
            AddTranscriptRequest(source_id=source, timestamp_us=timestamp, text=text)
        )

    result = await memory.recall_conversation.execute(
        RecallConversationRequest(participant_id="alice", end_us=10)
    )

    assert [(entry.role, entry.text) for entry in result.entries] == [
        ("user", "hello"),
        ("agent", "hi"),
    ]


async def test_text_memory_disambiguates_sanitized_source_names(tmp_path: Path) -> None:
    memory = TextMemoryTools(tmp_path)
    await memory.add_transcript.execute(
        AddTranscriptRequest(source_id="room/a", timestamp_us=1, text="slash")
    )
    await memory.add_transcript.execute(
        AddTranscriptRequest(source_id="room?a", timestamp_us=2, text="question")
    )

    slash = await memory.query_transcripts.execute(
        QueryTranscriptsRequest(source_id="room/a", start_us=0, end_us=10)
    )
    question = await memory.query_transcripts.execute(
        QueryTranscriptsRequest(source_id="room?a", start_us=0, end_us=10)
    )

    assert [segment.text for segment in slash.segments] == ["slash"]
    assert [segment.text for segment in question.segments] == ["question"]
    assert (tmp_path / "room_a.identity").read_text() == "room/a"
    assert (tmp_path / "room_a_2.identity").read_text() == "room?a"


def test_text_memory_store_does_not_follow_identity_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "transcripts"
    outside = tmp_path / "outside.identity"
    outside.write_text("malicious", encoding="utf-8")
    store = _TranscriptStore(root)
    (root / "malicious.jsonl").write_text("", encoding="utf-8")
    (root / "malicious.identity").symlink_to(outside)

    with pytest.raises(ValueError, match="escapes transcript directory"):
        store.query("malicious", 0, 1)
