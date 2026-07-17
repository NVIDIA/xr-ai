# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for native text memory and its generic MCP adapter."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastmcp import Client as McpClient
from nat.builder.workflow_builder import WorkflowBuilder
from xr_ai_nat.adapters.mcp import create_mcp_server
from xr_ai_nat.functions.text_memory import TextMemoryError, TextMemoryFunctionsConfig


@asynccontextmanager
async def _functions(directory):
    async with WorkflowBuilder() as builder:
        await builder.add_function_group(
            "text_memory",
            TextMemoryFunctionsConfig(directory=directory),
        )
        group = await builder.get_function_group("text_memory")
        yield await group.get_all_functions()


async def test_text_memory_functions_persist_query_and_summarize(tmp_path) -> None:
    async with _functions(tmp_path) as functions:
        assert set(functions) == {
            "text_memory__add_transcript",
            "text_memory__get_transcript_stats",
            "text_memory__list_sources",
            "text_memory__query_transcripts",
        }
        add = functions["text_memory__add_transcript"]
        query = functions["text_memory__query_transcripts"]
        sources = functions["text_memory__list_sources"]
        stats = functions["text_memory__get_transcript_stats"]
        assert sources.input_schema.model_json_schema()["properties"] == {}

        await add.ainvoke({"source_id": "alice@home", "timestamp_us": 20, "text": "later"})
        await add.ainvoke({"source_id": "alice@home", "timestamp_us": 10, "text": "first"})

        segments = await query.ainvoke(
            {"source_id": "alice@home", "start_us": 0, "end_us": 20}
        )
        source_ids = await sources.ainvoke({})
        summary = await stats.ainvoke({"source_id": "alice@home"})
        missing = await stats.ainvoke({"source_id": "missing"})
    assert [segment.text for segment in segments] == ["first", "later"]
    assert source_ids == ["alice@home"]
    assert summary.model_dump() == {
        "source_id": "alice@home",
        "count": 2,
        "total_chars": 10,
        "earliest_us": 10,
        "latest_us": 20,
    }
    assert isinstance(missing, TextMemoryError)
    assert (tmp_path / "alice_home.identity").read_text() == "alice@home"


async def test_generic_mcp_adapter_preserves_list_and_object_results(tmp_path) -> None:
    async with _functions(tmp_path) as functions:
        exports = [
            functions["text_memory__add_transcript"],
            functions["text_memory__query_transcripts"],
        ]
        server = create_mcp_server(
            "text-memory-test",
            exports,
            tool_names={
                "text_memory__add_transcript": "add_transcript",
                "text_memory__query_transcripts": "query_transcripts",
            },
        )
        async with McpClient(server) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            added = await client.call_tool(
                "add_transcript",
                {"source_id": "agent", "timestamp_us": 100, "text": "remember this"},
            )
            queried = await client.call_tool(
                "query_transcripts",
                {"source_id": "agent", "start_us": 0, "end_us": 200},
            )
    assert set(tools) == {"add_transcript", "query_transcripts"}
    assert set(tools["add_transcript"].inputSchema["properties"]) == {
        "source_id",
        "timestamp_us",
        "text",
    }
    assert added.structured_content == {"result": {"ok": True}}
    assert queried.structured_content == {
        "result": [{"timestamp_us": 100, "text": "remember this"}]
    }


async def test_text_memory_disambiguates_sanitized_source_names(tmp_path) -> None:
    async with _functions(tmp_path) as functions:
        add = functions["text_memory__add_transcript"]
        query = functions["text_memory__query_transcripts"]
        await add.ainvoke({"source_id": "room/a", "timestamp_us": 1, "text": "slash"})
        await add.ainvoke({"source_id": "room?a", "timestamp_us": 2, "text": "question"})
        slash = await query.ainvoke({"source_id": "room/a", "start_us": 0, "end_us": 10})
        question = await query.ainvoke(
            {"source_id": "room?a", "start_us": 0, "end_us": 10}
        )

    assert [segment.text for segment in slash] == ["slash"]
    assert [segment.text for segment in question] == ["question"]
    assert (tmp_path / "room_a.identity").read_text() == "room/a"
    assert (tmp_path / "room_a_2.identity").read_text() == "room?a"


async def test_mcp_adapter_rejects_ambiguous_or_unknown_names(tmp_path) -> None:
    async with _functions(tmp_path) as functions:
        add = functions["text_memory__add_transcript"]
        with pytest.raises(ValueError, match="unique"):
            create_mcp_server("duplicate", [add, add])
        with pytest.raises(ValueError, match="unknown functions"):
            create_mcp_server("unknown", [add], tool_names={"missing": "add"})
        with pytest.raises(ValueError, match="unknown functions"):
            create_mcp_server("unknown", [add], untyped_outputs={"missing"})
