# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NAT↔voice adapters, and the conversation-recall producer/consumer loop.

`record_voice_transcripts` is the producer that writes ``{pid}:user`` /
``{pid}:agent`` transcript sources; `xr_conversation_memory.recall_conversation`
is the consumer that reads them back. This exercises both ends end-to-end.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from nat.builder.workflow_builder import WorkflowBuilder
from xr_ai_nat.adapters.voice import as_voice_handler, record_voice_transcripts
from xr_ai_nat.functions.text_memory import (
    ConversationMemoryFunctionsConfig,
    RecallConversationRequest,
    TextMemoryFunctionsConfig,
)
from xr_ai_voice import VoiceQuery, VoiceTurn


class _EchoFunction:
    """Duck-typed NAT ``Function`` stand-in for the adapter unit tests."""

    async def ainvoke(self, request: object) -> str:
        return f"answer:{request}"

    async def astream(self, request: object) -> AsyncIterator[str]:
        del request
        for part in ("one ", "two"):
            yield part


async def test_as_voice_handler_maps_request_and_response() -> None:
    handler = as_voice_handler(
        _EchoFunction(),
        request=lambda query: query.text.upper(),
        response=str,
    )
    response = await handler(
        VoiceQuery(participant_id="alice", text="hi", fresh_match=True, timestamp_us=1)
    )
    assert response == "answer:HI"


async def test_as_voice_handler_streams_and_drops_empty_chunks() -> None:
    handler = as_voice_handler(
        _EchoFunction(),
        request=lambda query: query.text,
        response=lambda chunk: str(chunk).strip(),  # "one " -> "one", "two" -> "two"
        streaming=True,
    )
    stream = await handler(
        VoiceQuery(participant_id="alice", text="go", fresh_match=True, timestamp_us=1)
    )
    assert [chunk async for chunk in stream] == ["one", "two"]


async def test_record_voice_transcripts_then_recall_conversation(tmp_path) -> None:
    async with WorkflowBuilder() as builder:
        await builder.add_function_group(
            "text_memory", TextMemoryFunctionsConfig(directory=tmp_path)
        )
        await builder.add_function_group(
            "conversation_memory", ConversationMemoryFunctionsConfig()
        )
        text_memory = await builder.get_function_group("text_memory")
        conversation = await builder.get_function_group("conversation_memory")
        add_transcript = (await text_memory.get_all_functions())["text_memory__add_transcript"]
        recall = (await conversation.get_all_functions())["conversation_memory__recall_conversation"]

        record = record_voice_transcripts(add_transcript)
        await record(VoiceTurn(participant_id="alice", role="user", timestamp_us=10, text="hello"))
        await record(VoiceTurn(participant_id="alice", role="agent", timestamp_us=20, text="hi there"))
        await record(VoiceTurn(participant_id="alice", role="user", timestamp_us=30, text="how are you"))
        # Whitespace-only turns are not persisted.
        await record(VoiceTurn(participant_id="alice", role="agent", timestamp_us=40, text="   "))
        # A different participant must not leak into alice's recall.
        await record(VoiceTurn(participant_id="bob", role="user", timestamp_us=15, text="not alice"))

        result = await recall.ainvoke(RecallConversationRequest(participant_id="alice"))

    assert [(entry.timestamp_us, entry.role, entry.text) for entry in result.entries] == [
        (10, "user", "hello"),
        (20, "agent", "hi there"),
        (30, "user", "how are you"),
    ]
    # The producer stored role-scoped sources under the participant id.
    assert (tmp_path / "alice_user.identity").read_text() == "alice:user"
    assert (tmp_path / "alice_agent.identity").read_text() == "alice:agent"


async def test_recall_conversation_respects_time_window(tmp_path) -> None:
    async with WorkflowBuilder() as builder:
        await builder.add_function_group(
            "text_memory", TextMemoryFunctionsConfig(directory=tmp_path)
        )
        await builder.add_function_group(
            "conversation_memory", ConversationMemoryFunctionsConfig()
        )
        add_transcript = (await (await builder.get_function_group("text_memory")).get_all_functions())[
            "text_memory__add_transcript"
        ]
        recall = (await (await builder.get_function_group("conversation_memory")).get_all_functions())[
            "conversation_memory__recall_conversation"
        ]
        record = record_voice_transcripts(add_transcript)
        await record(VoiceTurn(participant_id="alice", role="user", timestamp_us=10, text="early"))
        await record(VoiceTurn(participant_id="alice", role="user", timestamp_us=100, text="late"))

        result = await recall.ainvoke(
            RecallConversationRequest(participant_id="alice", start_us=50, end_us=200)
        )

    assert [entry.text for entry in result.entries] == ["late"]
