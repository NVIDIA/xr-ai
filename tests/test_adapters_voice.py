# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NAT↔voice adapters, and the conversation-recall producer/consumer loop.

`record_voice_transcripts` is the producer that writes ``{pid}:user`` /
``{pid}:agent`` transcript sources; `xr_conversation_memory.recall_conversation`
is the consumer that reads them back. This exercises both ends end-to-end.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from nat.builder.workflow_builder import WorkflowBuilder
from pydantic import BaseModel
from xr_ai_nat.adapters import as_voice_event_handler, as_voice_handler, record_voice_transcripts
from xr_ai_nat.events import EventDispatcher, EventEnvelope, EventTopic
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


class _Request(BaseModel):
    text: str


class _EventFunction:
    def __init__(self) -> None:
        self.event: EventEnvelope | None = None

    async def ainvoke(self, event: EventEnvelope) -> str:
        self.event = event
        return "event answer"


async def test_as_voice_handler_maps_request_and_response() -> None:
    handler = as_voice_handler(
        _EchoFunction(),
        request=lambda query: query.text.upper(),
        response=str,
    )
    response = await handler(VoiceQuery(participant_id="alice", text="hi", fresh_match=True, timestamp_us=1))
    assert response == "answer:HI"


async def test_as_voice_handler_streams_and_drops_empty_chunks() -> None:
    handler = as_voice_handler(
        _EchoFunction(),
        request=lambda query: query.text,
        response=lambda chunk: str(chunk).strip(),  # "one " -> "one", "two" -> "two"
        streaming=True,
    )
    stream = await handler(VoiceQuery(participant_id="alice", text="go", fresh_match=True, timestamp_us=1))
    assert [chunk async for chunk in stream] == ["one", "two"]


async def test_as_voice_event_handler_publishes_transport_neutral_request() -> None:
    topic = EventTopic("application.request", _Request)
    dispatcher = EventDispatcher()
    function = _EventFunction()
    dispatcher.subscribe(topic, subscriber_id="application", function=function)  # type: ignore[arg-type]
    handler = as_voice_event_handler(
        dispatcher,
        topic,
        payload=lambda query: _Request(text=query.text),
        subscribers={"application"},
    )

    response = await handler(
        VoiceQuery(
            participant_id="alice",
            text="start monitoring",
            fresh_match=True,
            timestamp_us=42,
        )
    )

    assert response == "event answer"
    assert function.event is not None
    assert function.event.participant_id == "alice"
    assert function.event.producer == "voice.input"
    assert function.event.timestamp_us == 42
    assert topic.payload_from(function.event) == _Request(text="start monitoring")


async def test_record_voice_transcripts_then_recall_conversation(tmp_path) -> None:
    async with WorkflowBuilder() as builder:
        await builder.add_function_group("text_memory", TextMemoryFunctionsConfig(directory=tmp_path))
        await builder.add_function_group("conversation_memory", ConversationMemoryFunctionsConfig())
        text_memory = await builder.get_function_group("text_memory")
        conversation = await builder.get_function_group("conversation_memory")
        add_transcript = (await text_memory.get_all_functions())["text_memory__add_transcript"]
        recall = (await conversation.get_all_functions())["conversation_memory__recall_conversation"]

        record = record_voice_transcripts(add_transcript)
        # A real exchange gives the user turn and the agent turn the SAME
        # timestamp — both carry the originating query's time — so recall has to
        # order the tie user-before-agent rather than relying on distinct stamps.
        await record(VoiceTurn(participant_id="alice", role="user", timestamp_us=10, text="hello"))
        await record(VoiceTurn(participant_id="alice", role="agent", timestamp_us=10, text="hi there"))
        await record(VoiceTurn(participant_id="alice", role="user", timestamp_us=30, text="how are you"))
        # Whitespace-only turns are not persisted.
        await record(VoiceTurn(participant_id="alice", role="agent", timestamp_us=30, text="   "))
        # A different participant must not leak into alice's recall.
        await record(VoiceTurn(participant_id="bob", role="user", timestamp_us=10, text="not alice"))

        result = await recall.ainvoke(RecallConversationRequest(participant_id="alice"))

    assert [(entry.timestamp_us, entry.role, entry.text) for entry in result.entries] == [
        (10, "user", "hello"),
        (10, "agent", "hi there"),
        (30, "user", "how are you"),
    ]
    # The producer stored role-scoped sources under the participant id.
    assert (tmp_path / "alice_user.identity").read_text() == "alice:user"
    assert (tmp_path / "alice_agent.identity").read_text() == "alice:agent"


def test_voice_adapters_are_reachable_from_the_public_adapters_namespace() -> None:
    """Applications are told to use ``xr_ai_nat.adapters.as_voice_handler``; the
    package must actually export both adapters (they resolve lazily because they
    need the optional ``[voice]`` extra)."""
    from xr_ai_nat import adapters
    from xr_ai_nat.adapters import voice as voice_module

    assert adapters.as_voice_event_handler is voice_module.as_voice_event_handler
    assert adapters.as_voice_handler is voice_module.as_voice_handler
    assert adapters.record_voice_transcripts is voice_module.record_voice_transcripts
    assert sorted(adapters.__all__) == [
        "as_voice_event_handler",
        "as_voice_handler",
        "record_voice_transcripts",
    ]
    assert "as_voice_handler" in dir(adapters)
    with pytest.raises(AttributeError):
        adapters.not_an_adapter


async def test_recall_conversation_respects_time_window(tmp_path) -> None:
    async with WorkflowBuilder() as builder:
        await builder.add_function_group("text_memory", TextMemoryFunctionsConfig(directory=tmp_path))
        await builder.add_function_group("conversation_memory", ConversationMemoryFunctionsConfig())
        add_transcript = (await (await builder.get_function_group("text_memory")).get_all_functions())[
            "text_memory__add_transcript"
        ]
        recall = (await (await builder.get_function_group("conversation_memory")).get_all_functions())[
            "conversation_memory__recall_conversation"
        ]
        record = record_voice_transcripts(add_transcript)
        await record(VoiceTurn(participant_id="alice", role="user", timestamp_us=10, text="early"))
        await record(VoiceTurn(participant_id="alice", role="user", timestamp_us=100, text="late"))

        result = await recall.ainvoke(RecallConversationRequest(participant_id="alice", start_us=50, end_us=200))

    assert [entry.text for entry in result.entries] == ["late"]


async def test_recall_conversation_generated_contract_is_fully_described(tmp_path) -> None:
    """The recall surface is consumed by agents, so its generated schemas must
    describe every field and constrain ``role`` to the two roles that exist.

    An undescribed field gives the model nothing to reason about, and an
    unconstrained ``role`` invites it to invent a third value.
    """
    async with WorkflowBuilder() as builder:
        await builder.add_function_group("text_memory", TextMemoryFunctionsConfig(directory=tmp_path))
        await builder.add_function_group("conversation_memory", ConversationMemoryFunctionsConfig())
        conversation = await builder.get_function_group("conversation_memory")
        recall = (await conversation.get_all_functions())["conversation_memory__recall_conversation"]

        request = recall.input_schema.model_json_schema()
        result = recall.single_output_schema.model_json_schema()

    # Every request field is described.
    for name, prop in request["properties"].items():
        assert prop.get("description"), f"request field {name} has no description"

    # The result and its nested entry model are fully described.
    entry = result["$defs"]["ConversationEntry"]
    assert result["properties"]["entries"].get("description")
    for name in ("timestamp_us", "role", "text"):
        assert entry["properties"][name].get("description"), f"{name} has no description"

    # role is constrained to exactly the two roles the producer writes.
    assert entry["properties"]["role"]["enum"] == ["user", "agent"]


def test_conversation_memory_config_documents_its_text_memory_reference() -> None:
    field = ConversationMemoryFunctionsConfig.model_fields["text_memory"]
    assert field.description and "xr_text_memory" in field.description
