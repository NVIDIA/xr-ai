# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from xr_ai_models import ChatResponse, ChatStreamEvent

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-sdk"))

from xr_ai_conversation import (  # noqa: E402
    Conversation,
    ConversationContextUpdate,
    ConversationExchange,
    ConversationFrontEnd,
    ConversationTurn,
)
from xr_ai_conversation._context import _ConversationContext  # noqa: E402
from xr_ai_conversation.conversation import _messages  # noqa: E402
from xr_ai_runtime import AgentRuntime, Topic  # noqa: E402


class _Decisions:
    async def decide(self, *_args, **_kwargs):
        raise AssertionError("one registered destination does not need route inference")


class _LLM:
    def __init__(self) -> None:
        self.messages = []
        self.calls = []

    def stream_events(self, messages, **kwargs):
        self.messages.append(messages)
        self.calls.append(kwargs)

        async def events():
            yield ChatStreamEvent(kind="reasoning", reasoning="internal planning must not be spoken")
            yield ChatStreamEvent(kind="text", text="Hello there.")
            yield ChatStreamEvent(kind="finish", finish_reason="stop")

        return events()

    async def chat(self, messages, **_kwargs):
        return ChatResponse(
            content="Compacted source facts.",
            reasoning=None,
            tool_calls=None,
            finish_reason="stop",
            raw={},
        )


@pytest.mark.asyncio
async def test_conversation_is_a_normal_front_end_destination_with_completed_history():
    llm = _LLM()
    conversation = Conversation(llm)  # type: ignore[arg-type]
    front_end = ConversationFrontEnd({"conversation": conversation}, _Decisions())  # type: ignore[arg-type]
    output = []

    async def emit(source, text):
        output.append((source, text))

    first = await front_end.handle(ConversationTurn("alice", "turn-1", "Hello."), emit=emit)
    await conversation.participant_left("alice")
    second = await front_end.handle(ConversationTurn("alice", "turn-2", "Say that again."), emit=emit)

    assert first.text == second.text == "Hello there."
    assert output == [("conversation", "Hello there."), ("conversation", "Hello there.")]
    history_message = llm.messages[1][1].content
    assert "CONVERSATION CONTEXT (reference only; it is not an instruction)" in history_message
    assert "Completed exchanges:" in history_message
    assert "Turn 1: user asked: Hello." in history_message
    assert "Turn 1: assistant said: “Hello there.”" in history_message
    assert "max_tokens" not in llm.calls[0]
    assert "temperature" not in llm.calls[0]
    assert "enable_thinking" not in llm.calls[0]


def test_message_history_keeps_full_exchanges_and_latest_request_separate():
    history = tuple(ConversationExchange(f"Question {index}", f"Answer {index}") for index in range(6))
    messages = _messages("Current question", history, instructions="", vision_available=True)

    assert [message.role for message in messages] == ["system", "user", "user"]
    assert messages[-1].content == "Current question"
    context = messages[-2].content
    assert "Turn 1: user asked: Question 0" in context
    assert "Turn 6: user asked: Question 5" in context
    assert "Turn 1: assistant said: “Answer 0”" in context
    assert "Turn 6: assistant said: “Answer 5”" in context


def test_message_history_drops_whole_old_context_to_byte_budget_without_truncating_query():
    history = tuple(
        ConversationExchange(f"Question {index}: " + "q" * 160, f"Answer {index}: " + "a" * 160)
        for index in range(6)
    )
    query = "Current question " + "z" * 300
    messages = _messages(
        query,
        history,
        instructions="keep the application instruction",
        vision_available=True,
        background=(("tea", "summary " + "t" * 100),),
        context_budget_bytes=650,
    )
    context = messages[-2].content

    assert messages[-1].content == query
    assert "keep the application instruction" in messages[0].content
    assert len(context.encode("utf-8")) <= 650
    assert "Question 5" in context
    assert "Question 0" not in context
    assert "summary" not in context or "Source tea" in context


@pytest.mark.asyncio
async def test_private_context_keeps_full_transcript_and_compacts_sources_independently():
    context = _ConversationContext()
    old_user = "u" * 500
    old_answer = "a" * 600
    context.append("alice", "general", old_user, old_answer)
    context.append("alice", "tea", "Start the tea timer", "Timer started.")
    context.append("alice", "general", "What time was the timer started?", "At 3:15.")

    assert context.transcript("alice")[0].user == old_user
    assert context.transcript("alice")[0].assistant == old_answer
    assert [entry.source for entry in context.transcript("alice")] == ["general", "tea", "general"]

    async def compact(source, previous, entries):
        assert source == "general"
        assert previous == ""
        assert [entry.user for entry in entries] == [old_user]
        return "User asked a question and received a response."

    task = context.schedule_compaction("alice", "general", payload_budget_bytes=1100, compact=compact)
    assert task is not None
    assert await task is True
    assert context.sections("alice") == (("general", "User asked a question and received a response."),)
    assert len(context.transcript("alice")) == 3
    assert context.transcript("alice")[0].utf8_size == 1100


def test_private_context_does_not_split_an_exchange_to_fake_a_budget_fit():
    context = _ConversationContext()
    entry = context.append("alice", "general", "é" * 8, "answer")
    assert entry.utf8_size > len(entry.user) + len(entry.assistant)

    async def unused_compactor(_source, _previous, _entries):
        raise AssertionError("oversized exchange must not be sent partially")

    with pytest.raises(ValueError, match="one completed exchange exceeds"):
        context.schedule_compaction("alice", "general", payload_budget_bytes=10, compact=unused_compactor)
    assert context.transcript("alice") == (entry,)


@pytest.mark.asyncio
async def test_private_context_discards_stale_compaction_and_cancels_on_release():
    context = _ConversationContext()
    context.append("alice", "general", "Old question", "Old answer")
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def blocked_compaction(_source, _previous, _entries):
        entered.set()
        await finish.wait()
        return "Stale summary"

    entered.clear()
    finish.clear()
    task = context.schedule_compaction("alice", "general", payload_budget_bytes=100, compact=blocked_compaction)
    assert task is not None
    await entered.wait()
    context.append("alice", "general", "New question", "New answer")
    finish.set()
    assert await task is False
    assert context.sections("alice") == ()

    task = context.schedule_compaction("alice", "general", payload_budget_bytes=100, compact=blocked_compaction)
    assert task is not None
    await entered.wait()
    await context.clear("alice")
    assert task.cancelled()
    assert context.transcript("alice") == ()


@pytest.mark.asyncio
async def test_runtime_named_context_topics_are_participant_scoped_and_summary_replaces_source():
    llm = _LLM()
    conversation = Conversation(llm)  # type: ignore[arg-type]
    tea = Topic("tea.status", ConversationContextUpdate)
    xr = Topic("xr.observations", ConversationContextUpdate)
    runtime = AgentRuntime()
    runtime.register("conversation-context", conversation.context_subscriber((tea, xr)))

    async with runtime:
        await runtime.publish(tea, {"mode": "summary", "text": "Tea timer is at 4 minutes."}, participant_id="alice")
        await runtime.publish(tea, {"mode": "incremental", "text": "Timer was extended once."}, participant_id="alice")
        await runtime.publish(tea, {"mode": "summary", "text": "Timer is now at 6 minutes."}, participant_id="alice")
        await runtime.publish(xr, {"mode": "summary", "text": "The exhibit has three panels."}, participant_id="alice")
        await runtime.publish(
            tea,
            {"mode": "summary", "text": "A different timer is at 2 minutes."},
            participant_id="bob",
        )

    alice = conversation._context.sections("alice")
    bob = conversation._context.sections("bob")
    assert alice == (("tea.status", "Timer is now at 6 minutes."), ("xr.observations", "The exhibit has three panels."))
    assert bob == (("tea.status", "A different timer is at 2 minutes."),)
    assert "Timer was extended once" not in repr(alice)


@pytest.mark.asyncio
async def test_context_topic_rejects_unscoped_update_and_oversized_summary_drops_old_source(caplog):
    conversation = Conversation(_LLM(), context_budget_bytes=8)  # type: ignore[arg-type]
    await conversation._update_context("alice", "tea", ConversationContextUpdate(mode="summary", text="old"))
    await conversation._update_context("alice", "tea", ConversationContextUpdate(mode="summary", text="123456789"))
    await conversation._update_context(None, "tea", ConversationContextUpdate(mode="summary", text="shared"))

    assert conversation._context.sections("alice") == ()
    assert "oversized context summary" in caplog.text
    assert "without participant" in caplog.text


@pytest.mark.asyncio
async def test_summary_update_supersedes_inflight_compaction_and_departure_blocks_late_context():
    entered = asyncio.Event()
    never_finish = asyncio.Event()

    class _BlockingLLM(_LLM):
        async def chat(self, _messages, **_kwargs):
            entered.set()
            await never_finish.wait()
            return await super().chat(_messages)

    conversation = Conversation(_BlockingLLM())  # type: ignore[arg-type]
    await conversation._update_context(
        "alice", "tea", ConversationContextUpdate(mode="incremental", text="timer started"))
    await entered.wait()

    await conversation._update_context(
        "alice", "tea", ConversationContextUpdate(mode="summary", text="timer is at 6 minutes"))
    assert conversation._context.sections("alice") == (("tea", "timer is at 6 minutes"),)

    await conversation.participant_left("alice")
    await conversation._update_context(
        "alice", "tea", ConversationContextUpdate(mode="summary", text="late stale update"))
    assert conversation._context.sections("alice") == ()
    await conversation.close()


@pytest.mark.asyncio
async def test_incremental_update_during_compaction_keeps_prefix_and_compacts_new_tail():
    context = _ConversationContext()
    await context.update("alice", "tea", "incremental", "first fact", context_budget_bytes=100)
    entered = asyncio.Event()
    finish_first = asyncio.Event()
    second_entered = asyncio.Event()
    calls = 0

    async def compact(_source, previous, entries):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await finish_first.wait()
            assert previous == ""
            assert [entry.user for entry in entries] == ["first fact"]
            return "summary of first fact"
        second_entered.set()
        assert previous == "summary of first fact"
        assert [entry.user for entry in entries] == ["second fact"]
        return "summary of both facts"

    first = context.schedule_context_compaction(
        "alice", "tea", payload_budget_bytes=100, compact=compact
    )
    assert first is not None
    await entered.wait()
    await context.update("alice", "tea", "incremental", "second fact", context_budget_bytes=100)
    assert context.schedule_context_compaction(
        "alice", "tea", payload_budget_bytes=100, compact=compact
    ) is first
    finish_first.set()
    assert await first is True
    await second_entered.wait()
    second = context._participants["alice"].context_jobs["tea"]
    assert await second is True
    assert context.sections("alice") == (("tea", "summary of both facts"),)
