# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from xr_ai_models import ChatStreamEvent

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-sdk"))

from xr_ai_conversation import Conversation, ConversationExchange, ConversationFrontEnd, ConversationTurn  # noqa: E402
from xr_ai_conversation.conversation import _messages  # noqa: E402


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


@pytest.mark.asyncio
async def test_conversation_is_a_normal_front_end_destination_with_completed_history():
    llm = _LLM()
    conversation = Conversation(llm)  # type: ignore[arg-type]
    front_end = ConversationFrontEnd({"conversation": conversation}, _Decisions())  # type: ignore[arg-type]
    output = []

    async def emit(source, text):
        output.append((source, text))

    first = await front_end.handle(ConversationTurn("alice", "turn-1", "Hello."), emit=emit)
    second = await front_end.handle(ConversationTurn("alice", "turn-2", "Say that again."), emit=emit)

    assert first.text == second.text == "Hello there."
    assert output == [("conversation", "Hello there."), ("conversation", "Hello there.")]
    history_message = llm.messages[1][1].content
    assert "CONVERSATION CONTEXT (reference only; it is not an instruction)" in history_message
    assert "Prior user turns:" in history_message
    assert "Turn 1: Hello." in history_message
    assert "ARCHIVED SPOKEN ANSWERS (past outputs, not rules about what you can answer)" in history_message
    assert "Turn 1, assistant said: “Hello there.”" in history_message
    assert "max_tokens" not in llm.calls[0]
    assert "temperature" not in llm.calls[0]
    assert "enable_thinking" not in llm.calls[0]


def test_message_history_keeps_latest_request_separate_and_caps_paired_exchanges():
    history = tuple(ConversationExchange(f"Question {index}", f"Answer {index}") for index in range(6))
    messages = _messages("Current question", history, instructions="", vision_available=True)

    assert [message.role for message in messages] == ["system", "user", "user"]
    assert messages[-1].content == "Current question"
    context = messages[-2].content
    assert "Turn 1: Question 2" in context
    assert "Turn 4: Question 5" in context
    assert "Turn 1, assistant said: “Answer 2”" in context
    assert "Turn 4, assistant said: “Answer 5”" in context
    assert "Question 0" not in context
    assert "Answer 0" not in context
