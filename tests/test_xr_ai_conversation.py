# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from xr_ai_models import ChatStreamEvent

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-sdk"))

from xr_ai_conversation import Conversation, ConversationFrontEnd, ConversationTurn  # noqa: E402


class _Decisions:
    async def decide(self, *_args, **_kwargs):
        raise AssertionError("one registered destination does not need route inference")


class _LLM:
    def __init__(self) -> None:
        self.messages = []

    def stream_events(self, messages, **_kwargs):
        self.messages.append(messages)

        async def events():
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
    assert "completed_exchanges" in llm.messages[1][1].content
    assert "Hello there." in llm.messages[1][1].content
