# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alert compilation: what the model is given and which answers become a watch."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from xr_ai_models import Capabilities, ChatResponse
from xr_ai_routing import AgentExchange

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "agent-samples/conversation-example/worker"))

from conversation_example_worker._alert_compile import CompiledAlert, compile_alert  # noqa: E402

_READY = {"status": "ready", "condition": " the lamp is blue ", "relevance": " the lamp "}


class _LLM:
    capabilities = Capabilities()

    def __init__(self, content: str) -> None:
        self.content = content
        self.requests: list[tuple] = []

    async def chat(self, messages, **_kwargs):
        self.requests.append(messages)
        return ChatResponse(content=self.content, reasoning=None, tool_calls=None, finish_reason="stop", raw={})


@pytest.mark.asyncio
@pytest.mark.parametrize("wrap", [lambda text: text, lambda text: f"```json\n{text}\n```", lambda text: f"\n{text}\n"])
async def test_a_ready_answer_becomes_a_trimmed_watch(wrap):
    llm = _LLM(wrap(json.dumps(_READY)))

    assert await compile_alert(llm, "Alert me about the lamp.") == CompiledAlert("the lamp is blue", "the lamp")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        json.dumps({"status": "unclear"}),
        "not json",
        "[]",
        json.dumps({"status": "ready", "condition": "the lamp is blue"}),
        json.dumps({"status": "ready", "condition": "  ", "relevance": "the lamp"}),
        json.dumps({"status": "ready", "condition": 3, "relevance": "the lamp"}),
    ],
)
async def test_anything_but_a_complete_ready_answer_is_no_watch(content):
    assert await compile_alert(_LLM(content), "Alert me about that.") is None


@pytest.mark.asyncio
async def test_the_model_sees_the_request_recent_exchanges_and_saved_context():
    llm = _LLM(json.dumps(_READY))
    history = (AgentExchange("The part is the red cable.", "Noted."),)

    await compile_alert(llm, "Alert me about the part.", history=history, context="Note: bench 3 is free.")

    (messages,) = llm.requests
    user = messages[-1].content
    assert messages[0].role == "system" and messages[-1].role == "user"
    assert user.index("Alert me about the part.") < user.index("The part is the red cable.") < user.index("bench 3")


@pytest.mark.asyncio
async def test_sections_without_content_are_left_out():
    llm = _LLM(json.dumps(_READY))

    await compile_alert(llm, "Alert me about the lamp.")

    user = llm.requests[0][-1].content
    assert "Alert me about the lamp." in user
    assert "Recent exchanges" not in user and "Saved context" not in user
