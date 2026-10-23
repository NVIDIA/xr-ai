# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the model-backed voice update combiner."""

from __future__ import annotations

import asyncio

import pytest
from xr_ai_models import ChatMessage, ChatResponse, ModelServiceError
from xr_ai_voice import llm_update_combiner


class _LLM:
    def __init__(
        self,
        content: str = "Combined update.",
        *,
        error: Exception | None = None,
        gate: asyncio.Event | None = None,
        tool_calls: object = None,
    ) -> None:
        self.content = content
        self.error = error
        self.gate = gate
        self.tool_calls = tool_calls
        self.cancelled = False
        self.calls: list[tuple[tuple[ChatMessage, ...], dict]] = []

    async def chat(self, messages, **kwargs) -> ChatResponse:
        self.calls.append((tuple(messages), kwargs))
        if self.error is not None:
            raise self.error
        if self.gate is not None:
            try:
                await self.gate.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return ChatResponse(self.content, None, self.tool_calls, "stop", {})


async def test_single_update_bypasses_the_model() -> None:
    llm = _LLM()

    assert await llm_update_combiner(llm)(["  Only one. "]) == "Only one."  # type: ignore[arg-type]
    assert llm.calls == []


async def test_several_updates_are_rewritten_once() -> None:
    llm = _LLM()

    text = await llm_update_combiner(llm, max_tokens=64)(["First.", "Second."])  # type: ignore[arg-type]

    assert text == "Combined update."
    messages, options = llm.calls[0]
    assert messages[0].role == "system"
    assert '["First.", "Second."]' in messages[1].content
    assert options["max_tokens"] == 64
    assert options["temperature"] == 0.0
    assert options["enable_thinking"] is False


@pytest.mark.parametrize(
    "llm",
    [
        _LLM(error=ModelServiceError("unavailable")),
        _LLM(content="   "),
        _LLM(content="Calling a tool.", tool_calls=[object()]),
    ],
)
async def test_unusable_model_output_falls_back_to_joined_updates(llm: _LLM) -> None:
    combiner = llm_update_combiner(llm)  # type: ignore[arg-type]

    assert await combiner(["First.", "Second."]) == "First. Second."


async def test_timeout_cancels_the_model_call_and_falls_back() -> None:
    llm = _LLM(gate=asyncio.Event())
    combiner = llm_update_combiner(llm, timeout_s=0.02)  # type: ignore[arg-type]

    assert await combiner(["First.", "Second."]) == "First. Second."
    assert llm.cancelled is True
    assert llm.calls[0][1]["timeout"] == 0.02


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prompt": " "}, "prompt"),
        ({"max_tokens": 0}, "token limit"),
        ({"timeout_s": 0}, "timeout"),
    ],
)
def test_configuration_validation(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        llm_update_combiner(_LLM(), **kwargs)  # type: ignore[arg-type]
