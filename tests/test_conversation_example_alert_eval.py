# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The replay records real compiler outcomes without equating ready with semantic correctness."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest
from xr_ai_models import ChatResponse

_PATH = Path(__file__).resolve().parents[1] / "agent-samples/conversation-example/eval/run_alert_compile_eval.py"
_SPEC = importlib.util.spec_from_file_location("conversation_alert_eval", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = runner
_SPEC.loader.exec_module(runner)

_CASE = {
    "name": "presence",
    "query": "Alert me when a suitcase is visible.",
    "expected_status": "ready",
    "expected_condition": "A suitcase is visible.",
}


class _LLM:
    def __init__(self, content):
        self.content = content
        self.options = None

    async def chat(self, messages, **kwargs):
        self.options = kwargs
        return ChatResponse(
            content=self.content,
            reasoning=None,
            tool_calls=None,
            finish_reason="stop",
            raw={"usage": {"completion_tokens": 12}},
        )


@pytest.mark.asyncio
async def test_ready_with_wrong_condition_is_recorded_for_semantic_review():
    llm = _LLM('{"status":"ready","condition":"a suitcase is open","relevance":"the suitcase"}')
    row = await runner.replay(llm, _CASE)

    assert row["status_matches"]
    assert row["semantic_review_required"]
    assert row["watch"]["condition"] == "a suitcase is open"
    assert row["expected_condition"] == "A suitcase is visible."
    assert row["usage"] == {"completion_tokens": 12}
    assert row["request"]["options"] == llm.options
    assert llm.options == {"enable_thinking": True, "thinking_budget": 1024, "max_tokens": 1280}


@pytest.mark.asyncio
async def test_budget_comparison_preserves_messages_and_total_output_allowance():
    llm = _LLM('{"status":"unclear"}')
    low = await runner.replay(llm, _CASE, budget=256, max_tokens=1280)
    high = await runner.replay(llm, _CASE, budget=1024, max_tokens=1280)

    assert low["messages_sha256"] == high["messages_sha256"]
    assert low["request"]["messages"] == high["request"]["messages"]
    assert low["request"]["options"] == {**high["request"]["options"], "thinking_budget": 256}
    assert low["actual_status"] == high["actual_status"] == "unclear"
    assert not low["status_matches"]


@pytest.mark.asyncio
async def test_malformed_output_is_not_counted_as_unclear():
    row = await runner.replay(_LLM("broken json"), {**_CASE, "expected_status": "unclear"})
    assert row["actual_status"] == "invalid"
    assert not row["status_matches"]


@pytest.mark.asyncio
async def test_replay_deadline_cancels_inference_and_records_timeout(monkeypatch):
    cancelled = asyncio.Event()

    class WaitingLLM:
        async def chat(self, messages, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    monkeypatch.setattr(runner, "_COMPILE_TIMEOUT_SECS", 0.01)
    row = await runner.replay(WaitingLLM(), _CASE)
    assert cancelled.is_set()
    assert row["actual_status"] == "timeout"
    assert not row["status_matches"]
    assert row["usage"] is None
