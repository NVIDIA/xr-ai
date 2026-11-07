# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alert compilation: what the model is given and which answers become a watch."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest
import yaml
from xr_ai_models import Capabilities, ChatResponse, load_models_config, make_llm
from xr_ai_routing import AgentExchange

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "agent-samples/conversation-example/worker"))

from conversation_example_worker._alert_compile import CompiledAlert, compile_alert  # noqa: E402
from conversation_example_worker.app import _alert_compile_models  # noqa: E402

_READY = {"status": "ready", "condition": " the lamp is blue ", "relevance": " the lamp "}
_EVAL_CASES = yaml.safe_load(
    (_ROOT / "agent-samples/conversation-example/eval/alert-compile-cases.yaml").read_text(encoding="utf-8")
)


class _LLM:
    capabilities = Capabilities()

    def __init__(self, content: str) -> None:
        self.content = content
        self.requests: list[tuple] = []
        self.options: list[dict] = []

    async def chat(self, messages, **kwargs):
        self.requests.append(messages)
        self.options.append(kwargs)
        return ChatResponse(content=self.content, reasoning=None, tool_calls=None, finish_reason="stop", raw={})


@pytest.mark.asyncio
@pytest.mark.parametrize("wrap", [lambda text: text, lambda text: f"```json\n{text}\n```", lambda text: f"\n{text}\n"])
async def test_a_ready_answer_becomes_a_trimmed_watch(wrap):
    llm = _LLM(wrap(json.dumps(_READY)))

    assert await compile_alert(llm, "Alert me about the lamp.") == CompiledAlert("the lamp is blue", "the lamp")
    assert llm.options == [{"enable_thinking": True, "thinking_budget": 1024, "max_tokens": 1280}]


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
    for label in ("Alert request:", "Recent exchanges, oldest first:", "Saved context:"):
        assert label in messages[0].content
        assert label in user
    assert "saved notes" in messages[0].content
    assert "supply identifying details that the current request depends on" in messages[0].content
    assert "The current request determines what to watch for" in messages[0].content


@pytest.mark.asyncio
async def test_sections_without_content_are_left_out():
    llm = _LLM(json.dumps(_READY))

    await compile_alert(llm, "Alert me about the lamp.")

    user = llm.requests[0][-1].content
    assert "Alert me about the lamp." in user
    assert "Recent exchanges" not in user and "Saved context" not in user


@pytest.mark.asyncio
async def test_prompt_defines_future_conditions_without_requiring_current_evidence():
    llm = _LLM('{"status":"unclear"}')

    await compile_alert(llm, "Alert me when a delivery cart is visible.")

    system = llm.requests[0][0].content
    assert "Define a trigger for future camera frames" in system
    assert "No image is supplied to this compilation step, and none is required" in system
    assert "A subject missing from a previous scene description is not a reason to reject a future watch" in system
    assert "prior alert requests and prior assistant refusals do not replace it" in system
    assert "resolve references or supply identifying details that the current request depends on" in system


def test_alert_eval_cases_have_explicit_semantic_expectations():
    assert len({case["name"] for case in _EVAL_CASES}) == len(_EVAL_CASES)
    for case in _EVAL_CASES:
        assert case["query"].strip()
        assert case["expected_status"] in {"ready", "unclear"}
        assert ("expected_condition" in case) == (case["expected_status"] == "ready")
        if case["expected_status"] == "ready":
            assert case["expected_condition"].strip()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "history", "context"),
    [
        ("Alert me when you see a kettle.", (AgentExchange("What color is the mat?", "Green."),), ""),
        ("Alert me about the shape in my notes.", (), "Watch for a purple hexagon."),
        ("Alert me when the kettle is on the hob.", (), "The kettle is the blue one on the left."),
        ("Alert me when that thing happens.", (), ""),
    ],
)
async def test_replay_inputs_keep_the_current_request_separate_from_context(query, history, context):
    """Preserve replay inputs and framing, not a fake model-accuracy guarantee."""
    llm = _LLM('{"status":"unclear"}')

    await compile_alert(llm, query, history=history, context=context)

    user = llm.requests[0][-1].content
    assert user.split("\n\n", 1)[0] == f"Alert request:\n{query}"
    if history:
        assert f"Recent exchanges, oldest first:\nUser: {history[0].user}\nAssistant: {history[0].assistant}" in user
    if context:
        assert f"Saved context:\n{context}" in user


@pytest.mark.asyncio
async def test_sample_compiler_limits_reach_the_serialized_request(monkeypatch):
    models = load_models_config(_ROOT / "agent-samples/conversation-example/yaml/models.json")
    original = models.llm("llm")
    compile_models = _alert_compile_models(models)
    spec = compile_models.llm("alert_compile_llm")
    assert spec.endpoint == original.endpoint
    assert spec.adapter.thinking == original.adapter.thinking
    assert spec.adapter.model_name == original.adapter.model_name
    payloads = []

    def handle(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(_READY)}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: http)
        llm = make_llm(compile_models, "alert_compile_llm")
        try:
            assert await compile_alert(llm, "Alert me when the lamp is blue.") == CompiledAlert(
                "the lamp is blue", "the lamp"
            )
        finally:
            await llm.close()

    (payload,) = payloads
    assert payload["max_tokens"] == 1280
    assert payload["thinking_token_budget"] == 1024
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert payload["temperature"] == 0.25
    assert original.default_extras == {
        "max_tokens": 4608,
        "temperature": 0.25,
        "chat_template_kwargs": {"reasoning_budget": 4096},
    }
