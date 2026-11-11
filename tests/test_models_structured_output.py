# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured output and forced tool calls on the OpenAI-compatible LLM client.

Uses :class:`_stub_openai.StubOpenAI`, so no model server is required. The
live behaviour is measured by ``agent-sdk/xr-ai-models/eval``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml
from _stub_openai import StubOpenAI
from xr_ai_models import (
    Capabilities,
    ChatMessage,
    ImagePart,
    OpenAICompatLLM,
    TextPart,
    ThinkingSpec,
    ToolDef,
    load_models_config_from_dict,
    make_llm,
)
from xr_ai_models.presets import available_presets, get_preset

SCHEMA = {
    "title": "Change Decision",
    "type": "object",
    "properties": {"important": {"type": "boolean"}},
    "required": ["important"],
}
TOOL = ToolDef("commit", "Commit the answer.", SCHEMA)
BOTH = Capabilities(json_schema=True, tool_choice=True)
USER = [ChatMessage(role="user", content="Did it change?")]


def _llm(stub: StubOpenAI, *, capabilities: Capabilities = BOTH, thinking: ThinkingSpec | None = None):
    return OpenAICompatLLM(
        "http://stub", "llm", client=stub.client(), capabilities=capabilities, thinking=thinking,
    )


async def test_response_schema_without_reasoning_sends_only_the_constraint() -> None:
    stub = StubOpenAI()
    stub.set_chat_message(content='{"important": true}')
    async with _llm(stub) as llm:
        response = await llm.chat(USER, response_schema=SCHEMA)
    assert response.content == '{"important": true}'
    body = stub.last_json()
    assert body["response_format"] == {
        "type": "json_schema", "json_schema": {"name": "Change_Decision", "schema": SCHEMA},
    }
    assert body["messages"] == [{"role": "user", "content": "Did it change?"}]
    assert "tool_choice" not in body and "tools" not in body


@pytest.mark.parametrize("thinking", [
    ThinkingSpec(control="chat_template", default=True),
    ThinkingSpec(control="system_prompt", default=True),
    ThinkingSpec(control="none"),
    ThinkingSpec(),
])
@pytest.mark.parametrize("enable", [False, True, None])
async def test_output_constraints_do_not_infer_reasoning_policy(thinking, enable) -> None:
    stub = StubOpenAI()
    messages = [ChatMessage(role="system", content="Watch the kettle."), *USER]
    capabilities = Capabilities(json_schema=True, reasoning=True)
    async with _llm(stub, thinking=thinking, capabilities=capabilities) as llm:
        await llm.chat(messages, enable_thinking=enable)
        unconstrained = stub.last_json()
        await llm.chat(messages, response_schema=SCHEMA, enable_thinking=enable)
        constrained = stub.last_json()
    assert constrained.pop("response_format")["json_schema"]["schema"] == SCHEMA
    assert constrained == unconstrained
    assert messages[0].content == "Watch the kettle."


async def test_untitled_schema_gets_a_default_name() -> None:
    stub = StubOpenAI()
    async with _llm(stub) as llm:
        await llm.chat(USER, response_schema={"type": "object"})
    assert stub.last_json()["response_format"]["json_schema"]["name"] == "response"


@pytest.mark.parametrize(("choice", "wire"), [
    ("auto", "auto"),
    ("none", "none"),
    ("required", "required"),
    ("commit", {"type": "function", "function": {"name": "commit"}}),
])
async def test_tool_choice_wire_format(choice, wire) -> None:
    stub = StubOpenAI()
    async with _llm(stub) as llm:
        await llm.chat(USER, tools=[TOOL], tool_choice=choice)
    body = stub.last_json()
    assert body["tool_choice"] == wire
    assert body["messages"] == [{"role": "user", "content": "Did it change?"}]  # tools carry the schema


@pytest.mark.parametrize("choice", ["auto", "none"])
async def test_unforced_tool_choice_needs_no_capability(choice) -> None:
    stub = StubOpenAI()
    async with _llm(stub, capabilities=Capabilities()) as llm:
        await llm.chat(USER, tools=[TOOL], tool_choice=choice)
    assert stub.last_json()["tool_choice"] == choice


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"response_schema": SCHEMA}, "json_schema support"),
    ({"tools": [TOOL], "tool_choice": "required"}, "tool_choice support"),
    ({"tools": [TOOL], "tool_choice": "commit"}, "tool_choice support"),
])
async def test_undeclared_constraints_are_rejected_before_sending(kwargs, message) -> None:
    stub = StubOpenAI()
    async with _llm(stub, capabilities=Capabilities(tool_calls=True)) as llm:
        with pytest.raises(ValueError, match=message):
            await llm.chat(USER, **kwargs)
    assert not stub.requests


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"tool_choice": "required"}, "requires tools"),
    ({"tools": [TOOL], "tool_choice": "other"}, "names no supplied tool"),
    ({"tools": [TOOL], "tool_choice": ""}, "tool_choice must be"),
    ({"tools": [TOOL], "response_schema": SCHEMA}, "cannot be combined with tools"),
    ({"response_schema": ["not", "a", "schema"]}, "must be a JSON Schema mapping"),
])
async def test_invalid_constraint_requests_are_rejected(kwargs, message) -> None:
    stub = StubOpenAI()
    async with _llm(stub) as llm:
        with pytest.raises(ValueError, match=message):
            await llm.chat(USER, **kwargs)
    assert not stub.requests


@pytest.mark.parametrize("method", ["chat", "stream", "stream_events"])
async def test_explicit_request_fields_override_defaults(method) -> None:
    stub = StubOpenAI()
    stub.set_stream_chunks([
        {"choices": [{"index": 0, "delta": {"content": '{"important": true}'}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ])
    extras = {
        "model": "wrong", "messages": [{"role": "user", "content": "wrong"}],
        "stream": method == "chat", "max_tokens": 64, "temperature": 0.9,
        "response_format": {"type": "text"},
        "chat_template_kwargs": {"enable_thinking": True, "custom": "kept"},
        "thinking_token_budget": 2048,
    }
    async with OpenAICompatLLM(
        "http://stub", "llm", client=stub.client(), capabilities=BOTH, default_extras=extras,
    ) as llm:
        result = getattr(llm, method)(
            USER, response_schema=SCHEMA, max_tokens=1024, temperature=0.0,
            enable_thinking=True, thinking_budget=256,
        )
        if method == "chat":
            await result
        else:
            _ = [chunk async for chunk in result]
    body = stub.last_json()
    assert body["model"] == "llm" and body["messages"] == [{"role": "user", "content": "Did it change?"}]
    assert body.get("stream", False) == (method != "chat")
    assert body["max_tokens"] == 1024 and body["temperature"] == 0.0
    assert body["thinking_token_budget"] == 256
    assert body["chat_template_kwargs"] == {"enable_thinking": True, "custom": "kept"}
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert extras["max_tokens"] == 64 and extras["response_format"] == {"type": "text"}


@pytest.mark.parametrize(("extras", "message"), [
    ({"tools": [TOOL.to_openai()], "response_format": {
        "type": "json_schema", "json_schema": {"name": "answer", "schema": SCHEMA},
    }}, "cannot be combined with tools"),
    ({"tool_choice": "required", "tools": [TOOL.to_openai()]}, "tool_choice support"),
    ({"tool_choice": "required"}, "requires tools"),
    ({"response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": SCHEMA}}},
     "json_schema support"),
])
async def test_constraints_from_defaults_are_validated_before_transport(extras, message) -> None:
    stub = StubOpenAI()
    async with OpenAICompatLLM("http://stub", "llm", client=stub.client(), default_extras=extras) as llm:
        with pytest.raises(ValueError, match=message):
            await llm.chat(USER)
    assert not stub.requests


async def test_explicit_tools_replace_defaults() -> None:
    stub = StubOpenAI()
    async with OpenAICompatLLM(
        "http://stub", "llm", client=stub.client(), capabilities=BOTH,
        default_extras={"tools": [TOOL.to_openai()]},
    ) as llm:
        await llm.chat(USER, tools=[], response_schema=SCHEMA)
    assert stub.last_json()["tools"] == []


async def test_tool_choice_explicitly_overrides_default_choice() -> None:
    stub = StubOpenAI()
    async with OpenAICompatLLM(
        "http://stub", "llm", client=stub.client(),
        default_extras={"tools": [TOOL.to_openai()], "tool_choice": "required"},
    ) as llm:
        await llm.chat(USER, tool_choice="none")
    assert stub.last_json()["tool_choice"] == "none"


async def test_optional_limits_still_use_deployment_defaults() -> None:
    stub = StubOpenAI()
    async with OpenAICompatLLM(
        "http://stub", "llm", client=stub.client(), capabilities=BOTH,
        default_extras={"max_tokens": 128, "temperature": 0.2},
    ) as llm:
        await llm.chat(USER, response_schema=SCHEMA)
    assert stub.last_json()["max_tokens"] == 128 and stub.last_json()["temperature"] == 0.2


async def test_stream_paths_send_the_same_constraints() -> None:
    stub = StubOpenAI()
    stub.set_stream_tokens(['{"important":', " true}"])
    async with _llm(stub) as llm:
        text = "".join([chunk async for chunk in llm.stream(USER, response_schema=SCHEMA)])
        assert text == '{"important": true}'
        streamed = stub.last_json()
        stub.set_stream_chunks([
            {"choices": [{"index": 0, "delta": {"content": '{"important": true}'}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ])
        events = [event async for event in llm.stream_events(USER, response_schema=SCHEMA)]
    assert streamed["response_format"]["json_schema"]["schema"] == SCHEMA
    assert stub.last_json()["response_format"] == streamed["response_format"]
    assert "".join(e.text for e in events if e.kind == "text") == '{"important": true}'


def _tool_stream(finish: str) -> list[dict]:
    return [
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "commit", "arguments": '{"important": true}'}},
        ]}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
    ]


async def test_named_tool_choice_stream_accepts_stop_finish() -> None:
    # OpenAI-compatible servers report "stop" when the call was forced by name.
    stub = StubOpenAI()
    stub.set_stream_chunks(_tool_stream("stop"))
    async with _llm(stub) as llm:
        events = [event async for event in llm.stream_events(USER, tools=[TOOL], tool_choice="commit")]
    assert [e.tool_call.arguments for e in events if e.kind == "tool_call"] == ['{"important": true}']
    assert events[-1].finish_reason == "stop"


async def test_default_named_tool_choice_uses_effective_stream_finish_contract() -> None:
    stub = StubOpenAI()
    stub.set_stream_chunks(_tool_stream("stop"))
    async with OpenAICompatLLM(
        "http://stub", "llm", client=stub.client(), capabilities=BOTH,
        default_extras={"tools": [TOOL.to_openai()], "tool_choice": {
            "type": "function", "function": {"name": "commit"},
        }},
    ) as llm:
        events = [event async for event in llm.stream_events(USER)]
    assert [e.tool_call.name for e in events if e.kind == "tool_call"] == ["commit"]
    assert events[-1].finish_reason == "stop"


@pytest.mark.parametrize("choice", [None, "auto", "required"])
async def test_unnamed_tool_choice_stream_still_rejects_stop_with_calls(choice) -> None:
    stub = StubOpenAI()
    stub.set_stream_chunks(_tool_stream("stop"))
    async with _llm(stub) as llm:
        with pytest.raises(ValueError, match="finish_reason 'stop'"):
            _ = [event async for event in llm.stream_events(USER, tools=[TOOL], tool_choice=choice)]


async def test_multimodal_user_content_is_kept_with_a_schema() -> None:
    stub = StubOpenAI()
    messages = [ChatMessage(role="user", content=[TextPart("What is shown?"), ImagePart("data:image/png;base64,AAAA")])]
    async with _llm(stub) as llm:
        await llm.chat(messages, response_schema=SCHEMA)
    sent = stub.last_json()["messages"]
    assert sent[0]["role"] == "user" and sent[0]["content"][0] == {"type": "text", "text": "What is shown?"}


def test_capabilities_declared_by_presets_are_known_fields() -> None:
    for name in available_presets():
        Capabilities(**get_preset(name).get("capabilities", {}))  # raises on an unknown key
    config = {"models": {"llm": {"adapter": {"preset": "nemotron_omni"}, "endpoint": {"base_url": "http://x"}}}}
    llm = make_llm(load_models_config_from_dict(config, source="test"), "llm")
    assert not llm.capabilities.json_schema and llm.capabilities.tool_choice


def test_a_profile_declares_capabilities_on_top_of_its_preset() -> None:
    # The documented way to qualify another model keeps the preset's features.
    config = {"models": {"llm": {
        "adapter": {"preset": "nemotron3_nano", "capabilities": {"json_schema": True, "tool_choice": True}},
        "endpoint": {"base_url": "http://x"},
    }}}
    llm = make_llm(load_models_config_from_dict(config, source="test"), "llm")
    preset = Capabilities(**get_preset("nemotron3_nano")["capabilities"])
    assert not preset.json_schema and not preset.tool_choice
    assert llm.capabilities == Capabilities(**{**preset.__dict__, "json_schema": True, "tool_choice": True})


# ── Eval harness, checked offline ─────────────────────────────────────────

EVAL_DIR = Path(__file__).resolve().parents[1] / "agent-sdk" / "xr-ai-models" / "eval"


def _eval_module():
    spec = importlib.util.spec_from_file_location("structured_output_eval", EVAL_DIR / "structured_output_eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_eval_cases_are_well_formed_and_graders_separate_examples() -> None:
    evaluation = _eval_module()
    cases = yaml.safe_load((EVAL_DIR / "cases" / "structured_output.yaml").read_text(encoding="utf-8"))["cases"]
    assert len({case["name"] for case in cases}) == len(cases)
    for case in cases:
        assert case["system"] and case["user"] and case["checks"], case["name"]
        assert case["schema"].get("type") == "object", case["name"]
        for answer in case["accept"]:
            parsed = json.loads(answer)
            assert all(evaluation.check(parsed, rule) for rule in case["checks"]), (case["name"], answer)
        for answer in case["reject"]:
            parsed = json.loads(answer)
            assert not all(evaluation.check(parsed, rule) for rule in case["checks"]), (case["name"], answer)
