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


REASONING = ThinkingSpec(control="chat_template", default=True)


def _instruction() -> str:
    return "Respond with only a JSON value that conforms to this JSON Schema:\n" + json.dumps(
        SCHEMA, separators=(",", ":"),
    )


async def test_response_schema_without_reasoning_sends_only_the_constraint() -> None:
    # A stated schema biased enum answers when no reasoning precedes them.
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


async def test_response_schema_with_reasoning_states_it_in_a_new_system_message() -> None:
    # After reasoning, an unstated constraint let answers fill with whitespace.
    stub = StubOpenAI()
    async with _llm(stub, thinking=REASONING) as llm:
        await llm.chat(USER, response_schema=SCHEMA, thinking_budget=256)
    body = stub.last_json()
    assert body["messages"] == [
        {"role": "system", "content": _instruction()},
        {"role": "user", "content": "Did it change?"},
    ]
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA


@pytest.mark.parametrize(("content", "expected"), [
    ("Watch the kettle.", f"Watch the kettle.\n\n{_instruction()}"),
    ("", _instruction()),
    ([TextPart("Watch the kettle.")], [
        {"type": "text", "text": "Watch the kettle."}, {"type": "text", "text": _instruction()},
    ]),
])
async def test_response_schema_extends_the_callers_system_message(content, expected) -> None:
    stub = StubOpenAI()
    messages = [ChatMessage(role="system", content=content), *USER]
    async with _llm(stub, thinking=REASONING) as llm:
        await llm.chat(messages, response_schema=SCHEMA, thinking_budget=256)
    sent = stub.last_json()["messages"]
    assert sent[0] == {"role": "system", "content": expected}
    assert sent[1:] == [{"role": "user", "content": "Did it change?"}]
    assert messages[0].content == content  # the caller's messages are not changed


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


CONSTRAINTS = [{"response_schema": SCHEMA}, {"tools": [TOOL], "tool_choice": "commit"}]


@pytest.mark.parametrize("constraint", CONSTRAINTS)
@pytest.mark.parametrize(("thinking", "enable", "capabilities"), [
    (ThinkingSpec(control="chat_template", default=True), None, BOTH),
    (ThinkingSpec(control="chat_template", default=False), True, BOTH),
    # An unknown default on a reasoning model may think.
    (ThinkingSpec(), None, Capabilities(json_schema=True, tool_choice=True, reasoning=True)),
])
async def test_reasoning_with_a_constraint_requires_a_budget(constraint, thinking, enable, capabilities) -> None:
    stub = StubOpenAI()
    async with _llm(stub, thinking=thinking, capabilities=capabilities) as llm:
        with pytest.raises(ValueError, match="requires thinking_budget"):
            await llm.chat(USER, enable_thinking=enable, **constraint)
        with pytest.raises(ValueError, match="max_tokens must exceed thinking_budget"):
            await llm.chat(USER, enable_thinking=enable, thinking_budget=256, max_tokens=256, **constraint)
        assert not stub.requests
        await llm.chat(USER, enable_thinking=enable, thinking_budget=256, max_tokens=1024, **constraint)
    assert stub.last_json()["thinking_token_budget"] == 256


@pytest.mark.parametrize("constraint", CONSTRAINTS)
@pytest.mark.parametrize(("thinking", "enable"), [
    (ThinkingSpec(control="chat_template", default=True), False),
    (ThinkingSpec(control="chat_template", default=False), None),
    (ThinkingSpec(control="none", default=True), None),
    (ThinkingSpec(), None),  # unknown default on a model without declared reasoning
])
async def test_constraints_without_reasoning_need_no_budget(constraint, thinking, enable) -> None:
    stub = StubOpenAI()
    async with _llm(stub, thinking=thinking) as llm:
        await llm.chat(USER, enable_thinking=enable, **constraint)
    assert "thinking_token_budget" not in stub.last_json()


@pytest.mark.parametrize("constraint", CONSTRAINTS)
@pytest.mark.parametrize(("thinking", "capabilities"), [
    (ThinkingSpec(control="system_prompt", default=True), BOTH),
    # No switch reaches the model, and it declares reasoning output.
    (ThinkingSpec(control="none"), Capabilities(json_schema=True, tool_choice=True, reasoning=True)),
])
async def test_reasoning_without_an_enforced_budget_cannot_be_constrained(constraint, thinking, capabilities) -> None:
    stub = StubOpenAI()
    async with _llm(stub, thinking=thinking, capabilities=capabilities) as llm:
        with pytest.raises(ValueError, match="enforces a thinking budget"):
            await llm.chat(USER, thinking_budget=256, **constraint)
    assert not stub.requests


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
    async with _llm(stub, thinking=REASONING) as llm:
        await llm.chat(messages, response_schema=SCHEMA, thinking_budget=256)
    sent = stub.last_json()["messages"]
    assert sent[0]["role"] == "system" and sent[1]["content"][0] == {"type": "text", "text": "What is shown?"}


def test_capabilities_declared_by_presets_are_known_fields() -> None:
    for name in available_presets():
        Capabilities(**get_preset(name).get("capabilities", {}))  # raises on an unknown key
    config = {"models": {"llm": {"adapter": {"preset": "nemotron_omni"}, "endpoint": {"base_url": "http://x"}}}}
    llm = make_llm(load_models_config_from_dict(config, source="test"), "llm")
    assert llm.capabilities.json_schema and llm.capabilities.tool_choice


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


def test_eval_states_the_schema_exactly_as_the_client_does() -> None:
    # The prompt_only reference must differ from json_schema only by the constraint.
    from xr_ai_models._openai_compat import _with_schema_instruction

    evaluation = _eval_module()
    stated = _with_schema_instruction([{"role": "system", "content": "S"}], SCHEMA)[0]["content"]
    assert stated == "S\n\n" + evaluation._schema_instruction(SCHEMA)
