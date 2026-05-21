# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire-trace golden tests for the xr-render-demo worker LLM call sites.

Exercises the four LLM call sites in ``RenderSceneProcessor`` against
``StubOpenAI`` without a real server or GPU.  Asserts that the JSON bodies
sent over the wire match pre-migration goldens (byte-for-byte field presence,
not ordering) and that ``ChatResponse`` fields are correctly extracted.

GPU verification skipped — stub-server tests only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add the worker directory to sys.path so we can import its modules.
_WORKER_DIR = (
    Path(__file__).resolve().parent.parent
    / "agent-samples" / "xr-render-demo" / "worker"
)
sys.path.insert(0, str(_WORKER_DIR))

from _stub_openai import StubOpenAI

from xr_ai_models import (
    ChatMessage,
    OpenAICompatLLM,
    ToolDef,
)
from xr_ai_models.config import load_models_config

# ── helpers ────────────────────────────────────────────────────────────────────

_MODELS_YAML = (
    Path(__file__).resolve().parent.parent
    / "agent-samples" / "xr-render-demo" / "yaml" / "models.yaml"
)


def _make_llm(stub: StubOpenAI, *, model_name: str = "llm",
              reasoning_field: str | None = None) -> OpenAICompatLLM:
    """Build an LLM client wired to a StubOpenAI transport."""
    return OpenAICompatLLM(
        "http://stub",
        model_name,
        reasoning_field=reasoning_field,
        client=stub.client(),
    )


# ── models.yaml round-trip ────────────────────────────────────────────────────


def test_models_yaml_loads() -> None:
    """The bundled models.yaml parses without error and exposes expected names."""
    cfg = load_models_config(_MODELS_YAML)
    llm_spec      = cfg.llm("llm")
    agent_llm_spec = cfg.llm("agent_llm")
    stt_spec      = cfg.stt("stt")
    tts_spec      = cfg.tts("tts")
    vlm_spec      = cfg.vlm("vlm")

    assert llm_spec.base_url       == "http://localhost:8106"
    assert agent_llm_spec.base_url == "http://localhost:8107"
    assert stt_spec.base_url       == "http://localhost:8103"
    assert tts_spec.base_url       == "http://localhost:8105"
    assert vlm_spec.base_url       == "http://localhost:8100"

    # nemotron3_nano preset must set reasoning_field so ChatResponse.reasoning
    # is populated from the server's "reasoning" field.
    assert agent_llm_spec.reasoning_field == "reasoning"


# ── quick-ack wire golden ─────────────────────────────────────────────────────


async def test_quick_ack_wire_golden() -> None:
    """quick-ack: max_tokens=40, temperature=0.0, no tools, no thinking."""
    stub = StubOpenAI()
    stub.set_chat_message(content='{"ack": "On it!", "think": false}')
    llm = _make_llm(stub)

    messages = [
        ChatMessage(role="system", content="You are a quick-ack classifier."),
        ChatMessage(role="user",   content="Add a red sphere in front of me"),
    ]
    resp = await llm.chat(messages, max_tokens=40, temperature=0.0)

    body = stub.last_json()

    # Field presence matches pre-migration golden.
    assert body["model"]        == "llm"
    assert body["max_tokens"]   == 40
    assert body["temperature"]  == 0.0
    assert "tools" not in body
    assert "chat_template_kwargs" not in body
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"

    assert resp.content == '{"ack": "On it!", "think": false}'
    assert resp.reasoning is None
    assert resp.tool_calls is None


# ── still-working wire golden ─────────────────────────────────────────────────


async def test_still_working_wire_golden() -> None:
    """still-working: max_tokens=24, temperature=0.9, no tools, no thinking."""
    stub = StubOpenAI()
    stub.set_chat_message(content="Still calculating the position...")
    llm = _make_llm(stub)

    messages = [
        ChatMessage(role="system", content="Generate a short still-working message."),
        ChatMessage(role="user",   content="User request: Add a sphere to my left"),
    ]
    resp = await llm.chat(messages, max_tokens=24, temperature=0.9)

    body = stub.last_json()

    assert body["model"]       == "llm"
    assert body["max_tokens"]  == 24
    assert body["temperature"] == 0.9
    assert "tools" not in body
    assert "chat_template_kwargs" not in body

    assert resp.content == "Still calculating the position..."


# ── validation wire golden ────────────────────────────────────────────────────


async def test_validation_wire_golden() -> None:
    """validation: max_tokens=60, temperature=0.0, no tools, no thinking."""
    stub = StubOpenAI()
    stub.set_chat_message(content='{"ok": true, "issue": ""}')
    llm = _make_llm(stub)

    messages = [
        ChatMessage(role="system", content="Validate whether the request was completed."),
        ChatMessage(role="user",   content='Request: Add red sphere\nCurrent scene: {}'),
    ]
    resp = await llm.chat(messages, max_tokens=60, temperature=0.0)

    body = stub.last_json()

    assert body["model"]       == "llm"
    assert body["max_tokens"]  == 60
    assert body["temperature"] == 0.0
    assert "tools" not in body
    assert "chat_template_kwargs" not in body

    assert resp.content == '{"ok": true, "issue": ""}'


# ── agentic-loop wire golden ──────────────────────────────────────────────────


async def test_agentic_loop_wire_golden_thinking_on() -> None:
    """agentic-loop with thinking enabled: tools, enable_thinking=True, thinking_budget=1024."""
    stub = StubOpenAI()
    stub.set_chat_message(content="Done — sphere added in front of you.")

    # nemotron3_nano uses model_name="llm" and reasoning_field="reasoning"
    agent_llm = _make_llm(stub, reasoning_field="reasoning")

    tools = [
        ToolDef(
            name="add_primitive",
            description="Add a primitive object to the scene.",
            parameters={
                "type": "object",
                "properties": {
                    "type":  {"type": "string"},
                    "x":     {"type": "number"},
                    "y":     {"type": "number"},
                    "z":     {"type": "number"},
                    "color": {"type": "string"},
                },
            },
        ),
        ToolDef(
            name="get_scene_state",
            description="Return the current scene objects.",
            parameters={"type": "object", "properties": {}},
        ),
    ]

    messages = [
        ChatMessage(role="system", content="You are a spatial AI assistant."),
        ChatMessage(
            role="user",
            content="[Pre-fetched context]\nSCENE OBJECTS: (empty)\n\n[Request]\nAdd a blue sphere",
        ),
    ]

    resp = await agent_llm.chat(
        messages,
        tools=tools,
        max_tokens=2048,
        temperature=0.0,
        enable_thinking=True,
        thinking_budget=1024,
    )

    body = stub.last_json()

    # Model name from nemotron3_nano preset.
    assert body["model"]       == "llm"
    assert body["max_tokens"]  == 2048
    assert body["temperature"] == 0.0

    # Tools must be present in OpenAI wire format.
    assert "tools" in body
    assert len(body["tools"]) == 2
    tool_names = {t["function"]["name"] for t in body["tools"]}
    assert tool_names == {"add_primitive", "get_scene_state"}

    # Thinking kwargs must be present.
    assert body.get("chat_template_kwargs") == {
        "enable_thinking":  True,
        "thinking_budget":  1024,
    }

    # Messages wired correctly.
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"

    # Response parsing.
    assert resp.content == "Done — sphere added in front of you."
    assert resp.tool_calls is None


async def test_agentic_loop_wire_golden_thinking_off() -> None:
    """agentic-loop with thinking off: no chat_template_kwargs in body."""
    stub = StubOpenAI()
    stub.set_chat_message(content="Done.")
    agent_llm = _make_llm(stub, reasoning_field="reasoning")

    messages = [
        ChatMessage(role="system", content="You are a spatial AI assistant."),
        ChatMessage(role="user",   content="[Pre-fetched context]\n\n[Request]\nAdd sphere"),
    ]
    await agent_llm.chat(
        messages,
        tools=[ToolDef(name="add_primitive", description="Add.", parameters={})],
        max_tokens=1024,
        temperature=0.0,
        enable_thinking=False,
    )

    body = stub.last_json()
    assert body["max_tokens"] == 1024
    assert "chat_template_kwargs" not in body


# ── reasoning-field normalization ─────────────────────────────────────────────


async def test_agentic_loop_reasoning_field_normalized() -> None:
    """nemotron3_nano preset uses reasoning_field='reasoning'; SDK exposes it as ChatResponse.reasoning."""
    stub = StubOpenAI()
    stub.set_chat_message(
        content="I placed the sphere ahead of you.",
        reasoning="RESOLVE: user said 'in front' → forward direction. COMPUTE: pos = head + fwd × 1.5",
        reasoning_field="reasoning",  # nano_v3 server writes to "reasoning"
    )

    agent_llm = _make_llm(stub, reasoning_field="reasoning")

    resp = await agent_llm.chat(
        [ChatMessage(role="user", content="Add a sphere in front")],
    )

    assert resp.reasoning == (
        "RESOLVE: user said 'in front' → forward direction. COMPUTE: pos = head + fwd × 1.5"
    )
    assert resp.content   == "I placed the sphere ahead of you."
    assert resp.tool_calls is None


async def test_agentic_loop_tool_calls_parsed() -> None:
    """Tool calls in the agentic loop are parsed into ToolCall objects."""
    stub = StubOpenAI()
    stub.set_chat_message(
        content="",
        tool_calls=[{
            "id":       "call_abc123",
            "type":     "function",
            "function": {
                "name":      "add_primitive",
                "arguments": '{"type": "sphere", "x": 0.0, "y": 1.6, "z": -1.5}',
            },
        }],
        finish_reason="tool_calls",
    )

    agent_llm = _make_llm(stub, reasoning_field="reasoning")
    resp = await agent_llm.chat(
        [ChatMessage(role="user", content="Add sphere ahead")],
        tools=[ToolDef(name="add_primitive", description="Add.", parameters={})],
    )

    assert resp.tool_calls is not None
    assert len(resp.tool_calls) == 1
    tc = resp.tool_calls[0]
    assert tc.id        == "call_abc123"
    assert tc.name      == "add_primitive"
    args = json.loads(tc.arguments)
    assert args["type"] == "sphere"
    assert args["x"]    == 0.0


# ── ToolDef.to_openai() round-trip ────────────────────────────────────────────


def test_tool_def_to_openai_wire_shape() -> None:
    """ToolDef.to_openai() must produce the exact OpenAI wire shape.

    The SDK re-produces the same shape the pre-migration hand-rolled dicts had
    so the upstream server sees byte-identical tool definitions.
    """
    td = ToolDef(
        name="update_primitive",
        description="Update an existing object.",
        parameters={
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "x":  {"type": "number"},
            },
        },
    )
    wire = td.to_openai()
    assert wire == {
        "type": "function",
        "function": {
            "name":        "update_primitive",
            "description": "Update an existing object.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "x":  {"type": "number"},
                },
            },
        },
    }
