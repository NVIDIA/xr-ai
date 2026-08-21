# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire-trace golden tests for the xr-render-demo model-service contracts.

Exercises the worker's direct model calls against ``StubOpenAI`` without a
real server or GPU. Asserts that the JSON bodies sent over the wire retain the
required fields and that ``ChatResponse`` fields are correctly extracted.

GPU verification skipped — stub-server tests only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from xr_ai_models import (
    ChatMessage,
    OpenAICompatLLM,
    ToolDef,
    load_models_config,
)
from xr_render_demo_worker.config import load_config
from xr_render_demo_worker.models import SceneReply, SceneRequest, SubagentResult, SubagentTask

# Add the worker directory to sys.path so we can import its modules.
_WORKER_DIR = (
    Path(__file__).resolve().parent.parent
    / "agent-samples" / "xr-render-demo" / "worker"
)
sys.path.insert(0, str(_WORKER_DIR))

from _stub_openai import StubOpenAI  # noqa: E402

_SAMPLE = Path(__file__).resolve().parent.parent / "agent-samples" / "xr-render-demo"
_PACKAGE = _WORKER_DIR / "xr_render_demo_worker"

# ── helpers ────────────────────────────────────────────────────────────────────

_MODELS_PROFILE = (
    Path(__file__).resolve().parent.parent
    / "agent-samples" / "xr-render-demo" / "yaml" / "models.local.json"
)


def _make_llm(stub: StubOpenAI, *, model_name: str = "llm",
              reasoning_field: str | None = None,
              default_extras: dict | None = None) -> OpenAICompatLLM:
    """Build an LLM client wired to a StubOpenAI transport."""
    return OpenAICompatLLM(
        "http://stub",
        model_name,
        reasoning_field=reasoning_field,
        default_extras=default_extras,
        client=stub.client(),
    )


def _make_spec_llm(stub: StubOpenAI, name: str) -> OpenAICompatLLM:
    """Build an LLM client from the shipped local profile spec for *name*."""
    spec = load_models_config(_MODELS_PROFILE).llm(name)
    return _make_llm(
        stub,
        model_name=spec.model_name,
        reasoning_field=spec.reasoning_field,
        default_extras=spec.default_extras,
    )


def test_worker_config_loads_sample_yaml() -> None:
    config = load_config(_SAMPLE / "yaml/xr_render_demo_worker.yaml")
    assert config.voice_gate_yaml.exists()


def test_all_agent_modules_export_descriptions() -> None:
    from xr_render_demo_worker.agents import appearance, memory, object, placement, vision  # noqa: F401


def test_prompt_files_exist_and_are_nonempty() -> None:
    prompts = sorted(_PACKAGE.rglob("*prompt*.txt"))
    assert len(prompts) == 6  # supervisor + five subagents
    for prompt in prompts:
        assert prompt.read_text(encoding="utf-8").strip(), prompt


def test_models_round_trip() -> None:
    request = SceneRequest(transcript="hi", participant_id="p", timestamp_us=1)
    assert SceneRequest.model_validate(request.model_dump()) == request
    task = SubagentTask(instruction="do")
    assert SubagentTask.model_validate(task.model_dump()) == task
    assert SceneReply(response="ok").response == "ok"
    assert SubagentResult(result="ok").result == "ok"



def test_prompt_audit_is_clean() -> None:
    import io
    from contextlib import redirect_stdout

    from xr_render_demo_eval import harness

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        harness.audit_prompts()
    warnings = [line for line in buffer.getvalue().splitlines() if line.startswith("AUDIT WARNING")]
    assert not warnings, warnings


@pytest.mark.parametrize("entry", ["xr_render_demo_worker.__main__"])
def test_entry_module_imports(entry: str) -> None:
    __import__(entry)


def test_truncation_replies_resolve() -> None:
    from xr_render_demo_worker.supervisor import _resolve_truncation_reply, _splice_completion

    assert _splice_completion("Put the sphere on the", "On the box.") == "Put the sphere on the box."
    assert _splice_completion("Put the sphere on the", "The box.") == "Put the sphere on the box."
    assert _splice_completion("Move it towards", "The window.") == "Move it towards The window."
    assert _resolve_truncation_reply("Put the sphere on the", "Never mind.") is None
    assert _resolve_truncation_reply("Put the sphere on the", "Cancel") is None
    fresh = _resolve_truncation_reply("Put the sphere on the", "Put the sphere on the box.")
    assert fresh == "Put the sphere on the box."


def test_truncated_transcripts_detected() -> None:
    from xr_render_demo_worker.supervisor import _is_truncated, _truncated_reply

    assert _is_truncated("Add a red sphere in")
    assert not _is_truncated("Add a red sphere in front of me.")
    assert _truncated_reply("Add a red sphere in") is not None


# ── models profile round-trip ─────────────────────────────────────────────────


def test_models_profile_loads() -> None:
    """The bundled local profile parses without error and exposes expected names."""
    cfg = load_models_config(_MODELS_PROFILE)
    llm_spec      = cfg.llm("llm")
    agent_llm_spec = cfg.llm("agent_llm")
    stt_spec      = cfg.stt("stt")
    tts_spec      = cfg.tts("tts")
    vlm_spec      = cfg.vlm("vlm")

    assert llm_spec.base_url       == "http://localhost:8108"
    assert agent_llm_spec.base_url == "http://localhost:8108"
    assert stt_spec.base_url       == "http://localhost:8103"
    assert tts_spec.base_url       == "http://localhost:8105"
    assert vlm_spec.base_url       == "http://localhost:8100"

    # nemotron_omni preset must set reasoning_field so ChatResponse.reasoning
    # is populated from vLLM's "reasoning_content" field.
    assert agent_llm_spec.reasoning_field == "reasoning_content"

    # Both logical LLMs share the Omni server. The preset must pin thinking off
    # at the wire level: Nemotron-3-Nano-Omni's template defaults thinking-on,
    # which burns short reply budgets on hidden reasoning and returns empty
    # content with finish_reason="length".
    for spec in (llm_spec, agent_llm_spec):
        assert spec.model_name == "llm"
        assert spec.default_extras["chat_template_kwargs"] == {"enable_thinking": False}
    assert vlm_spec.model_name == "vlm"
    assert vlm_spec.capabilities["vision"] is True


def test_worker_config_idle_timeout_disabled_by_default() -> None:
    """The shipped worker YAML ships idle_timeout_secs: 0, which the loader
    maps to None (disabled) so a quiet session is never auto-cancelled."""
    from xr_render_demo_worker.config import load_config

    worker_yaml = (
        Path(__file__).resolve().parent.parent
        / "agent-samples" / "xr-render-demo" / "yaml" / "xr_render_demo_worker.yaml"
    )
    cfg = load_config(worker_yaml)
    assert cfg.idle_timeout_secs is None


def test_worker_config_idle_timeout_opt_in(tmp_path) -> None:
    """A positive idle_timeout_secs in the YAML is parsed to a float."""
    from xr_render_demo_worker.config import load_config

    y = tmp_path / "w.yaml"
    y.write_text("idle_timeout_secs: 300\n")
    cfg = load_config(y)
    assert cfg.idle_timeout_secs == 300.0


# ── untooled chat wire golden ─────────────────────────────────────────────────


async def test_untooled_chat_wire_golden() -> None:
    """Untooled chat on the llm preset: params pass through, thinking pinned off."""
    stub = StubOpenAI()
    stub.set_chat_message(content="On it!")
    llm = _make_spec_llm(stub, "llm")

    messages = [
        ChatMessage(role="system", content="Reply in one short sentence."),
        ChatMessage(role="user",   content="Add a red sphere in front of me"),
    ]
    resp = await llm.chat(messages, max_tokens=40, temperature=0.0)

    body = stub.last_json()

    assert body["model"]        == "llm"
    assert body["max_tokens"]   == 40
    assert body["temperature"]  == 0.0
    assert "tools" not in body
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"

    assert resp.content == "On it!"
    assert resp.reasoning is None
    assert resp.tool_calls is None


# ── agentic-loop wire golden ──────────────────────────────────────────────────


async def test_agentic_loop_wire_golden_thinking_on() -> None:
    """agentic-loop with thinking enabled: tools, enable_thinking=True, thinking_budget=1024."""
    stub = StubOpenAI()
    stub.set_chat_message(content="Done — sphere added in front of you.")

    agent_llm = _make_spec_llm(stub, "agent_llm")

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

    # Model name from the nemotron_omni preset.
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
    """agentic-loop with thinking off: the preset's wire-level default applies."""
    stub = StubOpenAI()
    stub.set_chat_message(content="Done.")
    agent_llm = _make_spec_llm(stub, "agent_llm")

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
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


# ── reasoning-field normalization ─────────────────────────────────────────────


async def test_agentic_loop_reasoning_field_normalized() -> None:
    """The Omni reasoning field is normalized to ChatResponse.reasoning."""
    stub = StubOpenAI()
    stub.set_chat_message(
        content="I placed the sphere ahead of you.",
        reasoning="RESOLVE: user said 'in front' → forward direction. COMPUTE: pos = head + fwd × 1.5",
        reasoning_field="reasoning_content",
    )

    agent_llm = _make_llm(stub, reasoning_field="reasoning_content")

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
