# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire-trace golden tests for the xr-render-demo model-service contracts.

Exercises the worker's direct model calls against ``StubOpenAI`` without a
real server or GPU. Asserts that the JSON bodies sent over the wire retain the
required fields and that ``ChatResponse`` fields are correctly extracted.

GPU verification skipped — stub-server tests only.
"""
from __future__ import annotations

import asyncio
import json
import runpy
import sys
import tomllib
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

# Add the worker directory to sys.path so we can import its modules.
_WORKER_DIR = (
    Path(__file__).resolve().parent.parent
    / "agent-samples" / "xr-render-demo" / "worker"
)
sys.path.insert(0, str(_WORKER_DIR))

from _stub_openai import StubOpenAI

from xr_ai_hub import DataMessage
from xr_ai_models import (
    ChatMessage,
    OpenAICompatLLM,
    ToolDef,
    load_models_config,
)
from xr_ai_tools import ToolSet
from xr_ai_tools.tracking import TrackingTools
from xr_ai_tools.tool_calling import tool_definitions

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
    # at the wire level: Nemotron-3-Nano-Omni's template defaults
    # thinking-on, which would burn the quick-ack's 40-token budget on hidden
    # reasoning and return empty content with finish_reason="length".
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
    assert cfg.web_events_host == "127.0.0.1"
    assert cfg.web_events_port == 8092


def test_worker_depends_on_web_events_sdk() -> None:
    project = tomllib.loads((_WORKER_DIR / "pyproject.toml").read_text())

    assert "xr-ai-web-events" in project["project"]["dependencies"]
    assert project["tool"]["uv"]["sources"]["xr-ai-web-events"]["path"] == (
        "../../../agent-sdk/xr-ai-web-events"
    )


def test_worker_config_idle_timeout_opt_in(tmp_path) -> None:
    """A positive idle_timeout_secs in the YAML is parsed to a float."""
    from xr_render_demo_worker.config import load_config

    y = tmp_path / "w.yaml"
    y.write_text("idle_timeout_secs: 300\n")
    cfg = load_config(y)
    assert cfg.idle_timeout_secs == 300.0


def test_eval_harness_imports_public_perception_contracts() -> None:
    eval_path = (Path(__file__).resolve().parent.parent / "agent-samples"
                 / "xr-render-demo" / "eval" / "eval.py")
    namespace = runpy.run_path(str(eval_path), run_name="xr_render_eval_contract")

    assert namespace["LIVE_PERCEPTION_TOOL"] == _loop.LIVE_PERCEPTION_TOOL
    assert namespace["PAST_PERCEPTION_TOOL"] == _loop.PAST_PERCEPTION_TOOL
    assert namespace["PERCEPTION_TOOL_DEFS"] == _loop.PERCEPTION_TOOL_DEFS


# ── quick-ack wire golden ─────────────────────────────────────────────────────


async def test_quick_ack_wire_golden() -> None:
    """quick-ack: max_tokens=40, temperature=0.0, no tools, thinking pinned off."""
    stub = StubOpenAI()
    stub.set_chat_message(content='{"ack": "On it!", "think": false}')
    llm = _make_spec_llm(stub, "llm")

    messages = [
        ChatMessage(role="system", content="You are a quick-ack classifier."),
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

    assert resp.content == '{"ack": "On it!", "think": false}'
    assert resp.reasoning is None
    assert resp.tool_calls is None


# ── still-working wire golden ─────────────────────────────────────────────────


async def test_still_working_wire_golden() -> None:
    """still-working: max_tokens=24, temperature=0.9, no tools, thinking pinned off."""
    stub = StubOpenAI()
    stub.set_chat_message(content="Still calculating the position...")
    llm = _make_spec_llm(stub, "llm")

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
    assert body["chat_template_kwargs"] == {"enable_thinking": False}

    assert resp.content == "Still calculating the position..."


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


# ── XR-launch-failure notice delivery (runtime voice + panel) ─────────────────
#
# When start_xr / the LOVR-spawn poll fails, XRSessionLifecycle publishes a
# notice. The scene loop must deliver it with
# the same shape as a normal final answer: a yielded voice chunk and an
# agent.response data message for the panel.

_PROMPTS_DIR = _WORKER_DIR / "xr_render_demo_worker" / "prompts"
_SYSTEM_PROMPT = _PROMPTS_DIR / "system.txt"

_LAUNCH_FAIL_MSG = "I couldn't start the XR session — try Launch XR again."


class _CaptureTransport:
    """HubVoiceTransport double — records send_return_data and owns the
    target participant. Only the surface the notice path touches."""

    def __init__(self) -> None:
        self.target_participant = ""
        self.sent: list[DataMessage] = []

    def set_target_participant(self, pid: str) -> None:
        self.target_participant = pid

    async def send_return_data(self, msg: DataMessage) -> None:
        self.sent.append(msg)


def _make_brain(transport: _CaptureTransport, llm=None):
    """Build a real SceneModelLoop whose service clients are unused.

    The notice path never dereferences them. The constructor eagerly reads
    the real prompt files, so point at the bundled prompts/ directory.
    Pass ``llm`` to exercise the real _quick_ack parse paths against a stub.
    """
    return _loop.SceneModelLoop(
        transport   = transport,
        cfg         = None,
        tools       = ToolSet(()),
        release_vision = lambda _pid: None,
        text_memory = None,
        prompt_path = _SYSTEM_PROMPT,
        model_tools = [],
        llm         = llm,
        agent_llm   = None,
    )


async def test_launch_failure_notice_spoken_and_paneled() -> None:
    """A lifecycle notice yields voice and sends panel data to the same pid."""
    transport = _CaptureTransport()
    transport.set_target_participant("pid-1")
    brain = _make_brain(transport)

    spoken = [text async for text in brain.handle_notice("pid-1", _LAUNCH_FAIL_MSG)]

    assert spoken == [_LAUNCH_FAIL_MSG]

    # Panel: exactly one agent.response send to the originating pid.
    assert len(transport.sent) == 1
    sent = transport.sent[0]
    assert sent.topic == "agent.response"
    assert sent.participant_id == "pid-1"
    assert sent.data.decode() == _LAUNCH_FAIL_MSG


async def test_quick_ack_spoken_on_non_thinking_turn() -> None:
    """ACK-SPEAK POLICY: the quick-ack is yielded (→ TTS) on EVERY turn,
    including a non-thinking one, so a tool-using turn is never silent until
    the final reply. Pre-change the ack was spoken only when needs_thinking.

    _quick_ack and _agentic_loop are stubbed so no LLM or tool execution is touched.
    """
    transport = _CaptureTransport()
    transport.set_target_participant("pid-1")
    brain = _make_brain(transport)

    async def _fake_quick_ack(_pid, _text):
        return ("On it.", False)  # ack present, needs_thinking = False

    async def _fake_loop(*_a, **_k):
        return "All set."

    brain._quick_ack = _fake_quick_ack      # noqa: SLF001
    brain._agentic_loop = _fake_loop        # noqa: SLF001

    gen = await brain.handle_query("pid-1", "place a cube")
    spoken = [s async for s in gen]

    # Ack is spoken first (so the turn isn't silent), then the final reply.
    assert spoken and spoken[0] == "On it."
    assert "All set." in spoken
    # Ack is also mirrored to the panel on agent.progress.
    progress = [m for m in transport.sent if m.topic == "agent.progress"]
    assert any(m.data.decode() == "On it." for m in progress)


def test_tool_result_json_is_sanitized() -> None:
    """A final response that is nothing but a JSON object (e.g. an echoed
    tool result) must be flagged so it never reaches TTS; prose that merely
    contains JSON passes."""
    from xr_render_demo_worker.model_io import looks_like_leaked_tool_call

    assert looks_like_leaked_tool_call('{"id": "box-1", "ok": true, "reason": null}')
    assert looks_like_leaked_tool_call('[{"id": "box-1", "ok": true}]')
    assert not looks_like_leaked_tool_call('Added box-1 ({"ok": true} from the scene).')
    assert not looks_like_leaked_tool_call("Added a blue sphere.")


async def test_quick_ack_parses_wellformed_json() -> None:
    """_quick_ack returns the ack string and a strict-bool think flag."""
    stub = StubOpenAI()
    stub.set_chat_message(content='{"ack": "On it", "think": true}')
    brain = _make_brain(_CaptureTransport(), llm=_make_spec_llm(stub, "llm"))
    assert await brain._quick_ack("pid-1", "move the cube") == ("On it", True)  # noqa: SLF001


async def test_quick_ack_string_think_is_not_truthy() -> None:
    """A model emitting "think": "false" (a string) must not enable thinking."""
    stub = StubOpenAI()
    stub.set_chat_message(content='{"ack": "On it", "think": "false"}')
    brain = _make_brain(_CaptureTransport(), llm=_make_spec_llm(stub, "llm"))
    assert await brain._quick_ack("pid-1", "move the cube") == ("On it", False)  # noqa: SLF001


async def test_quick_ack_truncated_json_not_spoken() -> None:
    """A truncated JSON payload has no closing brace; the raw fragment must
    not be returned as a speakable ack."""
    stub = StubOpenAI()
    stub.set_chat_message(content='{"ack": "Let me ta')
    brain = _make_brain(_CaptureTransport(), llm=_make_spec_llm(stub, "llm"))
    assert await brain._quick_ack("pid-1", "what am I holding") == ("", False)  # noqa: SLF001


async def test_quick_ack_bare_prose_fallback() -> None:
    """Non-JSON prose is used as the ack with thinking off."""
    stub = StubOpenAI()
    stub.set_chat_message(content="On it")
    brain = _make_brain(_CaptureTransport(), llm=_make_spec_llm(stub, "llm"))
    assert await brain._quick_ack("pid-1", "add a sphere") == ("On it", False)  # noqa: SLF001


async def test_quick_ack_transport_error_falls_back_silent_fast() -> None:
    """LLM failure → no ack, thinking off (fail toward the tool-trusting mode)."""

    class _BoomLLM:
        async def chat(self, *_a, **_k):
            raise TimeoutError

    brain = _make_brain(_CaptureTransport(), llm=_BoomLLM())
    assert await brain._quick_ack("pid-1", "move it up") == ("", False)  # noqa: SLF001


async def test_already_punctuated_ack_not_doubled() -> None:
    """An ack ending in !/? passes through unchanged (no "On it!.")."""
    transport = _CaptureTransport()
    transport.set_target_participant("pid-1")
    brain = _make_brain(transport)

    async def _fake_quick_ack(_pid, _text):
        return ("On it!", False)

    async def _fake_loop(*_a, **_k):
        return "Done."

    brain._quick_ack = _fake_quick_ack      # noqa: SLF001
    brain._agentic_loop = _fake_loop        # noqa: SLF001

    gen = await brain.handle_query("pid-1", "add a cube")
    spoken = [s async for s in gen]
    assert spoken == ["On it!", "Done."]


async def test_empty_ack_yields_no_spoken_line() -> None:
    """The quick-ack failure fallback ("", False) must not yield an empty
    ack line or post an empty progress message; the final reply still lands."""
    transport = _CaptureTransport()
    transport.set_target_participant("pid-1")
    brain = _make_brain(transport)

    async def _fake_quick_ack(_pid, _text):
        return ("", False)

    async def _fake_loop(*_a, **_k):
        return "All set."

    brain._quick_ack = _fake_quick_ack      # noqa: SLF001
    brain._agentic_loop = _fake_loop        # noqa: SLF001

    gen = await brain.handle_query("pid-1", "add a cube")
    spoken = [s async for s in gen]
    assert spoken == ["All set."]
    assert not [m for m in transport.sent if m.topic == "agent.progress"]


async def test_unpunctuated_ack_gets_terminal_period() -> None:
    """Acks without terminal punctuation are normalized before being yielded
    to TTS; the panel copy stays verbatim."""
    transport = _CaptureTransport()
    transport.set_target_participant("pid-1")
    brain = _make_brain(transport)

    async def _fake_quick_ack(_pid, _text):
        return ("Let me take a look", True)

    async def _fake_loop(*_a, **_k):
        return "Done."

    brain._quick_ack = _fake_quick_ack      # noqa: SLF001
    brain._agentic_loop = _fake_loop        # noqa: SLF001

    gen = await brain.handle_query("pid-1", "what am I holding")
    spoken = [s async for s in gen]

    assert spoken[0] == "Let me take a look."
    progress = [m for m in transport.sent if m.topic == "agent.progress"]
    assert any(m.data.decode() == "Let me take a look" for m in progress)


# ── live-frame perception routing (look_at_current_frame) ──────────────────────
#
# A real-world visual question must reach the live-frame VLM path. These tests
# stub the VLM client and hub frame path to verify that routing contract.

from xr_ai_hub import FrameData, FrameSignal, PixelFormat  # noqa: E402
from xr_ai_models import ChatResponse, ToolCall  # noqa: E402

from xr_render_demo_worker import scene_loop as _loop  # noqa: E402
from xr_render_demo_worker import spatial_tools as _spatial_tools  # noqa: E402
from xr_render_demo_worker import tools as _tools  # noqa: E402
from xr_render_demo_worker import __main__ as _worker  # noqa: E402


class _WarmupLLM:
    def __init__(self, *, healthy: bool, fail_warmup: bool = False) -> None:
        self.healthy = healthy
        self.fail_warmup = fail_warmup
        self.calls: list[tuple[list[ChatMessage], dict]] = []

    async def health(self) -> bool:
        return self.healthy

    async def chat(self, messages: list[ChatMessage], **kwargs):
        self.calls.append((messages, kwargs))
        if self.fail_warmup:
            raise RuntimeError("still loading")


async def test_llm_warmup_waits_for_health() -> None:
    llm = _WarmupLLM(healthy=False)

    assert await _worker._probe_warmed_llm(llm, warmup=True) is False
    assert llm.calls == []


async def test_llm_readiness_requires_successful_warmup() -> None:
    llm = _WarmupLLM(healthy=True, fail_warmup=True)

    assert await _worker._probe_warmed_llm(llm, warmup=True) is False
    assert len(llm.calls) == 1


async def test_llm_warmup_probe_uses_first_turn_contract() -> None:
    llm = _WarmupLLM(healthy=True)

    assert await _worker._probe_warmed_llm(llm, warmup=True) is True
    messages, options = llm.calls[0]
    assert [message.content for message in messages] == ["Add a small cube."]
    assert options == {"max_tokens": 40, "timeout": 120.0}


async def test_render_spatial_native_toolbox_builds() -> None:
    """The sample's prompt-compatible spatial surface uses native Tool schemas."""
    tracking = TrackingTools("tcp://127.0.0.1:65530", timeout_s=0.1)
    try:
        spatial = _spatial_tools.RenderSpatialTools(tracking)
        definitions = {tool.name: tool for tool in tool_definitions(spatial.tools)}
        expected_parameters = {
            "along_direction": {
                "origin_x", "origin_y", "origin_z", "target_x", "target_y", "target_z", "distance",
            },
            "between_anchors": {"a_x", "a_y", "a_z", "b_x", "b_y", "b_z"},
            "displace_object": {"current_x", "current_y", "current_z", "right", "up", "forward"},
            "displace_objects": {
                "object_ids", "current_xs", "current_ys", "current_zs", "right", "up", "forward",
            },
            "get_head_pose": set(),
            "place_inside_by_id": {"movee_id", "container_x", "container_y", "container_z"},
            "place_object_relative": {"origin_x", "origin_y", "origin_z", "direction", "distance"},
            "place_user_relative": {"direction", "distance"},
            "position_ahead": {"distance"},
            "position_relative": {
                "forward", "right", "up", "origin_x", "origin_y", "origin_z",
            },
            "scale_value": {"current", "factor"},
            "world_offset": {"origin_x", "origin_y", "origin_z", "dx", "dy", "dz"},
        }
        assert set(definitions) == set(expected_parameters)
        for name, parameters in expected_parameters.items():
            properties = definitions[name].parameters["properties"]
            assert set(properties) == parameters
            assert all(prop.get("description") for prop in properties.values())
    finally:
        await tracking.close()


async def test_render_spatial_vector_tools_execute_edge_cases() -> None:
    tracking = TrackingTools("tcp://127.0.0.1:65530", timeout_s=0.1)
    try:
        spatial = _spatial_tools.RenderSpatialTools(tracking)
        offset = await spatial.world_offset.execute(
            _spatial_tools.WorldOffsetRequest(
                origin_x=1.0, origin_y=2.0, origin_z=3.0,
                dx=-0.5, dy=1.0, dz=2.0,
            )
        )
        scaled = await spatial.scale_value.execute(
            _spatial_tools.ScaleValueRequest(current=1.25, factor=2.0)
        )
        away = await spatial.along_direction.execute(
            _spatial_tools.AlongDirectionRequest(
                origin_x=0.0, origin_y=0.0, origin_z=0.0,
                target_x=1.0, target_y=0.0, target_z=0.0, distance=-2.0,
            )
        )

        assert offset.model_dump() == {"x": 0.5, "y": 3.0, "z": 5.0}
        assert scaled.value == 2.5
        assert away.model_dump() == {"x": -2.0, "y": 0.0, "z": 0.0}
        with pytest.raises(RuntimeError, match="origin and target coincide"):
            await spatial.along_direction.execute(
                _spatial_tools.AlongDirectionRequest(
                    origin_x=1.0, origin_y=1.0, origin_z=1.0,
                    target_x=1.0, target_y=1.0, target_z=1.0,
                    distance=0.5,
                )
            )
    finally:
        await tracking.close()


async def test_live_worker_and_eval_share_native_toolbox_assembly() -> None:
    """NativeCapabilities exposes the complete runtime tool surface without MCP."""
    capabilities = _tools.NativeCapabilities(
        scene_endpoint="tcp://127.0.0.1:65527",
        openxr_endpoint="tcp://127.0.0.1:65528",
        video_memory_endpoint="tcp://127.0.0.1:65529",
        frame_endpoint=_FakeEndpoint(),
        vlm=_FakeVLM(),
        text_memory_dir="/tmp/xr-render-test-memory",
    )
    try:
        names = {name for name, _tool in capabilities.all.items()}
    finally:
        await capabilities.close()

    assert names == {
        "add_primitive",
        "along_direction",
        "between_anchors",
        "displace_object",
        "displace_objects",
        "get_current_frame",
        "get_historical_frame",
        "get_historical_frames",
        "get_historical_video",
        "get_head_pose",
        "get_health",
        "get_latest_video",
        "get_scene_state",
        "get_video_stats",
        "list_recorded_participants",
        "place_inside_by_id",
        "place_object_relative",
        "place_user_relative",
        "position_ahead",
        "position_relative",
        "query_image",
        "query_images",
        "query_video",
        "get_latest_frames",
        "remove_primitive",
        "scale_value",
        "start_xr",
        "update_primitive",
        "world_offset",
    }


async def test_model_facing_perception_schema_is_trimmed() -> None:
    """The model sees participant-free perception facades, not internal tools.

    The worker injects participant and absolute-time context, selects an image,
    and then calls ``query_image`` without exposing either internal contract to
    the model.
    """
    capabilities = _tools.NativeCapabilities(
        scene_endpoint="tcp://127.0.0.1:65527",
        openxr_endpoint="tcp://127.0.0.1:65528",
        video_memory_endpoint="tcp://127.0.0.1:65529",
        frame_endpoint=_FakeEndpoint(),
        vlm=_FakeVLM(),
        text_memory_dir="/tmp/xr-render-test-memory",
    )
    try:
        native = {tool.name: tool for tool in tool_definitions(capabilities.model)}
        assert "get_current_frame" not in native
        assert "query_image" not in native
        assert "look_at_current_frame" not in native
        assert "look_at_past_frame" not in native
        assert not {
            "get_historical_frame",
            "get_historical_frames",
            "get_historical_video",
            "get_latest_frames",
            "get_latest_video",
            "query_images",
            "query_video",
        } & native.keys()

        # Assemble the model-facing list exactly as the worker does.
        tools = [
            tool
            for tool in tool_definitions(capabilities.model)
            if tool.name not in {_loop.LIVE_PERCEPTION_TOOL, _loop.PAST_PERCEPTION_TOOL}
        ]
        tools.extend(_loop.PERCEPTION_TOOL_DEFS)
    finally:
        await capabilities.close()

    model_facing = {tool.name: tool for tool in tools}
    live = model_facing["look_at_current_frame"].parameters
    past = model_facing["look_at_past_frame"].parameters
    assert set(live["properties"]) == {"question"}
    assert live["required"] == ["question"]
    assert set(past["properties"]) == {"question", "second_ago"}
    assert set(past["required"]) == {"question", "second_ago"}
    # No injected context leaks to the model.
    for schema in (live, past):
        assert "participant_id" not in schema["properties"]
        assert "reference_time_us" not in schema["properties"]
        assert "start_us" not in schema["properties"]


async def test_model_dispatch_rejects_tools_outside_the_advertised_schema() -> None:
    brain = _make_brain(_CaptureTransport())

    result = await brain._execute_tool(  # noqa: SLF001
        "query_image",
        {"image": {"uri": "file:///etc/passwd"}, "query": "Read this"},
    )

    assert result == {"error": "Unknown model tool: query_image"}


class _FakeEndpoint:
    """Hub ProcessorEndpoint double — frame callback, pixel request, status, and
    return-data send. Native vision functions acquire frames through this endpoint;
    the transport delegates return-data sends to it, so camera-control messages are
    recorded in the shared ``sent`` list."""

    def __init__(self, sent: list[DataMessage] | None = None) -> None:
        self.frame_cbs: list = []
        self.frame: FrameData | None = None
        self.frame_requests: list[FrameSignal] = []
        self.statuses: list[tuple[str, str]] = []
        self.sent: list[DataMessage] = sent if sent is not None else []

    def on_frame(self, cb) -> None:
        self.frame_cbs.append(cb)

    def on_participant(self, _cb) -> None:
        pass

    async def request_frame(self, sig: FrameSignal, timeout: float = 0.0):
        self.frame_requests.append(sig)
        return self.frame

    async def set_status(self, status: str, pid: str | None = None) -> None:
        self.statuses.append((status, pid or ""))

    async def send_return_data(self, msg: DataMessage) -> None:
        self.sent.append(msg)


class _CaptureTransportWithEndpoint(_CaptureTransport):
    """Capture transport that also exposes a fake hub endpoint so the brain
    can register its frame callback and pull pixels. The endpoint shares this
    transport's ``sent`` list so endpoint sends show up in ``transport.sent``."""

    def __init__(self) -> None:
        super().__init__()
        self.endpoint = _FakeEndpoint(sent=self.sent)


class _FakeVLM:
    """VLMService double — records image calls and returns a canned
    ChatResponse so we can assert the perception path reached the VLM."""

    def __init__(self, answer: str = "It's a red mug.") -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    async def ask_images(self, images, question, *, system_prompt: str = "",
                         **_kw) -> ChatResponse:
        self.calls.append((images, question))
        return ChatResponse(
            content=self.answer, reasoning=None, tool_calls=None,
            finish_reason="stop", raw={},
        )

    async def close(self) -> None:
        pass


def _rgb_frame(pid: str, *, w: int = 4, h: int = 4) -> tuple[FrameSignal, FrameData]:
    """A tiny solid-colour RGB24 frame + its matching signal for *pid*."""
    pts = _now_us_test()
    data = bytes([200, 30, 30]) * (w * h)  # solid red
    sig = FrameSignal(
        slot=0, seq=1, pts_us=pts, width=w, height=h,
        fmt=PixelFormat.RGB24, data_sz=len(data), participant_id=pid,
    )
    fd = FrameData(
        seq=1, pts_us=pts, width=w, height=h,
        fmt=PixelFormat.RGB24, data=data, participant_id=pid,
    )
    return sig, fd


def _now_us_test() -> int:
    import time as _t
    return _t.time_ns() // 1_000


@asynccontextmanager
async def _perception_brain(transport, vlm: _FakeVLM):
    capabilities = _tools.NativeCapabilities(
        scene_endpoint="tcp://127.0.0.1:65527",
        openxr_endpoint="tcp://127.0.0.1:65528",
        video_memory_endpoint="tcp://127.0.0.1:65529",
        frame_endpoint=transport.endpoint,
        vlm=vlm,
        text_memory_dir="/tmp/xr-render-test-memory",
        frame_max_age_s=60.0,
        frame_timeout_s=0.2,
    )
    try:
        yield _loop.SceneModelLoop(
            transport=transport,
            cfg=None,
            tools=capabilities.all,
            release_vision=capabilities.release,
            text_memory=None,
            prompt_path=_SYSTEM_PROMPT,
            model_tools=list(_loop.PERCEPTION_TOOL_DEFS),
            llm=None,
            agent_llm=None,
        )
    finally:
        await capabilities.close()


def test_perception_tool_def_in_prompt_and_classifier() -> None:
    """The perception tool is named in the system prompt, and the quick-ack
    classifier treats camera lookups as tool-settled (think=false territory):
    thinking is reserved for requests no tool pattern covers."""
    prompt = _SYSTEM_PROMPT.read_text(encoding="utf-8")
    assert "look_at_current_frame" in prompt
    ack = (_PROMPTS_DIR / "quick_ack.txt").read_text(encoding="utf-8").lower()
    assert "default is false" in ack and "camera" in ack


async def test_perception_query_reaches_vlm_frame_path() -> None:
    """A vision question routed to look_at_current_frame pulls the current
    always-on live frame and runs the VLM — returning the VLM answer to the
    loop (NOT a generic reasoning-loop fallback)."""
    transport = _CaptureTransportWithEndpoint()
    transport.set_target_participant("pid-1")
    vlm = _FakeVLM(answer="It's a red mug.")
    async with _perception_brain(transport, vlm) as brain:
        sig, fd = _rgb_frame("pid-1")
        for cb in transport.endpoint.frame_cbs:
            await cb(sig)
        transport.endpoint.frame = fd

        result = await brain._execute_tool(  # noqa: SLF001
            "look_at_current_frame",
            {"question": "What colour is this thing I'm holding?"},
            pid="pid-1",
        )

    # Reached the VLM with selected JPEG bytes + the question.
    assert len(vlm.calls) == 1
    images, question = vlm.calls[0]
    assert len(images) == 1
    assert isinstance(images[0], bytes)
    assert "colour" in question
    # The pixel request used the seeded live frame.
    assert transport.endpoint.frame_requests == [sig]
    # The VLM answer is returned to the loop, not a generic fallback.
    assert result == {"answer": "It's a red mug."}


async def test_perception_unavailable_frame_ends_turn_gracefully() -> None:
    """A failed image query ends the turn via the graceful no-frame path.

    The native ``look_at_current_frame`` raises (no frame / no VLM answer); the
    processor converts any failure into a ``_PerceptionUnavailableError`` carrying
    the short spoken message rather than feeding an error back to the model."""
    transport = _CaptureTransport()
    brain = _make_brain(transport)  # _UnusedToolbox.invoke raises on call

    with pytest.raises(_loop._PerceptionUnavailableError) as excinfo:
        await brain._look_at_current_frame("pid-1", "What is shown?")  # noqa: SLF001

    assert excinfo.value.spoken == _loop._NO_FRAME_MSG


def _stub_turn(brain, loop) -> None:
    """Stub the LLM-driven parts so _run_turn exercises only the status bracket."""
    async def _ack(_pid, _text):
        return "", False
    brain._quick_ack = _ack        # noqa: SLF001
    brain._agentic_loop = loop     # noqa: SLF001


async def test_run_turn_brackets_client_status_processing_then_idle() -> None:
    """The render turn owns the per-client UI status: 'processing' at entry and
    'idle' when it ends. (Native vision functions never emit status.)"""
    transport = _CaptureTransportWithEndpoint()
    transport.set_target_participant("pid-1")
    brain = _make_brain(transport)

    async def _loop(_text, _pid, *, ref_us, needs_thinking, thinking_ctx):
        return "All set."

    _stub_turn(brain, _loop)
    async for _ in brain._run_turn("pid-1", "add a red sphere"):  # noqa: SLF001
        pass

    assert transport.endpoint.statuses == [("processing", "pid-1"), ("idle", "pid-1")]


async def test_run_turn_status_clears_on_failure() -> None:
    """'idle' still fires when the turn fails (finally path)."""
    transport = _CaptureTransportWithEndpoint()
    transport.set_target_participant("pid-1")
    brain = _make_brain(transport)

    async def _boom(_text, _pid, *, ref_us, needs_thinking, thinking_ctx):
        raise RuntimeError("loop failed")

    _stub_turn(brain, _boom)
    async for _ in brain._run_turn("pid-1", "q"):  # noqa: SLF001
        pass

    assert transport.endpoint.statuses == [("processing", "pid-1"), ("idle", "pid-1")]


async def test_run_turn_status_clears_on_barge_in_cancellation() -> None:
    """A barge-in cancels the turn; 'idle' must still fire and the
    CancelledError must propagate."""
    transport = _CaptureTransportWithEndpoint()
    transport.set_target_participant("pid-1")
    brain = _make_brain(transport)

    async def _cancel(_text, _pid, *, ref_us, needs_thinking, thinking_ctx):
        raise asyncio.CancelledError

    _stub_turn(brain, _cancel)
    with pytest.raises(asyncio.CancelledError):
        async for _ in brain._run_turn("pid-1", "q"):  # noqa: SLF001
            pass

    assert transport.endpoint.statuses == [("processing", "pid-1"), ("idle", "pid-1")]


async def test_run_turn_status_clears_when_cancelled_during_initial_publish() -> None:
    """If a barge-in lands while the initial 'processing' publish is still in
    flight, the turn must still clear to 'idle' — the publish sits inside the
    protected region, so the finally always runs."""
    transport = _CaptureTransportWithEndpoint()
    transport.set_target_participant("pid-1")
    brain = _make_brain(transport)

    in_processing = asyncio.Event()
    statuses = transport.endpoint.statuses

    async def _blocking_status(status, pid=None):
        statuses.append((status, pid or ""))
        if status == "processing":
            in_processing.set()
            await asyncio.sleep(3600)  # hold the publish open until cancelled

    transport.endpoint.set_status = _blocking_status  # type: ignore[assignment]

    async def _loop(_text, _pid, *, ref_us, needs_thinking, thinking_ctx):
        return "unreached"

    _stub_turn(brain, _loop)

    async def _run() -> None:
        async for _ in brain._run_turn("pid-1", "q"):  # noqa: SLF001
            pass

    task = asyncio.create_task(_run())
    await in_processing.wait()  # cancellation now lands mid-'processing'-publish
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert ("idle", "pid-1") in statuses


async def test_agentic_loop_reports_agent_llm_failure() -> None:
    """A failed agent-LLM request must not fall through to a success reply."""
    transport = _CaptureTransport()
    brain = _make_brain(transport)

    async def _context(_pid: str, *, ref_us: int) -> str:
        return "scene context"

    class _FailingAgentLLM:
        async def chat(self, *_args, **_kwargs):
            raise RuntimeError("backend unavailable")

    brain._build_turn_context = _context  # noqa: SLF001
    brain._agent_llm = _FailingAgentLLM()  # noqa: SLF001

    answer = await brain._agentic_loop("add a cube", "pid-1")  # noqa: SLF001

    assert answer == "Something went wrong — please try again."


async def test_perception_no_frame_yields_graceful_message() -> None:
    """When no live camera frame can be obtained, the perception turn ends with
    a short spoken+panel message — never a hang or a silent failure.

    Driven through _agentic_loop so the full graceful path is exercised:
    look_at_current_frame → _PerceptionUnavailableError → the loop returns the
    spoken message (which _run_turn then speaks and panels)."""
    transport = _CaptureTransportWithEndpoint()
    transport.set_target_participant("pid-1")
    vlm = _FakeVLM()
    async def _fake_call_tool(_tool, _args, *, silent=False):
        return {}

    call_count = {"n": 0}

    async def _fake_chat(messages, **kwargs):
        call_count["n"] += 1
        return ChatResponse(
            content="",
            reasoning=None,
            tool_calls=[ToolCall(
                id="call_look",
                name="look_at_current_frame",
                arguments='{"question": "What colour is this?"}',
            )],
            finish_reason="tool_calls",
            raw={},
        )

    class _LLM:
        async def chat(self, messages, **kw):
            return await _fake_chat(messages, **kw)

    async with _perception_brain(transport, vlm) as brain:
        brain._call_tool = _fake_call_tool  # noqa: SLF001
        brain._agent_llm = _LLM()  # noqa: SLF001
        answer = await brain._agentic_loop(  # noqa: SLF001
            "what colour is this thing I'm holding?", "pid-1",
            ref_us=_now_us_test(), needs_thinking=True, thinking_ctx=[""],
        )

    # Graceful spoken message, not a hang or a generic "Done." fallback.
    assert answer == _loop._NO_FRAME_MSG
    # Camera is always-on streaming — no startCamera/stopCamera messages sent.
    controls = [m for m in transport.sent if m.topic == "clientControl"]
    assert not any(b'"startCamera"' in m.data for m in controls)
    # VLM was never reached — there was no frame to ask about.
    assert vlm.calls == []


def test_scene_loop_resets_only_the_target_participant_state() -> None:
    brain = _make_brain(_CaptureTransport())
    brain._history = {  # noqa: SLF001
        "alice": [("a", "one")],
        "bob": [("b", "two")],
    }
    brain._recent_moves = {  # noqa: SLF001
        "alice": [("sphere-0", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))],
        "bob": [("box-0", (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))],
    }
    brain._pre_move_positions = {  # noqa: SLF001
        "alice": {"sphere-0": (1.0, 0.0, 0.0)},
        "bob": {"box-0": (0.0, 1.0, 0.0)},
    }

    brain.reset_history("alice")

    assert "alice" not in brain._history  # noqa: SLF001
    assert "alice" not in brain._recent_moves  # noqa: SLF001
    assert "alice" not in brain._pre_move_positions  # noqa: SLF001
    assert brain._history["bob"] == [("b", "two")]  # noqa: SLF001
    assert "bob" in brain._recent_moves  # noqa: SLF001
    assert "bob" in brain._pre_move_positions  # noqa: SLF001


def test_scene_loop_bounds_participant_state_as_one_lru_unit() -> None:
    brain = _make_brain(_CaptureTransport())
    brain._participant_capacity = 2  # noqa: SLF001

    def seed(participant_id: str) -> None:
        brain._touch_participant_state(participant_id)  # noqa: SLF001
        brain._history[participant_id] = [("user", participant_id)]  # noqa: SLF001
        brain._recent_moves[participant_id] = []  # noqa: SLF001
        brain._pre_move_positions[participant_id] = {}  # noqa: SLF001

    seed("alice")
    seed("bob")
    brain._touch_participant_state("alice")  # noqa: SLF001
    seed("carol")

    assert list(brain._participant_order) == ["alice", "carol"]  # noqa: SLF001
    for state in (brain._history, brain._recent_moves,  # noqa: SLF001
                  brain._pre_move_positions):  # noqa: SLF001
        assert set(state) == {"alice", "carol"}
