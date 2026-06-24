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

import asyncio
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

from pipecat.frames.frames import EndFrame, Frame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

from xr_ai_agent import DataMessage
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


def test_worker_config_idle_timeout_disabled_by_default() -> None:
    """The shipped worker YAML ships idle_timeout_secs: 0, which the loader
    maps to None (disabled) so a quiet session is never auto-cancelled."""
    from config import load_config

    worker_yaml = (
        Path(__file__).resolve().parent.parent
        / "agent-samples" / "xr-render-demo" / "yaml" / "xr_render_demo_worker.yaml"
    )
    cfg = load_config(worker_yaml)
    assert cfg.idle_timeout_secs is None


def test_worker_config_idle_timeout_opt_in(tmp_path) -> None:
    """A positive idle_timeout_secs in the YAML is parsed to a float."""
    from config import load_config

    y = tmp_path / "w.yaml"
    y.write_text("idle_timeout_secs: 300\n")
    cfg = load_config(y)
    assert cfg.idle_timeout_secs == 300.0


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


# ── XR-launch-failure notice delivery (yield → TTS + _send → panel) ────────────
#
# When start_xr / the LOVR-spawn poll fails, RenderDemoAgent calls
# RenderSceneProcessor.enqueue_notice(pid, msg). The notice must be delivered
# with the SAME shape as a normal final answer: spoken (yielded → TextFrame at
# the TTS-facing sink) AND on the agent.response data topic (panel). The notice
# path runs no LLM/MCP, so the brain is built with None clients and the real
# prompt files — only the transport is faked to capture _send.

_PROMPTS_DIR = _WORKER_DIR / "prompts"
_SYSTEM_PROMPT = _PROMPTS_DIR / "system.txt"

_LAUNCH_FAIL_MSG = "I couldn't start the XR session — try Launch XR again."


class _CaptureSink(FrameProcessor):
    """Tail processor — collects every downstream frame it sees."""

    def __init__(self) -> None:
        super().__init__(enable_direct_mode=True)
        self.frames: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        self.frames.append(frame)
        await self.push_frame(frame, direction)


class _CaptureTransport:
    """XRMediaHubTransport double — records send_return_data and owns the
    target participant. Only the surface the notice path touches."""

    def __init__(self) -> None:
        self.target_participant = ""
        self.sent: list[DataMessage] = []

    def set_target_participant(self, pid: str) -> None:
        self.target_participant = pid

    async def send_return_data(self, msg: DataMessage) -> None:
        self.sent.append(msg)


def _make_brain(transport: _CaptureTransport):
    """Build a real RenderSceneProcessor whose LLM/MCP clients are None.

    The notice path (enqueue_notice → handle_query short-circuit →
    _emit_notice) never dereferences them. The constructor eagerly reads
    the real prompt files, so point at the bundled prompts/ directory.
    """
    return _proc.RenderSceneProcessor(
        transport   = transport,
        cfg         = None,
        render      = None,
        oxr         = None,
        vlm         = None,
        video       = None,
        vec         = None,
        prompt_path = _SYSTEM_PROMPT,
        tools       = [],
        llm         = None,
        agent_llm   = None,
    )


async def _drive_notice(brain, transport: _CaptureTransport,
                        pid: str, msg: str) -> _CaptureSink:
    """Run brain → sink in a PipelineWorker and call enqueue_notice once the
    pipeline has started, then drain with EndFrame. Returns the sink."""
    sink = _CaptureSink()
    pipeline = Pipeline([brain, sink])
    worker = PipelineWorker(
        pipeline, cancel_on_idle_timeout=False, enable_rtvi=False,
    )
    runner = WorkerRunner()
    await runner.add_workers(worker)

    async def drive() -> None:
        # Let StartFrame propagate before injecting the notice.
        await asyncio.sleep(0.05)
        await brain.enqueue_notice(pid, msg)
        await asyncio.sleep(0.15)
        await worker.queue_frame(EndFrame())

    await asyncio.gather(runner.run(), drive())
    return sink


async def test_launch_failure_notice_spoken_and_paneled() -> None:
    """enqueue_notice delivers BOTH: a yielded TextFrame (→ TTS) and an
    agent.response data message (→ panel), routed to the originating pid.

    Because agent_llm is None, if the query had wrongly fallen through to
    the agentic loop it would have returned "Done." — so seeing the exact
    notice string at the sink proves the LLM loop was skipped.
    """
    transport = _CaptureTransport()
    transport.set_target_participant("pid-1")
    brain = _make_brain(transport)

    sink = await _drive_notice(brain, transport, "pid-1", _LAUNCH_FAIL_MSG)

    # Spoken: exactly one TextFrame carrying the notice verbatim.
    texts = [f.text for f in sink.frames if isinstance(f, TextFrame)]
    assert texts == [_LAUNCH_FAIL_MSG]

    # Panel: exactly one agent.response send to the originating pid.
    assert len(transport.sent) == 1
    sent = transport.sent[0]
    assert sent.topic == "agent.response"
    assert sent.participant_id == "pid-1"
    assert sent.data.decode() == _LAUNCH_FAIL_MSG
    # No brain.close() — that closes the (None) LLM clients; the notice
    # path never opened them.


async def test_pending_notice_not_consumed_by_real_query() -> None:
    """Exact-text match: a real query that interleaves before the notice
    task runs must NOT be mistaken for the pending notice. The pending
    entry survives a non-matching handle_query; the matching one drains it.

    handle_query returns the generator without iterating, so no LLM fires —
    we only assert on _pending_notices bookkeeping here.
    """
    transport = _CaptureTransport()
    transport.set_target_participant("pid-1")
    brain = _make_brain(transport)

    # Register a pending notice the way enqueue_notice does, without spawning.
    brain._pending_notices.setdefault("pid-1", []).append(_LAUNCH_FAIL_MSG)

    # A real, different query for the same pid must not consume the notice.
    await brain.handle_query("pid-1", "move the cube left", False)
    assert brain._pending_notices.get("pid-1") == [_LAUNCH_FAIL_MSG]

    # The matching text drains it.
    await brain.handle_query("pid-1", _LAUNCH_FAIL_MSG, False)
    assert "pid-1" not in brain._pending_notices


async def test_quick_ack_spoken_on_non_thinking_turn() -> None:
    """ACK-SPEAK POLICY: the quick-ack is yielded (→ TTS) on EVERY turn,
    including a non-thinking one, so a tool-using turn is never silent until
    the final reply. Pre-change the ack was spoken only when needs_thinking.

    _quick_ack and _agentic_loop are stubbed so no LLM/MCP client is touched.
    """
    transport = _CaptureTransport()
    transport.set_target_participant("pid-1")
    brain = _make_brain(transport)

    async def _fake_quick_ack(_text):
        return ("On it.", False)  # ack present, needs_thinking = False

    async def _fake_loop(*_a, **_k):
        return "All set."

    brain._quick_ack = _fake_quick_ack      # noqa: SLF001
    brain._agentic_loop = _fake_loop        # noqa: SLF001

    gen = await brain.handle_query("pid-1", "place a cube", False)
    spoken = [s async for s in gen]

    # Ack is spoken first (so the turn isn't silent), then the final reply.
    assert spoken and spoken[0] == "On it."
    assert "All set." in spoken
    # Ack is also mirrored to the panel on agent.progress.
    progress = [m for m in transport.sent if m.topic == "agent.progress"]
    assert any(m.data.decode() == "On it." for m in progress)


# ── live-frame perception routing (look_at_current_frame) ──────────────────────
#
# A real-world visual question — "what colour is this thing I'm holding?" — must
# reach the LIVE-FRAME VLM path, not stall in the reasoning loop. Before the fix
# the render-demo worker had no frame tracking and never turned the camera on, so
# a perception query looped on an unanswerable tool (get_frame_from_time, which
# isn't even registered when recording is disabled) and hung. These tests stub
# the VLM client + the hub frame path and assert the routing mechanically.

from xr_ai_agent import FrameData, FrameSignal, PixelFormat  # noqa: E402
from xr_ai_models import ChatResponse, ToolCall  # noqa: E402

import processors as _proc  # noqa: E402


class _FakeEndpoint:
    """Hub ProcessorEndpoint double — frame callback, pixel request, status, and
    return-data send. VisionModule now talks to the endpoint directly (the real
    transport.send_return_data is a pure delegate to endpoint.send_return_data),
    so camera-control messages are recorded here into the shared ``sent`` list."""

    def __init__(self, sent: list[DataMessage] | None = None) -> None:
        self.frame_cbs: list = []
        self.frame: FrameData | None = None
        self.frame_requests: list[FrameSignal] = []
        self.statuses: list[tuple[str, str]] = []
        self.sent: list[DataMessage] = sent if sent is not None else []

    def on_frame(self, cb) -> None:
        self.frame_cbs.append(cb)

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
    """VLMService double — records the ask_image call and returns a canned
    ChatResponse so we can assert the perception path reached the VLM."""

    def __init__(self, answer: str = "It's a red mug.") -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    async def ask_image(self, image, question, *, system_prompt: str = "",
                        **_kw) -> ChatResponse:
        self.calls.append((image, question))
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


def _make_perception_brain(transport, vlm: _FakeVLM):
    return _proc.RenderSceneProcessor(
        transport   = transport,
        cfg         = None,
        render      = None,
        oxr         = None,
        vlm         = None,
        video       = None,
        vec         = None,
        prompt_path = _SYSTEM_PROMPT,
        tools       = [_proc._PERCEPTION_TOOL_DEF],
        llm         = None,
        agent_llm   = None,
        vlm_service = vlm,
        frame_max_age_s     = 60.0,   # generous so the seeded frame stays fresh
        camera_on_timeout_s = 0.2,    # short — the no-frame test must not hang
        camera_grace_s      = 0.05,
    )


def test_perception_tool_def_in_prompt_and_classifier() -> None:
    """The new perception tool is named in the system prompt, and the
    quick-ack classifier still flags real-world visual lookups as think=true
    (so they enter the reasoning loop where the tool lives)."""
    prompt = _SYSTEM_PROMPT.read_text(encoding="utf-8")
    assert "look_at_current_frame" in prompt
    # The classifier prompt routes real-world camera queries to think=true.
    ack = (_PROMPTS_DIR / "quick_ack.txt").read_text(encoding="utf-8").lower()
    assert "real world" in ack and "camera" in ack


async def test_perception_query_reaches_vlm_frame_path() -> None:
    """A vision question routed to look_at_current_frame turns the camera on,
    pulls the live frame, and runs the VLM — returning the VLM answer to the
    loop (NOT a generic reasoning-loop fallback)."""
    transport = _CaptureTransportWithEndpoint()
    transport.set_target_participant("pid-1")
    vlm = _FakeVLM(answer="It's a red mug.")
    brain = _make_perception_brain(transport, vlm)

    # Seed a fresh live frame for the participant (as if the hub delivered one).
    # The VisionModule owns the frame cache now, so deliver via the registered
    # frame callback rather than poking brain internals.
    sig, fd = _rgb_frame("pid-1")
    for cb in transport.endpoint.frame_cbs:
        await cb(sig)
    transport.endpoint.frame = fd

    result = await brain._execute_tool(  # noqa: SLF001
        "look_at_current_frame",
        {"question": "What colour is this thing I'm holding?"},
        pid="pid-1",
    )

    # Reached the VLM with the encoded frame + the question.
    assert len(vlm.calls) == 1
    image, question = vlm.calls[0]
    assert image.startswith("data:image/jpeg;base64,")
    assert "colour" in question
    # The pixel request used the seeded live frame.
    assert transport.endpoint.frame_requests == [sig]
    # The VLM answer is returned to the loop, not a generic fallback.
    assert result == {"answer": "It's a red mug."}


async def test_perception_no_frame_yields_graceful_message() -> None:
    """When no live camera frame can be obtained, the perception turn ends with
    a short spoken+panel message — never a hang or a silent failure.

    Driven through _agentic_loop so the full graceful path is exercised:
    look_at_current_frame → _PerceptionUnavailableError → the loop returns the
    spoken message (which _run_turn then speaks and panels)."""
    transport = _CaptureTransportWithEndpoint()
    transport.set_target_participant("pid-1")
    vlm = _FakeVLM()
    brain = _make_perception_brain(transport, vlm)

    # No frame seeded → _wait_for_camera_frame times out (camera_on_timeout=0.2s).
    # Stub the MCP prefetch (None clients) and script the agent LLM to emit a
    # single look_at_current_frame tool call.
    async def _fake_call_mcp(_client, tool, _args, *, silent=False):
        return {}  # empty scene / pose

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

    brain._call_mcp = _fake_call_mcp        # noqa: SLF001
    class _LLM:
        async def chat(self, messages, **kw):
            return await _fake_chat(messages, **kw)
    brain._agent_llm = _LLM()               # noqa: SLF001

    answer = await brain._agentic_loop(     # noqa: SLF001
        "what colour is this thing I'm holding?", "pid-1",
        ref_us=_now_us_test(), needs_thinking=True, thinking_ctx=[""],
    )

    # Graceful spoken message, not a hang or a generic "Done." fallback.
    assert answer == _proc._NO_FRAME_MSG
    # The camera WAS turned on (startCamera sent) before giving up.
    controls = [m for m in transport.sent if m.topic == "clientControl"]
    assert any(b'"startCamera"' in m.data for m in controls)
    # VLM was never reached — there was no frame to ask about.
    assert vlm.calls == []
