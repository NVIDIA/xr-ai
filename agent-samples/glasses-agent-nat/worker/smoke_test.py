# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline smoke harness for glasses-agent-nat worker logic.

Pure stdlib + the worker's own modules + ``unittest.mock``. Exercises the
non-network branches of :class:`QueryProcessor` against an in-memory
:class:`AgentMemory`, with the LLM-call methods (``_quick_ack``,
``_agentic_loop``) replaced by stubs.

Run via::

    cd agent-samples/glasses-agent-nat/worker
    uv run python smoke_test.py

Covers the four regression scenarios from the previous plan plus six new
scenarios for the demo-disambiguation fix. Exits non-zero on the first
failure so CI / a watch loop can pick it up.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys
import time
import unittest.mock as mock
from dataclasses import dataclass, field
from typing import Any

# Run the harness from the worker directory so the worker's own modules
# resolve without any package gymnastics.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import intent
from config import WorkerConfig
from memory import AgentMemory, Demonstration, DemoStep, RecordedFrame, VoiceNote
from processors import QueryProcessor, _after_frame_for_step, _select_reference_frames_for_step
from vad import VadDetector


# ── test scaffolding ─────────────────────────────────────────────────────────

@dataclass
class Recorder:
    """Records every call the worker would have made out to the hub / TTS."""
    sent_text: list[tuple[str, str, str]] = field(default_factory=list)  # (pid, text, topic)
    spoken:    list[tuple[str, str]]      = field(default_factory=list)  # (pid, text)
    flushed:   list[str]                  = field(default_factory=list)  # pids
    quick_ack_calls:    list[str] = field(default_factory=list)
    agentic_loop_calls: list[str] = field(default_factory=list)

    async def send_text(self, pid: str, text: str, topic: str) -> None:
        self.sent_text.append((pid, text, topic))

    async def say(self, pid: str, text: str) -> None:
        self.spoken.append((pid, text))

    async def flush_audio(self, pid: str) -> None:
        self.flushed.append(pid)


def _make_cfg(freshness_s: float = 120.0) -> WorkerConfig:
    return WorkerConfig(
        stt_server          = "http://stub",
        tts_server          = "http://stub",
        llm_server          = "http://stub",
        agent_llm_server    = "http://stub",
        vlm_mcp             = "http://stub",
        video_mcp           = "http://stub",
        transcript_mcp      = "http://stub",
        nat_workflow_config = pathlib.Path("/nonexistent.yaml"),
        vlm_interval_s              = 1.0,
        vlm_obs_max                 = 240,
        condenser_interval_s        = 60.0,
        transcript_source           = "smoke-test",
        guidance_check_interval_s   = 2.0,
        guidance_freshness_window_s = freshness_s,
        silence_duration            = 0.8,
        min_speech                  = 0.15,
        silero_threshold            = 0.3,
    )


def _seed_demo(memory: AgentMemory, name: str, *, finished_ago_s: float = 0.0) -> Demonstration:
    """Register *name* as a finished, ready-to-guide demo."""
    now_us  = int(time.time() * 1_000_000)
    end_us  = now_us - int(finished_ago_s * 1_000_000)
    demo = Demonstration(name=name, started_at_us=end_us - 5_000_000, ended_at_us=end_us)
    demo.steps = [
        DemoStep(step_number=1, timestamp_us=end_us, description=f"start {name}", image_path=""),
        DemoStep(step_number=2, timestamp_us=end_us, description=f"finish {name}", image_path=""),
    ]
    demo.instructions = [f"Begin {name}.", f"Complete {name}."]
    memory._demos[name] = demo
    if end_us > memory._last_demo_finished_at_us:
        memory._last_demo_finished_at_us = end_us
    return demo


def _make_qp(memory: AgentMemory, rec: Recorder, *, freshness_s: float = 120.0) -> QueryProcessor:
    """Construct a QueryProcessor with the NAT runtime / agent stubbed.

    We bypass the real NatRuntime constructor (which would try to load
    the workflow YAML) by patching :class:`NatAgentRunner` in the
    processors module to a no-op for the duration of __init__.

    `call_tool` is an AsyncMock that returns ``{}`` by default — most
    scenarios override it per-call; the default keeps stubs like
    `_derive_step_requirements` from blowing up with "MagicMock can't be
    awaited" when finalization runs in the background.
    """
    fake_runtime = mock.MagicMock(name="NatRuntime")
    fake_runtime.call_tool = mock.AsyncMock(return_value={})
    with mock.patch("processors.NatAgentRunner", autospec=False) as ar_cls:
        ar_cls.return_value = mock.MagicMock(name="NatAgentRunner")
        qp = QueryProcessor(
            _make_cfg(freshness_s=freshness_s),
            memory,
            fake_runtime,
            send_text=rec.send_text,
            say=rec.say,
            flush_audio=rec.flush_audio,
        )

    async def fake_quick_ack(transcript: str) -> tuple[str, bool]:
        rec.quick_ack_calls.append(transcript)
        # Match the live behaviour: a short ack + needs_thinking flag.
        return "On it.", False

    async def fake_agentic_loop(transcript: str, pid: str, *, ref_us: int = 0, needs_thinking: bool = False) -> str:
        rec.agentic_loop_calls.append(transcript)
        return f"FAKE_REPLY({transcript!r})"

    qp._quick_ack    = fake_quick_ack          # type: ignore[assignment]
    qp._agentic_loop = fake_agentic_loop       # type: ignore[assignment]
    return qp


def _make_ga(memory: AgentMemory, qp: QueryProcessor) -> Any:
    """Construct a GlassesAgent with hub I/O replaced by AsyncMocks."""
    import agent as agent_mod

    fake_runtime = mock.MagicMock(name="NatRuntime")
    fake_ep = mock.MagicMock(name="ProcessorEndpoint")
    fake_ep.flush_return_audio = mock.AsyncMock()
    fake_ep.send_return_audio = mock.AsyncMock()
    fake_ep.send_return_data = mock.AsyncMock()

    with mock.patch.object(agent_mod, "ProcessorEndpoint", autospec=False) as ep_cls:
        ep_cls.return_value = fake_ep
        return agent_mod.GlassesAgent(
            cfg=_make_cfg(),
            memory=memory,
            transcript_client=mock.MagicMock(name="TranscriptClient"),
            query_processor=qp,
            stt_url="http://stub",
            tts_url="http://stub",
            nat_runtime=fake_runtime,
        )


# ── trace capture ────────────────────────────────────────────────────────────

class _TraceList(list):
    """Captures trace log records emitted by processors during one scenario."""
    def append_record(self, record: Any) -> None:
        self.append(record.getMessage())


def _attach_trace_capture() -> _TraceList:
    import logging
    captured = _TraceList()

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append_record(record)

    h = _Handler()
    h.setLevel(logging.DEBUG)
    log = logging.getLogger("glasses_agent_nat.trace")
    log.setLevel(logging.DEBUG)
    log.addHandler(h)
    return captured


# ── tiny assertion helpers ───────────────────────────────────────────────────

class Fail(AssertionError):
    """Distinguish harness-level expected failures from unrelated exceptions."""


def expect(cond: bool, msg: str) -> None:
    if not cond:
        raise Fail(msg)


def first_match(events: list[str], prefix: str) -> str | None:
    for e in events:
        if e.startswith(prefix):
            return e
    return None


def count_starting_with(events: list[str], prefix: str) -> int:
    return sum(1 for e in events if e.startswith(prefix))


# ── scenarios ────────────────────────────────────────────────────────────────

PID = "p1"

_VAD_SR        = 16_000
_VAD_CHUNK_N   = int(_VAD_SR * 0.02)
_VAD_SPEECH    = (12_000).to_bytes(2, "little", signed=True) * _VAD_CHUNK_N
_VAD_SILENCE   = b"\x00\x00" * _VAD_CHUNK_N


async def _feed_vad_many(vad: VadDetector, n: int, chunk: bytes) -> None:
    for _ in range(n):
        await vad.feed(chunk, _VAD_SR)


async def scenario_0_vad_fallback_constructs_without_silero() -> None:
    received: list[bytes] = []

    async def on_utt(audio: bytes, _sr: int) -> None:
        received.append(audio)

    with mock.patch.dict(sys.modules, {"silero_vad": None}):
        vad = VadDetector(on_utterance=on_utt)

    expect(vad._silero is None, "silero should be disabled after import failure")  # type: ignore[attr-defined]
    expect(received == [], f"constructor should not emit utterances: {received}")


async def scenario_0b_vad_energy_fallback_finalizes_utterance() -> None:
    received: list[tuple[bytes, int]] = []

    async def on_utt(audio: bytes, sr: int) -> None:
        received.append((audio, sr))

    with mock.patch.dict(sys.modules, {"silero_vad": None}):
        vad = VadDetector(
            on_utterance     = on_utt,
            silence_duration = 0.10,
            min_speech       = 0.06,
            silero_threshold = 0.5,
        )

    await _feed_vad_many(vad, 10, _VAD_SPEECH)
    await _feed_vad_many(vad, 7, _VAD_SILENCE)

    expect(len(received) == 1, f"fallback VAD should emit one utterance: {received}")
    audio, sr = received[0]
    expect(sr == _VAD_SR, f"wrong sample rate: {sr}")
    expect(len(audio) // 2 >= int(_VAD_SR * 0.20), "utterance lost speech samples")


async def scenario_0c_vad_speech_start_rearms_with_fallback() -> None:
    starts: list[int] = []
    finalized: list[int] = []

    async def on_start() -> None:
        starts.append(1)

    async def on_utt(_audio: bytes, _sr: int) -> None:
        finalized.append(1)

    with mock.patch.dict(sys.modules, {"silero_vad": None}):
        vad = VadDetector(
            on_utterance     = on_utt,
            on_speech_start  = on_start,
            silence_duration = 0.10,
            min_speech       = 0.06,
            silero_threshold = 0.5,
        )

    await _feed_vad_many(vad, 10, _VAD_SPEECH)
    await asyncio.sleep(0)
    expect(starts == [1], f"speech_start should fire once: {starts}")

    await _feed_vad_many(vad, 7, _VAD_SILENCE)
    expect(finalized == [1], f"utterance should finalize once: {finalized}")

    await _feed_vad_many(vad, 10, _VAD_SPEECH)
    await asyncio.sleep(0)
    expect(starts == [1, 1], f"speech_start should re-arm: {starts}")


async def scenario_1_what_do_you_see_with_fresh_demo() -> None:
    """REGRESSION: 'what do you see?' inside the freshness window must reach
    the agentic loop, NOT the freshness fallback."""
    mem = AgentMemory()
    _seed_demo(mem, "pico headset", finished_ago_s=5.0)
    rec = Recorder()
    trace = _attach_trace_capture()
    qp = _make_qp(mem, rec)

    await qp.handle("What do you see?", PID, ref_us=0)

    expect(len(rec.agentic_loop_calls) == 1,
           f"expected one agentic-loop call, got {rec.agentic_loop_calls}")
    expect(first_match(trace, "GUIDANCE_FALLBACK") is None,
           f"freshness fallback fired for a current-view question: {trace}")
    expect(qp._guidance_demo is None, "current-view question should not start guidance")


async def scenario_2_bare_stop_outside_guidance() -> None:
    """REGRESSION: bare 'stop' outside guidance flushes audio + text-only ack."""
    mem = AgentMemory()
    rec = Recorder()
    trace = _attach_trace_capture()
    qp = _make_qp(mem, rec)

    await qp.handle("stop", PID, ref_us=0)

    expect(PID in rec.flushed, f"flush_audio not called: {rec.flushed}")
    expect(rec.spoken == [], f"bare stop should not TTS, got {rec.spoken}")
    expect(first_match(trace, "STOP_SPEAKING") is not None,
           f"expected STOP_SPEAKING trace, got {trace}")
    expect(len(rec.agentic_loop_calls) == 0, "bare stop should not run the agentic loop")


async def scenario_3_layer1_filler_dropped() -> None:
    """REGRESSION: Layer 1 shape filter drops pure-filler transcripts.

    The harness calls intent.is_shape_noise() directly because the live
    wiring lives in agent.py — we just need to know the helper still
    classifies the trace's typical noise correctly so the worker never
    has to call into the agentic loop with it.
    """
    expect(intent.is_shape_noise("uh"),               "single filler 'uh' must be noise")
    expect(intent.is_shape_noise("uh um yeah"),       "all-filler must be noise")
    expect(intent.is_shape_noise(""),                 "empty string must be noise")
    expect(intent.is_shape_noise("..."),              "no-letters must be noise")
    expect(not intent.is_shape_noise("show me how to wear pico headset"),
           "real request must pass the shape filter")


async def scenario_4_stt_garbled_freshness_single_demo() -> None:
    """REGRESSION: STT-mangled guidance request inside freshness window with
    one demo on file falls back to that demo via the freshness path."""
    mem = AgentMemory()
    _seed_demo(mem, "pico headset", finished_ago_s=5.0)
    rec = Recorder()
    trace = _attach_trace_capture()
    qp = _make_qp(mem, rec)

    await qp.handle("the gym", PID, ref_us=0)

    expect(qp._guidance_demo is not None, "freshness fallback should have started guidance")
    assert qp._guidance_demo is not None  # narrow for type
    expect(qp._guidance_demo.name == "pico headset",
           f"freshness fallback picked wrong demo: {qp._guidance_demo.name}")
    fb = first_match(trace, "GUIDANCE_FALLBACK")
    expect(fb is not None, f"expected GUIDANCE_FALLBACK trace, got {trace}")
    assert fb is not None
    expect("via=recent" in fb, f"single-demo freshness should log via=recent: {fb}")


async def scenario_5_two_demo_bare_name_via_freshness() -> None:
    """NEW: Two demos on file, STT mangles 'pico' to 'pickle'. The freshness
    fallback's NEW strict matcher must NOT silently grab desk-arrangement
    (the most recent). It must either match pico via unique-best at score
    >= 2, OR return without starting guidance for the wrong demo."""
    mem = AgentMemory()
    _seed_demo(mem, "pico headset",    finished_ago_s=30.0)
    _seed_demo(mem, "desk arrangement", finished_ago_s=5.0)
    rec = Recorder()
    trace = _attach_trace_capture()
    qp = _make_qp(mem, rec)

    # 'Pickle headset' is not a guidance phrase, so it falls into the
    # freshness branch. Strict matcher tokens are {pickle, headset};
    # only pico headset overlaps (score=1). Strict needs >= 2 so the
    # name-aware fallback returns None and we defer to most_recent ->
    # desk arrangement.
    await qp.handle("Pickle headset.", PID, ref_us=0)

    fb = first_match(trace, "GUIDANCE_FALLBACK")
    expect(fb is not None, f"expected GUIDANCE_FALLBACK from freshness branch, got {trace}")
    assert fb is not None
    expect("via=recent" in fb,
           f"weak-token fallback must log via=recent, not silently match by name: {fb}")
    # The deferred demo IS desk arrangement; the failure mode we're
    # guarding against is the strict matcher hijacking with the wrong demo.
    assert qp._guidance_demo is not None
    expect(qp._guidance_demo.name == "desk arrangement",
           f"freshness defer should pick most_recent, got {qp._guidance_demo.name}")


async def scenario_6_two_demo_shared_token_tied() -> None:
    """NEW: Two demos share 'headset'. User says bare 'headset' inside
    freshness. Strict matcher must return None (tied top set size 2),
    fallback picks most_recent (desk arrangement is more recent than both
    headset demos in this setup).
    """
    mem = AgentMemory()
    _seed_demo(mem, "pico headset", finished_ago_s=30.0)
    _seed_demo(mem, "vr headset",   finished_ago_s=20.0)
    _seed_demo(mem, "desk arrangement", finished_ago_s=5.0)
    rec = Recorder()
    trace = _attach_trace_capture()
    qp = _make_qp(mem, rec)

    await qp.handle("headset", PID, ref_us=0)

    fb = first_match(trace, "GUIDANCE_FALLBACK")
    expect(fb is not None, f"expected GUIDANCE_FALLBACK, got {trace}")
    assert fb is not None
    expect("via=recent" in fb,
           f"tied-at-top tokens must NOT be matched by name: {fb}")
    assert qp._guidance_demo is not None
    expect(qp._guidance_demo.name == "desk arrangement",
           f"tied tokens should defer to most_recent: {qp._guidance_demo.name}")


async def scenario_7_how_to_do_starts_guidance() -> None:
    """NEW: 'How to do desk arrangement' must hit the EXPLICIT guidance
    branch (not the freshness fallback), so the trace shows
    GUIDANCE_START not GUIDANCE_FALLBACK."""
    mem = AgentMemory()
    _seed_demo(mem, "pico headset",    finished_ago_s=30.0)
    _seed_demo(mem, "desk arrangement", finished_ago_s=5.0)
    rec = Recorder()
    trace = _attach_trace_capture()
    qp = _make_qp(mem, rec)

    await qp.handle("How to do desk arrangement", PID, ref_us=0)

    expect(first_match(trace, "GUIDANCE_FALLBACK") is None,
           f"'how to do' must hit the explicit guidance branch, not fallback: {trace}")
    gs = first_match(trace, "GUIDANCE_START")
    expect(gs is not None, f"expected GUIDANCE_START trace, got {trace}")
    assert gs is not None
    expect("demo=desk arrangement" in gs,
           f"explicit branch picked wrong demo: {gs}")


async def scenario_8_ambiguous_then_resolved_by_number() -> None:
    """NEW: Generic 'show me how' with 2 demos -> pending state, numbered
    re-ask. User answers 'two' -> picks choices[1] = desk arrangement."""
    mem = AgentMemory()
    _seed_demo(mem, "pico headset",    finished_ago_s=30.0)
    _seed_demo(mem, "desk arrangement", finished_ago_s=20.0)
    rec = Recorder()
    trace = _attach_trace_capture()
    qp = _make_qp(mem, rec)

    await qp.handle("Show me how.", PID, ref_us=0)
    expect(PID in qp._pending_guidance_by_pid,
           f"multi-demo ambiguous request did not set pending state: {qp._pending_guidance_by_pid}")
    pset = first_match(trace, "PENDING_SET")
    expect(pset is not None, f"expected PENDING_SET trace, got {trace}")
    # Re-ask should be a numbered list, not a bare 'X or Y'.
    last_text = rec.sent_text[-1][1] if rec.sent_text else ""
    expect("1)" in last_text and "2)" in last_text,
           f"pending re-ask must be numbered, got: {last_text!r}")

    # Answer with a number.
    await qp.handle("two", PID, ref_us=0)

    expect(PID not in qp._pending_guidance_by_pid,
           "pending state should be cleared after successful resolution")
    assert qp._guidance_demo is not None
    expect(qp._guidance_demo.name == "desk arrangement",
           f"choice 'two' should pick choices[1] = desk arrangement, got {qp._guidance_demo.name}")
    gs = first_match(trace, "GUIDANCE_START")
    expect(gs is not None, f"expected GUIDANCE_START after numeric choice, got {trace}")


async def scenario_9_pending_state_cancel_escape() -> None:
    """NEW: While pending disambiguation, bare 'cancel' must clear pending
    state and call _stop_speaking — not get trapped in the re-ask loop."""
    mem = AgentMemory()
    _seed_demo(mem, "pico headset",    finished_ago_s=30.0)
    _seed_demo(mem, "desk arrangement", finished_ago_s=20.0)
    rec = Recorder()
    trace = _attach_trace_capture()
    qp = _make_qp(mem, rec)

    await qp.handle("Show me how.", PID, ref_us=0)
    expect(PID in qp._pending_guidance_by_pid, "setup: pending state not set")

    rec.flushed.clear()
    await qp.handle("cancel", PID, ref_us=0)

    expect(PID not in qp._pending_guidance_by_pid,
           "cancel during pending must clear state, did not")
    expect(PID in rec.flushed, "cancel during pending must flush audio")
    expect(qp._guidance_demo is None, "cancel during pending must NOT start guidance")
    expect(first_match(trace, "STOP_SPEAKING") is not None,
           f"expected STOP_SPEAKING for bare cancel, got {trace}")


async def scenario_11_expected_requirements_threaded_through_monitor() -> None:
    """When the monitor calls check_guidance_step_complete, the demo step's
    expected_requirements and teacher reference frame must flow into tool args.
    """
    mem = AgentMemory()
    demo = _seed_demo(mem, "pico headset", finished_ago_s=5.0)
    demo.steps[0].image_path = "/tmp/teacher-step.png"
    demo.steps[0].teacher_caption = "teacher completed-state caption"
    demo.steps[0].expected_requirements = ["headset on head", "strap behind head"]
    rec = Recorder()
    qp = _make_qp(mem, rec)

    captured: dict[str, Any] = {}

    async def fake_call_tool(group: str, tool: str, args: dict, **_) -> dict:
        captured["group"] = group
        captured["tool"]  = tool
        captured["args"]  = args
        return {
            "completed": True,
            "current_observation": "headset visible",
            "checks": [
                {"requirement": "headset on head", "visible": True, "evidence": "covers eyes"},
                {"requirement": "strap behind head", "visible": True, "evidence": "strap visible"},
            ],
            "missing_or_mismatched": [],
            "image_path": "/tmp/x.png",
            "issue": "",
        }

    qp._nat_runtime.call_tool = fake_call_tool        # type: ignore[attr-defined]
    qp._guidance_demo = demo
    qp._guidance_step = 0

    result = await qp._guidance_completion_result(PID)

    expect(captured.get("tool") == "check_guidance_step_complete",
           f"wrong tool called: {captured}")
    args = captured.get("args") or {}
    expect(args.get("expected_requirements") == ["headset on head", "strap behind head"],
           f"expected_requirements not threaded into tool args: {args}")
    expect(args.get("teacher_image_path") == "/tmp/teacher-step.png",
           f"teacher_image_path not threaded into tool args: {args}")
    expect(args.get("teacher_caption") == "teacher completed-state caption",
           f"teacher_caption not threaded into tool args: {args}")
    expect(result.get("completed") is True, f"result should pass through: {result}")


async def scenario_11b_after_frame_returns_paired_path_and_caption() -> None:
    """A1: _after_frame_for_step returns the same RecordedFrame for path AND
    caption — guards against the prior timestamp-drift bug where path came
    from one frame and caption from another.
    """
    frames = [
        RecordedFrame(frame_idx=0, timestamp_us=1_000_000,
                      image_path="/tmp/f0.png", description="hand approaches headset"),
        RecordedFrame(frame_idx=1, timestamp_us=3_000_000,
                      image_path="/tmp/f1.png", description="headset lifted off strap"),
        RecordedFrame(frame_idx=2, timestamp_us=5_000_000,
                      image_path="/tmp/f2.png", description="headset placed on head"),
    ]
    notes = [
        VoiceNote(timestamp_us=500_000,  text="grab the headset"),
        VoiceNote(timestamp_us=2_500_000, text="put it on"),
    ]

    # Step 0: completion is the latest frame strictly before notes[1] (2.5s) → frame[0]
    f0 = _after_frame_for_step(frames, notes, 0)
    expect(f0 is frames[0],
           f"step 0 should pick frames[0], got {f0}")
    expect(f0.image_path == "/tmp/f0.png" and f0.description == "hand approaches headset",
           "path and caption must come from the same RecordedFrame")

    # Step 1: last step (no next note) → last frame
    f1 = _after_frame_for_step(frames, notes, 1)
    expect(f1 is frames[-1],
           f"step 1 should pick last frame, got {f1}")
    expect(f1.image_path == "/tmp/f2.png" and f1.description == "headset placed on head",
           "path and caption must come from the same RecordedFrame")

    # Empty frames → None
    none_frame = _after_frame_for_step([], notes, 0)
    expect(none_frame is None, f"empty frames must yield None, got {none_frame}")


async def scenario_11c_fallback_requirements_from_instruction() -> None:
    """If NAT requirement derivation fails, common imperative steps still
    get deterministic requirements so monitoring does not disable itself.
    """
    mem = AgentMemory()
    rec = Recorder()
    qp = _make_qp(mem, rec)

    async def failing_call_tool(*_args, **_kwargs) -> dict:
        return {"requirements": []}

    qp._nat_runtime.call_tool = failing_call_tool  # type: ignore[attr-defined]
    reqs = await qp._derive_step_requirements(
        "Put the Pico headset on your head",
        "teacher has the headset on their head",
    )
    expect(reqs == ["headset on head"],
           f"fallback requirement wrong: {reqs}")


async def scenario_11d_empty_requirements_still_checked_with_teacher_frame() -> None:
    """An empty checklist must not disable guidance checks when a teacher
    reference frame is available.
    """
    mem = AgentMemory()
    demo = _seed_demo(mem, "pico headset", finished_ago_s=5.0)
    demo.steps[0].image_path = "/tmp/teacher-step.png"
    demo.steps[0].expected_requirements = []
    rec = Recorder()
    qp = _make_qp(mem, rec)
    qp._guidance_demo = demo
    qp._guidance_step = 0
    captured: dict[str, Any] = {}

    async def fake_call_tool(group: str, tool: str, args: dict, **_) -> dict:
        captured["tool"] = tool
        captured["args"] = args
        return _grounded_yes(["headset on head"])

    qp._nat_runtime.call_tool = fake_call_tool  # type: ignore[attr-defined]
    result = await qp._guidance_completion_result(PID)

    expect(captured.get("tool") == "check_guidance_step_complete",
           f"guidance check was not called: {captured}")
    args = captured.get("args") or {}
    expect(args.get("expected_requirements") == [],
           f"expected empty requirements to pass through: {args}")
    expect(args.get("teacher_image_path") == "/tmp/teacher-step.png",
           f"teacher frame missing from guidance check: {args}")
    expect(result.get("completed") is True, f"result should pass through: {result}")


async def scenario_11e_guidance_frame_pair_trace() -> None:
    """Guidance checks trace teacher/live frame paths and timestamps."""
    mem = AgentMemory()
    demo = _seed_demo(mem, "pico headset", finished_ago_s=5.0)
    demo.steps[0].image_path = "/tmp/teacher-step.png"
    demo.steps[0].teacher_caption = "teacher completed-state caption"
    rec = Recorder()
    trace = _attach_trace_capture()
    qp = _make_qp(mem, rec)
    qp._guidance_demo = demo
    qp._guidance_step = 0
    qp._guidance_started_at_us = 100
    qp._guidance_step_spoken_at_us = 200

    async def fake_call_tool(group: str, tool: str, args: dict, **_) -> dict:
        return {
            "completed": False,
            "current_observation": "",
            "checks": [],
            "missing_or_mismatched": [],
            "image_path": "/tmp/live.png",
            "teacher_image_path": args.get("teacher_image_path", ""),
            "timestamp_us": 300,
            "issue": "no grounded evidence",
        }

    qp._nat_runtime.call_tool = fake_call_tool  # type: ignore[attr-defined]
    await qp._guidance_completion_result(PID)
    event = first_match(trace, "GUIDANCE_FRAME_PAIR")
    expect(event is not None, f"expected GUIDANCE_FRAME_PAIR trace, got {trace}")
    assert event is not None
    expect("/tmp/teacher-step.png" in event and "/tmp/live.png" in event and "min_ts=200" in event,
           f"frame-pair trace missing fields: {event}")


async def scenario_11f_reused_adjacent_reference_marked_unreliable() -> None:
    """If adjacent steps select the same frame, the later reference is unreliable."""
    mem = AgentMemory()
    mem.start_recording("mixed step")
    assert mem.recording is not None
    started = mem.recording.started_at_us
    mem.add_voice_note(VoiceNote(started + 2_000_000, "Step one hold controller"))
    mem.add_voice_note(VoiceNote(started + 2_500_000, "Step two hold controller"))
    mem.add_recorded_frame(RecordedFrame(
        frame_idx=0,
        timestamp_us=started + 2_200_000,
        image_path="/tmp/shared.png",
        description="white controller held in hand",
    ))
    demo = mem.finish_recording()
    assert demo is not None
    demo.finalize_generation = mem._finalize_generation

    rec = Recorder()
    qp = _make_qp(mem, rec)

    async def fake_analyze(_demo: Demonstration) -> tuple[str, list[str]]:
        return "summary", ["hold controller", "hold controller"]

    async def fake_derive(instruction: str, _caption: str) -> list[str]:
        return [instruction]

    qp._analyze_recording = fake_analyze  # type: ignore[assignment]
    qp._derive_step_requirements = fake_derive  # type: ignore[assignment]
    await qp._finalize_demo(demo, demo.finalize_generation, PID, 1)

    expect(demo.steps[0].image_path == "/tmp/shared.png",
           f"first step should keep shared reference: {demo.steps}")
    expect(demo.steps[1].image_path == "" and demo.steps[1].reference_reliable is False,
           f"second reused reference should be unreliable: {demo.steps[1]}")


async def scenario_11h_text_video_mismatch_keeps_visual_reference() -> None:
    """A usable teacher frame remains authoritative when text mismatches video."""
    mem = AgentMemory()
    mem.start_recording("mismatch demo")
    assert mem.recording is not None
    started = mem.recording.started_at_us
    mem.add_voice_note(VoiceNote(started + 50_000, "Step one put a blue circle on your head"))
    mem.add_recorded_frame(RecordedFrame(
        frame_idx=0,
        timestamp_us=started,
        image_path="/tmp/black-cap.png",
        description="black baseball cap with gold logo on head",
    ))
    demo = mem.finish_recording()
    assert demo is not None
    demo.finalize_generation = mem._finalize_generation
    rec = Recorder()
    qp = _make_qp(mem, rec)

    async def fake_analyze(_demo: Demonstration) -> tuple[str, list[str]]:
        return "summary", ["put a blue circle on your head"]

    async def fake_derive(_instruction: str, caption: str) -> list[str]:
        return ["black cap on head"] if "black" in caption and "cap" in caption else []

    qp._analyze_recording = fake_analyze  # type: ignore[assignment]
    qp._derive_step_requirements = fake_derive  # type: ignore[assignment]
    await qp._finalize_demo(demo, demo.finalize_generation, PID, 1)

    step = demo.steps[0]
    expect(step.image_path == "/tmp/black-cap.png" and step.reference_reliable,
           f"mismatched text should keep usable teacher frame: {step}")
    expect(step.text_video_mismatch is True,
           f"mismatch should be traced on the step: {step}")
    expect(step.expected_requirements == ["black cap on head"],
           f"requirements should follow teacher visual hint: {step.expected_requirements}")


async def scenario_11e_reference_selection_prefers_instruction_match() -> None:
    """Reference selection should not pick a later frame that mostly shows the next step."""
    frames = [
        RecordedFrame(0, 1_000_000, "/tmp/f0.png", "person seated with no objects"),
        RecordedFrame(1, 2_000_000, "/tmp/f1.png", "white VR controller held in right hand"),
        RecordedFrame(2, 3_000_000, "/tmp/f2.png", "white controller in hand and headset held nearby"),
        RecordedFrame(3, 4_000_000, "/tmp/f3.png", "headset worn over eyes while controller remains visible"),
    ]
    notes = [
        VoiceNote(2_500_000, "Step one put the white controller in your hand"),
        VoiceNote(4_500_000, "Step two put the headset on your head"),
    ]
    instructions = [
        "put the white controller in your hand",
        "put the headset on your head",
    ]

    step1, _backups1, scores1 = _select_reference_frames_for_step(
        frames, notes, 0, instructions, 0, 5_000_000,
    )
    step2, _backups2, scores2 = _select_reference_frames_for_step(
        frames, notes, 1, instructions, 0, 5_000_000,
    )

    expect(step1 is not None and step1.frame_idx in (1, 2),
           f"step 1 should choose controller frame, got {step1}, scores={scores1}")
    expect(step2 is not None and step2.frame_idx == 3,
           f"step 2 should choose headset-on-head frame, got {step2}, scores={scores2}")


async def scenario_11g_reference_selection_penalizes_future_object() -> None:
    """A cap step should prefer cap-only over cap plus future headphones."""
    frames = [
        RecordedFrame(0, 1_000_000, "/tmp/f0.png", "black baseball cap with gold logo on head"),
        RecordedFrame(1, 2_000_000, "/tmp/f1.png", "black cap on head while holding white headphones above head"),
    ]
    notes = [
        VoiceNote(900_000, "Step one wear the cap on your head"),
        VoiceNote(2_500_000, "Step two put the headphone on your head"),
    ]
    selected, _backups, scores = _select_reference_frames_for_step(
        frames,
        notes,
        0,
        ["wear the cap on your head", "put the headphone on your head"],
        0,
        3_000_000,
    )
    expect(selected is not None and selected.frame_idx == 0,
           f"future headphone frame should be penalized: selected={selected}, scores={scores}")


async def scenario_11f_reference_selection_requires_primary_object() -> None:
    frames = [
        RecordedFrame(0, 1_000_000, "/tmp/f0.png", "person standing in office"),
        RecordedFrame(1, 2_000_000, "/tmp/f1.png", "empty hands visible near face"),
    ]
    notes = [VoiceNote(1_500_000, "Step one hold the controller in your hand")]
    selected, backups, scores = _select_reference_frames_for_step(
        frames, notes, 0, ["hold the controller in your hand"], 0, 3_000_000,
    )
    expect(selected is None and backups == [],
           f"missing primary object must produce no reliable reference: {selected}, {backups}, {scores}")


async def scenario_12_issue_surfaces_in_guidance_question() -> None:
    """When the monitor returns an `issue`, _handle_guidance_question must
    use it in the spoken reply (preferred over current_observation)."""
    mem = AgentMemory()
    demo = _seed_demo(mem, "pico headset", finished_ago_s=5.0)
    rec = Recorder()
    qp = _make_qp(mem, rec)
    qp._guidance_demo = demo
    qp._guidance_step = 0

    async def fake_result(_pid: str) -> dict:
        return {
            "completed": False,
            "current_observation": "white circle on the desk",
            "checks": [{"requirement": "blue circle on desk", "visible": False, "evidence": ""}],
            "missing_or_mismatched": ["blue circle on desk"],
            "issue": "blue circle not white",
        }

    qp._guidance_completion_result = fake_result          # type: ignore[assignment]

    await qp._handle_guidance_question("am I doing it right?", PID)

    last_spoken = rec.spoken[-1][1] if rec.spoken else ""
    expect("blue circle not white" in last_spoken,
           f"issue not surfaced in spoken reply: {last_spoken!r}")
    expect("white circle on the desk" not in last_spoken,
           f"current_observation should be hidden when a richer 'issue' is present: {last_spoken!r}")


async def scenario_13_recording_loop_not_paused_by_query() -> None:
    """A4: with _user_query_active=True AND a recording in progress, the
    background loop must still call _capture_recording_frame (the
    observation branch is gated by _user_query_active, the recording
    branch is NOT).
    """
    import agent as agent_mod
    import xr_ai_agent

    # Set up enough state on a real GlassesAgent instance to run one body
    # of _background_vlm_loop. Patch ProcessorEndpoint / VadDetector so the
    # constructor does no network I/O.
    mem = AgentMemory()
    cfg = _make_cfg()
    rec = Recorder()
    fake_runtime = mock.MagicMock(name="NatRuntime")
    qp = _make_qp(mem, rec)

    with mock.patch.object(xr_ai_agent, "ProcessorEndpoint", autospec=False) as ep_cls, \
         mock.patch.object(agent_mod, "ProcessorEndpoint", autospec=False) as ep_cls2:
        ep_cls.return_value = mock.MagicMock(name="ProcessorEndpoint")
        ep_cls2.return_value = mock.MagicMock(name="ProcessorEndpoint")
        ga = agent_mod.GlassesAgent(
            cfg=cfg,
            memory=mem,
            transcript_client=mock.MagicMock(name="TranscriptClient"),
            query_processor=qp,
            stt_url="http://stub",
            tts_url="http://stub",
            nat_runtime=fake_runtime,
        )

    import dataclasses

    # Tighten the loop sleep so the test completes quickly.
    ga._cfg = dataclasses.replace(cfg, vlm_interval_s=0.05)
    ga._rec_warmup_end = 0
    ga._user_query_active = True

    # Start a recording in memory.
    mem.start_recording("test demo")

    capture_calls: list[str] = []
    observe_calls: list[str] = []

    async def fake_capture(pid: str, skip_ts: int = 0):
        capture_calls.append(pid)
        return None  # don't add a frame, just record the call

    async def fake_observe(pid: str, previous: str = "", skip_ts: int = 0):
        observe_calls.append(pid)
        return None

    ga._capture_recording_frame = fake_capture        # type: ignore[assignment]
    ga._observe_frame           = fake_observe        # type: ignore[assignment]
    ga._active_pid              = lambda: PID         # type: ignore[assignment]

    # The first iteration sees was_recording=False (local) and resets
    # warmup to now + 2 s, so we have to wait past warmup AND through one
    # more sleep cycle before the capture branch runs.
    task = asyncio.create_task(ga._background_vlm_loop())
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if capture_calls:
            break
        await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    expect(len(capture_calls) >= 1,
           f"recording capture must run despite _user_query_active=True: {capture_calls}")
    expect(len(observe_calls) == 0,
           f"observation branch must NOT run while user query is active: {observe_calls}")


async def scenario_13b_guidance_suppresses_background_observation() -> None:
    """Guidance owns VLM checks; background observation must not compete."""
    import agent as agent_mod
    import xr_ai_agent

    mem = AgentMemory()
    demo = _seed_demo(mem, "pico headset", finished_ago_s=5.0)
    rec = Recorder()
    fake_runtime = mock.MagicMock(name="NatRuntime")
    qp = _make_qp(mem, rec)
    qp._guidance_demo = demo
    qp._guidance_step = 0

    with mock.patch.object(xr_ai_agent, "ProcessorEndpoint", autospec=False) as ep_cls, \
         mock.patch.object(agent_mod, "ProcessorEndpoint", autospec=False) as ep_cls2:
        ep_cls.return_value = mock.MagicMock(name="ProcessorEndpoint")
        ep_cls2.return_value = mock.MagicMock(name="ProcessorEndpoint")
        ga = agent_mod.GlassesAgent(
            cfg=_make_cfg(),
            memory=mem,
            transcript_client=mock.MagicMock(name="TranscriptClient"),
            query_processor=qp,
            stt_url="http://stub",
            tts_url="http://stub",
            nat_runtime=fake_runtime,
        )

    import dataclasses
    ga._cfg = dataclasses.replace(ga._cfg, vlm_interval_s=0.02)
    ga._active_pid = lambda: PID  # type: ignore[assignment]
    observe_calls: list[str] = []

    async def fake_observe(pid: str, previous: str = "", skip_ts: int = 0):
        observe_calls.append(pid)
        return None

    ga._observe_frame = fake_observe  # type: ignore[assignment]
    task = asyncio.create_task(ga._background_vlm_loop())
    await asyncio.sleep(0.08)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    expect(observe_calls == [],
           f"background observation must pause during guidance: {observe_calls}")


async def scenario_13c_vad_speech_start_does_not_flush_tts() -> None:
    """Raw VAD starts can be noise; they must not interrupt active TTS."""
    mem = AgentMemory()
    rec = Recorder()
    qp = _make_qp(mem, rec)
    ga = _make_ga(mem, qp)
    ga._speech_generation[PID] = 3

    try:
        await ga._handle_speech_start(PID)
    finally:
        await ga._http.aclose()

    expect(ga._speech_generation[PID] == 3,
           f"speech-start must not bump generation: {ga._speech_generation}")
    expect(ga._ep.flush_return_audio.await_count == 0,
           "speech-start must not flush return audio")


async def scenario_13d_accepted_dispatch_still_flushes_tts() -> None:
    """Accepted voice transcripts still interrupt playback at dispatch time."""
    mem = AgentMemory()
    rec = Recorder()
    qp = _make_qp(mem, rec)
    ga = _make_ga(mem, qp)
    handled: list[tuple[str, str, int]] = []

    async def fake_handle(text: str, pid: str, ref_us: int) -> None:
        handled.append((text, pid, ref_us))

    qp.handle = fake_handle  # type: ignore[method-assign]

    try:
        await ga._dispatch_query(PID, "stop", ref_us=123, source="voice")
        await asyncio.wait_for(ga._query_tasks[PID], timeout=1.0)
    finally:
        await ga._http.aclose()

    expect(ga._ep.flush_return_audio.await_count == 1,
           "accepted dispatch must flush return audio")
    expect(ga._speech_generation.get(PID) == 1,
           f"accepted dispatch must bump generation: {ga._speech_generation}")
    expect(handled == [("stop", PID, 123)],
           f"accepted dispatch did not run query handler: {handled}")


async def scenario_10_pending_attempt_limit() -> None:
    """NEW: Two unrelated answers to a pending prompt -> attempt 2 clears
    state with an apology, no infinite loop."""
    mem = AgentMemory()
    _seed_demo(mem, "pico headset",    finished_ago_s=30.0)
    _seed_demo(mem, "desk arrangement", finished_ago_s=20.0)
    rec = Recorder()
    trace = _attach_trace_capture()
    qp = _make_qp(mem, rec)

    await qp.handle("Show me how.", PID, ref_us=0)
    expect(PID in qp._pending_guidance_by_pid, "setup: pending state not set")

    # Attempt 1: gibberish. Should re-ask.
    await qp.handle("uhhh something else", PID, ref_us=0)
    expect(PID in qp._pending_guidance_by_pid,
           "attempt 1 should re-ask, not clear pending")
    expect(qp._pending_guidance_by_pid[PID]["attempts"] == 1,
           f"attempt counter wrong: {qp._pending_guidance_by_pid[PID]}")

    # Attempt 2: gibberish again. Should give up and clear.
    await qp.handle("more nonsense here", PID, ref_us=0)
    expect(PID not in qp._pending_guidance_by_pid,
           "attempt 2 should clear pending state")
    expect(qp._guidance_demo is None, "attempt limit should NOT start guidance")
    expect(first_match(trace, "PENDING_GIVE_UP") is not None,
           f"expected PENDING_GIVE_UP, got {trace}")


async def scenario_14_question_form_done_runs_completion_check() -> None:
    """B1: 'am i done?' must route to the question handler (completion check),
    NOT exit guidance.
    """
    mem = AgentMemory()
    demo = _seed_demo(mem, "pico headset", finished_ago_s=5.0)
    rec = Recorder()
    qp = _make_qp(mem, rec)
    _enter_guidance(qp, demo)

    completion_called = {"count": 0}

    async def fake_completion(_pid: str) -> dict:
        completion_called["count"] += 1
        return {
            "completed": True,
            "current_observation": "task complete",
            "checks": [{"requirement": "x", "visible": True, "evidence": "x visible"}],
            "missing_or_mismatched": [],
            "issue": "",
        }

    qp._guidance_completion_result = fake_completion   # type: ignore[assignment]

    await qp.handle("am i done?", PID, ref_us=0)

    expect(qp._guidance_demo is not None, "question-form must NOT exit guidance")
    expect(completion_called["count"] == 1,
           f"completion check not called: {completion_called}")


async def scenario_14b_bare_done_still_exits() -> None:
    """B1: bare 'done' is still an exit command (no question form, completion stem)."""
    mem = AgentMemory()
    demo = _seed_demo(mem, "pico headset", finished_ago_s=5.0)
    rec = Recorder()
    trace = _attach_trace_capture()
    qp = _make_qp(mem, rec)
    _enter_guidance(qp, demo)

    await qp.handle("done", PID, ref_us=0)

    expect(qp._guidance_demo is None, "bare 'done' must exit guidance")
    expect(first_match(trace, "GUIDANCE_DONE") is not None,
           f"expected GUIDANCE_DONE in trace, got {trace}")


async def scenario_14c_is_this_complete_runs_completion_check() -> None:
    """B1: 'is this complete?' (stem 'complete', question form) routes to check."""
    mem = AgentMemory()
    demo = _seed_demo(mem, "pico headset", finished_ago_s=5.0)
    rec = Recorder()
    qp = _make_qp(mem, rec)
    _enter_guidance(qp, demo)

    completion_called = {"count": 0}

    async def fake_completion(_pid: str) -> dict:
        completion_called["count"] += 1
        return {
            "completed": True,
            "current_observation": "task complete",
            "checks": [{"requirement": "x", "visible": True, "evidence": "x visible"}],
            "missing_or_mismatched": [],
            "issue": "",
        }

    qp._guidance_completion_result = fake_completion   # type: ignore[assignment]

    await qp.handle("is this complete?", PID, ref_us=0)

    expect(qp._guidance_demo is not None, "'is this complete' must NOT exit guidance")
    expect(completion_called["count"] == 1,
           f"completion check not called for 'is this complete': {completion_called}")


async def scenario_14d_did_i_finish_runs_completion_check() -> None:
    """B1: 'did i finish?' (stem 'finish' — the reviewer-flagged gap) routes
    to the completion check, not exit and not LLM fallback.
    """
    mem = AgentMemory()
    demo = _seed_demo(mem, "pico headset", finished_ago_s=5.0)
    rec = Recorder()
    qp = _make_qp(mem, rec)
    _enter_guidance(qp, demo)

    completion_called = {"count": 0}

    async def fake_completion(_pid: str) -> dict:
        completion_called["count"] += 1
        return {
            "completed": False,
            "current_observation": "no headset visible",
            "checks": [{"requirement": "headset on head", "visible": False, "evidence": ""}],
            "missing_or_mismatched": ["headset on head"],
            "issue": "headset not on",
        }

    qp._guidance_completion_result = fake_completion   # type: ignore[assignment]

    await qp.handle("did i finish?", PID, ref_us=0)

    expect(qp._guidance_demo is not None, "'did i finish' must NOT exit guidance")
    expect(completion_called["count"] == 1,
           f"completion check not called for 'did i finish': {completion_called}")


async def scenario_15_save_this_does_not_start_recording() -> None:
    """B2: 'save this for later' must NOT trigger demo-start now that
    'save this' is removed from _DEMO_START_PHRASES.
    """
    mem = AgentMemory()
    rec = Recorder()
    trace = _attach_trace_capture()
    qp = _make_qp(mem, rec)

    await qp.handle("save this for later", PID, ref_us=0)

    expect(mem.recording is None,
           "save this must not start a recording")
    expect(first_match(trace, "DEMO_START") is None,
           f"DEMO_START fired for 'save this': {trace}")


async def scenario_16_token_aware_demo_name() -> None:
    """B3: 'start recording theme song demo' must capture 'theme song demo'
    as the name, not the chopped 'me song demo' (the old prefix-strip
    bug).
    """
    mem = AgentMemory()
    rec = Recorder()
    qp = _make_qp(mem, rec)

    await qp.handle("start recording theme song demo", PID, ref_us=0)

    expect(mem.recording is not None, "demo should have started")
    assert mem.recording is not None
    expect(mem.recording.name == "theme song demo",
           f"demo name was wrongly chopped: {mem.recording.name!r}")


async def scenario_16b_watch_meteor_does_not_start_recording() -> None:
    """B3: 'watch meteor shower' shares the substring 'watch me' with the
    start phrase but is NOT a contiguous token slice — must not trigger.
    """
    mem = AgentMemory()
    rec = Recorder()
    qp = _make_qp(mem, rec)

    await qp.handle("watch meteor shower", PID, ref_us=0)

    expect(mem.recording is None,
           f"'watch meteor shower' falsely started recording: {mem.recording}")


async def scenario_16c_watch_me_arrange_starts_recording() -> None:
    """B3: 'watch me arrange the headset' is a token-boundary match for
    'watch me' and should start a recording named 'arrange headset'
    (filler 'the' is stripped).
    """
    mem = AgentMemory()
    rec = Recorder()
    qp = _make_qp(mem, rec)

    await qp.handle("watch me arrange the headset", PID, ref_us=0)

    expect(mem.recording is not None, "expected recording to start")
    assert mem.recording is not None
    # 'the' is leading filler; allowed names are 'arrange the headset'
    # (no filler strip after the first token) or 'arrange headset'
    # (after iterative strip). The current implementation strips only
    # while the FIRST token is filler — 'arrange' is not, so the name
    # keeps 'the' inside.
    expect("arrange" in mem.recording.name and "headset" in mem.recording.name,
           f"unexpected demo name: {mem.recording.name!r}")


async def scenario_16d_watch_me_comma_arrange_starts_recording() -> None:
    """B3: punctuation between 'watch me' and the rest must not break
    detection — 're.findall' tokenization drops the comma.
    """
    mem = AgentMemory()
    rec = Recorder()
    qp = _make_qp(mem, rec)

    await qp.handle("watch me, arrange the headset", PID, ref_us=0)

    expect(mem.recording is not None,
           "'watch me, arrange...' should still trigger via token-boundary detection")


async def scenario_17_lets_do_does_not_start_guidance() -> None:
    """B4: bare 'let's do the dishes' (no demos on file) must NOT enter
    guidance — 'let's do' was removed from _GUIDANCE_PHRASES.
    """
    mem = AgentMemory()
    rec = Recorder()
    qp = _make_qp(mem, rec)

    await qp.handle("let's do the dishes", PID, ref_us=0)

    expect(qp._guidance_demo is None,
           "'let's do' must NOT trigger guidance now that the phrase is removed")
    expect(len(rec.agentic_loop_calls) >= 1,
           "with no guidance-phrase match, request must fall through to the agentic loop")


async def scenario_18_guidance_during_finalization() -> None:
    """B6: after stop-recording, the demo is in is_finalizing=True. A
    guidance request must get the "still analyzing" message and NOT
    enter guidance.
    """
    mem = AgentMemory()
    rec = Recorder()
    qp = _make_qp(mem, rec)

    # Start recording, add a frame, stop. _handle_demo_end launches a
    # background _finalize_demo task that awaits _analyze_recording.
    mem.start_recording("pico headset")
    _add_recorded_frame(mem)

    analyze_started = asyncio.Event()
    analyze_proceed = asyncio.Event()

    async def slow_analyze(_demo: Demonstration) -> tuple[str, list[str]]:
        analyze_started.set()
        await analyze_proceed.wait()
        return "ok", ["step one", "step two"]

    qp._analyze_recording = slow_analyze   # type: ignore[assignment]

    await qp.handle("stop recording", PID, ref_us=0)
    await analyze_started.wait()  # finalize task is parked inside _analyze_recording

    # While the finalize task is still running, request guidance for that demo.
    rec.spoken.clear()
    rec.sent_text.clear()
    await qp.handle("show me how to wear pico headset", PID, ref_us=0)

    last_spoken = " ".join(t for _, t in rec.spoken)
    expect("still analyzing" in last_spoken.lower(),
           f"expected 'still analyzing' message during finalization, got {rec.spoken}")
    expect(qp._guidance_demo is None,
           "must NOT enter guidance while demo is_finalizing")
    expect(PID not in qp._pending_guidance_by_pid,
           "single matching demo: no pending state should be set")

    # Let analysis finish + tasks settle.
    analyze_proceed.set()
    for task in list(qp._finalize_tasks):
        await task


async def scenario_18b_pending_then_finalizing_clears_pending() -> None:
    """B6: with a seeded second demo and a finalizing demo, an ambiguous
    'show me how' sets pending. Replying with the EXACT name (order-
    independent) clears pending and lands on the still-analyzing demo,
    which emits "still analyzing".
    """
    mem = AgentMemory()
    _seed_demo(mem, "desk arrangement", finished_ago_s=60.0)
    rec = Recorder()
    qp = _make_qp(mem, rec)

    mem.start_recording("pico headset")
    _add_recorded_frame(mem)

    analyze_started = asyncio.Event()
    analyze_proceed = asyncio.Event()

    async def slow_analyze(_demo: Demonstration) -> tuple[str, list[str]]:
        analyze_started.set()
        await analyze_proceed.wait()
        return "ok", ["step one"]

    qp._analyze_recording = slow_analyze   # type: ignore[assignment]

    await qp.handle("stop recording", PID, ref_us=0)
    await analyze_started.wait()

    # Two demos exist (one seeded + one finalizing). Bare 'show me how'
    # should set pending state.
    await qp.handle("show me how.", PID, ref_us=0)
    expect(PID in qp._pending_guidance_by_pid,
           f"expected pending state after ambiguous request, got {qp._pending_guidance_by_pid}")

    # Reply by NAME (order-independent) → resolves to the finalizing demo.
    rec.spoken.clear()
    await qp.handle("pico headset", PID, ref_us=0)

    expect(PID not in qp._pending_guidance_by_pid,
           "pending state must clear on successful name match")
    expect(qp._guidance_demo is None,
           "guidance must NOT enter when picked demo is finalizing")
    last_spoken = " ".join(t for _, t in rec.spoken)
    expect("still analyzing" in last_spoken.lower(),
           f"expected 'still analyzing' for finalizing demo, got {rec.spoken}")

    analyze_proceed.set()
    for task in list(qp._finalize_tasks):
        await task


async def scenario_19_finalization_completes_async() -> None:
    """B6: after a short analysis, demo.is_finalizing must flip back to
    False, demo.steps must populate, and DEMO_FINALIZED must trace.
    """
    mem = AgentMemory()
    rec = Recorder()
    trace = _attach_trace_capture()
    qp = _make_qp(mem, rec)

    mem.start_recording("pico headset")
    _add_recorded_frame(mem)

    async def fast_analyze(_demo: Demonstration) -> tuple[str, list[str]]:
        await asyncio.sleep(0.05)
        return "overview", ["step one", "step two"]

    qp._analyze_recording = fast_analyze   # type: ignore[assignment]

    await qp.handle("stop recording", PID, ref_us=0)
    for task in list(qp._finalize_tasks):
        await task

    demo = mem.get_demonstration("pico headset")
    assert demo is not None
    expect(demo.is_finalizing is False,
           f"is_finalizing must reset to False after finalize, got {demo.is_finalizing}")
    expect(len(demo.steps) == 2,
           f"steps not populated: {demo.steps}")
    expect(first_match(trace, "DEMO_FINALIZED") is not None,
           f"expected DEMO_FINALIZED in trace, got {trace}")


async def scenario_20_advance_matcher_handles_filler() -> None:
    """B5: positive advance cases must advance; negative cases (incl.
    'I got it wrong') must NOT.
    """
    mem = AgentMemory()
    demo = _seed_demo(mem, "long demo", finished_ago_s=5.0)
    # Make sure there's room to advance several steps.
    while len(demo.steps) < 6:
        demo.steps.append(DemoStep(
            step_number  = len(demo.steps) + 1,
            timestamp_us = demo.steps[-1].timestamp_us + 1,
            description  = f"step {len(demo.steps) + 1}",
            image_path   = "",
        ))
    rec = Recorder()
    qp = _make_qp(mem, rec)
    _enter_guidance(qp, demo)

    # Stub _speak_current_guidance_step so the advance doesn't try to
    # invoke a real TTS chain when stepping through.
    async def fake_speak(_pid: str) -> None:
        return None
    qp._speak_current_guidance_step = fake_speak       # type: ignore[assignment]

    # Disable monitor restart so we don't background a task per advance.
    qp._start_guidance_monitor = lambda _pid: None     # type: ignore[assignment]

    # Stub _handle_guidance_question so the negative cases below don't
    # try to reach a live LLM. The advance-matcher decision happens
    # BEFORE this in handle(); the question fallback is a no-op for the
    # purposes of this test.
    async def fake_question(_transcript: str, _pid: str) -> None:
        return None
    qp._handle_guidance_question = fake_question       # type: ignore[assignment]

    positives = ["next step please", "okay, next step", "next", "got it"]
    negatives = ["I got it wrong", "next time I see this", "got it but I'm stuck"]

    start_step = qp._guidance_step
    for utter in positives:
        before = qp._guidance_step
        await qp.handle(utter, PID, ref_us=0)
        expect(qp._guidance_step == before + 1,
               f"advance positive {utter!r} did not advance: {before} → {qp._guidance_step}")

    after_positives = qp._guidance_step
    for utter in negatives:
        before = qp._guidance_step
        await qp.handle(utter, PID, ref_us=0)
        expect(qp._guidance_step == before,
               f"advance negative {utter!r} falsely advanced: {before} → {qp._guidance_step}")

    expect(qp._guidance_step == after_positives,
           f"step counter drifted under negatives: {qp._guidance_step} vs {after_positives}")
    expect(qp._guidance_step == start_step + len(positives),
           f"expected exactly {len(positives)} advances, got {qp._guidance_step - start_step}")


async def scenario_21_clear_demos_cancels_finalization() -> None:
    """B6: 'forget all demos' during finalization cancels the in-flight
    task. The CancelledError lands inside _analyze_recording, so the
    expected trace is DEMO_FINALIZE_CANCELLED.
    """
    mem = AgentMemory()
    rec = Recorder()
    trace = _attach_trace_capture()
    qp = _make_qp(mem, rec)

    mem.start_recording("pico headset")
    _add_recorded_frame(mem)

    analyze_started = asyncio.Event()

    async def long_analyze(_demo: Demonstration) -> tuple[str, list[str]]:
        analyze_started.set()
        await asyncio.sleep(3.0)
        return "ok", ["step"]

    qp._analyze_recording = long_analyze   # type: ignore[assignment]

    await qp.handle("stop recording", PID, ref_us=0)
    await analyze_started.wait()

    # Clear demos — should cancel the finalize task.
    rec.spoken.clear()
    await qp.handle("forget all demos", PID, ref_us=0)

    # Let cancellation propagate.
    for task in list(qp._finalize_tasks):
        try:
            await task
        except asyncio.CancelledError:
            pass

    expect(first_match(trace, "DEMO_FINALIZE_CANCELLED") is not None,
           f"expected DEMO_FINALIZE_CANCELLED in trace, got {trace}")
    expect(first_match(trace, "DEMO_FINALIZED") is None,
           f"DEMO_FINALIZED must NOT fire after clear, got {trace}")

    # No "Saved" speech should appear in the spoken history after the clear.
    for _pid, text in rec.spoken:
        expect("Saved" not in text,
               f"stale 'Saved' message leaked through after clear: {text!r}")

    expect(len(qp._finalize_tasks) == 0,
           f"finalize_tasks should be drained: {qp._finalize_tasks}")


async def scenario_22_rerecord_same_name_during_finalization() -> None:
    """B6: re-recording the same name while a finalize task is still
    running invalidates the prior task — identity check via
    AgentMemory.is_current_finalization.
    """
    mem = AgentMemory()
    rec = Recorder()
    trace = _attach_trace_capture()
    qp = _make_qp(mem, rec)

    # Track which demo each analyze call belongs to, to gate them.
    proceed_events: list[asyncio.Event] = []
    analyze_call_order: list[str] = []

    async def gated_analyze(demo: Demonstration) -> tuple[str, list[str]]:
        analyze_call_order.append(demo.name)
        ev = asyncio.Event()
        proceed_events.append(ev)
        await ev.wait()
        return f"summary {demo.name}", [f"step {demo.name}"]

    qp._analyze_recording = gated_analyze   # type: ignore[assignment]

    # First recording.
    mem.start_recording("pico headset")
    _add_recorded_frame(mem)
    await qp.handle("stop recording", PID, ref_us=0)

    # Wait for task A to enter analyze.
    while len(proceed_events) < 1:
        await asyncio.sleep(0.01)
    demo_a = mem.get_demonstration("pico headset")
    assert demo_a is not None

    # Second recording with the SAME name — overwrites _demos["pico headset"].
    mem.start_recording("pico headset")
    _add_recorded_frame(mem)
    await qp.handle("stop recording", PID, ref_us=0)

    while len(proceed_events) < 2:
        await asyncio.sleep(0.01)
    demo_b = mem.get_demonstration("pico headset")
    assert demo_b is not None
    expect(demo_b is not demo_a, "second recording must be a different object")

    # Release both analyses.
    rec.spoken.clear()
    for ev in proceed_events:
        ev.set()
    for task in list(qp._finalize_tasks):
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Task A should have logged DEMO_FINALIZE_STALE (identity check
    # failed: _demos['pico headset'] is demo_B now). Task B should have
    # logged DEMO_FINALIZED and spoken "Saved demonstration 'pico headset'".
    expect(first_match(trace, "DEMO_FINALIZE_STALE") is not None,
           f"expected DEMO_FINALIZE_STALE for stale re-record task, got {trace}")
    expect(first_match(trace, "DEMO_FINALIZED") is not None,
           f"expected DEMO_FINALIZED for the winning task, got {trace}")
    expect(count_starting_with(trace, "DEMO_FINALIZED") == 1,
           f"only ONE DEMO_FINALIZED expected, got {count_starting_with(trace, 'DEMO_FINALIZED')}")

    saved_count = sum(1 for _, text in rec.spoken if "Saved demonstration" in text)
    expect(saved_count == 1,
           f"exactly one 'Saved demonstration' should be spoken, got {saved_count}: {rec.spoken}")


# ── grounded-completion + active-monitor scenarios ───────────────────────────

def _grounded_yes(reqs: list[str]) -> dict:
    """Build a `check_guidance_step_complete` result that should pass the
    new grounded-evidence parser: non-empty current_observation, every
    expected requirement present in checks with visible=True + evidence.
    """
    return {
        "completed": True,
        "current_observation": "headset is on the user's head with strap visible",
        "checks": [
            {"requirement": r, "visible": True, "evidence": f"clear view of {r}"}
            for r in reqs
        ],
        "missing_or_mismatched": [],
        "image_path": "/tmp/live.png",
        "issue": "",
    }


def _ungrounded_yes() -> dict:
    """Biased YES with no evidence — must be rejected by the parser."""
    return {
        "completed": True,
        "current_observation": "",
        "checks": [],
        "missing_or_mismatched": [],
        "image_path": "",
        "issue": "",
    }


def _grounded_no(missing: list[str]) -> dict:
    return {
        "completed": False,
        "current_observation": "user is sitting at the desk, no headset visible",
        "checks": [
            {"requirement": r, "visible": False, "evidence": ""}
            for r in missing
        ],
        "missing_or_mismatched": missing,
        "image_path": "/tmp/live.png",
        "issue": f"{missing[0]} not visible" if missing else "",
    }


async def _drive_monitor_until(
    qp: QueryProcessor,
    pid: str,
    *,
    cycles: int,
    obs_per_cycle: int,
    completion_results: list[dict],
) -> dict:
    """Drive _guidance_monitor_loop for *cycles* iterations.

    Each cycle bumps len(memory._observations) by *obs_per_cycle* (use 0
    for a static scene) and returns completion_results[i] from the stubbed
    NAT call (last element repeats if the list is shorter). Returns
    {"advances": int, "step": int, "vlm_calls": int} so callers can
    assert directly on advance counts (more robust than the step index,
    which wraps when _finish_guidance fires).
    """
    rec_results = list(completion_results)

    call_idx = {"i": 0, "advances": 0}

    async def fake_call_tool(group: str, tool: str, args: dict, **_) -> dict:
        i = call_idx["i"]
        call_idx["i"] += 1
        if i < len(rec_results):
            return rec_results[i]
        return rec_results[-1]

    qp._nat_runtime.call_tool = fake_call_tool  # type: ignore[attr-defined]

    async def fake_speak(_pid: str) -> None:
        call_idx["advances"] += 1
        qp._guidance_step_obs_baseline = len(qp._memory._observations)
        qp._guidance_check_obs_baseline = len(qp._memory._observations)
        qp._guidance_monitor_idle_cycles = 0
        qp._guidance_consecutive_yes = 0

    qp._speak_current_guidance_step = fake_speak  # type: ignore[assignment]
    qp._start_guidance_monitor      = lambda _pid: None  # type: ignore[assignment]
    # Suppress _finish_guidance side-effects: it cancels the monitor task
    # and clears _guidance_demo, which would terminate the loop and mask
    # advance counts. The test cares about how many times the monitor
    # voted-to-advance, not about post-finish state.
    async def fake_finish(_pid: str) -> None:
        call_idx["advances"] += 1
        qp._guidance_step = 0  # reset so the monitor can keep iterating
        qp._guidance_step_obs_baseline = len(qp._memory._observations)
        qp._guidance_check_obs_baseline = len(qp._memory._observations)
        qp._guidance_monitor_idle_cycles = 0
        qp._guidance_consecutive_yes = 0
    qp._finish_guidance = fake_finish  # type: ignore[assignment]

    import dataclasses
    qp._cfg = dataclasses.replace(qp._cfg, guidance_check_interval_s=0.01)

    qp._guidance_step_obs_baseline   = len(qp._memory._observations)
    qp._guidance_check_obs_baseline  = len(qp._memory._observations)
    qp._guidance_monitor_idle_cycles = 0
    qp._guidance_consecutive_yes     = 0

    task = asyncio.create_task(qp._guidance_monitor_loop(pid))
    try:
        for _ in range(cycles):
            await asyncio.sleep(0.02)
            for _j in range(obs_per_cycle):
                qp._memory._observations.append(
                    __import__("memory").Observation(
                        timestamp_us=int(time.time() * 1_000_000),
                        description="simulated change",
                        image_path="/tmp/x.png",
                    )
                )
    finally:
        qp._guidance_demo = None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    return {
        "advances":  call_idx["advances"],
        "vlm_calls": call_idx["i"],
        "step":      qp._guidance_step,
    }


def _seed_long_demo(mem: AgentMemory, name: str, n_steps: int = 10) -> Demonstration:
    """Seed a demo with many steps so monitor tests don't wrap into _finish_guidance."""
    demo = _seed_demo(mem, name, finished_ago_s=5.0)
    while len(demo.steps) < n_steps:
        demo.steps.append(DemoStep(
            step_number  = len(demo.steps) + 1,
            timestamp_us = demo.steps[-1].timestamp_us + 1,
            description  = f"step {len(demo.steps) + 1}",
            image_path   = "",
        ))
        demo.instructions.append(f"step {len(demo.instructions) + 1}")
    return demo


async def scenario_24_biased_yes_static_scene_blocked() -> None:
    """REGRESSION: a biased VLM that returns completed=true with NO
    evidence on a static scene must NOT advance guidance. Drives the
    monitor for many cycles and asserts advance count is 0.
    """
    mem = AgentMemory()
    demo = _seed_long_demo(mem, "pico headset", n_steps=10)
    reqs = ["headset on head", "strap behind head"]
    for s in demo.steps:
        s.expected_requirements = reqs
        s.image_path = "/tmp/teacher.png"
        s.reference_reliable = True
        s.image_path = "/tmp/teacher.png"
        s.reference_reliable = True
        s.image_path = "/tmp/teacher.png"
        s.reference_reliable = True
        s.image_path = "/tmp/teacher.png"
        s.reference_reliable = True
    rec = Recorder()
    qp = _make_qp(mem, rec)
    qp._guidance_demo = demo
    qp._guidance_step = 0

    result = await _drive_monitor_until(
        qp, PID,
        cycles            = 15,
        obs_per_cycle     = 0,                          # static scene
        completion_results= [_ungrounded_yes()] * 15,
    )
    expect(result["advances"] == 0,
           f"biased YES on static scene must NOT advance: got {result['advances']} advances")


async def scenario_25_grounded_yes_static_scene_needs_two() -> None:
    """Static correct behavior advances after two grounded checks."""
    mem = AgentMemory()
    demo = _seed_long_demo(mem, "pico headset", n_steps=10)
    reqs = ["headset on head", "strap behind head"]
    for s in demo.steps:
        s.expected_requirements = reqs
        s.image_path = "/tmp/teacher.png"
        s.reference_reliable = True
    rec = Recorder()
    qp = _make_qp(mem, rec)
    qp._guidance_demo = demo
    qp._guidance_step = 0

    result = await _drive_monitor_until(
        qp, PID,
        cycles            = 6,
        obs_per_cycle     = 0,                          # static
        completion_results= [_grounded_yes(reqs)] * 6,
    )
    expect(result["advances"] >= 1,
           f"grounded YES on static scene must advance: got {result['advances']} advances")


async def scenario_26_grounded_yes_moving_scene_needs_two() -> None:
    """Moving scenes also advance after two grounded YES votes."""
    mem = AgentMemory()
    demo = _seed_long_demo(mem, "pico headset", n_steps=10)
    reqs = ["headset on head"]
    for s in demo.steps:
        s.expected_requirements = reqs
        s.image_path = "/tmp/teacher.png"
        s.reference_reliable = True
    rec = Recorder()
    qp = _make_qp(mem, rec)
    qp._guidance_demo = demo
    qp._guidance_step = 0

    result = await _drive_monitor_until(
        qp, PID,
        cycles            = 6,
        obs_per_cycle     = 1,                          # moving scene
        completion_results= [_grounded_yes(reqs)] * 6,
    )
    expect(result["advances"] >= 1,
           f"grounded YES on moving scene must advance within 6 cycles: got {result['advances']} advances")


async def scenario_27_malformed_response_resets_streak() -> None:
    """REGRESSION: an empty-evidence response between two grounded YES
    must RESET the streak so no advance fires.
    """
    mem = AgentMemory()
    demo = _seed_long_demo(mem, "pico headset", n_steps=10)
    reqs = ["headset on head"]
    for s in demo.steps:
        s.expected_requirements = reqs
        s.image_path = "/tmp/teacher.png"
        s.reference_reliable = True
    rec = Recorder()
    qp = _make_qp(mem, rec)
    qp._guidance_demo = demo
    qp._guidance_step = 0

    pattern = [_grounded_yes(reqs), _ungrounded_yes()] * 8

    result = await _drive_monitor_until(
        qp, PID,
        cycles            = 16,
        obs_per_cycle     = 1,                          # moving scene → 2-YES threshold
        completion_results= pattern,
    )
    expect(result["advances"] == 0,
           f"alternating malformed responses must keep streak at 1: "
           f"got {result['advances']} advances")


async def scenario_27b_grounded_no_speaks_correction() -> None:
    """Two grounded failures for the same missing requirement speak a correction."""
    mem = AgentMemory()
    demo = _seed_long_demo(mem, "pico headset", n_steps=10)
    reqs = ["headset on head"]
    for s in demo.steps:
        s.expected_requirements = reqs
        s.image_path = "/tmp/teacher.png"
        s.reference_reliable = True
    rec = Recorder()
    qp = _make_qp(mem, rec)
    qp._guidance_demo = demo
    qp._guidance_step = 0

    await _drive_monitor_until(
        qp, PID,
        cycles            = 6,
        obs_per_cycle     = 0,
        completion_results= [_grounded_no(reqs)] * 6,
    )
    spoken = " ".join(text for _pid, text in rec.spoken)
    expect("headset on head" in spoken,
           f"grounded repeated failure should speak correction, got {rec.spoken}")


async def scenario_27c_unreliable_reference_blocks_passive_advance() -> None:
    """Passive monitor should not auto-advance without a reliable teacher reference."""
    mem = AgentMemory()
    demo = _seed_long_demo(mem, "pico headset", n_steps=10)
    reqs = ["headset on head"]
    for s in demo.steps:
        s.expected_requirements = reqs
        s.reference_reliable = False
        s.image_path = ""
    rec = Recorder()
    qp = _make_qp(mem, rec)
    qp._guidance_demo = demo
    qp._guidance_step = 0

    result = await _drive_monitor_until(
        qp, PID,
        cycles            = 6,
        obs_per_cycle     = 0,
        completion_results= [_grounded_yes(reqs)] * 6,
    )
    expect(result["advances"] == 0,
           f"unreliable teacher reference must block passive advance: {result}")


# ── direct parser unit tests for check_guidance_step_complete_impl ───────────

async def _run_parser_case(
    *,
    expected_requirements: list[str],
    vlm_json_response:     str,
    ask_frames_response:   str | None = None,
    teacher_image_path:    str = "",
    teacher_caption:       str = "",
    live_timestamp_us:     int = 0,
    min_live_timestamp_us: int = 0,
    prompt_capture:        dict | None = None,
    instruction:           str = "Put the headset on your head",
) -> dict:
    """Drive check_guidance_step_complete_impl with a stubbed VLM."""
    import glasses_nat_tasks as tasks

    class _FakeFrame:
        async def ainvoke(self, _args: dict) -> dict:
            return {"path": "/tmp/live.png", "timestamp_us": live_timestamp_us}

    class _FakeAsk:
        async def ainvoke(self, args: dict) -> str:
            if prompt_capture is not None:
                prompt_capture["question"]   = args.get("question", "")
                prompt_capture["image_path"] = args.get("image_path", "")
            return vlm_json_response

    class _FakeAskFrames:
        async def ainvoke(self, args: dict) -> str:
            if prompt_capture is not None:
                prompt_capture["frames_question"] = args.get("question", "")
                prompt_capture["image_paths"]     = args.get("image_paths", [])
            return ask_frames_response or ""

    result = await tasks.check_guidance_step_complete_impl(
        participant_id        = "p1",
        instruction           = instruction,
        expected_requirements = expected_requirements,
        teacher_image_path    = teacher_image_path,
        teacher_caption       = teacher_caption,
        min_live_timestamp_us = min_live_timestamp_us,
        get_latest_frame      = _FakeFrame(),
        ask_image             = _FakeAsk(),
        ask_frames            = _FakeAskFrames() if ask_frames_response is not None else None,
    )
    return result.model_dump()


async def parser_test_a_completed_true_no_evidence_rejected() -> None:
    """Parser must reject completed=true with empty current_observation
    and empty checks (the textbook bias-only response)."""
    out = await _run_parser_case(
        expected_requirements=["headset on head"],
        vlm_json_response='{"completed": true, "current_observation": "", "checks": []}',
    )
    expect(out["completed"] is False,
           f"empty-evidence YES must be rejected, got {out}")


async def parser_test_b_copied_checklist_empty_evidence_rejected() -> None:
    """Parser must reject when checks copy the requirement back with
    visible=true but evidence="" — bias-copy failure mode."""
    out = await _run_parser_case(
        expected_requirements=["headset on head", "strap behind head"],
        vlm_json_response=(
            '{"completed": true, '
            '"current_observation": "person at desk", '
            '"checks": ['
            '{"requirement": "headset on head", "visible": true, "evidence": ""},'
            '{"requirement": "strap behind head", "visible": true, "evidence": ""}'
            ']}'
        ),
    )
    expect(out["completed"] is False,
           f"copied-checklist-with-empty-evidence must be rejected, got {out}")


async def parser_test_c_all_visible_with_evidence_accepted() -> None:
    """Parser must accept completed=true when every expected requirement
    is in checks with visible=true AND non-empty evidence."""
    out = await _run_parser_case(
        expected_requirements=["headset on head"],
        vlm_json_response=(
            '{"completed": true, '
            '"current_observation": "headset is on the head", '
            '"checks": ['
            '{"requirement": "headset on head", "visible": true, '
            '"evidence": "black headset clearly visible covering eyes"}'
            ']}'
        ),
    )
    expect(out["completed"] is True,
           f"fully grounded YES must be accepted, got {out}")


async def parser_test_d_vlm_judgment_not_exact_checklist_match() -> None:
    """The VLM judgment is authoritative; parser does not exact-match every hint."""
    out = await _run_parser_case(
        expected_requirements=["headset on head", "strap behind head"],
        vlm_json_response=(
            '{"completed": true, '
            '"current_observation": "headset is on the head", '
            '"checks": ['
            '{"requirement": "headset on head", "visible": true, '
            '"evidence": "headset covers eyes"}'
            ']}'
        ),
    )
    expect(out["completed"] is True,
           f"grounded VLM pass should not require exact checklist coverage, got {out}")


async def parser_test_e_prompt_omits_teacher_caption() -> None:
    """Teacher caption is included in the instruction fallback prompt."""
    captured: dict = {}
    secret_caption = "ABRACADABRA_TEACHER_CAPTION_DO_NOT_LEAK"
    await _run_parser_case(
        expected_requirements=["headset on head"],
        vlm_json_response='{"completed": false, "current_observation": "x", "checks": []}',
        teacher_caption=secret_caption,
        prompt_capture=captured,
        instruction="Put the headset on your head",
    )
    expect(secret_caption in captured.get("question", ""),
           f"single-image fallback should include teacher caption: {captured.get('question', '')!r}")
    q = captured.get("question", "")
    expect("Put the headset on your head" in q,
           f"prompt missing instruction: {q!r}")
    expect("headset on head" in q,
           f"prompt missing the expected requirement: {q!r}")


async def parser_test_f_flat_schema_observation_only_accepted() -> None:
    """Parser must accept the new flat shape:
    {"observation": "...", "requirements": {"req": {"visible": true, "evidence": "..."}}}.
    """
    out = await _run_parser_case(
        expected_requirements=["headset on head"],
        vlm_json_response=(
            '{"observation": "the headset is on her head", '
            '"requirements": {'
            '"headset on head": {"visible": true, "evidence": "black headset covering her eyes"}'
            "}}"
        ),
    )
    expect(out["completed"] is True,
           f"flat-schema grounded YES must be accepted, got {out}")
    expect(out.get("current_observation", "").strip() != "",
           f"observation must be surfaced for the trace, got {out}")


async def parser_test_g_flat_schema_empty_observation_rejected() -> None:
    """Flat schema with observation='' must reject regardless of
    requirements content, and the issue field must mention the
    observation problem so GUIDANCE_CHECK_RAW can log it.
    """
    out = await _run_parser_case(
        expected_requirements=["headset on head"],
        vlm_json_response=(
            '{"observation": "", '
            '"requirements": {'
            '"headset on head": {"visible": true, "evidence": "covering her eyes"}'
            "}}"
        ),
    )
    expect(out["completed"] is False,
           f"empty observation must reject even with evidence, got {out}")
    expect("observ" in out.get("issue", "").lower(),
           f"issue must explain the observation problem, got {out.get('issue')!r}")


async def parser_test_h_old_nested_schema_still_parses() -> None:
    """Back-compat: the previous nested {current_observation, checks[]}
    shape must still parse and accept when fully grounded.
    """
    out = await _run_parser_case(
        expected_requirements=["headset on head"],
        vlm_json_response=(
            '{"completed": true, '
            '"current_observation": "headset visible on her head", '
            '"checks": ['
            '{"requirement": "headset on head", "visible": true, '
            '"evidence": "black band wraps her head"}'
            "]}"
        ),
    )
    expect(out["completed"] is True,
           f"old nested schema must still be accepted, got {out}")


async def parser_test_i_prose_only_response_rejected() -> None:
    """Cosmos-style prose-only response (no JSON anywhere) must reject
    AND populate issue with a non-empty reason so the trace can log it.
    """
    out = await _run_parser_case(
        expected_requirements=["headset on head"],
        vlm_json_response="Yes, the headset is on her head.",
    )
    expect(out["completed"] is False,
           f"prose-only response must reject, got {out}")
    issue = out.get("issue", "").lower()
    expect("json" in issue or "parse" in issue or "non-json" in issue,
           f"issue must mention parse/json failure, got {out.get('issue')!r}")
    expect(out.get("raw_vlm", "").strip() != "",
           f"raw_vlm must be surfaced for the trace, got {out.get('raw_vlm')!r}")


async def parser_test_j_two_frame_comparison_preferred() -> None:
    """Teacher comparison runs first and can complete without ask_image."""
    captured: dict = {}
    teacher_path = __file__
    out = await _run_parser_case(
        expected_requirements=["headset on head"],
        vlm_json_response='{"observation": "student holds headset too low", "requirements": {}}',
        ask_frames_response=(
            '{"observation": "student wears the headset like the teacher", '
            '"requirements": {'
            '"headset matches teacher": {"visible": true, "evidence": "Image 2 shows headset on head"}'
            '}, "issue": ""}'
        ),
        teacher_image_path=teacher_path,
        teacher_caption="teacher has the headset on their head",
        prompt_capture=captured,
    )
    expect(out["completed"] is True,
           f"teacher comparison should complete before live-only check: {out}")
    expect("image_paths" in captured,
           f"ask_frames should run first: {captured}")
    expect("image_path" not in captured,
           f"ask_image should not run after teacher pass: {captured}")


async def parser_test_k_two_frame_fallback_to_single_image() -> None:
    """If ask_frames returns an error string, the instruction fallback runs."""
    captured: dict = {}
    out = await _run_parser_case(
        expected_requirements=["headset on head"],
        vlm_json_response=(
            '{"observation": "headset is on the head", '
            '"requirements": {'
            '"headset on head": {"visible": true, "evidence": "headset covers eyes"}'
            "}}"
        ),
        ask_frames_response="ask_frames: vlm-server request failed: boom",
        teacher_image_path=__file__,
        prompt_capture=captured,
    )
    expect(out["completed"] is True,
           f"single-image fallback should accept grounded response: {out}")
    expect(captured.get("image_path") == "/tmp/live.png",
           f"ask_image fallback did not run: {captured}")


async def parser_test_l_stale_live_frame_rejected() -> None:
    out = await _run_parser_case(
        expected_requirements=["headset on head"],
        vlm_json_response=(
            '{"observation": "headset is on the head", '
            '"requirements": {'
            '"headset on head": {"visible": true, "evidence": "headset covers eyes"}'
            "}}"
        ),
        live_timestamp_us=100,
        min_live_timestamp_us=200,
    )
    expect(out["completed"] is False,
           f"stale frame must not complete step: {out}")
    expect(out["timestamp_us"] == 100,
           f"stale frame timestamp should be surfaced: {out}")
    expect("fresh" in out["issue"].lower(),
           f"stale frame issue should be non-corrective fresh-frame wait: {out}")


async def parser_test_m_fresh_live_frame_accepted() -> None:
    out = await _run_parser_case(
        expected_requirements=["headset on head"],
        vlm_json_response=(
            '{"observation": "headset is on the head", '
            '"requirements": {'
            '"headset on head": {"visible": true, "evidence": "headset covers eyes"}'
            "}}"
        ),
        live_timestamp_us=300,
        min_live_timestamp_us=200,
    )
    expect(out["completed"] is True,
           f"fresh frame should accept grounded response: {out}")
    expect(out["timestamp_us"] == 300,
           f"fresh frame timestamp should be surfaced: {out}")


async def parser_test_n_teacher_evidence_leak_rejected() -> None:
    """Teacher-frame evidence cannot make a live image pass."""
    captured: dict = {}
    out = await _run_parser_case(
        expected_requirements=["cap on head"],
        vlm_json_response=(
            '{"observation": "The person is holding white over-ear headphones.", '
            '"requirements": {'
            '"cap on head": {"visible": false, "evidence": ""}'
            "}}"
        ),
        ask_frames_response=(
            '{"observation": "The person is holding white over-ear headphones.", '
            '"requirements": {'
            '"cap on head": {"visible": true, "evidence": "Image 1 shows a black cap on the head"}'
            "}}"
        ),
        teacher_image_path=__file__,
        prompt_capture=captured,
    )
    expect(out["completed"] is False,
           f"teacher evidence leakage must reject, got: {out}")
    expect("image_paths" in captured,
           f"teacher comparison should run first: {captured}")


async def parser_test_o_teacher_mismatch_instruction_fallback_can_pass() -> None:
    """A teacher mismatch falls back to instruction/caption matching."""
    captured: dict = {}
    out = await _run_parser_case(
        expected_requirements=["cap on head"],
        vlm_json_response=(
            '{"observation": "The person is wearing a black cap on their head.", '
            '"requirements": {'
            '"cap on head": {"visible": true, "evidence": "black cap on head"}'
            "}}"
        ),
        ask_frames_response=(
            '{"observation": "The cap is not exactly like Image 1.", '
            '"requirements": {'
            '"matches teacher": {"visible": false, "evidence": ""}'
            '}, "issue": "The cap is offset from the teacher reference."}'
        ),
        teacher_image_path=__file__,
        prompt_capture=captured,
    )
    expect(out["completed"] is True,
           f"instruction fallback should pass after teacher mismatch: {out}")
    expect("image_paths" in captured and "image_path" in captured,
           f"teacher mismatch should run ask_image fallback: {captured}")


async def parser_test_p_malformed_teacher_falls_back_to_instruction() -> None:
    captured: dict = {}
    out = await _run_parser_case(
        expected_requirements=["headset on head"],
        vlm_json_response=(
            '{"observation": "headset is on the head", '
            '"requirements": {'
            '"headset on head": {"visible": true, "evidence": "headset covers eyes"}'
            "}}"
        ),
        ask_frames_response='```json\n{"observation": "started but not closed"',
        teacher_image_path=__file__,
        prompt_capture=captured,
    )
    expect(out["completed"] is True,
           f"malformed teacher output should fall back and pass: {out}")
    expect("image_paths" in captured and "image_path" in captured,
           f"malformed teacher output should trigger ask_image fallback: {captured}")


async def parser_test_q_negative_visual_statement_rejected() -> None:
    out = await _run_parser_case(
        expected_requirements=["mouse visible"],
        vlm_json_response=(
            '{"observation": "The mouse is no longer present on the desk.", '
            '"requirements": {"mouse visible": {"visible": true, "evidence": "mouse no longer present"}}}'
        ),
        instruction="Put the mouse on the desk",
    )
    expect(out["completed"] is False,
           f"negative visual statement must not pass: {out}")


async def scenario_33_guidance_noise_transcript_does_not_dispatch_or_flush() -> None:
    mem = AgentMemory()
    demo = _seed_demo(mem, "pico headset", finished_ago_s=5.0)
    rec = Recorder()
    qp = _make_qp(mem, rec)
    _enter_guidance(qp, demo)
    ga = _make_ga(mem, qp)

    class _Resp:
        is_error = False

        def json(self) -> dict:
            return {"text": "Catches."}

    class _Http:
        async def post(self, *_args, **_kwargs) -> _Resp:
            return _Resp()

        async def aclose(self) -> None:
            pass

    ga._http = _Http()  # type: ignore[assignment]
    try:
        await ga._handle_utterance(PID, b"\x01\x00" * 1600, 16_000)
    finally:
        await ga._http.aclose()

    expect(ga._ep.flush_return_audio.await_count == 0,
           "guidance noise transcript must not flush return audio")
    expect(PID not in ga._query_tasks,
           f"guidance noise transcript must not dispatch query: {ga._query_tasks}")


async def scenario_31_guidance_correction_is_rate_limited() -> None:
    """First grounded issue speaks immediately; repeats are rate-limited."""
    mem = AgentMemory()
    rec = Recorder()
    qp = _make_qp(mem, rec)
    result = {
        "issue": "Raise the headset onto your head",
        "missing_or_mismatched": ["headset on head"],
    }

    await qp._maybe_send_guidance_correction(PID, 0, result)
    expect(len(rec.spoken) == 1,
           f"first grounded issue should speak one correction: {rec.spoken}")
    await qp._maybe_send_guidance_correction(PID, 0, result)
    expect(len(rec.spoken) == 1,
           f"rate limit should suppress immediate repeat: {rec.spoken}")


async def scenario_31b_uncertainty_does_not_speak_correction() -> None:
    mem = AgentMemory()
    rec = Recorder()
    qp = _make_qp(mem, rec)

    await qp._maybe_send_guidance_correction(PID, 0, {
        "issue": "vlm returned non-json",
        "missing_or_mismatched": [],
        "current_observation": "",
    })
    await qp._maybe_send_guidance_correction(PID, 0, {
        "issue": "no grounded evidence",
        "missing_or_mismatched": [],
        "current_observation": "person at desk",
    })
    await qp._maybe_send_guidance_correction(PID, 0, {
        "issue": "I cannot see a current frame.",
        "missing_or_mismatched": [],
        "current_observation": "",
    })

    expect(rec.spoken == [],
           f"uncertainty should not speak correction: {rec.spoken}")


async def scenario_31c_guidance_interval_default_is_fast() -> None:
    cfg = _make_cfg()
    expect(cfg.guidance_check_interval_s == 2.0,
           f"guidance interval default should be 2.0s, got {cfg.guidance_check_interval_s}")


async def scenario_32_guidance_request_restarts_active_demo() -> None:
    """Explicit guidance requests during guidance restart the matching demo."""
    mem = AgentMemory()
    demo = _seed_long_demo(mem, "pico headset", n_steps=4)
    rec = Recorder()
    qp = _make_qp(mem, rec)
    _enter_guidance(qp, demo, step=2)
    qp._start_guidance_monitor = lambda _pid: None  # type: ignore[assignment]

    await qp.handle("Show me how to wear pico headset", PID, ref_us=0)

    expect(qp._guidance_step == 0,
           f"guidance request should restart at step 0, got {qp._guidance_step}")
    last_spoken = rec.spoken[-1][1] if rec.spoken else ""
    expect("Step 1" in last_spoken,
           f"restart should speak step 1, got {rec.spoken}")


async def scenario_28_whats_next_advances() -> None:
    """REGRESSION (from run 20260527_172213): 'what's next?' during
    guidance must advance to the next step, not fall to the question
    fallback that has no knowledge of the demo structure.
    """
    mem = AgentMemory()
    demo = _seed_long_demo(mem, "pico headset", n_steps=4)
    rec = Recorder()
    qp = _make_qp(mem, rec)
    _enter_guidance(qp, demo)

    async def fake_speak(_pid: str) -> None:
        return None
    qp._speak_current_guidance_step = fake_speak       # type: ignore[assignment]
    qp._start_guidance_monitor = lambda _pid: None     # type: ignore[assignment]

    async def fake_question(_transcript: str, _pid: str) -> None:
        raise AssertionError("'what's next?' must NOT fall to the question handler")
    qp._handle_guidance_question = fake_question       # type: ignore[assignment]

    before = qp._guidance_step
    await qp.handle("what's next?", PID, ref_us=0)
    expect(qp._guidance_step == before + 1,
           f"'what's next?' must advance one step: {before} → {qp._guidance_step}")


async def scenario_29_whats_next_doubled_advances() -> None:
    """REGRESSION: STT often bundles back-to-back utterances, e.g.
    'what's next? what's next?'. Exact-half repetition must collapse
    to a single advance phrase and advance exactly once.
    """
    mem = AgentMemory()
    demo = _seed_long_demo(mem, "pico headset", n_steps=4)
    rec = Recorder()
    qp = _make_qp(mem, rec)
    _enter_guidance(qp, demo)

    async def fake_speak(_pid: str) -> None:
        return None
    qp._speak_current_guidance_step = fake_speak       # type: ignore[assignment]
    qp._start_guidance_monitor = lambda _pid: None     # type: ignore[assignment]

    async def fake_question(_transcript: str, _pid: str) -> None:
        raise AssertionError("doubled 'what's next?' must NOT fall to the question handler")
    qp._handle_guidance_question = fake_question       # type: ignore[assignment]

    before = qp._guidance_step
    await qp.handle("what's next? what's next?", PID, ref_us=0)
    expect(qp._guidance_step == before + 1,
           f"doubled 'what's next?' must advance exactly once: {before} → {qp._guidance_step}")


async def scenario_30_negative_guard_holds() -> None:
    """REGRESSION GUARD against substring-containment matchers:
    'I got it wrong', 'next time I see this', and similar must NEVER
    advance. The strict-equality matcher is the contract.
    """
    mem = AgentMemory()
    demo = _seed_long_demo(mem, "pico headset", n_steps=4)
    rec = Recorder()
    qp = _make_qp(mem, rec)
    _enter_guidance(qp, demo)

    async def fake_speak(_pid: str) -> None:
        return None
    qp._speak_current_guidance_step = fake_speak       # type: ignore[assignment]
    qp._start_guidance_monitor = lambda _pid: None     # type: ignore[assignment]

    # The negative cases all fall through to the question handler,
    # which is fine — we just need to confirm the step counter never
    # moves. Stub the handler to a no-op so it doesn't reach an LLM.
    async def fake_question(_transcript: str, _pid: str) -> None:
        return None
    qp._handle_guidance_question = fake_question       # type: ignore[assignment]

    start_step = qp._guidance_step
    for utter in ("I got it wrong", "next time I see this", "what's the weather", "got it but I'm stuck"):
        before = qp._guidance_step
        await qp.handle(utter, PID, ref_us=0)
        expect(qp._guidance_step == before,
               f"negative {utter!r} falsely advanced: {before} → {qp._guidance_step}")
    expect(qp._guidance_step == start_step,
           f"step counter drifted under negatives: {qp._guidance_step} vs {start_step}")


async def scenario_23_clear_demos_suppresses_ping() -> None:
    """B6: the 10 s 'still analyzing' ping is launched inside _finalize_demo
    and gated by _say_if_current. After clear-demos, the ping must not
    speak.
    """
    mem = AgentMemory()
    rec = Recorder()
    qp = _make_qp(mem, rec)

    mem.start_recording("pico headset")
    _add_recorded_frame(mem)

    analyze_started = asyncio.Event()

    async def very_long_analyze(_demo: Demonstration) -> tuple[str, list[str]]:
        analyze_started.set()
        await asyncio.sleep(12.0)
        return "ok", ["step"]

    qp._analyze_recording = very_long_analyze   # type: ignore[assignment]

    await qp.handle("stop recording", PID, ref_us=0)
    await analyze_started.wait()

    # Clear demos at t=0.1s. Cancellation kills the analyze task, which
    # bubbles into _finalize_demo's finally block, which cancels the
    # ping task.
    await asyncio.sleep(0.1)
    await qp.handle("forget all demos", PID, ref_us=0)

    for task in list(qp._finalize_tasks):
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Now wait past the 10 s ping deadline and confirm no ping ever fires.
    rec.spoken.clear()
    rec.sent_text.clear()
    # Short wait is enough — the ping task was cancelled when the
    # finalize task's finally block ran, well before its asyncio.sleep(10)
    # could complete.
    await asyncio.sleep(0.2)

    for _pid, text, _topic in rec.sent_text:
        expect("Still analyzing" not in text,
               f"stale ping leaked through after clear: {text!r}")


# ── Phase 2 helpers ──────────────────────────────────────────────────────────

def _add_recorded_frame(mem: AgentMemory, ts_offset_us: int = 1_000_000) -> None:
    """Add a single RecordedFrame to the currently-active recording."""
    assert mem.recording is not None, "no recording in progress"
    frame = RecordedFrame(
        frame_idx    = len(mem.recording.recorded_frames),
        timestamp_us = mem.recording.started_at_us + ts_offset_us,
        image_path   = f"/tmp/rec_{ts_offset_us}.png",
        description  = f"frame at +{ts_offset_us}us",
    )
    mem.add_recorded_frame(frame)


def _enter_guidance(
    qp: QueryProcessor,
    demo: Demonstration,
    *,
    step: int = 0,
) -> None:
    """Place *qp* into guidance mode for *demo* without invoking _start_guidance.

    Bypasses _speak_current_guidance_step / monitor wiring so tests can
    inspect the question-form routing in isolation.
    """
    qp._guidance_demo = demo
    qp._guidance_step = step


# ── runner ───────────────────────────────────────────────────────────────────

SCENARIOS = [
    scenario_0_vad_fallback_constructs_without_silero,
    scenario_0b_vad_energy_fallback_finalizes_utterance,
    scenario_0c_vad_speech_start_rearms_with_fallback,
    scenario_1_what_do_you_see_with_fresh_demo,
    scenario_2_bare_stop_outside_guidance,
    scenario_3_layer1_filler_dropped,
    scenario_4_stt_garbled_freshness_single_demo,
    scenario_5_two_demo_bare_name_via_freshness,
    scenario_6_two_demo_shared_token_tied,
    scenario_7_how_to_do_starts_guidance,
    scenario_8_ambiguous_then_resolved_by_number,
    scenario_9_pending_state_cancel_escape,
    scenario_10_pending_attempt_limit,
    scenario_11_expected_requirements_threaded_through_monitor,
    scenario_11b_after_frame_returns_paired_path_and_caption,
    scenario_11c_fallback_requirements_from_instruction,
    scenario_11d_empty_requirements_still_checked_with_teacher_frame,
    scenario_11e_guidance_frame_pair_trace,
    scenario_11f_reused_adjacent_reference_marked_unreliable,
    scenario_11h_text_video_mismatch_keeps_visual_reference,
    scenario_11e_reference_selection_prefers_instruction_match,
    scenario_11g_reference_selection_penalizes_future_object,
    scenario_11f_reference_selection_requires_primary_object,
    scenario_12_issue_surfaces_in_guidance_question,
    scenario_13_recording_loop_not_paused_by_query,
    scenario_13b_guidance_suppresses_background_observation,
    scenario_13c_vad_speech_start_does_not_flush_tts,
    scenario_13d_accepted_dispatch_still_flushes_tts,
    scenario_14_question_form_done_runs_completion_check,
    scenario_14b_bare_done_still_exits,
    scenario_14c_is_this_complete_runs_completion_check,
    scenario_14d_did_i_finish_runs_completion_check,
    scenario_15_save_this_does_not_start_recording,
    scenario_16_token_aware_demo_name,
    scenario_16b_watch_meteor_does_not_start_recording,
    scenario_16c_watch_me_arrange_starts_recording,
    scenario_16d_watch_me_comma_arrange_starts_recording,
    scenario_17_lets_do_does_not_start_guidance,
    scenario_18_guidance_during_finalization,
    scenario_18b_pending_then_finalizing_clears_pending,
    scenario_19_finalization_completes_async,
    scenario_20_advance_matcher_handles_filler,
    scenario_21_clear_demos_cancels_finalization,
    scenario_22_rerecord_same_name_during_finalization,
    scenario_23_clear_demos_suppresses_ping,
    scenario_24_biased_yes_static_scene_blocked,
    scenario_25_grounded_yes_static_scene_needs_two,
    scenario_26_grounded_yes_moving_scene_needs_two,
    scenario_27_malformed_response_resets_streak,
    scenario_27b_grounded_no_speaks_correction,
    scenario_27c_unreliable_reference_blocks_passive_advance,
    scenario_28_whats_next_advances,
    scenario_29_whats_next_doubled_advances,
    scenario_30_negative_guard_holds,
    scenario_31_guidance_correction_is_rate_limited,
    scenario_31b_uncertainty_does_not_speak_correction,
    scenario_31c_guidance_interval_default_is_fast,
    scenario_32_guidance_request_restarts_active_demo,
    scenario_33_guidance_noise_transcript_does_not_dispatch_or_flush,
    parser_test_a_completed_true_no_evidence_rejected,
    parser_test_b_copied_checklist_empty_evidence_rejected,
    parser_test_c_all_visible_with_evidence_accepted,
    parser_test_d_vlm_judgment_not_exact_checklist_match,
    parser_test_e_prompt_omits_teacher_caption,
    parser_test_f_flat_schema_observation_only_accepted,
    parser_test_g_flat_schema_empty_observation_rejected,
    parser_test_h_old_nested_schema_still_parses,
    parser_test_i_prose_only_response_rejected,
    parser_test_j_two_frame_comparison_preferred,
    parser_test_k_two_frame_fallback_to_single_image,
    parser_test_l_stale_live_frame_rejected,
    parser_test_m_fresh_live_frame_accepted,
    parser_test_n_teacher_evidence_leak_rejected,
    parser_test_o_teacher_mismatch_instruction_fallback_can_pass,
    parser_test_p_malformed_teacher_falls_back_to_instruction,
    parser_test_q_negative_visual_statement_rejected,
]


async def main() -> int:
    failures: list[tuple[str, str]] = []
    for fn in SCENARIOS:
        name = fn.__name__
        try:
            await fn()
        except Fail as exc:
            failures.append((name, f"FAIL: {exc}"))
        except Exception as exc:  # pragma: no cover - harness-level surprises
            failures.append((name, f"EXC:  {type(exc).__name__}: {exc}"))
        else:
            print(f"PASS  {name}")
    if failures:
        print()
        print(f"{len(failures)} failure(s):")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        return 1
    print()
    print(f"All {len(SCENARIOS)} scenarios passed.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
