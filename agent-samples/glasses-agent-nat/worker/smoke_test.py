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
from memory import AgentMemory, Demonstration, DemoStep
from processors import QueryProcessor


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
        guidance_check_interval_s   = 4.0,
        guidance_freshness_window_s = freshness_s,
        silence_duration            = 0.8,
        min_speech                  = 0.3,
        silero_threshold            = 0.5,
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
    """
    fake_runtime = mock.MagicMock(name="NatRuntime")
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


# ── runner ───────────────────────────────────────────────────────────────────

SCENARIOS = [
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
