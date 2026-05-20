# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
QueryProcessor — handles all user utterances for the glasses LangChain agent.

Flow per utterance:
  1. Demo-mode detection (start / end / guidance request / guidance advance).
  2. _quick_ack() — fast Llama-Nemotron call: spoken ack + think=True/False.
  3. _agentic_loop() — LangChain agent loop over the existing MCP tools:
       - Pre-fetch latest frame concurrently while building runtime context.
       - Context = AgentMemory snapshot + frame path via LangChain middleware.
       - Tools: ask_image (vlm-mcp), get_frame_from_time / get_video_stats /
                list_recorded_participants (video-mcp).
       - Finish: LLM text response → TTS + data message to participant.

Demonstration detection phrases:
  Start:    "let me show you", "watch what i", "i'll demonstrate",
            "start recording", "remember this", "watch me", "watch how i"
  End:      "that's it", "done", "stop recording", "end demonstration",
            "finished", "end recording"
  Guidance: "how do i", "walk me through", "show me how",
            "teach me how to", "guide me through", "step by step"
  Advance:  "next", "continue", "got it", "okay next"
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import string
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from fastmcp import Client as McpClient
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

from config import WorkerConfig
from memory import (
    AgentMemory,
    Demonstration,
    DemoStep,
    MemorySnapshot,
    Observation,
    VoiceNote,
    parse_mcp_result,
)

log = logging.getLogger("glasses_agent_langchain.processors")

_trace_log = logging.getLogger("glasses_agent_langchain.trace")

_MAX_LOOP = 8

# ── demo detection phrases ────────────────────────────────────────────────────

_DEMO_START_PHRASES = (
    # Explicit start commands
    "start recording",
    "begin recording",
    "start demo",
    "start a demo",
    "begin demo",
    # Natural "record X" phrasing
    "record demo",
    "record a demo",
    "record steps",
    "record how",
    "record me",
    # Capture / save variants
    "capture a demo",
    "capture how",
    "save these steps",
    "save this",
    # Show / demonstrate
    "let me show you",
    "watch what i",
    "watch me",
    "watch how i",
    "i'll demonstrate",
    "demonstrate how",
    # Remember variants
    "remember this",
    "remember how",
    "remember these",
    "remember steps",
    "can you remember",
)

_DEMO_END_PHRASES = (
    "that's it",
    "stop recording",
    "finish recording",
    "end demonstration",
    "finished",
    "end recording",
    # "done" is short — match only if it's a standalone word to avoid false positives
)

_GUIDANCE_PHRASES = (
    "how do i",
    "walk me through",
    "show me how",
    "teach me how to",
    "guide me through",
    "step by step",
)

_GUIDANCE_ADVANCE_PHRASES = (
    "next",
    "continue",
    "got it",
    "okay next",
    "ok next",
    "next step",
    "go on",
)

_GUIDANCE_DONE_PHRASES = (
    "done",
    "finished",
    "all done",
    "that's it",
    "complete",
    "stop guiding",
    "stop guide",
    "stop guidance",
    "exit guidance",
    "exit",
    "stop",
    "quit",
    "cancel",
)

_TASK_NUMBER_WORDS = {
    "one": 1,
    "first": 1,
    "two": 2,
    "second": 2,
    "three": 3,
    "third": 3,
    "four": 4,
    "fourth": 4,
    "five": 5,
    "fifth": 5,
    "six": 6,
    "sixth": 6,
    "seven": 7,
    "seventh": 7,
    "eight": 8,
    "eighth": 8,
    "nine": 9,
    "ninth": 9,
    "ten": 10,
    "tenth": 10,
}

_AGENT_RESPONSE_TOPIC = "agent.response"
_AGENT_PROGRESS_TOPIC = "agent.progress"
_HISTORY_TURNS = 4

_QUICK_ACK_SCHEMA = {
    "name": "quick_ack_decision",
    "description": "Return a spoken acknowledgement and whether the request needs deeper thinking.",
    "parameters": {
        "type": "object",
        "properties": {
            "ack": {
                "type": "string",
                "description": "A short natural spoken acknowledgement, 3-6 words.",
            },
            "think": {
                "type": "boolean",
                "description": "True if the request needs the large reasoning/tool agent.",
            },
        },
        "required": ["ack", "think"],
    },
}

_DEMO_ANALYSIS_SCHEMA = {
    "name": "demo_analysis",
    "description": "Summarize a recorded demonstration into voice-grounded instructions.",
    "parameters": {
        "type": "object",
        "properties": {
            "overview": {
                "type": "string",
                "description": "One sentence describing the demonstrated procedure.",
            },
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Chronological action steps grounded in voice notes.",
            },
        },
        "required": ["overview", "steps"],
    },
}

_GUIDANCE_RESPONSE_SCHEMA = {
    "name": "guidance_response",
    "description": "Answer a user's question while they are following a guidance step.",
    "parameters": {
        "type": "object",
        "properties": {
            "reply": {
                "type": "string",
                "description": "A direct 1-2 sentence spoken answer.",
            },
        },
        "required": ["reply"],
    },
}


def _now_us() -> int:
    return time.time_ns() // 1_000


# ── QueryProcessor ────────────────────────────────────────────────────────────

SendTextCb = Callable[[str, str, str], Awaitable[None]]  # (pid, text, topic)


@dataclass(frozen=True)
class GlassesRuntimeContext:
    memory_snapshot:      MemorySnapshot
    pid:                  str
    ref_us:               int
    needs_thinking:       bool = False
    recording_name:       str | None = None
    guidance_instruction: str | None = None
    guidance_step:        int = 0
    guidance_total:       int = 0
    frame_path:           str | None = None
    frame_description:    str | None = None


class GuidanceTaskResolver:
    """Pure helpers for deterministic task-number and task-name matching."""

    @staticmethod
    def extract_task_index(text: str) -> int | None:
        lower = text.lower()
        patterns = (
            r"\btask\s*(\d+)\b",
            r"\bnumber\s*(\d+)\b",
            r"\btask\s+([a-z]+)\b",
            r"\b([a-z]+)\s+task\b",
        )
        for pattern in patterns:
            match = re.search(pattern, lower)
            if not match:
                continue
            value = match.group(1)
            if value.isdigit():
                return int(value)
            if value in _TASK_NUMBER_WORDS:
                return _TASK_NUMBER_WORDS[value]
        stripped = lower.strip()
        if stripped.isdigit():
            return int(stripped)
        return _TASK_NUMBER_WORDS.get(stripped)

    @staticmethod
    def strip_task_number_prefix(text: str) -> str:
        text = re.sub(r"^\s*for\s+", "", text.strip())
        text = re.sub(r"^\s*task\s+\d+\s+", "", text)
        text = re.sub(
            r"^\s*task\s+(" + "|".join(_TASK_NUMBER_WORDS) + r")\s+",
            "",
            text,
        )
        text = re.sub(r"^\s*task\s+", "", text)
        return text.strip()


def _format_glasses_runtime_context(ctx: GlassesRuntimeContext) -> str:
    parts = [ctx.memory_snapshot.to_system_message().content]

    if ctx.recording_name:
        parts.append(f"[Recording active — demo: {ctx.recording_name!r}]")

    if ctx.guidance_instruction:
        parts.append(
            f"[Guidance mode — step {ctx.guidance_step} of {ctx.guidance_total}]\n"
            f"Current instruction: {ctx.guidance_instruction}\n"
            "Help the user complete THIS step. Be concise and direct."
        )

    if ctx.pid:
        parts.append(f"Participant: {ctx.pid}")
    if ctx.ref_us:
        parts.append(f"Reference time (when user spoke): {ctx.ref_us} µs")

    if ctx.frame_path:
        if ctx.frame_description:
            parts.append(
                f"[Current camera view — fresh as of this turn]\n{ctx.frame_description}\n"
                f"(image path for ask_image: {ctx.frame_path})"
            )
        else:
            parts.append(f"[Latest camera frame]\n{ctx.frame_path}")

    return "\n\n".join(parts)


def _trim_to_recent_turns(messages: list[Any]) -> list[Any] | None:
    human_seen = 0
    keep_from = 0
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], HumanMessage):
            human_seen += 1
            if human_seen == _HISTORY_TURNS:
                keep_from = idx
                break
    if human_seen < _HISTORY_TURNS or keep_from == 0:
        return None
    return [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages[keep_from:]]


class _GlassesAgentMiddleware(AgentMiddleware[AgentState, GlassesRuntimeContext]):
    def __init__(self, model_factory: Callable[[bool], ChatOpenAI]) -> None:
        self._model_factory = model_factory

    def before_model(
        self, state: AgentState, runtime: Runtime[GlassesRuntimeContext]
    ) -> dict[str, Any] | None:
        trimmed = _trim_to_recent_turns(state["messages"])
        if trimmed is None:
            return None
        return {"messages": trimmed}

    async def awrap_model_call(
        self,
        request: ModelRequest[GlassesRuntimeContext],
        handler: Callable[[ModelRequest[GlassesRuntimeContext]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        runtime_context = request.runtime.context
        needs_thinking = bool(runtime_context and runtime_context.needs_thinking)

        base_prompt = request.system_prompt or ""
        if needs_thinking:
            base_prompt = (
                "Use your private <think> block to reason through the question. "
                "NEVER output these thoughts in your final response - "
                "only output a concise 1-3 sentence answer for the wearer.\n\n"
                + base_prompt
            )

        if runtime_context is not None:
            base_prompt = (
                f"{base_prompt}\n\n"
                "[Runtime XR context — use this before calling tools]\n"
                f"{_format_glasses_runtime_context(runtime_context)}"
            )

        request = request.override(
            model=self._model_factory(needs_thinking),
            system_message=SystemMessage(content=base_prompt),
        )
        return await handler(request)


class QueryProcessor:
    """Handles text utterances: demo detection → quick-ack → LangChain loop."""

    def __init__(
        self,
        cfg:          WorkerConfig,
        memory:       AgentMemory,
        vlm_client:   McpClient,
        video_client: McpClient,
        system_prompt: str,
        langchain_tools: list[BaseTool],
        *,
        send_text:    SendTextCb,
        say:          Callable[[str, str], Awaitable[None]],   # (pid, text)
    ) -> None:
        self._cfg          = cfg
        self._memory       = memory
        self._vlm          = vlm_client
        self._video        = video_client
        self._system_prompt = system_prompt
        self._send_text    = send_text
        self._say          = say
        self._last_turn_by_pid: dict[str, tuple[str, str]] = {}

        self._langchain_tools:  list[BaseTool] = langchain_tools
        self._langchain_agent_graph: Any | None = None
        self._checkpointer = InMemorySaver()

        # Guidance mode state.
        self._guidance_demo: Demonstration | None = None
        self._guidance_step: int = 0
        self._guidance_monitor_task: asyncio.Task | None = None
        self._guidance_advancing:    bool                = False
        self._guidance_step_obs_baseline:   int = 0
        self._guidance_monitor_idle_cycles: int = 0
        self._guidance_consecutive_yes:     int = 0  # consecutive 2-frame YES responses
        self._pending_guidance_query_by_pid: dict[str, str] = {}

    async def handle(
        self,
        transcript: str,
        pid:        str,
        ref_us:     int,
        *,
        source:      str = "voice",
    ) -> None:
        """Entry point: dispatch one transcribed utterance."""
        text  = transcript.strip()
        lower = text.lower().strip().rstrip(string.punctuation)
        if not lower:
            return

        _trace_log.info("USER  %s", text)

        # ── forget-all-demos ─────────────────────────────────────────────────
        if any(p in lower for p in ("forget all demos", "clear all demos",
                                    "delete all demos", "forget demos",
                                    "clear demos", "reset demos")):
            n = self._memory.clear_demonstrations()
            response = f"Done — {n} demonstration{'s' if n != 1 else ''} cleared."
            await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
            await self._say(pid, response)
            _trace_log.info("DEMOS_CLEARED  count=%d", n)
            return

        pending_key = pid or "default"
        if pending_key in self._pending_guidance_query_by_pid:
            demo = self._resolve_guidance_demo(lower)
            if demo is not None:
                self._pending_guidance_query_by_pid.pop(pending_key, None)
                await self._start_guidance_demo(demo, pid)
            else:
                await self._ask_for_task_number(pid)
            return

        # ── demo-end detection ───────────────────────────────────────────────
        if self._memory.recording and self._is_demo_end(lower):
            # If the utterance contains narration before the stop phrase, save it.
            for phrase in _DEMO_END_PHRASES:
                if phrase in lower:
                    narration = text[:lower.index(phrase)].strip().rstrip(",.")
                    if len(narration) > 3:
                        self._memory.add_voice_note(
                            VoiceNote(timestamp_us=ref_us, text=narration)
                        )
                        _trace_log.info("VOICE_NOTE(pre-stop)  %s", narration[:80])
                    break
            await self._handle_demo_end(pid, ref_us)
            return

        # ── voice narration during recording ─────────────────────────────────
        if self._memory.recording is not None:
            if source != "voice":
                await self._answer_query_during_recording(text, pid, ref_us)
                return
            import os as _os, json as _json, re as _re
            note = VoiceNote(timestamp_us=ref_us, text=text)
            self._memory.add_voice_note(note)
            _trace_log.info("VOICE_NOTE  t=%d  %s", ref_us, text[:80])
            # Append to the same JSONL log as the frames so everything is on disk.
            rec      = self._memory.recording
            run_dir  = _os.environ.get("XR_RUN_DIR", "/tmp")
            safe     = _re.sub(r"[^a-zA-Z0-9_-]", "_", rec.name)[:40]
            log_dir  = _os.path.join(run_dir, "recordings")
            log_path = _os.path.join(log_dir, f"{safe}_{rec.started_at_us}.jsonl")
            try:
                _os.makedirs(log_dir, exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(_json.dumps({
                        "type":    "voice",
                        "ts_us":   ref_us,
                        "text":    text,
                    }) + "\n")
            except Exception as exc:
                log.warning("voice note log write failed: %s", exc)
            return

        # ── demo-start detection ─────────────────────────────────────────────
        demo_start = self._extract_demo_name(lower)
        if demo_start is not None:
            await self._handle_demo_start(demo_start, pid)
            return

        # ── guidance mode: owns ALL utterances — nothing falls through ────────
        if self._guidance_demo is not None:
            if self._is_guidance_done(lower):
                await self._finish_guidance(pid)
            elif self._is_guidance_advance(lower):
                await self._advance_guidance(pid)
            else:
                # Any other utterance (question, comment, noise) gets a brief
                # contextual reply focused on the current step.
                await self._handle_guidance_question(text, pid)
            return

        # ── guidance request detection ────────────────────────────────────────
        guidance_match = self._match_guidance_request(lower)
        if guidance_match is not None:
            await self._handle_guidance_request(guidance_match, pid)
            return

        # ── ordinary query ────────────────────────────────────────────────────
        try:
            ack, needs_thinking = await self._quick_ack(text, pid)
        except Exception:
            log.exception("quick-ack failed")
            ack, needs_thinking = "", False
        if not ack:
            ack, needs_thinking = self._fallback_ack(text)

        if ack:
            await self._send_text(pid, ack, _AGENT_PROGRESS_TOPIC)
            if needs_thinking:
                await self._say(pid, ack)

        try:
            response = await self._agentic_loop(
                text, pid, ref_us=ref_us, needs_thinking=needs_thinking
            )
        except Exception:
            log.exception("agentic loop failed")
            response = "Something went wrong — please try again."

        if response:
            self._last_turn_by_pid[pid or "default"] = (text, response)
            await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
            await self._say(pid, response)

    # ── demo detection helpers ────────────────────────────────────────────────

    def _is_demo_end(self, lower: str) -> bool:
        for phrase in _DEMO_END_PHRASES:
            if phrase in lower:
                return True
        # "done" alone (not part of a longer phrase)
        if lower.strip() == "done" or lower.startswith("done ") or lower.endswith(" done"):
            return True
        return False

    def _extract_demo_name(self, lower: str) -> str | None:
        """Return the demo name if a start phrase is detected, else None.

        Heuristic: the demo name is whatever comes after the start phrase.
        If nothing follows (bare trigger), use a timestamp-based name.
        """
        for phrase in _DEMO_START_PHRASES:
            if phrase in lower:
                # Everything after the triggering phrase is the name.
                after = lower.split(phrase, 1)[1].strip().rstrip(string.punctuation).strip()
                after = self._strip_task_number_prefix(after)
                if after and len(after) > 2:
                    return after
                # Bare trigger ("watch me") — generate a name from timestamp.
                ts = time.strftime("%H%M%S")
                return f"demo-{ts}"
        return None

    @staticmethod
    def _extract_task_index(text: str) -> int | None:
        return GuidanceTaskResolver.extract_task_index(text)

    @staticmethod
    def _strip_task_number_prefix(text: str) -> str:
        return GuidanceTaskResolver.strip_task_number_prefix(text)

    def _match_guidance_request(self, lower: str) -> str | None:
        """Return the user query if a guidance phrase is detected, else None."""
        for phrase in _GUIDANCE_PHRASES:
            if phrase in lower:
                return lower
        return None

    def _is_guidance_advance(self, lower: str) -> bool:
        for phrase in _GUIDANCE_ADVANCE_PHRASES:
            if lower.strip() == phrase or lower.strip().startswith(phrase):
                return True
        return False

    def _is_guidance_done(self, lower: str) -> bool:
        # Exact match for multi-word phrases; word-boundary match for single words
        # so "stop, stop" / "stop recording" / "okay stop" all exit guidance.
        words = set(lower.replace(",", " ").replace(".", " ").split())
        for phrase in _GUIDANCE_DONE_PHRASES:
            if lower.strip() == phrase:
                return True
            if " " not in phrase and phrase in words:
                return True
        return False

    async def _answer_query_during_recording(self, text: str, pid: str, ref_us: int) -> None:
        recording = self._memory.recording
        response_prefix = (
            f"[Recording active — demo: {recording.name!r}]\n"
            "Answer this typed question, but do not treat it as demo narration.\n\n"
            if recording else ""
        )
        ack, _ = self._fallback_ack(text)
        await self._send_text(pid, ack, _AGENT_PROGRESS_TOPIC)
        await self._say(pid, ack)
        try:
            response = await self._agentic_loop(
                response_prefix + text,
                pid,
                ref_us=ref_us,
                needs_thinking=True,
            )
        except Exception:
            log.exception("recording data query failed")
            response = "Something went wrong — please try again."
        if response:
            await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
            await self._say(pid, response)

    # ── demo actions ──────────────────────────────────────────────────────────

    async def _handle_demo_start(self, name: str, pid: str) -> None:
        self._memory.start_recording(name)
        task_index = self._memory.recording.task_index if self._memory.recording else 0
        response = (
            f"Okay, start recording task {task_index}: '{name}'. Go ahead and demonstrate — "
            "I'm watching your hands and will capture each step."
        )
        await self._send_text(pid, response, _AGENT_PROGRESS_TOPIC)
        await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
        await self._say(pid, response)
        log.info("demo start response  pid=%r  %s", pid, response)
        _trace_log.info("DEMO_START  task=%d  name=%s", task_index, name)

    async def _handle_demo_end(self, pid: str, ref_us: int) -> None:
        demo = self._memory.finish_recording()
        if demo is None:
            response = "No demonstration was being recorded."
            await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
            await self._say(pid, response)
            return
        if not demo.recorded_frames:
            response = (
                "Recording stopped but no camera frames were captured — "
                "is the camera on? Start the camera then try recording again."
            )
            await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
            await self._say(pid, response)
            return

        n_frames = len(demo.recorded_frames)
        _trace_log.info(
            "DEMO_END  task=%d  name=%s  frames=%d",
            demo.task_index,
            demo.name,
            n_frames,
        )

        await self._send_text(
            pid, f"Got it — {n_frames} frames captured. Analyzing now…",
            _AGENT_PROGRESS_TOPIC,
        )
        await self._say(pid, "Got it. Analyzing the recording now.")

        # Send a follow-up after 10 s so the user knows it's still working.
        async def _ping():
            await asyncio.sleep(10)
            await self._send_text(
                pid, "Still analyzing — reviewing key frames…",
                _AGENT_PROGRESS_TOPIC,
            )
        ping = asyncio.create_task(_ping())

        try:
            overview, instructions = await self._analyze_recording(demo)
        finally:
            ping.cancel()

        if not instructions:
            import os as _os, re as _re
            run_dir  = _os.environ.get("XR_RUN_DIR", "/tmp")
            safe     = _re.sub(r"[^a-zA-Z0-9_-]", "_", demo.name)[:40]
            log_path = f"{run_dir}/recordings/{safe}_{demo.started_at_us}.jsonl"
            _trace_log.info("DEMO_ANALYSIS_FAILED  log=%s", log_path)
            response = (
                f"Recorded {n_frames} frames but analysis failed. "
                f"Raw observations saved to: {log_path}"
            )
            await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
            await self._say(pid, "Recording saved but analysis failed.")
            return

        demo.summary      = overview
        demo.instructions = instructions

        # Create DemoStep objects and assign an "after" reference frame to each.
        # We want the frame that shows the COMPLETED state of each step (i.e.,
        # just before the next voice note starts), not the before-state.
        # Using the before-state as reference confuses the vlm-check because
        # the comparison question asks "has the step been done?" — if Image 1
        # shows the pre-step state, the VLM correctly says NO even when done.
        n       = len(instructions)
        notes   = demo.voice_notes  # sorted by timestamp (added in order)
        frames  = demo.recorded_frames

        def _after_frame(step_idx: int) -> str:
            """Frame just before the next voice note = completed state of step."""
            if not frames:
                return ""
            # Find the voice note that starts the NEXT step.
            next_note_ts = None
            if step_idx + 1 < len(notes):
                next_note_ts = notes[step_idx + 1].timestamp_us
            if next_note_ts is not None:
                # Latest frame before the next note.
                before = [f for f in frames if f.timestamp_us < next_note_ts]
                return before[-1].image_path if before else frames[-1].image_path
            # Last step: use the last recorded frame.
            return frames[-1].image_path

        span = max(demo.ended_at_us - demo.started_at_us, 1)
        for i, instr in enumerate(instructions):
            ts = demo.started_at_us + int(i / max(n - 1, 1) * span)
            demo.steps.append(DemoStep(
                step_number  = i + 1,
                timestamp_us = ts,
                description  = instr,
                image_path   = _after_frame(i),
            ))

        if demo.recorded_frames:
            _trace_log.info("STEP_FRAMES  %s",
                            " | ".join(
                                f"{s.step_number}:{s.image_path[-20:]}"
                                for s in demo.steps if s.image_path
                            ))

        _trace_log.info("DEMO_INSTRUCTIONS  %s",
                        " | ".join(f"{i+1}:{s[:40]}" for i, s in enumerate(instructions)))

        response = f"Saved task {demo.task_index} '{demo.name}' with {n} steps."
        if overview:
            response += f" {overview}"
        await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
        await self._say(
            pid,
            f"Saved task {demo.task_index}, '{demo.name}', with {n} steps.",
        )

    async def _analyze_recording(
        self, demo: Demonstration
    ) -> tuple[str, list[str]]:
        """LLM analysis with thinking on all recorded frames.

        The LLM receives all frame descriptions and can call ask_image on any
        recorded frame path to get additional detail.  Uses agent_llm_server
        (Nemotron-3-Nano-30B) with thinking enabled for best reasoning quality.

        Returns (overview, instructions_list) or ("", []) on failure.
        """
        import re as _re, os as _os
        frames = demo.recorded_frames
        if not frames:
            return "", []

        # (duplicate log removed — ANALYSIS_START is logged below with voice_notes count)

        # Subsample frames if too many, always include first and last.
        if len(frames) > 40:
            step  = len(frames) / 30.0
            idxs  = sorted({0, len(frames) - 1} |
                           {int(i * step) for i in range(1, 30)})
            shown = [frames[i] for i in idxs]
        else:
            shown = frames

        # Pre-filter filler voice notes before analysis.
        # Single-word reactions said at the start of a recording ("Next", "Okay",
        # "Yeah") are not steps. Removing them prevents the LLM from treating
        # them as step anchors and avoids duplicate/ghost steps.
        _FILLER = frozenset({
            "next", "okay", "ok", "yeah", "yes", "no", "and", "then",
            "um", "uh", "hmm", "right", "sure", "alright",
        })
        import string as _str
        def _is_filler(text: str) -> bool:
            words = [w.strip(_str.punctuation).lower() for w in text.split()]
            return all(w in _FILLER for w in words if w)

        meaningful_notes = [v for v in demo.voice_notes if not _is_filler(v.text)]

        # Build interleaved timeline: VLM frames + voice notes, sorted by time.
        t0 = demo.started_at_us
        timeline: list[tuple[int, str]] = []
        for f in shown:
            rel = (f.timestamp_us - t0) / 1_000_000
            timeline.append((
                f.timestamp_us,
                f"[Frame {f.frame_idx + 1}/{len(frames)} | +{rel:.1f}s]\n{f.description}",
            ))
        for v in meaningful_notes:
            rel = (v.timestamp_us - t0) / 1_000_000
            timeline.append((v.timestamp_us, f"[Voice +{rel:.1f}s] \"{v.text}\""))
        timeline.sort(key=lambda x: x[0])
        desc_block = "\n\n".join(entry for _, entry in timeline)

        has_voice = bool(meaningful_notes)
        _trace_log.info("ANALYSIS_START  demo=%r  frames=%d  voice_notes=%d",
                        demo.name, len(frames), len(demo.voice_notes))

        run_dir  = _os.environ.get("XR_RUN_DIR", "/tmp")
        safe     = _re.sub(r"[^a-zA-Z0-9_-]", "_", demo.name)[:40]
        log_path = f"{run_dir}/recordings/{safe}_{demo.started_at_us}.jsonl"

        if has_voice:
            voice_guidance = (
                "\n[Voice +Xs] entries are the user's spoken narration.\n"
                "[Frame N/M | +Xs] entries are VLM descriptions of what the camera saw.\n\n"
                "CRITICAL RULES:\n"
                "  - Voice notes are the ONLY source of steps. "
                "NEVER create a step from a frame description alone.\n"
                "  - Strip the '+Xs' timing prefix from all output.\n"
                "  - MERGE consecutive notes that together describe one action:\n"
                "    'Grab the glass.' + 'and put it to the right side.' "
                "→ ONE step: 'Grab the glass and place it to the right side.'\n"
                "    'Grab the knife.' + 'Put it to the left.' "
                "→ ONE step: 'Grab the knife and place it to the left.'\n"
                "    Any note starting with 'and', 'then', 'put it', 'place it' that "
                "refers to the previous note's object must be merged, not a new step.\n"
                "  - Each physical object should appear in at most one step unless "
                "explicitly picked up a second time.\n"
                "  - Use nearby frames only to add spatial detail to a voice-defined step.\n"
            )
        else:
            voice_guidance = ""

        system_content = (
            f"You are analyzing a recorded demonstration: {demo.name!r}\n\n"
            "Timeline is in ascending time order (smaller +Ns = earlier).\n"
            f"{voice_guidance}"
            "\nYOUR TASK:\n"
            "1. Steps MUST be in chronological order.\n"
            "2. Each step = one complete action. Merge action fragments.\n"
            "3. Output only steps grounded in voice notes.\n\n"
            "OUTPUT — a single JSON object, nothing else:\n"
            '{"overview": "one sentence", "steps": ["step 1", "step 2", ...]}'
        )

        user_content = (
            f"Demo: {demo.name!r}\n\n"
            f"Timeline ({len(timeline)} entries):\n\n"
            f"{desc_block}\n\n"
            "Return the structured overview and steps."
        )
        try:
            structured = self._agent_llm(
                needs_thinking=False,
                max_tokens=1024,
                timeout=90.0,
            ).with_structured_output(_DEMO_ANALYSIS_SCHEMA, include_raw=True)
            result = await asyncio.wait_for(
                structured.ainvoke([
                    SystemMessage(content=system_content),
                    HumanMessage(content=user_content),
                ]),
                timeout=90.0,
            )
            obj = result.get("parsed") if isinstance(result, dict) else result
            if isinstance(obj, dict):
                overview = str(obj.get("overview", "")).strip()
                steps = [
                    str(step).strip()
                    for step in obj.get("steps", [])
                    if str(step).strip()
                ]
                _trace_log.info(
                    "ANALYSIS_RESULT  overview=%s  steps=%d",
                    overview[:120],
                    len(steps),
                )
                return overview, steps
            raw = result.get("raw") if isinstance(result, dict) else None
            raw_text = self._message_text(raw) if raw is not None else ""
            log.warning("analysis structured output failed: %s", raw_text[:300])
        except Exception:
            log.exception("analysis LLM call failed")

        return "", []

    # ── guidance actions ──────────────────────────────────────────────────────

    async def _handle_guidance_request(self, query: str, pid: str) -> None:
        """Find a matching demo and enter guidance mode."""
        demo = self._resolve_guidance_demo(query)
        if demo is None:
            if not self._memory.list_demonstrations_with_indices():
                response = "I don't have any recorded demonstrations yet. Show me first!"
                await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
                await self._say(pid, response)
                return
            self._pending_guidance_query_by_pid[pid or "default"] = query
            await self._ask_for_task_number(pid)
            return

        if not demo.steps:
            response = f"The demonstration '{demo.name}' has no steps recorded."
            await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
            await self._say(pid, response)
            return

        await self._start_guidance_demo(demo, pid)

    def _resolve_guidance_demo(self, query: str) -> Demonstration | None:
        task_index = self._extract_task_index(query)
        if task_index is not None:
            return self._memory.get_demonstration_by_task_index(task_index)
        return self._memory.find_demonstration_fuzzy(query)

    async def _ask_for_task_number(self, pid: str) -> None:
        demos = self._memory.list_demonstrations_with_indices()
        if not demos:
            response = "I don't have any recorded demonstrations yet. Show me first!"
        else:
            choices = ", ".join(
                f"task {idx} -- {demo.name}" for idx, demo in demos
            )
            response = f"Please tell me the task number: {choices}."
        await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
        await self._say(pid, response)

    async def _start_guidance_demo(self, demo: Demonstration, pid: str) -> None:
        if not demo.steps:
            response = f"The demonstration '{demo.name}' has no steps recorded."
            await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
            await self._say(pid, response)
            return

        # Cancel any previously-running guidance session cleanly.
        if self._guidance_monitor_task and not self._guidance_monitor_task.done():
            self._guidance_monitor_task.cancel()
            self._guidance_monitor_task = None
        self._guidance_advancing = False

        self._guidance_demo = demo
        self._guidance_step = 0
        _trace_log.info(
            "GUIDANCE_START  task=%d  demo=%s  steps=%d",
            demo.task_index,
            demo.name,
            len(demo.steps),
        )
        await self._speak_current_guidance_step(pid)
        self._start_guidance_monitor(pid)

    async def _advance_guidance(self, pid: str) -> None:
        if self._guidance_demo is None or self._guidance_advancing:
            return
        self._guidance_advancing = True
        try:
            self._guidance_step += 1
            if self._guidance_step >= len(self._guidance_demo.steps):
                await self._finish_guidance(pid)
            else:
                await self._speak_current_guidance_step(pid)
        finally:
            self._guidance_advancing = False

    async def _finish_guidance(self, pid: str) -> None:
        if self._guidance_monitor_task and not self._guidance_monitor_task.done():
            self._guidance_monitor_task.cancel()
            self._guidance_monitor_task = None
        demo = self._guidance_demo
        self._guidance_demo = None
        self._guidance_step = 0
        if demo:
            response = f"You've completed all steps in '{demo.name}'. Well done!"
        else:
            response = "Guidance complete."
        await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
        await self._say(pid, response)
        _trace_log.info("GUIDANCE_DONE")

    async def _handle_guidance_question(self, transcript: str, pid: str) -> None:
        """Answer a question during guidance. Pre-fetches the camera frame so
        questions like 'where is the knife?' get a real VLM-based answer."""
        if self._guidance_demo is None:
            return
        instruction = self._instruction_for_step(self._guidance_demo)
        step_num    = self._guidance_step + 1
        total       = len(self._guidance_demo.steps)

        # Fetch the current live frame only.
        # Reference frame comparison was giving incorrect answers for "did I do
        # it correctly?" — the VLM matched visual intent (user near target) as
        # completion. Ask the VLM what it actually sees, then let the LLM judge.
        frame_path = await self._get_latest_frame_path(pid, ref_us=0)

        system_content = (
            f"You are guiding someone through a procedure. "
            f"They are on step {step_num} of {total}: \"{instruction}\". "
            "Answer their question honestly in 1-2 short sentences based on "
            "what the camera currently shows. Be direct — if it is not done "
            "correctly, say so clearly."
        )
        user_parts: list[str] = []
        if frame_path:
            try:
                result = await asyncio.wait_for(
                    self._call_mcp(
                        self._vlm, "ask_image",
                        {"question": (
                            f"Step to complete: \"{instruction}\"\n"
                            f"User asks: {transcript}\n"
                            "Describe exactly what you see in the current frame "
                            "relevant to this step."
                        ),
                         "image_path": frame_path},
                        silent=True,
                    ),
                    timeout=8.0,
                )
                if isinstance(result, dict):
                    result = (result.get("result") or result.get("text")
                              or next(iter(result.values()), ""))
                if isinstance(result, str) and result.strip():
                    user_parts.append(f"What the camera sees: {result.strip()}")
            except (asyncio.TimeoutError, Exception):
                pass
        user_parts.append(f"User question: {transcript}")
        try:
            structured = self._quick_llm(max_tokens=120, timeout=10.0).with_structured_output(
                _GUIDANCE_RESPONSE_SCHEMA,
                include_raw=True,
            )
            result = await asyncio.wait_for(
                structured.ainvoke([
                    SystemMessage(content=system_content),
                    HumanMessage(content="\n\n".join(user_parts)),
                ]),
                timeout=10.0,
            )
            obj = result.get("parsed") if isinstance(result, dict) else result
            reply = ""
            if isinstance(obj, dict):
                reply = str(obj.get("reply", "")).strip()
            if not reply and isinstance(result, dict):
                raw = result.get("raw")
                reply = self._message_text(raw) if raw is not None else ""
            if reply:
                await self._send_text(pid, reply, _AGENT_RESPONSE_TOPIC)
                await self._say(pid, reply)
        except Exception:
            log.exception("guidance question failed")

    def _instruction_for_step(self, demo: "Demonstration") -> str:
        """Return the clean instruction for the current step, falling back to raw description."""
        idx = self._guidance_step
        if demo.instructions and idx < len(demo.instructions):
            return demo.instructions[idx]
        return demo.steps[idx].description

    async def _speak_current_guidance_step(self, pid: str) -> None:
        demo = self._guidance_demo
        if demo is None:
            return
        self._guidance_step_obs_baseline   = len(self._memory._observations)
        self._guidance_monitor_idle_cycles = 0
        self._guidance_consecutive_yes     = 0
        total       = len(demo.steps)
        step_num    = self._guidance_step + 1
        instruction = self._instruction_for_step(demo)
        response = f"Step {step_num} of {total}: {instruction}"
        await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
        await self._say(pid, response)
        _trace_log.info("GUIDANCE_STEP  %d/%d  %s", step_num, total, instruction[:60])

    # ── guidance auto-advance monitor ─────────────────────────────────────────

    def _start_guidance_monitor(self, pid: str) -> None:
        if self._guidance_monitor_task and not self._guidance_monitor_task.done():
            self._guidance_monitor_task.cancel()
        self._guidance_monitor_task = asyncio.create_task(
            self._guidance_monitor_loop(pid), name="guidance-monitor"
        )

    async def _guidance_monitor_loop(self, pid: str) -> None:
        """Advance guidance automatically when the user completes a step.

        Primary signal: the background VLM observation loop adds a new entry
        when something visibly changes — obs_delta > 0 means the user did something.

        Fallback (fine-motor actions): after 3 idle cycles (~12 s) with no
        observation change, do a direct VLM check asking whether the step
        looks complete. This catches button presses / switch flips that don't
        produce large enough scene changes for the observation loop.
        """
        _trace_log.info("GUIDANCE_MONITOR  start  step=%d", self._guidance_step)
        try:
            while self._guidance_demo is not None:
                await asyncio.sleep(self._cfg.guidance_check_interval_s)
                if self._guidance_demo is None:
                    break

                current_obs = len(self._memory._observations)
                delta = current_obs - self._guidance_step_obs_baseline
                _trace_log.info(
                    "GUIDANCE_MONITOR  step=%d  obs_delta=%d  idle=%d",
                    self._guidance_step, delta, self._guidance_monitor_idle_cycles,
                )

                if delta > 0:
                    self._guidance_monitor_idle_cycles = 0
                    await self._advance_guidance(pid)
                else:
                    self._guidance_monitor_idle_cycles += 1
                    # After 1 idle cycle (~4 s) with no obs change, use direct
                    # VLM check.  Static gestures (finger counting, button holds)
                    # never trigger obs_delta > 0, so the VLM check is the primary
                    # advance signal for those steps.
                    if self._guidance_monitor_idle_cycles >= 1:
                        self._guidance_monitor_idle_cycles = 0
                        if await self._vlm_step_complete(pid):
                            self._guidance_consecutive_yes += 1
                            _trace_log.info(
                                "GUIDANCE_MONITOR  yes-count=%d  step=%d",
                                self._guidance_consecutive_yes, self._guidance_step,
                            )
                            # Require 2 consecutive YES before advancing — prevents
                            # false-positives when the scene happens to already match
                            # the target state from a previous arrangement.
                            if self._guidance_consecutive_yes >= 2:
                                self._guidance_consecutive_yes = 0
                                _trace_log.info("GUIDANCE_MONITOR  vlm-advance  step=%d",
                                                self._guidance_step)
                                await self._advance_guidance(pid)
                        else:
                            self._guidance_consecutive_yes = 0
        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("guidance monitor error")
        _trace_log.info("GUIDANCE_MONITOR  exit")

    async def _vlm_step_complete(self, pid: str) -> bool:
        """Ask the VLM whether the current step looks done.

        Strategy (in order):
        1. Two-frame visual comparison if a reference frame exists.
           Image 1 = completed demo state, Image 2 = live. Direct YES/NO.
        2. If that returns NO or fails, retry with the instruction text only
           (single live frame + voice-note description). Catches cases where
           the reference frame comparison is ambiguous.
        """
        import os as _os
        if self._guidance_demo is None:
            return False
        instruction = self._instruction_for_step(self._guidance_demo)
        frame_path  = await self._get_latest_frame_path(pid, ref_us=0)
        if not frame_path:
            return False

        def _unwrap(r) -> str:
            if isinstance(r, dict):
                r = r.get("result") or r.get("text") or next(iter(r.values()), "")
            return str(r).strip() if r else ""

        # Single live frame + instruction text only.
        # The 2-frame reference comparison caused false positives: the VLM would
        # say YES when the user was in the process of doing the step (intent
        # visible) rather than when it was actually complete.
        question = (
            f"Has this been done: {instruction}\n"
            "YES or NO."
        )
        try:
            result = _unwrap(await asyncio.wait_for(
                self._call_mcp(self._vlm, "ask_image",
                               {"question": question, "image_path": frame_path},
                               silent=True),
                timeout=8.0,
            ))
        except asyncio.TimeoutError:
            result = ""
        completed = result.upper().startswith("YES")
        _trace_log.info("GUIDANCE_MONITOR  vlm-check  %r → %s  raw=%r",
                        instruction[:40], "YES" if completed else "NO", result[:30])
        return completed

    # ── quick-ack ─────────────────────────────────────────────────────────────

    async def _quick_ack(self, transcript: str, pid: str = "") -> tuple[str, bool]:
        """Fast call to Llama-Nemotron: returns (ack_text, needs_thinking).

        think=True for: questions about past events, spatial/visual analysis,
                        demonstrations, corrections.
        think=False for: simple current-view questions, greetings, acknowledgements.
        """
        context = ""
        last_turn = self._last_turn_by_pid.get(pid or "default")
        if last_turn:
            last_user, last_agent = last_turn
            context = f"[Previous turn] User: {last_user} / Agent: {last_agent}\n"

        system_content = (
            "ack: a SHORT natural spoken acknowledgment (3-6 words, no period). "
            "Sound like a helpful smart-glasses assistant about to START working on it. "
            "ALWAYS use present or future tense — the task is NOT done yet. "
            "NEVER use past tense. "
            "Examples: 'On it.' / 'Let me look.' / 'Sure, checking now.' / "
            "'Let me think about that.' / 'Looking back at that.' / 'Got it.'\n"
            "think: true if ANY of:\n"
            "  (A) questions about past events, what happened earlier, or recall "
            "('what was that?', 'what did I see earlier?', 'did I do that?')\n"
            "  (B) spatial or visual analysis that requires examining an image "
            "('what is that?', 'describe what I see', 'is there a ...?')\n"
            "  (C) questions about demonstrations or procedures "
            "('how many steps?', 'what was step 2?')\n"
            "  (D) corrections, follow-ups, or ambiguous references "
            "('no, not that', 'the one I saw earlier')\n"
            "think: false for: greetings, simple yes/no, acknowledgements, "
            "immediate next-step advances in guidance."
        )
        try:
            structured = self._quick_llm(max_tokens=80, timeout=8.0).with_structured_output(
                _QUICK_ACK_SCHEMA,
                include_raw=True,
            )
            result = await asyncio.wait_for(
                structured.ainvoke([
                    SystemMessage(content=system_content),
                    HumanMessage(content=context + transcript),
                ]),
                timeout=8.0,
            )
            obj = result.get("parsed") if isinstance(result, dict) else result
            if isinstance(obj, dict):
                ack = str(obj.get("ack", "")).strip()
                think = bool(obj.get("think", False))
                if ack:
                    log.info("quick-ack: %r  think=%s", ack, think)
                    _trace_log.info("ACK   %s  [think=%s]", ack, think)
                    return ack, think
            raw = result.get("raw") if isinstance(result, dict) else None
            text = self._message_text(raw) if raw is not None else ""
            return text, False
        except Exception:
            log.debug("quick-ack call failed", exc_info=True)
        return "", False

    @staticmethod
    def _fallback_ack(transcript: str) -> tuple[str, bool]:
        lower = transcript.lower()
        if any(word in lower for word in ("look", "see", "image", "picture", "view", "describe")):
            return "Let me look.", True
        if "?" in transcript or any(lower.startswith(prefix) for prefix in ("what", "where", "how", "why", "can you")):
            return "Let me check.", True
        return "On it.", True

    # ── agentic loop ──────────────────────────────────────────────────────────

    async def _prefetch_frame_description(
        self, pid: str, ref_us: int
    ) -> tuple[str | None, str | None]:
        """Get the latest frame and a fresh VLM description of it.

        Returns (frame_path, description). Either may be None on failure.
        Called concurrently with context-building so the VLM answer is
        ready before the first LLM call.
        """
        path = await self._get_latest_frame_path(pid, ref_us)
        if not path:
            return None, None
        try:
            result = await asyncio.wait_for(
                self._call_mcp(
                    self._vlm, "ask_image",
                    {"question": "Describe what you see in this image in 1-2 sentences.",
                     "image_path": path},
                    silent=True,
                ),
                timeout=8.0,
            )
            if isinstance(result, dict):
                result = (result.get("result") or result.get("text")
                          or next(iter(result.values()), ""))
            if isinstance(result, str) and result.strip():
                return path, result.strip()
        except Exception:
            pass
        return path, None

    async def _get_latest_frame_path(self, pid: str, ref_us: int) -> str | None:
        """Get the latest frame path from video-mcp for *pid*.

        Tries get_latest_frame (live-only / recording-disabled mode) first;
        falls back to get_frame_from_time (recording-enabled mode).
        """
        candidates = [
            ("get_latest_frame", {"participant_id": pid}),
            (
                "get_frame_from_time",
                {"participant_id": pid, "second_ago": 0, "reference_time_us": ref_us or 0},
            ),
        ]
        for tool, args in candidates:
            try:
                result = await self._video.call_tool(tool, args)
                data = parse_mcp_result(result)
                if isinstance(data, dict) and "path" in data:
                    return data["path"]
            except Exception as exc:
                log.debug("pre-fetch frame via %s failed: %s", tool, exc)
        return None

    def _quick_llm(self, *, max_tokens: int = 128, timeout: float = 30.0) -> ChatOpenAI:
        return ChatOpenAI(
            base_url=self._cfg.llm_server.rstrip("/") + "/v1",
            api_key="EMPTY",
            model="llm",
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=0,
        )

    def _agent_llm(
        self, *, needs_thinking: bool = False, max_tokens: int | None = None, timeout: float = 180.0
    ) -> ChatOpenAI:
        chat_template_kwargs: dict[str, Any] = {"enable_thinking": needs_thinking}
        if needs_thinking:
            chat_template_kwargs["thinking_budget"] = 1024

        return ChatOpenAI(
            base_url=self._cfg.agent_llm_server.rstrip("/") + "/v1",
            api_key="EMPTY",
            model="llm",
            temperature=0.0,
            max_tokens=max_tokens or (1024 if needs_thinking else 512),
            timeout=timeout,
            max_retries=0,
            extra_body={"chat_template_kwargs": chat_template_kwargs},
        )

    def _langchain_agent(self):
        if self._langchain_agent_graph is not None:
            return self._langchain_agent_graph

        agent = create_agent(
            model=self._agent_llm(needs_thinking=False),
            tools=self._langchain_tools,
            system_prompt=self._system_prompt,
            middleware=[_GlassesAgentMiddleware(lambda needs: self._agent_llm(
                needs_thinking=needs
            ))],
            context_schema=GlassesRuntimeContext,
            checkpointer=self._checkpointer,
        )
        self._langchain_agent_graph = agent
        return agent

    @staticmethod
    def _message_text(message: Any) -> str:
        content = (
            message.get("content") if isinstance(message, dict)
            else getattr(message, "content", "")
        )
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(p for p in parts if p).strip()
        return str(content).strip() if content else ""

    def _langchain_response_text(self, result: Any) -> str:
        messages = result.get("messages", []) if isinstance(result, dict) else []
        for message in reversed(messages):
            msg_type = (
                message.get("role") if isinstance(message, dict)
                else getattr(message, "type", "")
            )
            if msg_type in ("ai", "assistant"):
                text = self._message_text(message)
                if text:
                    return text
        return ""

    async def _invoke_langchain_agent(
        self,
        transcript: str,
        *,
        pid: str,
        runtime_context: GlassesRuntimeContext,
        needs_thinking: bool,
    ) -> str:
        thread_id = f"participant:{pid or 'default'}"
        result = await self._langchain_agent().ainvoke(
            {"messages": [{"role": "user", "content": transcript}]},
            config={
                "recursion_limit": _MAX_LOOP * 2 + 2,
                "configurable": {"thread_id": thread_id},
            },
            context=runtime_context,
        )
        return self._langchain_response_text(result)

    async def _agentic_loop(
        self,
        transcript:     str,
        pid:            str,
        *,
        ref_us:         int  = 0,
        needs_thinking: bool = False,
    ) -> str:
        """Multi-turn tool-calling loop using LangChain's agent runtime.

        Pre-fetches the latest frame concurrently while building context so
        the frame is ready before the first LLM call.
        """
        # Pre-fetch the latest frame and get a fresh VLM description concurrently.
        frame_task = asyncio.create_task(self._prefetch_frame_description(pid, ref_us))

        recording_name: str | None = None
        if self._memory.recording is not None:
            recording_name = self._memory.recording.name

        guidance_instruction: str | None = None
        guidance_step = 0
        guidance_total = 0
        if self._guidance_demo is not None:
            guidance_total = len(self._guidance_demo.steps)
            guidance_step = self._guidance_step + 1
            guidance_instruction = self._instruction_for_step(self._guidance_demo)

        # Await the pre-fetched frame + fresh VLM description.
        frame_path, frame_desc = await frame_task
        runtime_context = GlassesRuntimeContext(
            memory_snapshot      = self._memory.snapshot(max_recent=8),
            pid                  = pid,
            ref_us               = ref_us,
            needs_thinking       = needs_thinking,
            recording_name       = recording_name,
            guidance_instruction = guidance_instruction,
            guidance_step        = guidance_step,
            guidance_total       = guidance_total,
            frame_path           = frame_path,
            frame_description    = frame_desc,
        )
        context_text = _format_glasses_runtime_context(runtime_context)
        _trace_log.info("CTX   %s", context_text.replace("\n", " | ")[:500])

        response = await self._invoke_langchain_agent(
            transcript,
            pid=pid,
            runtime_context=runtime_context,
            needs_thinking=needs_thinking,
        )
        if not response and needs_thinking:
            log.warning(
                "langchain agent returned empty response with thinking; "
                "retrying without thinking"
            )
            response = await self._invoke_langchain_agent(
                transcript,
                pid=pid,
                runtime_context=runtime_context,
                needs_thinking=False,
            )
        _trace_log.info("RESP  %s", response or "Done.")
        return response or "Done."

    async def _call_mcp(
        self, client: McpClient, tool: str, args: dict, *, silent: bool = False
    ) -> dict | str | None:
        try:
            result = await client.call_tool(tool, args)
            return parse_mcp_result(result)
        except Exception as exc:
            if not silent:
                log.error("mcp %s failed: %s", tool, exc)
            return {"error": str(exc)}

    async def close(self) -> None:
        if self._guidance_monitor_task and not self._guidance_monitor_task.done():
            self._guidance_monitor_task.cancel()


