# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
QueryProcessor — handles all user utterances for the glasses agent.

Flow per utterance:
  1. Demo-mode detection (start / end / guidance request / guidance advance).
  2. _quick_ack() — fast Llama-Nemotron call: spoken ack + think=True/False.
  3. _agentic_loop() — delegates to a NAT tool_calling_agent workflow:
       - Pre-fetch latest frame concurrently while building context.
       - Context = memory.build_context() + frame path + conversation history.
       - Tools: configured in yaml/glasses_agent_nat_workflow.yaml.
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
import string
import time
from typing import Callable, Awaitable

import httpx

from config import WorkerConfig
from memory import AgentMemory, Demonstration, DemoStep, Observation, VoiceNote
from nat_agent import NatAgentRunner
from nat_runtime import NatRuntime

log = logging.getLogger("glasses_agent_nat.processors")

_trace_log = logging.getLogger("glasses_agent_nat.trace")

# ── demo detection phrases ────────────────────────────────────────────────────

_DEMO_START_PHRASES = (
    # Explicit start commands
    "start recording",
    "start record",
    "star recording",
    "star record",
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

_DEMO_NAME_PREFIXES = (
    "for task where ",
    "for the task where ",
    "for task ",
    "for the task ",
    "for task ",
    "for the task ",
    "called ",
    "named ",
    "where ",
    "the task ",
    "task ",
    "a ",
    "the ",
)

_DEMO_END_PHRASES = (
    "that's it",
    "stop recording",
    "end demonstration",
    "finished",
    "end recording",
    "finish recording",
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

_AGENT_RESPONSE_TOPIC = "agent.response"
_AGENT_PROGRESS_TOPIC = "agent.progress"

def _now_us() -> int:
    return time.time_ns() // 1_000


def _extract_json(text: str) -> str | None:
    depth, start, in_string, escape = 0, -1, False, False
    for i, ch in enumerate(text):
        if in_string:
            if escape:        escape = False
            elif ch == "\\": escape = True
            elif ch == '"':  in_string = False
            continue
        if ch == '"':   in_string = True; continue
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}":
            if depth == 0: continue
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start:i + 1]
    return None


# ── QueryProcessor ────────────────────────────────────────────────────────────

SendTextCb = Callable[[str, str, str], Awaitable[None]]  # (pid, text, topic)


class QueryProcessor:
    """Handles text utterances: demo detection → quick-ack → agentic loop."""

    def __init__(
        self,
        cfg:          WorkerConfig,
        memory:       AgentMemory,
        nat_runtime:  NatRuntime,
        *,
        send_text:    SendTextCb,
        say:          Callable[[str, str], Awaitable[None]],   # (pid, text)
    ) -> None:
        self._cfg          = cfg
        self._memory       = memory
        self._nat_runtime  = nat_runtime
        self._send_text    = send_text
        self._say          = say
        self._http         = httpx.AsyncClient(timeout=180.0)
        self._nat_agent    = NatAgentRunner(nat_runtime)

        self._history:     list[tuple[str, str]] = []
        self._history_max  = 4

        # Guidance mode state.
        self._guidance_demo: Demonstration | None = None
        self._guidance_step: int = 0
        self._guidance_monitor_task: asyncio.Task | None = None
        self._guidance_advancing:    bool                = False
        self._guidance_step_obs_baseline:   int = 0
        self._guidance_monitor_idle_cycles: int = 0
        self._guidance_consecutive_yes:     int = 0  # consecutive 2-frame YES responses

    async def handle(
        self,
        transcript: str,
        pid:        str,
        ref_us:     int,
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
            import os as _os, json as _json, re as _re
            note = VoiceNote(timestamp_us=ref_us, text=text)
            self._memory.add_voice_note(note)
            _trace_log.info("VOICE_NOTE  t=%d  %s", ref_us, text[:80])
            # Append to the same JSONL log as the frames so everything is on disk.
            rec      = self._memory.recording
            run_dir  = _os.environ.get("XR_RUN_DIR", "/tmp")
            safe     = _re.sub(r"[^a-zA-Z0-9_-]", "_", rec.name)[:40]
            log_path = _os.path.join(run_dir, "recordings",
                                     f"{safe}_{rec.started_at_us}.jsonl")
            try:
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
            ack, needs_thinking = await self._quick_ack(text)
        except Exception:
            log.exception("quick-ack failed")
            ack, needs_thinking = "", False

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
            self._history.append((text, response))
            if len(self._history) > self._history_max:
                self._history.pop(0)
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
                for prefix in _DEMO_NAME_PREFIXES:
                    if after.startswith(prefix):
                        after = after[len(prefix):].strip().rstrip(string.punctuation).strip()
                        break
                if after and len(after) > 2:
                    return after
                # Bare trigger ("watch me") — generate a name from timestamp.
                ts = time.strftime("%H%M%S")
                return f"demo-{ts}"
        return None

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

    # ── demo actions ──────────────────────────────────────────────────────────

    async def _handle_demo_start(self, name: str, pid: str) -> None:
        self._memory.start_recording(name)
        response = (
            f"Recording '{name}'. Go ahead and demonstrate — "
            "I'm watching your hands and will capture each step."
        )
        await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
        await self._say(pid, response)
        _trace_log.info("DEMO_START  name=%s", name)

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
        _trace_log.info("DEMO_END  name=%s  frames=%d", demo.name, n_frames)

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

        response = f"Saved '{demo.name}' with {n} steps."
        if overview:
            response += f" {overview}"
        await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
        await self._say(pid, f"Saved demonstration '{demo.name}' with {n} steps.")

    async def _analyze_recording(
        self, demo: Demonstration
    ) -> tuple[str, list[str]]:
        """Analyze a recorded demonstration through the NAT worker task group."""
        frames = demo.recorded_frames
        if not frames:
            return "", []

        _trace_log.info("ANALYSIS_START  demo=%r  frames=%d  voice_notes=%d",
                        demo.name, len(frames), len(demo.voice_notes))
        try:
            result = await self._nat_runtime.call_tool("glasses_worker_tasks", "analyze_recording", {
                "name": demo.name,
                "started_at_us": demo.started_at_us,
                "frames": [
                    {
                        "frame_idx": f.frame_idx,
                        "timestamp_us": f.timestamp_us,
                        "image_path": f.image_path,
                        "description": f.description,
                    }
                    for f in frames
                ],
                "voice_notes": [
                    {"timestamp_us": v.timestamp_us, "text": v.text}
                    for v in demo.voice_notes
                ],
            })
        except Exception:
            log.exception("analysis NAT task failed")
            return "", []
        if not isinstance(result, dict):
            return "", []
        overview = str(result.get("overview", "")).strip()
        steps = [str(s).strip() for s in result.get("steps", []) if str(s).strip()]
        _trace_log.info("ANALYSIS_RESULT  %s", str(result)[:300])
        return overview, steps

    # ── guidance actions ──────────────────────────────────────────────────────

    async def _handle_guidance_request(self, query: str, pid: str) -> None:
        """Find a matching demo and enter guidance mode."""
        # Look for a demo name mentioned in the query, or use the first available.
        demo = self._memory.find_demonstration_fuzzy(query)
        if demo is None:
            demos = self._memory.list_demonstrations()
            if demos:
                demo = self._memory.get_demonstration(demos[0])
        if demo is None:
            response = "I don't have any recorded demonstrations yet. Show me first!"
            await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
            await self._say(pid, response)
            return

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
        _trace_log.info("GUIDANCE_START  demo=%s  steps=%d", demo.name, len(demo.steps))
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

        lower = transcript.lower()
        if any(p in lower for p in ("doing it right", "doing step", "correct", "done", "finished")):
            result = await self._guidance_completion_result(pid)
            raw = str(result.get("raw", "")) if isinstance(result, dict) else ""
            completed = bool(result.get("completed")) if isinstance(result, dict) else False
            if completed:
                reply = f"Yes, step {step_num} looks complete. {instruction}"
            else:
                detail = f" I see: {raw}" if raw and raw.upper() != "NO" else ""
                reply = f"Not yet. Step {step_num} is: {instruction}.{detail}"
            await self._send_text(pid, reply, _AGENT_RESPONSE_TOPIC)
            await self._say(pid, reply)
            return

        # Fetch the current live frame only.
        # Reference frame comparison was giving incorrect answers for "did I do
        # it correctly?" — the VLM matched visual intent (user near target) as
        # completion. Ask the VLM what it actually sees, then let the LLM judge.
        frame_path = await self._get_latest_frame_path(pid, ref_us=0)

        messages: list[dict] = [
            {
                "role": "system",
                "content": (
                    f"You are guiding someone through a procedure. "
                    f"They are on step {step_num} of {total}: \"{instruction}\". "
                    "Answer their question honestly in 1-2 short sentences based on "
                    "what the camera currently shows. Be direct — if it is not done "
                    "correctly, say so clearly."
                ),
            },
        ]
        user_content: list[dict] = []
        if frame_path:
            try:
                result = await asyncio.wait_for(
                    self._call_vlm(
                        "ask_image",
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
                    user_content.append({
                        "type": "text",
                        "text": (
                            f"What the camera sees: {result.strip()}\n\n"
                            f"User question: {transcript}"
                        ),
                    })
            except (asyncio.TimeoutError, Exception):
                pass
        if not user_content:
            user_content.append({"type": "text", "text": transcript})
        messages.append({"role": "user", "content": user_content})

        body = {
            "model": "llm",
            "messages": messages,
            "max_tokens": 100,
            "temperature": 0.1,
        }
        try:
            resp = await asyncio.wait_for(
                self._http.post(
                    self._cfg.llm_server.rstrip("/") + "/v1/chat/completions",
                    json=body,
                ),
                timeout=10.0,
            )
            if not resp.is_error:
                reply = resp.json()["choices"][0]["message"]["content"].strip()
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

                should_check = delta > 0
                if not should_check:
                    self._guidance_monitor_idle_cycles += 1
                    should_check = self._guidance_monitor_idle_cycles >= 1
                if not should_check:
                    continue

                self._guidance_monitor_idle_cycles = 0
                if await self._vlm_step_complete(pid):
                    self._guidance_consecutive_yes += 1
                    _trace_log.info(
                        "GUIDANCE_MONITOR  yes-count=%d  step=%d",
                        self._guidance_consecutive_yes, self._guidance_step,
                    )
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

    async def _guidance_completion_result(self, pid: str) -> dict:
        if self._guidance_demo is None:
            return {}
        instruction = self._instruction_for_step(self._guidance_demo)
        try:
            result = await asyncio.wait_for(
                self._nat_runtime.call_tool(
                    "glasses_worker_tasks",
                    "check_guidance_step_complete",
                    {"participant_id": pid, "instruction": instruction},
                    participant_id=pid,
                ),
                timeout=10.0,
            )
        except (asyncio.TimeoutError, Exception):
            result = {}
        return result if isinstance(result, dict) else {}

    async def _vlm_step_complete(self, pid: str) -> bool:
        """Ask the NAT guidance task whether the current step looks done."""
        result = await self._guidance_completion_result(pid)
        raw = str(result.get("raw", "")) if isinstance(result, dict) else ""
        completed = bool(result.get("completed")) if isinstance(result, dict) else False
        instruction = self._instruction_for_step(self._guidance_demo) if self._guidance_demo else ""
        _trace_log.info("GUIDANCE_MONITOR  vlm-check  %r → %s  raw=%r",
                        instruction[:40], "YES" if completed else "NO", raw[:30])
        return completed

    # ── quick-ack ─────────────────────────────────────────────────────────────

    async def _quick_ack(self, transcript: str) -> tuple[str, bool]:
        """Fast call to Llama-Nemotron: returns (ack_text, needs_thinking).

        think=True for: questions about past events, spatial/visual analysis,
                        demonstrations, corrections.
        think=False for: simple current-view questions, greetings, acknowledgements.
        """
        context = ""
        if self._history:
            last_user, last_agent = self._history[-1]
            context = f"[Previous turn] User: {last_user} / Agent: {last_agent}\n"

        body = {
            "model": "llm",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        'Output ONLY one JSON object: {"ack": "<spoken phrase>", "think": false}\n'
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
                    ),
                },
                {"role": "user", "content": context + transcript},
            ],
            "max_tokens": 40,
            "temperature": 0.0,
        }
        try:
            resp = await asyncio.wait_for(
                self._http.post(
                    self._cfg.llm_server.rstrip("/") + "/v1/chat/completions",
                    json=body,
                ),
                timeout=8.0,
            )
            if not resp.is_error:
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                obj_text = _extract_json(raw)
                if obj_text:
                    try:
                        obj   = json.loads(obj_text)
                        ack   = str(obj.get("ack", "")).strip()
                        think = bool(obj.get("think", False))
                        log.info("quick-ack: %r  think=%s", ack, think)
                        _trace_log.info("ACK   %s  [think=%s]", ack, think)
                        return ack, think
                    except json.JSONDecodeError:
                        pass
                return raw, False
        except Exception:
            log.debug("quick-ack call failed", exc_info=True)
        return "", False

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
                self._call_vlm(
                    "ask_image",
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
                data = await self._call_video(tool, args, silent=True)
                if isinstance(data, dict) and "path" in data:
                    return data["path"]
            except Exception as exc:
                log.debug("pre-fetch frame via %s failed: %s", tool, exc)
        return None

    async def _agentic_loop(
        self,
        transcript:     str,
        pid:            str,
        *,
        ref_us:         int  = 0,
        needs_thinking: bool = False,
    ) -> str:
        """Run one user request through the configured NAT tool-calling agent.

        The NAT workflow owns request-time tool selection. This method only
        packages XR memory, participant, time, and conversation context.
        """
        # Build context from memory.
        ctx_parts: list[str] = []
        ctx_parts.append(self._memory.build_context(max_recent=8))

        if self._memory.recording is not None:
            ctx_parts.append(
                f"[Recording active — demo: {self._memory.recording.name!r}]"
            )

        if self._guidance_demo is not None:
            total       = len(self._guidance_demo.steps)
            step_num    = self._guidance_step + 1
            instruction = self._instruction_for_step(self._guidance_demo)
            ctx_parts.append(
                f"[Guidance mode — step {step_num} of {total}]\n"
                f"Current instruction: {instruction}\n"
                f"Help the user complete THIS step. Be concise and direct."
            )

        if pid:
            ctx_parts.append(f"Participant: {pid}")
        if ref_us:
            ctx_parts.append(f"Reference time (when user spoke): {ref_us} µs")

        if self._history:
            hist_lines: list[str] = []
            for u, a in self._history:
                hist_lines.append(f"  User: {u}")
                hist_lines.append(f"  Agent: {a}")
            ctx_parts.append("[Recent conversation]\n" + "\n".join(hist_lines))

        context = "\n\n".join(ctx_parts)
        _trace_log.info("CTX   %s", context.replace("\n", " | ")[:500])

        return await self._nat_agent.run(
            context=context,
            transcript=transcript,
            needs_thinking=needs_thinking,
            participant_id=pid,
        )

    # ── internal NAT MCP calls ────────────────────────────────────────────────

    async def _call_vlm(
        self, tool: str, args: dict, *, silent: bool = False
    ) -> dict | str | None:
        try:
            return await self._nat_runtime.call_tool("vlm_mcp", tool, args)
        except Exception as exc:
            if not silent:
                log.error("vlm-mcp %s failed: %s", tool, exc)
            return {"error": str(exc)}

    async def _call_video(
        self, tool: str, args: dict, *, silent: bool = False
    ) -> dict | str | None:
        try:
            return await self._nat_runtime.call_tool("video_mcp", tool, args)
        except Exception as exc:
            if not silent:
                log.error("video-mcp %s failed: %s", tool, exc)
            return {"error": str(exc)}

    async def close(self) -> None:
        if self._guidance_monitor_task and not self._guidance_monitor_task.done():
            self._guidance_monitor_task.cancel()
        await self._http.aclose()

