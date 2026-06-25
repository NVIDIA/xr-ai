# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
QueryProcessor — handles all user utterances for the glasses agent.

Flow per utterance:
  1. Demo-mode detection (start / end / guidance request / guidance advance).
  2. _quick_ack() — fast Llama-Nemotron call: spoken ack + think=True/False.
  3. _agentic_loop() — delegates to a NAT tool_calling_agent workflow:
       - Build context from memory.build_context() + conversation history.
       - Tools: configured in yaml/glasses_agent_nat_workflow.yaml.
       - Finish: LLM text response → TTS + data message to participant.

Demonstration detection phrases are defined as module constants
(`_DEMO_START_PHRASES`, `_DEMO_END_PHRASES`, `_GUIDANCE_PHRASES`,
`_GUIDANCE_ADVANCE_PHRASES`, `_GUIDANCE_DONE_PHRASES`,
`_GUIDANCE_STOP_PHRASES`, `_STOP_SPEAKING_PHRASES`). Examples:
  Start    "start recording <name>", "let me show you …"
  End      "stop recording", "end demonstration"
  Guidance "how do i …", "walk me through …"
  Advance  "next", "continue"
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import string
import time
from typing import Awaitable, Callable

import httpx
from config import WorkerConfig
from memory import (
    AgentMemory,
    Demonstration,
    DemoStep,
    RecordedFrame,
    StepKeyInfo,
    VoiceNote,
)
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

# Filler words stripped from the leading edge of the captured demo name
# after the start phrase. Run iteratively so "for the task X" → "X".
_LEADING_NAME_FILLER = frozenset({
    "for", "task", "where", "called", "named", "the", "a", "an",
})

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
    "how to do",
    "walk me through",
    "show me how",
    "teach me how to",
    "teach me how",
    "guide me through",
    "guide me",
    "step by step",
    "instruct me",
    "help me do",
    "help me with",
    "do task",
)

# Sentinel emitted by the agent LLM when the freshness marker is in
# context and the wearer's utterance is not clearly a current-view
# question. The agentic loop intercepts this token and hands off to
# _handle_ambiguous_guidance_request instead of speaking it.
_DEFER_TO_WORKER_SENTINEL = "__defer_to_worker__"

_GUIDANCE_ADVANCE_PHRASES = (
    "next",
    "continue",
    "got it",
    "okay next",
    "ok next",
    "next step",
    "go on",
    "what's next",
    "whats next",
    "what next",
    "what's the next step",
    "whats the next step",
    "what is the next step",
    "move on",
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

# Explicit guidance-stop phrases that exit guidance even before the
# generic stop branch runs, so the demo-end → explicit-guidance-stop →
# in-guidance-bare-stop → outside-guidance-stop precedence stays in
# this order regardless of which mode the worker is in.
_GUIDANCE_STOP_PHRASES = (
    "stop guidance",
    "cancel guidance",
    "exit guidance",
    "stop guiding",
    "end guidance",
)

# Generic "shut up" phrases: outside guidance, these only flush queued TTS
# (no exit-guidance side effect). Inside guidance, the in-guidance bare
# stop case (_is_guidance_done) handles them and exits guidance instead,
# matching the existing UX.
_STOP_SPEAKING_PHRASES = (
    "stop talking",
    "be quiet",
    "shut up",
    "stop speaking",
    "quiet",
    "shush",
    "hush",
    "stop",
    "cancel",
)

_AGENT_RESPONSE_TOPIC = "agent.response"
_AGENT_PROGRESS_TOPIC = "agent.progress"

# Spoken / ordinal forms accepted when the wearer is answering a numbered
# "did you mean…?" prompt. Only used inside the pending-disambiguation
# branch — outside that branch a bare "one" is treated as ordinary speech.
_NUMERIC_CHOICE_WORDS: dict[str, int] = {
    "one": 1, "first": 1, "1st": 1,
    "two": 2, "second": 2, "2nd": 2,
    "three": 3, "third": 3, "3rd": 3,
    "four": 4, "fourth": 4, "4th": 4,
    "five": 5, "fifth": 5, "5th": 5,
    "six": 6, "sixth": 6, "6th": 6,
}

# Cap pending-disambiguation re-asks so an uncooperative user (or a
# stuck mic) can't trap the wearer in a loop forever.
_PENDING_DISAMBIGUATION_MAX_ATTEMPTS = 2


def _extract_choice_number(lower: str) -> int | None:
    """If *lower* is a short numeric/ordinal answer, return its 1-based index.

    Examples: "1" / "one" / "first" / "the first one" / "number two" -> 1, 1, 1, 1, 2.
    A whole-utterance match is preferred so a casual mention of "first"
    in a longer sentence doesn't get hijacked.
    """
    stripped = lower.strip().rstrip(string.punctuation).strip()
    if not stripped:
        return None
    if stripped.isdigit():
        n = int(stripped)
        if 1 <= n <= 9:
            return n
    tokens = [t.strip(string.punctuation) for t in stripped.split()]
    if any(t in _NUMERIC_CHOICE_WORDS or (t.isdigit() and 1 <= int(t) <= 9)
           for t in tokens):
        # Drop common filler around the number — but only when at least
        # one numeric token survives, so "one two" stays a non-match.
        filler = {"the", "a", "an", "number", "option"}
        useful = [t for t in tokens if t and t not in filler]
        if len(useful) == 1:
            t = useful[0]
            if t.isdigit():
                n = int(t)
                if 1 <= n <= 9:
                    return n
            return _NUMERIC_CHOICE_WORDS.get(t)
    return None


# ── utterance tokenization / classification helpers ──────────────────────────

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokenize(s: str) -> list[str]:
    """Split *s* into lowercase alphanumeric tokens, preserving apostrophes.

    Used by every "did the user say X?" matcher so utterance tokens and
    phrase tokens agree on word boundaries — `"watch me, X"` splits to
    `["watch", "me", "x"]`, not `["watch", "me,", "x"]`.
    """
    return _TOKEN_RE.findall(s)


# Question-form prefixes that turn a completion phrase into a question
# rather than a command. `"am I done?"` asks for a check; bare `"done"`
# exits guidance.
_QUESTION_PREFIXES = ("am i", "is it", "is this", "did i", "do i", "have i")

# Word stems for "is the step finished?" — used by both the question-form
# completion check and the bare-command exit branch. Covers "did I finish?"
# (stem `finish`) and "have I completed it?" (stem `completed`) that the
# raw `done / finished / complete` set misses.
_COMPLETION_STEMS = frozenset({
    "done", "complete", "completed", "finish", "finished",
})


def _is_question_form(lower: str) -> bool:
    """Return True if *lower* begins with a recognized question prefix."""
    s = lower.strip()
    return any(s.startswith(p + " ") or s == p for p in _QUESTION_PREFIXES)


def _has_completion_stem(lower: str) -> bool:
    """Return True if *lower* contains any word from `_COMPLETION_STEMS`."""
    return bool(set(_tokenize(lower)) & _COMPLETION_STEMS)


# Filler tokens stripped from BOTH ends of an utterance before matching
# against `_GUIDANCE_ADVANCE_PHRASES`. Interior tokens are content, not
# filler: `"got it but I'm stuck"` keeps `but` and must NOT advance.
_ADVANCE_FILLER = frozenset({
    "please", "thanks", "okay", "ok", "just", "alright", "now", "then",
})


def _after_frame_for_step(
    frames: list[RecordedFrame],
    notes:  list[VoiceNote],
    step_idx: int,
) -> RecordedFrame | None:
    """Pick the recorded frame that captures the completed state of *step_idx*.

    Returns the latest frame strictly before the NEXT voice note (the one
    that begins the *following* step). For the last step, returns the
    last recorded frame. Returns ``None`` only when *frames* is empty.

    Lifted out of ``_handle_demo_end`` so the picked frame's
    ``image_path`` AND ``description`` can be threaded into ``DemoStep``
    together — no interpolated-timestamp drift between them — and so the
    helper is directly unit-testable.
    """
    if not frames:
        return None
    next_note_ts: int | None = None
    if step_idx + 1 < len(notes):
        next_note_ts = notes[step_idx + 1].timestamp_us
    if next_note_ts is not None:
        before = [f for f in frames if f.timestamp_us < next_note_ts]
        return before[-1] if before else frames[-1]
    return frames[-1]


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


def _fallback_step_requirements(instruction: str) -> list[str]:
    """Small deterministic fallback for common imperative guidance steps."""
    lower = instruction.lower()
    clean = re.sub(r"[^a-z0-9\s]", " ", lower)
    clean = re.sub(r"\s+", " ", clean).strip()
    out: list[str] = []

    if "headset" in clean and (
        "on your head" in clean
        or "on head" in clean
        or "wear" in clean
        or "put" in clean
    ):
        out.append("headset on head")

    if "controller" in clean and ("hold" in clean or "hand" in clean):
        out.append("controller in hand")

    patterns = (
        (r"\b(?:put|place|set)\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+on\s+(?:your\s+)?head\b", " on head"),
        (r"\b(?:hold|grip|pick up)\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+in\s+(?:your\s+)?hand\b", " in hand"),
    )
    for pattern, suffix in patterns:
        match = re.search(pattern, clean)
        if not match:
            continue
        obj = match.group(1).strip()
        obj = re.sub(r"\b(pico|vr|the|a|an)\b", "", obj)
        obj = re.sub(r"\s+", " ", obj).strip()
        if obj:
            out.append(obj + suffix)

    deduped: list[str] = []
    for req in out:
        if req not in deduped:
            deduped.append(req)
    return deduped[:4]


_FRAME_SCORE_STOPWORDS = frozenset({
    "a", "an", "and", "is", "it", "of", "on", "in", "into", "to", "the",
    "this", "that", "your", "you", "step", "one", "two", "three", "put",
    "place", "set", "hold", "grab", "pick", "up", "with", "for",
})

_FRAME_STATE_TOKENS = frozenset({
    "hand", "hands", "head", "eyes", "eye", "face", "left", "right",
})

_KNOWN_OBJECT_TOKENS = frozenset({
    "controller", "headset", "case", "mug", "cup", "phone", "lid",
    "switch", "button", "strap", "cap", "hat", "beret", "circle",
    "scissors", "scissor", "foam", "comb",
})


def _frame_score_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for token in _tokenize(text):
        if len(token) <= 1 or token in _FRAME_SCORE_STOPWORDS:
            continue
        out.add(token)
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            out.add(token[:-1])
    return out


def _primary_object_tokens(instruction: str) -> set[str]:
    tokens = _frame_score_tokens(instruction)
    known = tokens & _KNOWN_OBJECT_TOKENS
    if known:
        return known
    return tokens - _FRAME_STATE_TOKENS


def _target_state_tokens(instruction: str) -> set[str]:
    tokens = _frame_score_tokens(instruction)
    return tokens & _FRAME_STATE_TOKENS


def _candidate_score(
    frame: RecordedFrame,
    instruction: str,
    next_instruction: str = "",
) -> tuple[int, bool]:
    caption_tokens = _frame_score_tokens(frame.description)
    instruction_tokens = _frame_score_tokens(instruction)
    primary = _primary_object_tokens(instruction)
    state = _target_state_tokens(instruction)
    caption_objects = caption_tokens & _KNOWN_OBJECT_TOKENS

    primary_hits = primary & caption_tokens
    state_hits = state & caption_tokens
    primary_present = bool(primary_hits) if primary else bool(instruction_tokens & caption_tokens)
    mismatch_fallback = bool(caption_objects and state and (state & caption_tokens))
    primary_present = primary_present or mismatch_fallback
    state_plausible = True
    if "head" in state and "headset" in primary:
        state_plausible = bool({"head", "eyes", "eye", "face"} & caption_tokens)
    elif "hand" in state or "hands" in state:
        state_plausible = bool({"hand", "hands"} & caption_tokens)

    score = 0
    score += 3 * len(primary_hits)
    score += 2 * len(state_hits)
    score += len((instruction_tokens - primary - state) & caption_tokens)

    if next_instruction:
        next_primary = _primary_object_tokens(next_instruction) - primary
        next_state = _target_state_tokens(next_instruction) - state
        score -= 7 * len(next_primary & caption_tokens)
        score -= 4 * len(next_state & caption_tokens)

    return score, primary_present and state_plausible


def _visually_meaningful_frame(frame: RecordedFrame, instruction: str) -> bool:
    caption_tokens = _frame_score_tokens(frame.description)
    state = _target_state_tokens(instruction)
    if state and not (state & caption_tokens):
        return False
    return bool(caption_tokens & _KNOWN_OBJECT_TOKENS)


def _frame_candidates_for_step(
    frames: list[RecordedFrame],
    notes: list[VoiceNote],
    step_idx: int,
    demo_start_us: int,
    demo_end_us: int,
    *,
    preroll_us: int = 1_500_000,
) -> list[RecordedFrame]:
    if not frames:
        return []
    if step_idx < len(notes):
        start_us = max(demo_start_us, notes[step_idx].timestamp_us - preroll_us)
    else:
        start_us = demo_start_us if step_idx == 0 else frames[0].timestamp_us
    if step_idx + 1 < len(notes):
        end_us = notes[step_idx + 1].timestamp_us
    else:
        end_us = demo_end_us or frames[-1].timestamp_us
    return [f for f in frames if start_us <= f.timestamp_us <= end_us]


def _select_reference_frames_for_step(
    frames: list[RecordedFrame],
    notes: list[VoiceNote],
    step_idx: int,
    instructions: list[str],
    demo_start_us: int,
    demo_end_us: int,
) -> tuple[RecordedFrame | None, list[RecordedFrame], list[tuple[int, int, bool]]]:
    candidates = _frame_candidates_for_step(
        frames, notes, step_idx, demo_start_us, demo_end_us,
    )
    if not candidates:
        return None, [], []

    instruction = instructions[step_idx] if step_idx < len(instructions) else ""
    next_instruction = (
        instructions[step_idx + 1]
        if step_idx + 1 < len(instructions)
        else ""
    )
    scored: list[tuple[int, int, bool, RecordedFrame]] = []
    for frame in candidates:
        score, acceptable = _candidate_score(frame, instruction, next_instruction)
        scored.append((score, frame.frame_idx, acceptable, frame))

    acceptable = [item for item in scored if item[2] and item[0] > 0]
    if not acceptable:
        fallback_items = [
            item for item in scored
            if _visually_meaningful_frame(item[3], instruction)
        ]
        if not fallback_items:
            return None, [], [(idx, score, ok) for score, idx, ok, _ in scored]
        best_score = max(score for score, _idx, _ok, _frame in fallback_items)
        best_band = [item for item in fallback_items if item[0] >= best_score - 1]
        best = max(best_band, key=lambda item: item[3].timestamp_us)[3]
        backups = [
            item[3]
            for item in sorted(fallback_items, key=lambda item: (item[0], item[3].timestamp_us), reverse=True)
            if item[3] is not best
        ][:2]
        trace_scores = [(idx, score, ok) for score, idx, ok, _ in scored]
        return best, backups, trace_scores

    best_score = max(score for score, _idx, _ok, _frame in acceptable)
    best_band = [item for item in acceptable if item[0] >= best_score - 1]
    best = max(best_band, key=lambda item: item[3].timestamp_us)[3]
    backups = [
        item[3] for item in sorted(acceptable, key=lambda item: (item[0], item[3].timestamp_us), reverse=True)
        if item[3] is not best
    ][:2]
    trace_scores = [(idx, score, ok) for score, idx, ok, _ in scored]
    return best, backups, trace_scores


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
        flush_audio:  Callable[[str], Awaitable[None]],        # (pid,)
    ) -> None:
        self._cfg          = cfg
        self._memory       = memory
        self._nat_runtime  = nat_runtime
        self._send_text    = send_text
        self._say          = say
        self._flush_audio  = flush_audio
        self._http         = httpx.AsyncClient(timeout=180.0)
        self._nat_agent    = NatAgentRunner(nat_runtime)

        self._history:     list[tuple[str, str]] = []
        self._history_max  = 4

        # Last spoken ack per pid — used to drop a duplicate ack when the
        # LLM regenerates exactly the same phrase ("On it." / "On it.") for
        # back-to-back utterances, which sounds like a stutter.
        self._last_ack:    dict[str, str] = {}

        # Guidance mode state.
        self._guidance_demo: Demonstration | None = None
        self._guidance_step: int = 0
        self._guidance_monitor_task: asyncio.Task | None = None
        self._guidance_advancing:    bool                = False
        self._guidance_step_obs_baseline:   int = 0
        # Per-check baseline reset every monitor cycle so delta_since_last
        # measures "did the scene move during THIS window?" rather than
        # "has anything changed since the step started?".
        self._guidance_check_obs_baseline:  int = 0
        self._guidance_monitor_idle_cycles: int = 0
        self._guidance_consecutive_yes:     int = 0
        self._guidance_last_result:         dict = {}
        self._guidance_correction_state:    dict[str, object] = {}
        self._guidance_started_at_us:       int = 0
        self._guidance_step_spoken_at_us:   int = 0
        # Timestamp of the live frame the auto-advance monitor last ran a
        # grounded completion check against. Used to skip redundant VLM
        # checks when the hub hasn't delivered a newer frame (the verdict on
        # an identical frame is identical). Reset to 0 on each new step.
        self._guidance_last_checked_live_ts: int = 0

        # Pending guidance disambiguation per pid. When we ask "did you
        # mean 1) X or 2) Y?", the next non-stop utterance from that pid
        # is interpreted against the stored {query, choices, attempts}
        # instead of going through the normal handler. Cleared on
        # success, explicit stop, attempt cap, demo-set change, or any
        # branch that starts a different mode (guidance, recording).
        self._pending_guidance_by_pid: dict[str, dict] = {}

        # Outstanding _finalize_demo tasks (one per recording in
        # flight). Tracked so close() / clear-demos can cancel them and
        # so tests can observe completion.
        self._finalize_tasks: set[asyncio.Task] = set()

    def is_guiding(self, pid: str | None = None) -> bool:
        """Return True if guidance mode is active.

        *pid* is accepted for API parity with the per-participant agent —
        guidance state is currently global, so it is ignored.
        """
        return self._guidance_demo is not None

    def is_guidance_control_utterance(self, transcript: str) -> bool:
        lower = transcript.strip().lower().rstrip(string.punctuation)
        if not lower:
            return False
        return (
            self._is_guidance_stop(lower)
            or self._is_guidance_done(lower)
            or self._is_guidance_advance(lower)
            or self._match_guidance_request(lower) is not None
            or _is_question_form(lower) and _has_completion_stem(lower)
            or any(p in lower for p in ("doing it right", "doing step", "correct"))
        )

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
            # Demo set just changed underneath any in-flight pending
            # prompt — the stored choices list is now stale.
            self._pending_guidance_by_pid.clear()
            # Cancel any in-flight finalize tasks. The generation bump
            # inside clear_demonstrations() is the second line of defence;
            # cancellation here aborts the in-flight LLM call so it stops
            # using bandwidth, and emits DEMO_FINALIZE_CANCELLED in trace.
            for task in list(self._finalize_tasks):
                task.cancel()
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

        # ── explicit guidance-stop ───────────────────────────────────────────
        # Matches BEFORE the recording / guidance / generic-stop branches so
        # the wearer can always exit guidance with an unambiguous phrase.
        if self._is_guidance_stop(lower):
            self._clear_pending_guidance(pid)
            if self._guidance_demo is not None:
                await self._finish_guidance(pid)
            else:
                await self._stop_speaking(pid)
            return

        # ── pending guidance disambiguation ──────────────────────────────────
        # We previously asked "did you mean 1) X or 2) Y?" — interpret
        # this utterance as the answer. Order matters: bare "stop" /
        # "cancel" must still clear pending state and stop speaking
        # rather than being treated as a non-match and re-asked.
        if pid in self._pending_guidance_by_pid:
            if self._is_stop_speaking(lower):
                self._clear_pending_guidance(pid)
                await self._stop_speaking(pid)
                return
            demo = self._resolve_pending_choice(lower, pid)
            if demo is not None:
                self._clear_pending_guidance(pid)
                await self._start_guidance(demo, pid)
                return
            state = self._pending_guidance_by_pid[pid]
            attempts = int(state.get("attempts", 0)) + 1
            if attempts >= _PENDING_DISAMBIGUATION_MAX_ATTEMPTS:
                self._clear_pending_guidance(pid)
                response = (
                    "I'm having trouble matching that to a recorded demo. "
                    "Try again with the exact demo name when you're ready."
                )
                await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
                await self._say(pid, response)
                _trace_log.info("PENDING_GIVE_UP  pid=%s  utterance=%r", pid, text[:60])
                return
            state["attempts"] = attempts
            await self._reask_with_choices(pid, attempts)
            return

        # ── voice narration during recording ─────────────────────────────────
        if self._memory.recording is not None:
            import json as _json
            import os as _os
            import re as _re
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
            elif self._match_guidance_request(lower) is not None:
                await self._handle_guidance_request_during_guidance(lower, pid)
            else:
                # Any other utterance (question, comment, noise) gets a brief
                # contextual reply focused on the current step.
                await self._handle_guidance_question(text, pid)
            return

        # ── outside-guidance stop-speaking ───────────────────────────────────
        # We're not recording, not in guidance, and the user just said "stop"
        # / "be quiet" / "shut up" — they almost certainly want us to stop
        # talking, not to start a new query. flush_audio + text-only ack.
        if self._is_stop_speaking(lower):
            await self._stop_speaking(pid)
            return

        # ── guidance request detection ────────────────────────────────────────
        guidance_match = self._match_guidance_request(lower)
        if guidance_match is not None:
            await self._handle_guidance_request(guidance_match, pid)
            return

        # ── freshness-scoped guidance fallback ───────────────────────────────
        # Within `guidance_freshness_window_s` of a finished demo, treat
        # ambiguous utterances (no explicit guidance phrase, not a clear
        # current-view question) as "guide me through what I just
        # demonstrated". Catches STT-mangled requests like "the gym"
        # (heard from "show me how to wear pico headset") that would
        # otherwise fall into the agentic loop and get a wrong answer.
        if (
            self._memory.demo_is_fresh(self._cfg.guidance_freshness_window_s)
            and self._memory.most_recent_demo() is not None
            and not self._looks_like_current_view_question(lower)
        ):
            await self._handle_ambiguous_guidance_request(text, pid)
            return

        # ── ordinary query ────────────────────────────────────────────────────
        try:
            ack, needs_thinking = await self._quick_ack(text)
        except Exception:
            log.exception("quick-ack failed")
            ack, needs_thinking = "", False

        # ack: text-only progress message ("On it.", "Let me look.", …).
        # We deliberately do NOT TTS the ack here, even when needs_thinking
        # is True — TTSing both the ack and the eventual response makes the
        # agent talk over itself on long replies and double-speak the
        # acknowledgement on every stop. The progress text is enough.
        if ack:
            if self._last_ack.get(pid) == ack:
                _trace_log.info("ACK_DUP_SUPPRESS  %s", ack)
            else:
                self._last_ack[pid] = ack
                await self._send_text(pid, ack, _AGENT_PROGRESS_TOPIC)

        try:
            response = await self._agentic_loop(
                text, pid, ref_us=ref_us, needs_thinking=needs_thinking
            )
        except Exception:
            log.exception("agentic loop failed")
            response = "Something went wrong — please try again."

        # Sentinel: the agentic LLM signalled that this utterance is really
        # an ambiguous guidance request inside the freshness window. Hand
        # the original transcript off to the demo→guidance fallback rather
        # than speaking the sentinel back to the wearer.
        if response and _DEFER_TO_WORKER_SENTINEL in response:
            _trace_log.info("DEFER_TO_WORKER  %r", text[:60])
            await self._handle_ambiguous_guidance_request(text, pid)
            return

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

        Detection is token-boundary aware: a start phrase must appear as a
        contiguous SEQUENCE of word tokens, so ``"watch meteor shower"``
        does NOT trigger via ``"watch me"``. Once a start phrase matches,
        the tail tokens are stripped of leading filler (``the``,
        ``task``, …) so ``"theme song demo"`` is not chopped to
        ``"me song demo"``.
        """
        toks = _tokenize(lower)
        if not toks:
            return None
        for phrase in _DEMO_START_PHRASES:
            phrase_toks = _tokenize(phrase)
            if not phrase_toks:
                continue
            for i in range(len(toks) - len(phrase_toks) + 1):
                if toks[i:i + len(phrase_toks)] == phrase_toks:
                    tail = toks[i + len(phrase_toks):]
                    while tail and tail[0] in _LEADING_NAME_FILLER:
                        tail = tail[1:]
                    if len(" ".join(tail)) > 2:
                        return " ".join(tail)
                    # Bare trigger ("watch me") — generate a timestamp name.
                    ts = time.strftime("%H%M%S")
                    return f"demo-{ts}"
        return None

    def _match_guidance_request(self, lower: str) -> str | None:
        """Return the user query if a guidance phrase is detected, else None."""
        for phrase in _GUIDANCE_PHRASES:
            if phrase in lower:
                return lower
        return None

    _CURRENT_VIEW_PHRASES: tuple[str, ...] = (
        "what do you see",
        "what can you see",
        "describe this",
        "describe the view",
        "describe what you see",
        "what is in front of me",
        "what's in front of me",
        "what am i looking at",
        "what do i see",
    )

    def _looks_like_current_view_question(self, lower: str) -> bool:
        """Cheap text predicate for 'what do you see?'-style questions.

        Used by the freshness-scoped guidance fallback to avoid hijacking
        a legitimate current-view question with guidance on a recent demo.
        """
        s = lower.strip()
        for phrase in self._CURRENT_VIEW_PHRASES:
            if phrase in s:
                return True
        return False

    def _is_guidance_advance(self, lower: str) -> bool:
        """Whether *lower* is an explicit "advance to next step" command.

        Strict-equality matcher: tokenize, strip leading/trailing filler
        (``please``, ``okay``, ``now``, …), then accept only if the
        remaining tokens equal an advance phrase exactly. This prevents
        false positives like ``"I got it wrong"`` matching ``"got it"``
        or ``"next time I see this"`` matching ``"next"``.

        Doubled utterances (``"what's next? what's next?"``) are
        collapsed iff the first half equals the second half exactly —
        a common STT artifact when VAD bundles two short utterances.
        The collapse is bounded (no substring matching, no partial
        overlap) so ``"I got it wrong"`` is still rejected.
        """
        toks = _tokenize(lower)
        # Trim filler from both ends — interior tokens are content.
        while toks and toks[0] in _ADVANCE_FILLER:
            toks = toks[1:]
        while toks and toks[-1] in _ADVANCE_FILLER:
            toks = toks[:-1]
        if not toks:
            return False
        half = len(toks) // 2
        if half > 0 and len(toks) == half * 2 and toks[:half] == toks[half:]:
            toks = toks[:half]
        for phrase in _GUIDANCE_ADVANCE_PHRASES:
            if toks == _tokenize(phrase):
                return True
        return False

    def _is_guidance_done(self, lower: str) -> bool:
        """Whether *lower* is an explicit "exit guidance" command.

        Bare commands (``done``, ``finished``, ``stop guidance``) exit.
        Question forms containing a completion stem (``am I done?``,
        ``did I finish?``, ``is this complete?``) route to
        ``_handle_guidance_question`` for a real completion check
        instead — they are NOT exit commands.
        """
        # Multi-word explicit exits ("stop guidance", "exit guidance", …)
        # always exit, even when phrased as a question.
        multi_word_exits = [p for p in _GUIDANCE_DONE_PHRASES if " " in p]
        for phrase in multi_word_exits:
            if phrase in lower:
                return True
        # Question-form completion stems are NOT exits — let the
        # question handler check whether the step is actually done.
        if _is_question_form(lower) and _has_completion_stem(lower):
            return False
        # Bare-command branch: single-word phrases match on word boundary.
        words = set(lower.replace(",", " ").replace(".", " ").split())
        for phrase in _GUIDANCE_DONE_PHRASES:
            if lower.strip() == phrase:
                return True
            if " " not in phrase and phrase in words:
                return True
        return False

    def _is_guidance_stop(self, lower: str) -> bool:
        """Explicit 'stop guidance' / 'cancel guidance' / 'exit guidance'.

        Matches before the generic stop branch, so the wearer can always
        end guidance with an unambiguous phrase even if guidance state
        somehow got out of sync.
        """
        s = lower.strip()
        for phrase in _GUIDANCE_STOP_PHRASES:
            if phrase in s:
                return True
        return False

    def _is_stop_speaking(self, lower: str) -> bool:
        """Generic 'be quiet / stop / cancel'.

        Outside guidance only — inside guidance, ``_is_guidance_done``
        owns these phrases and exits guidance.
        """
        s = lower.strip()
        words = set(s.replace(",", " ").replace(".", " ").split())
        for phrase in _STOP_SPEAKING_PHRASES:
            if s == phrase:
                return True
            if " " in phrase and phrase in s:
                return True
            if " " not in phrase and phrase in words and len(words) <= 3:
                return True
        return False

    async def _stop_speaking(self, pid: str) -> None:
        """Flush queued TTS and send a brief text-only acknowledgement.

        Deliberately does NOT TTS the ack — the wearer just asked us to
        stop talking, speaking back would defeat the request.
        """
        await self._flush_audio(pid)
        await self._send_text(pid, "Okay.", _AGENT_PROGRESS_TOPIC)
        _trace_log.info("STOP_SPEAKING  pid=%s", pid)

    # ── demo actions ──────────────────────────────────────────────────────────

    async def _handle_demo_start(self, name: str, pid: str) -> None:
        # Starting a new recording supersedes any pending "did you
        # mean…?" prompt — the wearer is moving on.
        self._clear_pending_guidance(pid)
        self._memory.start_recording(name)
        response = (
            f"Recording '{name}'. Go ahead and demonstrate — "
            "I'm watching your hands and will capture each step."
        )
        await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
        await self._say(pid, response)
        _trace_log.info("DEMO_START  name=%s", name)

    async def _handle_demo_end(self, pid: str, ref_us: int) -> None:
        """Synchronous part of stop-recording: announce, then kick off the
        async analysis pipeline.

        Returns as soon as the "analyzing now" message has been sent. All
        subsequent user-facing sends (ping, analysis-failed message,
        "Saved" announcement) live in ``_finalize_demo`` and are gated
        by ``AgentMemory.is_current_finalization`` so a stale task whose
        demo was cleared / re-recorded cannot speak.
        """
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

        task = asyncio.create_task(
            self._finalize_demo(demo, demo.finalize_generation, pid, n_frames),
            name=f"finalize-{demo.name}",
        )
        self._finalize_tasks.add(task)
        task.add_done_callback(self._finalize_tasks.discard)

    async def _finalize_demo(
        self,
        demo:       Demonstration,
        generation: int,
        pid:        str,
        n_frames:   int,
    ) -> None:
        """Analyze *demo* and populate its steps, then announce.

        Every send_text / say in here goes through ``_say_if_current`` so
        a stale task (clear-demos OR re-record under the same name) is
        silently dropped instead of confusing the user with messages
        about a demo that no longer exists.

        The 10 s "still analyzing" ping is a child task launched here so
        it shares cancellation with this method.
        """
        async def _say_if_current(
            text:   str,
            *,
            topic:  str | None  = None,
            speak:  bool        = True,
        ) -> None:
            if not self._memory.is_current_finalization(demo, generation):
                _trace_log.info(
                    "DEMO_FINALIZE_SUPPRESSED_SEND  demo=%s  text=%r",
                    demo.name, text[:60],
                )
                return
            if topic is not None:
                await self._send_text(pid, text, topic)
            if speak:
                await self._say(pid, text)

        async def _ping() -> None:
            await asyncio.sleep(10)
            await _say_if_current(
                "Still analyzing — reviewing key frames…",
                topic=_AGENT_PROGRESS_TOPIC,
                speak=False,
            )

        ping = asyncio.create_task(_ping())
        try:
            try:
                overview, instructions = await self._analyze_recording(demo)
            except asyncio.CancelledError:
                _trace_log.info("DEMO_FINALIZE_CANCELLED  name=%s", demo.name)
                raise

            if not instructions:
                import os as _os
                run_dir  = _os.environ.get("XR_RUN_DIR", "/tmp")
                safe     = re.sub(r"[^a-zA-Z0-9_-]", "_", demo.name)[:40]
                log_path = f"{run_dir}/recordings/{safe}_{demo.started_at_us}.jsonl"
                _trace_log.info("DEMO_ANALYSIS_FAILED  log=%s", log_path)
                await _say_if_current(
                    f"Recorded {n_frames} frames but analysis failed. "
                    f"Raw observations saved to: {log_path}",
                    topic=_AGENT_RESPONSE_TOPIC,
                    speak=False,
                )
                await _say_if_current(
                    "Recording saved but analysis failed.",
                    speak=True,
                )
                return

            # Identity + generation check BEFORE mutating the demo — a
            # stale task (clear-demos or re-record under same name) must
            # not touch the demo dict's current entry.
            if not self._memory.is_current_finalization(demo, generation):
                _trace_log.info("DEMO_FINALIZE_STALE  name=%s", demo.name)
                return

            demo.summary      = overview
            demo.instructions = instructions

            n      = len(instructions)
            notes  = demo.voice_notes  # sorted by timestamp (added in order)
            frames = demo.recorded_frames

            span = max(demo.ended_at_us - demo.started_at_us, 1)
            previous_path = ""
            for i, instr in enumerate(instructions):
                ts    = demo.started_at_us + int(i / max(n - 1, 1) * span)
                frame, backups, scores = _select_reference_frames_for_step(
                    frames, notes, i, instructions, demo.started_at_us, demo.ended_at_us,
                )
                reliable = frame is not None
                text_video_mismatch = False
                if frame is not None:
                    _score, instruction_match = _candidate_score(
                        frame,
                        instr,
                        instructions[i + 1] if i + 1 < len(instructions) else "",
                    )
                    primary_tokens = _primary_object_tokens(instr)
                    caption_tokens = _frame_score_tokens(frame.description)
                    text_video_mismatch = (
                        not instruction_match
                        or bool(primary_tokens and not (primary_tokens & caption_tokens))
                    )
                if frame is not None and backups:
                    score_by_idx = {idx: score for idx, score, _ok in scores}
                    best_score = score_by_idx.get(frame.frame_idx)
                    if any(score_by_idx.get(b.frame_idx) == best_score for b in backups):
                        vlm_frame = await self._select_reference_frame_with_vlm(
                            instr, [frame, *backups[:3]],
                        )
                        if vlm_frame is not None:
                            frame = vlm_frame
                if frame is not None and previous_path and frame.image_path == previous_path:
                    reliable = False
                if frame is not None and not reliable and text_video_mismatch:
                    reliable = True
                path    = frame.image_path  if frame and reliable else ""
                caption = frame.description if frame and reliable else ""
                requirements = await self._derive_step_requirements(instr, caption)
                key_info = await self._extract_step_key_info(instr, caption, requirements)
                demo.steps.append(DemoStep(
                    step_number           = i + 1,
                    timestamp_us          = ts,
                    description           = instr,
                    image_path            = path,
                    teacher_caption       = caption,
                    expected_requirements = requirements,
                    key_info              = key_info,
                    reference_image_paths = (
                        [f.image_path for f in [frame, *backups] if f is not None]
                        if reliable else []
                    ),
                    reference_reliable    = reliable,
                    text_video_mismatch   = text_video_mismatch and reliable,
                ))
                if reliable:
                    previous_path = path
                _trace_log.info(
                    "STEP_FRAME_CANDIDATES  step=%d  selected=%s  reliable=%s  scores=%s",
                    i + 1,
                    frame.frame_idx if frame else "none",
                    "YES" if reliable else "NO",
                    " ".join(f"{idx}:{score}:{'Y' if ok else 'N'}" for idx, score, ok in scores),
                )
                if reliable and text_video_mismatch:
                    _trace_log.info(
                        "STEP_TEXT_VIDEO_MISMATCH  step=%d  instruction=%r  caption=%r",
                        i + 1, instr[:80], caption[:120],
                    )

            if demo.recorded_frames:
                _trace_log.info("STEP_FRAMES  %s",
                                " | ".join(
                                    f"{s.step_number}:{s.image_path[-20:]}"
                                    for s in demo.steps if s.image_path
                                ))

            _trace_log.info("DEMO_INSTRUCTIONS  %s",
                            " | ".join(f"{i+1}:{s[:40]}" for i, s in enumerate(instructions)))

            _trace_log.info(
                "STEP_REQUIREMENTS  %s",
                " | ".join(
                    f"{s.step_number}:[{', '.join(s.expected_requirements)}]"
                    for s in demo.steps
                ),
            )

            _trace_log.info(
                "STEP_KEY_INFO  %s",
                " | ".join(
                    f"{s.step_number}:obj={s.key_info.objects} act={s.key_info.action!r} "
                    f"pos={s.key_info.position!r} state={s.key_info.target_state!r}"
                    for s in demo.steps if s.key_info is not None
                ) or "none",
            )

            response = f"Saved '{demo.name}' with {n} steps."
            if overview:
                response += f" {overview}"
            await _say_if_current(response, topic=_AGENT_RESPONSE_TOPIC, speak=False)
            await _say_if_current(
                f"Saved demonstration '{demo.name}' with {n} steps.",
                speak=True,
            )
            _trace_log.info(
                "DEMO_FINALIZED  name=%s  steps=%d  duration=%.1fs",
                demo.name, n, (demo.ended_at_us - demo.started_at_us) / 1_000_000,
            )
        finally:
            ping.cancel()
            demo.is_finalizing = False

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

    async def _derive_step_requirements(
        self, instruction: str, teacher_caption: str
    ) -> list[str]:
        """Generate an atomic visual checklist for one step.

        Failure falls back to a small deterministic checklist for common
        imperative instructions.
        """
        if not instruction.strip():
            return []
        fallback = _fallback_step_requirements(instruction)
        try:
            result = await self._nat_runtime.call_tool(
                "glasses_worker_tasks",
                "derive_step_requirements",
                {
                    "instruction": instruction,
                    "teacher_caption": teacher_caption,
                },
            )
        except Exception:
            log.exception("derive_step_requirements call failed")
            if fallback:
                _trace_log.info("STEP_REQUIREMENTS_FALLBACK  %s", ", ".join(fallback))
            return fallback
        if not isinstance(result, dict):
            return fallback
        raw = result.get("requirements", [])
        if not isinstance(raw, list):
            return fallback
        out = [str(r).strip() for r in raw if str(r).strip()]
        if out:
            return out[:4]
        if fallback:
            _trace_log.info("STEP_REQUIREMENTS_FALLBACK  %s", ", ".join(fallback))
        return fallback

    async def _extract_step_key_info(
        self, instruction: str, teacher_caption: str, requirements: list[str],
    ) -> StepKeyInfo | None:
        """Distill a step into structured key info via the NAT worker task.

        Returns None on failure — guidance then falls back to the existing
        instruction + requirements checklist (no regression).
        """
        if not instruction.strip():
            return None
        try:
            result = await self._nat_runtime.call_tool(
                "glasses_worker_tasks",
                "derive_step_key_info",
                {
                    "instruction":     instruction,
                    "teacher_caption": teacher_caption,
                    "requirements":    list(requirements or []),
                },
            )
        except Exception:
            log.exception("derive_step_key_info call failed")
            return None
        if not isinstance(result, dict):
            return None
        info = StepKeyInfo(
            objects=[str(o).strip() for o in result.get("objects", []) if str(o).strip()][:3],
            action=str(result.get("action", "")).strip(),
            position=str(result.get("position", "")).strip(),
            target_state=str(result.get("target_state", "")).strip(),
            ignore=[str(i).strip() for i in result.get("ignore", []) if str(i).strip()],
        )
        return None if info.is_empty() else info

    async def _select_reference_frame_with_vlm(
        self, instruction: str, candidates: list[RecordedFrame],
    ) -> RecordedFrame | None:
        """Tie-break ambiguous teacher reference candidates with one VLM call."""
        candidates = candidates[:4]
        if len(candidates) <= 1:
            return candidates[0] if candidates else None
        labels = "\n".join(
            f"Image {i + 1}: frame_idx={frame.frame_idx}; caption={frame.description}"
            for i, frame in enumerate(candidates)
        )
        question = (
            f"Instruction: {instruction}\n\n"
            f"{labels}\n\n"
            "Choose the ONE image that best shows the completed teacher reference "
            "state for the instruction. Prefer end-state evidence over merely "
            "showing the object. Output only JSON: {\"best\": <image number>}."
        )
        try:
            result = await asyncio.wait_for(
                self._nat_runtime.call_tool(
                    "vlm_mcp",
                    "ask_frames",
                    {
                        "question": question,
                        "image_paths": [frame.image_path for frame in candidates],
                    },
                ),
                timeout=12.0,
            )
        except Exception:
            log.debug("reference-frame VLM tie-break failed", exc_info=True)
            return None

        if isinstance(result, dict):
            best = result.get("best")
        else:
            text = str(result or "")
            obj_text = _extract_json(text)
            if not obj_text:
                return None
            try:
                best = json.loads(obj_text).get("best")
            except Exception:
                return None
        try:
            idx = int(best) - 1
        except (TypeError, ValueError):
            return None
        if 0 <= idx < len(candidates):
            _trace_log.info(
                "STEP_FRAME_VLM_SELECT  instruction=%r  selected=%d",
                instruction[:50], candidates[idx].frame_idx,
            )
            return candidates[idx]
        return None

    # ── guidance actions ──────────────────────────────────────────────────────

    async def _handle_guidance_request(self, query: str, pid: str) -> None:
        """Find a matching demo and enter guidance mode.

        First tries strict fuzzy-matching against demo names. If no demo
        matches strictly and there's more than one on file, sets pending
        disambiguation state and re-asks with a numbered list — the
        wearer's next utterance is interpreted by the pending branch in
        ``handle()``. For ambiguous post-demo utterances inside the
        freshness window use ``_handle_ambiguous_guidance_request``
        instead — it explicitly prefers the most recent demo.
        """
        demo = self._memory.find_demonstration_fuzzy(query, min_confidence="strict")
        if demo is None:
            demos = self._memory.list_demonstrations()
            if len(demos) > 1:
                # Ambiguous: store choices keyed by pid so the wearer's
                # numeric / by-name reply gets routed back into
                # _start_guidance via the pending branch.
                self._pending_guidance_by_pid[pid] = {
                    "query":    query,
                    "choices":  list(demos),
                    "attempts": 0,
                }
                _trace_log.info(
                    "PENDING_SET  pid=%s  query=%r  choices=%s",
                    pid, query[:60], demos,
                )
                await self._reask_with_choices(pid, attempts=0)
                return
            if demos:
                demo = self._memory.get_demonstration(demos[0])
        if demo is None:
            response = "I don't have any recorded demonstrations yet. Show me first!"
            await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
            await self._say(pid, response)
            return
        await self._start_guidance(demo, pid)

    async def _handle_guidance_request_during_guidance(self, query: str, pid: str) -> None:
        """Restart/switch guidance when the wearer asks for guidance again."""
        current = self._guidance_demo
        demo = self._memory.find_demonstration_fuzzy(query, min_confidence="strict")
        if demo is not None:
            await self._start_guidance(demo, pid)
            return
        if current is None:
            await self._handle_guidance_request(query, pid)
            return

        query_tokens = set(_tokenize(query))
        other_names = [
            name for name in self._memory.list_demonstrations()
            if name != current.name
        ]
        mentions_other = any(
            query_tokens & set(_tokenize(name))
            for name in other_names
        )
        if mentions_other and other_names:
            self._pending_guidance_by_pid[pid] = {
                "query":    query,
                "choices":  [current.name, *other_names],
                "attempts": 0,
            }
            _trace_log.info(
                "PENDING_SET  pid=%s  query=%r  choices=%s",
                pid, query[:60], [current.name, *other_names],
            )
            await self._reask_with_choices(pid, attempts=0)
            return

        _trace_log.info("GUIDANCE_RESTART  demo=%s  query=%r", current.name, query[:60])
        await self._start_guidance(current, pid)

    def _clear_pending_guidance(self, pid: str) -> None:
        """Drop pending disambiguation state for *pid* if present.

        Safe to call unconditionally — used in every code path that
        starts a different mode, finishes guidance, or changes the demo
        set, so a stale "did you mean…?" prompt never lingers.
        """
        if self._pending_guidance_by_pid.pop(pid, None) is not None:
            _trace_log.info("PENDING_CLEAR  pid=%s", pid)

    def _resolve_pending_choice(self, lower: str, pid: str) -> Demonstration | None:
        """Interpret a pending-disambiguation reply.

        Accepts either a numeric/ordinal answer against the stored
        choices list (1-based), or a free-form name match against any
        demo in the choices list using lenient fuzzy matching (since the
        wearer is explicitly answering "which demo?", a single unique
        token is enough evidence).
        """
        state = self._pending_guidance_by_pid.get(pid)
        if not state:
            return None
        choices: list[str] = state.get("choices", [])
        if not choices:
            return None

        n = _extract_choice_number(lower)
        if n is not None and 1 <= n <= len(choices):
            return self._memory.get_demonstration(choices[n - 1])

        candidate = self._memory.find_demonstration_fuzzy(lower, min_confidence="lenient")
        if candidate is not None and candidate.name in choices:
            return candidate
        return None

    async def _reask_with_choices(self, pid: str, attempts: int) -> None:
        """Ask "did you mean 1) X or 2) Y?" — with a hint on re-asks."""
        state = self._pending_guidance_by_pid.get(pid)
        if not state:
            return
        choices: list[str] = state.get("choices", [])
        if not choices:
            return
        numbered = ", ".join(f"{i + 1}) '{name}'" for i, name in enumerate(choices))
        if attempts == 0:
            response = (
                f"I have a few demos saved — did you mean {numbered}? "
                "Reply with the number or the name."
            )
        else:
            response = (
                f"I didn't catch that. Please say the number or the exact name: {numbered}."
            )
        await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
        await self._say(pid, response)

    async def _handle_ambiguous_guidance_request(
        self, transcript: str, pid: str
    ) -> None:
        """Fallback for ambiguous post-demo utterances inside the freshness window.

        Prefer a strict name match against any recorded demo — if the
        utterance is STT-garbled but still contains enough distinctive
        tokens to identify one demo unambiguously (e.g. STT heard "show
        me how to wear pickle headset" with a 'pico headset' demo on
        file), guide that one. Only fall back to the most recent demo
        when no demo is a strict match. Strict mode here means we
        require either a (unique) substring match or two-token overlap,
        so a stray utterance with one weak token in common doesn't
        silently override the recent demo.
        """
        demo = self._memory.find_demonstration_fuzzy(transcript, min_confidence="strict")
        via = "name"
        if demo is None:
            demo = self._memory.most_recent_demo()
            via = "recent"
        if demo is None:
            log.info("ambiguous guidance request but no demo on file  pid=%s", pid)
            return
        _trace_log.info(
            "GUIDANCE_FALLBACK  via=%s  utterance=%r  demo=%s",
            via, transcript[:60], demo.name,
        )
        await self._start_guidance(demo, pid)

    async def _start_guidance(self, demo: Demonstration, pid: str) -> None:
        """Enter guidance mode for *demo*. Shared by name-match and freshness paths."""
        # We're committing to a demo — any pending "did you mean…?"
        # prompt is now resolved.
        self._clear_pending_guidance(pid)
        if demo.is_finalizing:
            response = (
                f"I'm still analyzing '{demo.name}' — "
                "try again in a few seconds."
            )
            await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
            await self._say(pid, response)
            _trace_log.info("GUIDANCE_BLOCKED_FINALIZING  demo=%s", demo.name)
            return
        if not demo.steps:
            response = f"The demonstration '{demo.name}' has no steps recorded."
            await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
            await self._say(pid, response)
            return

        if self._guidance_monitor_task and not self._guidance_monitor_task.done():
            self._guidance_monitor_task.cancel()
            self._guidance_monitor_task = None
        self._guidance_advancing = False

        self._guidance_demo = demo
        self._guidance_step = 0
        self._guidance_started_at_us = _now_us()
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
        self._clear_pending_guidance(pid)
        demo = self._guidance_demo
        self._guidance_demo = None
        self._guidance_step = 0
        self._guidance_started_at_us = 0
        self._guidance_step_spoken_at_us = 0
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
        if (_has_completion_stem(lower)
            or any(p in lower for p in ("doing it right", "doing step", "correct"))):
            result = await self._guidance_completion_result(pid)
            issue       = str(result.get("issue", "")).strip() if isinstance(result, dict) else ""
            current_obs = str(result.get("current_observation", "")).strip() if isinstance(result, dict) else ""
            completed   = bool(result.get("completed", False)) if isinstance(result, dict) else False
            missing = result.get("missing_or_mismatched", []) if isinstance(result, dict) else []
            # The parser may set `issue` to a debug-y reject reason
            # (``vlm returned non-json``, ``missing requirement: ...``,
            # ``visible without evidence``, ``no grounded evidence``).
            # Those are for the trace log, not the wearer. Speak `issue`
            # only when it is human-authored (e.g. ``blue circle not
            # white``) — i.e. it does NOT start with one of the parser
            # prefixes.
            _PARSER_REASONS = (
                "vlm ", "missing requirement", "visible without", "no grounded",
                "waiting for a fresh student frame", "text-video-mismatch",
            )
            if issue.lower().startswith("unreliable-reference"):
                reply = (
                    "I cannot reliably check this step from the demo video. "
                    "Please re-record this step or say next to continue."
                )
                await self._send_text(pid, reply, _AGENT_RESPONSE_TOPIC)
                await self._say(pid, reply)
                _trace_log.info(
                    "GUIDANCE_QUESTION_CHECK  step=%d  completed=NO  issue=%r  obs=%r",
                    self._guidance_step, issue[:80], current_obs[:80],
                )
                return
            if issue.lower().startswith(_PARSER_REASONS):
                if isinstance(missing, list) and missing:
                    spoken_issue = f"{missing[0]} not visible"
                else:
                    spoken_issue = ""
            else:
                spoken_issue = issue
            if completed:
                reply = f"Yes, step {step_num} looks complete. {instruction}"
            else:
                if spoken_issue:
                    detail = f" {spoken_issue}"
                elif current_obs:
                    detail = f" I see: {current_obs}"
                else:
                    detail = ""
                reply = f"Not yet. Step {step_num} is: {instruction}.{detail}"
            await self._send_text(pid, reply, _AGENT_RESPONSE_TOPIC)
            await self._say(pid, reply)
            _trace_log.info(
                "GUIDANCE_QUESTION_CHECK  step=%d  completed=%s  issue=%r  obs=%r",
                self._guidance_step, "YES" if completed else "NO",
                (spoken_issue or issue)[:80], current_obs[:80],
            )
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
        obs_now = len(self._memory._observations)
        self._guidance_step_spoken_at_us = _now_us()
        self._guidance_step_obs_baseline   = obs_now
        self._guidance_check_obs_baseline  = obs_now
        self._guidance_monitor_idle_cycles = 0
        self._guidance_consecutive_yes     = 0
        self._guidance_last_result         = {}
        self._guidance_correction_state    = {}
        self._guidance_last_checked_live_ts = 0
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
        """Actively check guidance progress on a fixed cadence."""
        _trace_log.info("GUIDANCE_MONITOR  start  step=%d", self._guidance_step)
        try:
            while self._guidance_demo is not None:
                await asyncio.sleep(self._cfg.guidance_check_interval_s)
                if self._guidance_demo is None:
                    break

                step_idx = self._guidance_step
                need = self._cfg.guidance_advance_confirmations

                # Static-frame skip: if the hub hasn't delivered a newer live
                # frame since the last grounded check, the verdict would be
                # identical — skip the expensive VLM compare. Cheap metadata
                # fetch only; leaves the yes-streak and correction state
                # untouched, so it is strictly behavior-preserving.
                if (self._cfg.guidance_skip_static_frames
                        and self._guidance_last_checked_live_ts):
                    live_ts = await self._latest_live_frame_ts(pid)
                    if live_ts and live_ts == self._guidance_last_checked_live_ts:
                        _trace_log.info(
                            "GUIDANCE_MONITOR  step=%d  skip=static-frame  ts=%d",
                            step_idx, live_ts,
                        )
                        continue

                current_obs = len(self._memory._observations)
                delta_since_last = current_obs - self._guidance_check_obs_baseline
                self._guidance_check_obs_baseline = current_obs

                _trace_log.info(
                    "GUIDANCE_MONITOR  step=%d  delta=%d  check=YES  need=%d",
                    step_idx, delta_since_last, need,
                )

                completed, has_evidence = await self._vlm_step_complete(
                    pid, require_reliable_reference=True,
                )
                checked_ts = int(self._guidance_last_result.get("timestamp_us", 0) or 0)
                if checked_ts:
                    self._guidance_last_checked_live_ts = checked_ts
                if completed and has_evidence:
                    self._guidance_consecutive_yes += 1
                    self._guidance_correction_state = {}
                    _trace_log.info(
                        "GUIDANCE_MONITOR  yes-count=%d/%d  step=%d",
                        self._guidance_consecutive_yes, need, step_idx,
                    )
                    if self._guidance_consecutive_yes >= need:
                        self._guidance_consecutive_yes = 0
                        _trace_log.info(
                            "GUIDANCE_MONITOR  vlm-advance  step=%d  need=%d",
                            step_idx, need,
                        )
                        await self._advance_guidance(pid)
                else:
                    if self._guidance_consecutive_yes:
                        _trace_log.info(
                            "GUIDANCE_MONITOR  streak-reset  step=%d  was=%d  reason=%s",
                            step_idx, self._guidance_consecutive_yes,
                            "no-evidence" if completed else "no",
                        )
                    self._guidance_consecutive_yes = 0
                    await self._maybe_send_guidance_correction(
                        pid, step_idx, self._guidance_last_result,
                    )
        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("guidance monitor error")
        _trace_log.info("GUIDANCE_MONITOR  exit")

    async def _latest_live_frame_ts(self, pid: str) -> int:
        """Cheap metadata-only fetch of the latest live frame timestamp.

        Used by the auto-advance monitor to detect whether a newer frame has
        arrived since the last grounded check. Returns 0 when no frame is
        available; never raises.
        """
        data = await self._call_video(
            "get_latest_frame", {"participant_id": pid}, silent=True,
        )
        if isinstance(data, dict):
            try:
                return int(data.get("timestamp_us", 0) or 0)
            except (TypeError, ValueError):
                return 0
        return 0

    async def _guidance_completion_result(
        self, pid: str, *, require_reliable_reference: bool = False,
    ) -> dict:
        if self._guidance_demo is None:
            return {}
        instruction = self._instruction_for_step(self._guidance_demo)
        expected_requirements: list[str] = []
        teacher_image_path = ""
        teacher_caption = ""
        reference_reliable = False
        key_objects: list[str] = []
        key_action = ""
        key_position = ""
        key_target_state = ""
        key_ignore: list[str] = []
        if 0 <= self._guidance_step < len(self._guidance_demo.steps):
            step = self._guidance_demo.steps[self._guidance_step]
            expected_requirements = list(step.expected_requirements)
            reference_reliable = bool(step.reference_reliable and step.image_path)
            teacher_image_path = step.image_path if reference_reliable else ""
            teacher_caption = step.teacher_caption
            if step.key_info is not None:
                key_objects      = list(step.key_info.objects)
                key_action       = step.key_info.action
                key_position     = step.key_info.position
                key_target_state = step.key_info.target_state
                key_ignore       = list(step.key_info.ignore)
        if require_reliable_reference and not reference_reliable:
            result = {
                "completed": False,
                "current_observation": "",
                "checks": [],
                "missing_or_mismatched": [],
                "image_path": "",
                "timestamp_us": 0,
                "issue": "unreliable-reference",
            }
            _trace_log.info(
                "GUIDANCE_FRAME_PAIR  step=%d  teacher=none  reliable=NO  live=none  live_ts=0  min_ts=%d  issue=%r",
                self._guidance_step,
                max(self._guidance_started_at_us, self._guidance_step_spoken_at_us),
                result["issue"],
            )
            return result
        min_live_timestamp_us = max(
            self._guidance_started_at_us,
            self._guidance_step_spoken_at_us,
        )
        try:
            result = await asyncio.wait_for(
                self._nat_runtime.call_tool(
                    "glasses_worker_tasks",
                    "check_guidance_step_complete",
                    {
                        "participant_id":        pid,
                        "instruction":           instruction,
                        "expected_requirements": expected_requirements,
                        "teacher_image_path":    teacher_image_path,
                        "teacher_caption":       teacher_caption,
                        "min_live_timestamp_us": min_live_timestamp_us,
                        "key_objects":           key_objects,
                        "key_action":            key_action,
                        "key_position":          key_position,
                        "key_target_state":      key_target_state,
                        "key_ignore":            key_ignore,
                    },
                    participant_id=pid,
                ),
                timeout=10.0,
            )
        except (asyncio.TimeoutError, Exception):
            result = {}
        if isinstance(result, dict):
            _trace_log.info(
                "GUIDANCE_FRAME_PAIR  step=%d  teacher=%s  reliable=%s  live=%s  live_ts=%s  min_ts=%d  issue=%r",
                self._guidance_step,
                teacher_image_path or "none",
                "YES" if reference_reliable else "NO",
                str(result.get("image_path", "") or "none"),
                str(result.get("timestamp_us", 0) or 0),
                min_live_timestamp_us,
                str(result.get("issue", ""))[:80],
            )
        return result if isinstance(result, dict) else {}

    async def _maybe_send_guidance_correction(
        self, pid: str, step_idx: int, result: dict,
    ) -> None:
        if not isinstance(result, dict):
            return
        current_obs = str(result.get("current_observation", "")).strip()
        missing = result.get("missing_or_mismatched", [])
        missing_key = ""
        if isinstance(missing, list) and missing:
            missing_key = str(missing[0]).strip()
        issue = str(result.get("issue", "")).strip()
        if not issue and missing_key:
            issue = f"{missing_key} not visible"
        if not issue:
            return
        parser_reasons = (
            "vlm ", "missing requirement", "visible without", "no grounded",
            "i cannot see", "waiting for a fresh student frame",
            "unreliable-reference", "text-video-mismatch",
        )
        if issue.lower().startswith(parser_reasons):
            if missing_key and current_obs:
                issue = f"{missing_key} not visible"
            else:
                if issue.lower().startswith("unreliable-reference"):
                    issue = (
                        "I cannot reliably check this step from the demo video. "
                        "Please re-record this step or say next to continue."
                    )
                else:
                    return

        now_us = _now_us()
        state = self._guidance_correction_state
        key = missing_key or issue
        same_key = state.get("step_idx") == step_idx and state.get("key") == key
        count = int(state.get("count", 0)) + 1 if same_key else 1
        last_spoken_us = int(state.get("last_spoken_us", 0)) if same_key else 0
        state.update({
            "step_idx": step_idx,
            "key": key,
            "issue": issue,
            "count": count,
            "last_spoken_us": last_spoken_us,
        })

        min_gap_us = int(max(8.0, self._cfg.guidance_check_interval_s * 4) * 1_000_000)
        if last_spoken_us and now_us - last_spoken_us < min_gap_us:
            return

        response = issue if issue.startswith("I cannot reliably") else f"Not yet. {issue}."
        await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
        await self._say(pid, response)
        state["last_spoken_us"] = now_us
        _trace_log.info("GUIDANCE_CORRECTION  step=%d  issue=%s", step_idx, issue[:80])

    async def _vlm_step_complete(
        self, pid: str, *, require_reliable_reference: bool = False,
    ) -> tuple[bool, bool]:
        """Ask the NAT guidance task whether the current step looks done.

        Returns ``(completed, has_evidence)``:
          - ``completed`` is True only when the parser accepted the
            grounded YES (every expected requirement covered with
            non-empty evidence).
          - ``has_evidence`` is True iff the response carried a non-empty
            ``current_observation`` AND at least one check has non-empty
            ``evidence``. Malformed / missing-fields collapse to
            ``(False, False)``.
        """
        result = await self._guidance_completion_result(
            pid, require_reliable_reference=require_reliable_reference,
        )
        self._guidance_last_result = result if isinstance(result, dict) else {}
        if not isinstance(result, dict):
            return False, False
        completed = bool(result.get("completed", False))
        current_obs = str(result.get("current_observation", "")).strip()
        checks = result.get("checks", [])
        if not isinstance(checks, list):
            checks = []
        has_evidence = bool(current_obs) and any(
            isinstance(c, dict)
            and c.get("visible")
            and str(c.get("evidence", "")).strip()
            for c in checks
        )
        instruction = self._instruction_for_step(self._guidance_demo) if self._guidance_demo else ""
        _trace_log.info(
            "GUIDANCE_MONITOR  vlm-check  %r → %s  evidence=%s  obs=%r",
            instruction[:40],
            "YES" if completed else "NO",
            "YES" if has_evidence else "NO",
            current_obs[:40],
        )
        if not (completed and has_evidence):
            _trace_log.info(
                "GUIDANCE_CHECK_RAW  step=%d  reason=%r  raw=%r",
                self._guidance_step,
                str(result.get("issue", ""))[:120],
                str(result.get("raw_vlm", ""))[:200],
            )
        return completed, has_evidence

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

        # Inject the fresh-demo marker ONLY inside the freshness window.
        # The system_prompt's __defer_to_worker__ rule is scoped to this
        # marker being present, so we never accidentally hand off when the
        # worker isn't actually expecting a guidance follow-up.
        if (
            self._guidance_demo is None
            and self._memory.recording is None
            and self._memory.demo_is_fresh(self._cfg.guidance_freshness_window_s)
        ):
            fresh = self._memory.most_recent_demo()
            if fresh is not None:
                ctx_parts.append(
                    f"[Fresh demo: {fresh.name!r}] "
                    "The wearer just finished recording this demo. If they're "
                    "asking to be guided through it (even ambiguously, even "
                    "if the transcript looks garbled), output "
                    "__defer_to_worker__ — the worker will start guidance."
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
        for task in list(self._finalize_tasks):
            task.cancel()
        if self._finalize_tasks:
            await asyncio.gather(*self._finalize_tasks, return_exceptions=True)
        await self._http.aclose()

