# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""QueryProcessor — entry point for transcribed user utterances.

Routes each utterance to the right controller:
  - DemoController  : recording start / end / post-recording analysis
  - GuidanceController : enter / advance / finish + per-step monitor
  - else            : quick-ack + NAT tool-calling agent (_agentic_loop)

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
from typing import Awaitable, Callable

from xr_ai_models import ChatMessage, load_models_config, make_llm

from config import WorkerConfig
from demo_lifecycle import DemoController
from demo_phrases import (
    DEMO_END_PHRASES,
    extract_demo_name,
    is_demo_end,
    is_guidance_advance,
    is_guidance_done,
    match_guidance_request,
)
from guidance import GuidanceController
from mcp_shims import call_vlm, get_latest_frame_path
from memory import AgentMemory, VoiceNote
from nat_agent import NatAgentRunner
from nat_runtime import NatRuntime

log = logging.getLogger("glasses_agent_nat.processors")
_trace_log = logging.getLogger("glasses_agent_nat.trace")

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


SendTextCb = Callable[[str, str, str], Awaitable[None]]  # (pid, text, topic)


class QueryProcessor:
    """Dispatches transcribed utterances to demo / guidance / agentic flows."""

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
        self._nat_agent    = NatAgentRunner(nat_runtime)

        # Worker-side LLM client (used by _quick_ack and shared with
        # GuidanceController.handle_question). Built from the sample's
        # models.yaml per AGENTS.md — no hand-rolled httpx allowed.
        models_cfg         = load_models_config(cfg.models_yaml)
        self._worker_llm   = make_llm(models_cfg, "worker_llm")

        self._history:     list[tuple[str, str]] = []
        self._history_max  = 4

        self._demo = DemoController(
            memory      = memory,
            nat_runtime = nat_runtime,
            send_text   = send_text,
            say         = say,
        )
        self._guidance = GuidanceController(
            cfg         = cfg,
            memory      = memory,
            nat_runtime = nat_runtime,
            worker_llm  = self._worker_llm,
            send_text   = send_text,
            say         = say,
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
            response = f"Done — {n} demonstration{'s' if n != 1 else ''} cleared."
            await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
            await self._say(pid, response)
            _trace_log.info("DEMOS_CLEARED  count=%d", n)
            return

        # ── demo-end detection ───────────────────────────────────────────────
        if self._memory.recording and is_demo_end(lower):
            # If the utterance contains narration before the stop phrase, save it.
            for phrase in DEMO_END_PHRASES:
                if phrase in lower:
                    narration = text[:lower.index(phrase)].strip().rstrip(",.")
                    if len(narration) > 3:
                        self._memory.add_voice_note(
                            VoiceNote(timestamp_us=ref_us, text=narration)
                        )
                        _trace_log.info("VOICE_NOTE(pre-stop)  %s", narration[:80])
                    break
            await self._demo.handle_end(pid, ref_us)
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
        demo_start = extract_demo_name(lower)
        if demo_start is not None:
            await self._demo.handle_start(demo_start, pid)
            return

        # ── guidance mode: owns ALL utterances — nothing falls through ────────
        if self._guidance.is_active:
            if is_guidance_done(lower):
                await self._guidance.finish(pid)
            elif is_guidance_advance(lower):
                await self._guidance.advance(pid)
            else:
                # Questions / comments / noise all get a brief contextual reply.
                await self._guidance.handle_question(text, pid)
            return

        # ── guidance request detection ────────────────────────────────────────
        guidance_match = match_guidance_request(lower)
        if guidance_match is not None:
            await self._guidance.handle_request(guidance_match, pid)
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

        messages = [
            ChatMessage(
                role="system",
                content=(
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
            ),
            ChatMessage(role="user", content=context + transcript),
        ]
        try:
            resp = await asyncio.wait_for(
                self._worker_llm.chat(messages, max_tokens=40, temperature=0.0),
                timeout=8.0,
            )
            raw = (resp.content or "").strip()
            if raw:
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
        path = await get_latest_frame_path(self._nat_runtime, pid, ref_us)
        if not path:
            return None, None
        try:
            result = await asyncio.wait_for(
                call_vlm(
                    self._nat_runtime,
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
        ctx_parts: list[str] = []
        ctx_parts.append(self._memory.build_context(max_recent=8))

        if self._memory.recording is not None:
            ctx_parts.append(
                f"[Recording active — demo: {self._memory.recording.name!r}]"
            )

        guidance_ctx = self._guidance.context_block()
        if guidance_ctx is not None:
            ctx_parts.append(guidance_ctx)

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

    async def close(self) -> None:
        await self._guidance.close()
        await self._worker_llm.close()
