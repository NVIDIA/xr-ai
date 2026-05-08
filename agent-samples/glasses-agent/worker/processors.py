# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
QueryProcessor — handles all user utterances for the glasses agent.

Flow per utterance:
  1. Demo-mode detection (start / end / guidance request / guidance advance).
  2. _quick_ack() — fast Llama-Nemotron call: spoken ack + think=True/False.
  3. _agentic_loop() — up to 8 iterations of OpenAI tool calling:
       - Pre-fetch latest frame concurrently while building context.
       - Context = memory.build_context() + frame path + conversation history.
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
import string
import time
from typing import Callable, Awaitable

import httpx
from fastmcp import Client as McpClient

from config import WorkerConfig
from memory import AgentMemory, Demonstration, Observation

log = logging.getLogger("glasses_agent.processors")

_trace_log = logging.getLogger("glasses_agent.trace")

_MAX_LOOP    = 8

# ── demo detection phrases ────────────────────────────────────────────────────

_DEMO_START_PHRASES = (
    "let me show you",
    "watch what i",
    "i'll demonstrate",
    "start recording",
    "start demo",
    "start a demo",
    "capture a demo",
    "record a demo",
    "begin demo",
    "begin recording",
    "remember this",
    "remember how",
    "watch me",
    "watch how i",
)

_DEMO_END_PHRASES = (
    "that's it",
    "stop recording",
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
)

_AGENT_RESPONSE_TOPIC = "agent.response"
_AGENT_PROGRESS_TOPIC = "agent.progress"

# Tools routed to video-mcp (union of live-only and recording-enabled sets).
_VIDEO_TOOLS = frozenset({
    "get_latest_frame",
    "get_frame_from_time",
    "list_live_participants",
    "list_recorded_participants",
    "get_video_stats",
    "query_video",
})
# Tools routed to vlm-mcp.
_VLM_TOOLS = frozenset({"ask_image"})


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
        vlm_client:   McpClient,
        video_client: McpClient,
        system_prompt: str,
        tools_openai:  list,
        *,
        send_text:    SendTextCb,
        say:          Callable[[str, str], Awaitable[None]],   # (pid, text)
    ) -> None:
        self._cfg          = cfg
        self._memory       = memory
        self._vlm          = vlm_client
        self._video        = video_client
        self._system_prompt = system_prompt
        self._tools_openai = tools_openai
        self._send_text    = send_text
        self._say          = say
        self._http         = httpx.AsyncClient(timeout=180.0)

        self._history:     list[tuple[str, str]] = []
        self._history_max  = 4

        # Guidance mode state.
        self._guidance_demo: Demonstration | None = None
        self._guidance_step: int = 0

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

        # ── demo-end detection ───────────────────────────────────────────────
        if self._memory.recording and self._is_demo_end(lower):
            await self._handle_demo_end(pid, ref_us)
            return

        # ── demo-start detection ─────────────────────────────────────────────
        demo_start = self._extract_demo_name(lower)
        if demo_start is not None:
            await self._handle_demo_start(demo_start, pid)
            return

        # ── guidance advance detection ────────────────────────────────────────
        if self._guidance_demo is not None:
            if self._is_guidance_done(lower):
                await self._finish_guidance(pid)
                return
            if self._is_guidance_advance(lower):
                await self._advance_guidance(pid)
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
        for phrase in _GUIDANCE_DONE_PHRASES:
            if lower.strip() == phrase:
                return True
        return False

    # ── demo actions ──────────────────────────────────────────────────────────

    async def _handle_demo_start(self, name: str, pid: str) -> None:
        self._memory.start_recording(name)
        response = f"Started recording '{name}'. I'll remember each step."
        await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
        await self._say(pid, response)
        _trace_log.info("DEMO_START  name=%s", name)

    async def _handle_demo_end(self, pid: str, ref_us: int) -> None:
        demo = self._memory.finish_recording()
        if demo is None or not demo.steps:
            response = "No demonstration was being recorded."
            await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
            await self._say(pid, response)
            return

        await self._send_text(pid, "Got it, processing demonstration…", _AGENT_PROGRESS_TOPIC)

        # Generate a concise summary via LLM.
        summary = await self._generate_demo_summary(demo)
        demo.summary = summary
        _trace_log.info("DEMO_END  name=%s  steps=%d  summary=%r",
                        demo.name, len(demo.steps), summary[:80])

        response = (
            f"Demonstration '{demo.name}' saved with {len(demo.steps)} steps. "
            + (f"Summary: {summary}" if summary else "")
        ).strip()
        await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
        await self._say(pid, f"Saved demonstration '{demo.name}' with {len(demo.steps)} steps.")

    async def _generate_demo_summary(self, demo: Demonstration) -> str:
        """Ask Llama-Nemotron to produce a concise summary + step descriptions."""
        steps_text = "\n".join(
            f"Step {s.step_number}: {s.description}" for s in demo.steps
        )
        body = {
            "model": "llm",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are summarizing a recorded demonstration for smart glasses. "
                        "Given the steps observed by the camera, write:\n"
                        "1. A 1-2 sentence summary of the overall procedure.\n"
                        "2. A numbered list of clear, actionable step descriptions "
                        "(one sentence each, imperative form: 'Place the X on Y').\n"
                        "Output ONLY the summary paragraph followed by the numbered list. "
                        "No markdown headers, no preamble."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Demonstration: '{demo.name}'\n\nObserved steps:\n{steps_text}",
                },
            ],
            "max_tokens": 512,
            "temperature": 0.3,
        }
        try:
            resp = await asyncio.wait_for(
                self._http.post(
                    self._cfg.llm_server.rstrip("/") + "/v1/chat/completions",
                    json=body,
                ),
                timeout=30.0,
            )
            if not resp.is_error:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            log.exception("demo summary generation failed")
        return ""

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

        self._guidance_demo = demo
        self._guidance_step = 0
        _trace_log.info("GUIDANCE_START  demo=%s  steps=%d", demo.name, len(demo.steps))
        await self._speak_current_guidance_step(pid)

    async def _advance_guidance(self, pid: str) -> None:
        if self._guidance_demo is None:
            return
        self._guidance_step += 1
        if self._guidance_step >= len(self._guidance_demo.steps):
            await self._finish_guidance(pid)
        else:
            await self._speak_current_guidance_step(pid)

    async def _finish_guidance(self, pid: str) -> None:
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

    async def _speak_current_guidance_step(self, pid: str) -> None:
        demo = self._guidance_demo
        if demo is None:
            return
        step = demo.steps[self._guidance_step]
        total = len(demo.steps)
        response = (
            f"Step {step.step_number} of {total}: {step.description} "
            f"Say 'next' when you're ready for the next step."
        )
        await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
        await self._say(pid, response)
        _trace_log.info("GUIDANCE_STEP  %d/%d  %s", step.step_number, total, step.description[:60])

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
        tool_names = {t["function"]["name"] for t in self._tools_openai}
        if "get_latest_frame" in tool_names:
            tool, args = "get_latest_frame", {"participant_id": pid}
        else:
            tool = "get_frame_from_time"
            args = {"participant_id": pid, "second_ago": 0, "reference_time_us": ref_us or 0}
        try:
            result = await self._video.call_tool(tool, args)
            data = _tool_payload(result)
            if isinstance(data, dict) and "path" in data:
                return data["path"]
        except Exception as exc:
            log.debug("pre-fetch frame failed: %s", exc)
        return None

    async def _agentic_loop(
        self,
        transcript:     str,
        pid:            str,
        *,
        ref_us:         int  = 0,
        needs_thinking: bool = False,
    ) -> str:
        """Multi-turn tool-calling loop using OpenAI tool calling protocol.

        Pre-fetches the latest frame concurrently while building context so
        the frame is ready before the first LLM call.
        """
        # Pre-fetch the latest frame concurrently.
        frame_task = asyncio.create_task(self._get_latest_frame_path(pid, ref_us))

        # Build context from memory.
        ctx_parts: list[str] = []
        ctx_parts.append(self._memory.build_context(max_recent=8))

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

        # Await the pre-fetched frame.
        frame_path = await frame_task
        if frame_path:
            ctx_parts.append(f"[Latest camera frame]\n{frame_path}")

        context = "\n\n".join(ctx_parts)
        _trace_log.info("CTX   %s", context.replace("\n", " | ")[:500])

        system_content = self._system_prompt
        if needs_thinking:
            system_content = (
                "Use your private <think> block to reason through the question. "
                "NEVER output these thoughts in your final response — "
                "only output a concise 1-3 sentence answer for the wearer.\n\n"
                + system_content
            )

        messages: list[dict] = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": (
                    f"[Context — use this before calling tools]\n"
                    f"{context}\n\n"
                    f"[User request]\n{transcript}"
                ),
            },
        ]

        for iteration in range(_MAX_LOOP):
            body = {
                "model": "llm",
                "messages": messages,
                "tools":       self._tools_openai,
                "max_tokens":  1024 if needs_thinking else 512,
                "temperature": 0.0,
                "chat_template_kwargs": {
                    "enable_thinking": needs_thinking,
                    **({"thinking_budget": 1024} if needs_thinking else {}),
                },
            }
            try:
                resp = await self._http.post(
                    self._cfg.agent_llm_server.rstrip("/") + "/v1/chat/completions",
                    json=body,
                )
                if resp.is_error:
                    log.error("agent-llm %s: %s", resp.status_code, resp.text[:300])
                    break
            except Exception:
                log.exception("agent-llm call failed on iteration %d", iteration)
                break

            choice     = resp.json()["choices"][0]
            message    = choice["message"]
            finish     = choice.get("finish_reason", "")
            tool_calls = message.get("tool_calls") or []
            content    = (message.get("content") or "").strip()

            log.info(
                "agent-llm iter=%d  finish=%s  tool_calls=%d  content=%r",
                iteration, finish, len(tool_calls), content[:200],
            )

            if not tool_calls:
                # Thinking exhausted the token budget without a tool call — retry
                # without thinking so the model can produce a text response.
                if finish == "length" and needs_thinking:
                    log.warning(
                        "iter=%d hit length limit during thinking — retrying without thinking",
                        iteration,
                    )
                    needs_thinking = False
                    continue
                _trace_log.info("RESP  %s", content or "Done.")
                return content or "Done."

            # Append assistant tool-call message.
            messages.append({
                "role":       "assistant",
                "content":    content or None,
                "tool_calls": tool_calls,
            })

            # Execute each tool call and append results.
            for tc in tool_calls:
                name     = tc["function"]["name"]
                args_str = tc["function"].get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {}

                log.info("tool call  iter=%d  tool=%s  args=%s", iteration, name, args)
                _trace_log.info(
                    "TOOL  [%d] %s(%s)", iteration, name,
                    ", ".join(f"{k}={v}" for k, v in args.items()),
                )
                result = await self._execute_tool(name, args)
                result_str = json.dumps(result, default=str)
                log.info("tool result  tool=%s  %s", name, result_str[:200])
                _trace_log.info("RES   [%d] %s → %s", iteration, name, result_str[:300])

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc["id"],
                    "content":      result_str,
                })

        return "Done."

    # ── tool routing ──────────────────────────────────────────────────────────

    async def _execute_tool(self, tool: str, args: dict) -> dict | str | None:
        if tool in _VLM_TOOLS:
            if tool == "ask_image":
                import os
                path = args.get("image_path", "")
                if path and not os.path.isfile(path):
                    return {
                        "error": (
                            f"File not found: {path!r}. "
                            "Call get_latest_frame or get_frame_from_time first to get a valid path."
                        )
                    }
            return await self._call_mcp(self._vlm, tool, args)
        if tool in _VIDEO_TOOLS:
            return await self._call_mcp(self._video, tool, args)
        return {"error": f"Unknown tool: {tool!r}"}

    async def _call_mcp(
        self, client: McpClient, tool: str, args: dict, *, silent: bool = False
    ) -> dict | str | None:
        try:
            result = await client.call_tool(tool, args)
            return _tool_payload(result)
        except Exception as exc:
            if not silent:
                log.error("mcp %s failed: %s", tool, exc)
            return {"error": str(exc)}

    async def close(self) -> None:
        await self._http.aclose()


# ── helpers ───────────────────────────────────────────────────────────────────

def _tool_payload(result) -> dict | list | str | None:
    # Prefer structured_content when FastMCP populates it (newer versions).
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured
    # Fall back to parsing the first TextContent item.
    items = getattr(result, "content", None) or []
    if items and hasattr(items[0], "text"):
        try:
            return json.loads(items[0].text)
        except Exception:
            return items[0].text
    return None
