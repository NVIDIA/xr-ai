# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Brain processor for xr-render-demo.

The voice pipeline (input → VadStt → VoiceGate → brain → StreamingTts → output)
is assembled by ``xr_ai_pipecat.make_voice_pipeline``. This module supplies
the sample-specific brain — a multi-step agentic loop over render-mcp,
oxr-mcp, vec-mcp, vlm-mcp, and video-mcp tools.

Agentic loop (max ``_MAX_LOOP`` iterations):
  - Llama-Nemotron emits an OpenAI ``tool_calls`` payload → execute tool,
    append result, continue.
  - When the model returns text instead of a tool call, that text is the
    final user-visible response.

A parallel "quick-ack" call to Minitron fires at the start of each turn
to (a) speak an immediate acknowledgment and (b) classify whether the
agentic loop needs thinking enabled. A periodic "still-working" loop
streams contextual progress messages to the data channel while the agent
reasons.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import AsyncIterator

from fastmcp import Client as McpClient
from loguru import logger

from xr_ai_agent import DataMessage
from xr_ai_logging import print_task_done_banner
from xr_ai_models import ChatMessage, LLMService, ToolCall, ToolDef, VLMService
from xr_ai_capabilities import VisionModule, VisionUnavailable
from xr_ai_pipecat import BrainProcessor
from xr_ai_pipecat.transport import XRMediaHubTransport

from config import WorkerConfig
from tooling import (
    extract_json,
    looks_like_leaked_tool_call,
    normalize_tool_args,
    tool_payload,
)

# Dedicated trace logger — writes a clean session transcript to a file.
# Key events: user speech, pre-fetched context, think flag, tool calls +
# results, agent response, validation.  Records bound with this binding
# are routed to ``/tmp/xr-agent-trace.log`` by the sink installed in
# ``xr_render_demo_worker.main()``; everything else is unaffected.
_trace_log = logger.bind(trace=True)

_MAX_LOOP = 10  # visual queries need up to 5 steps; give headroom


# Tools served by oxr-mcp (routed there, not to render-mcp).
_OXR_TOOLS = frozenset({
    "get_head_pose", "position_ahead", "position_relative",
    "place_user_relative", "place_object_relative",
    "place_inside_by_id", "displace_object", "displace_objects",
})

# Spatial primitive math tools served by vec-mcp. Routed there so
# the LLM offloads vector arithmetic.
_VEC_TOOLS = frozenset({
    "between_anchors", "world_offset",
    "along_direction", "scale_value",
})

# Tools served by vlm-mcp and video-mcp.
_VLM_TOOLS   = frozenset({"ask_image"})
_VIDEO_TOOLS = frozenset({
    "get_frame_from_time",
    # video-mcp exposes get_latest_frame instead of get_frame_from_time when
    # recording is disabled (the default for this demo), so it must still route
    # to video-mcp — otherwise the call falls through to render-mcp.
    "get_latest_frame",
    "list_live_participants", "list_recorded_participants",
    "get_video_stats", "query_video",
})

# Brain-executed perception tool. Not served by any MCP server — the brain
# intercepts it in _execute_tool, turns the camera on (if needed), grabs the
# latest live frame for the active participant, and runs the VLM on it. This
# replaces the broken get_frame_from_time→ask_image two-step (the camera was
# never turned on, and get_frame_from_time isn't even registered when video
# recording is disabled), mirroring simple-vlm-example's live-frame VLM path.
_PERCEPTION_TOOL = "look_at_current_frame"

_PERCEPTION_TOOL_DEF = ToolDef(
    name=_PERCEPTION_TOOL,
    description=(
        "Look at the user's LIVE camera feed right now and answer a question "
        "about the real world — what they are holding, pointing at, or looking "
        "at; a real-world colour, shape, text, or object. Turns the camera on "
        "automatically and inspects the current frame. Use this whenever the "
        "answer cannot be known from the XR scene state alone."
    ),
    parameters={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "The specific question to answer about the live camera "
                    "frame, e.g. 'What colour is the object the user is "
                    "holding?'"
                ),
            },
        },
        "required": ["question"],
    },
)

# Spoken when a perception query is asked but no live camera frame can be
# obtained. Short, user-actionable, never a hang or silent failure.
_NO_FRAME_MSG = "I can't see a camera feed right now — please check your camera."

# VLM guidance for the live-frame answer. Plain spoken English, terse.
_PERCEPTION_SYSTEM_PROMPT = (
    "You are looking at the user's live camera feed. Answer the question "
    "directly from what is visible in the image, in one short plain-English "
    "sentence. Never reply with JSON, code, or markdown."
)

# Brief human-readable progress message shown while a tool runs.
_TOOL_PROGRESS: dict[str, str] = {
    "get_head_pose":        "Checking your position...",
    "position_ahead":       "Computing gaze position...",
    "position_relative":    "Computing relative position...",
    "place_user_relative":  "Placing relative to you...",
    "place_object_relative":"Placing relative to object...",
    "place_inside_by_id":   "Placing inside container...",
    "displace_object":      "Shifting object...",
    "displace_objects":     "Shifting objects...",
    "between_anchors":      "Computing midpoint...",
    "world_offset":         "Computing offset...",
    "along_direction":      "Computing position...",
    "scale_value":          "Computing size...",
    "get_scene_state":      "Scanning the scene...",
    "add_primitive":        "Creating object...",
    "update_primitive":     "Updating object...",
    "remove_primitive":     "Removing object...",
}

_AGENT_RESPONSE_TOPIC  = "agent.response"
_AGENT_PROGRESS_TOPIC  = "agent.progress"


class _SceneNotReadyError(Exception):
    """Raised when render-mcp returns not_started — LOVR hasn't launched yet."""


class _PerceptionUnavailableError(Exception):
    """Raised when a perception (look_at_current_frame) query cannot be
    answered — no camera feed, no frame, or VLM failure. Carries a short
    user-facing spoken message so the turn ends with a graceful spoken+panel
    line instead of hanging in the reasoning loop or failing silently."""

    def __init__(self, spoken: str) -> None:
        super().__init__(spoken)
        self.spoken = spoken


def _now_us() -> int:
    return time.time_ns() // 1_000


# ── RenderSceneProcessor ──────────────────────────────────────────────────────

class RenderSceneProcessor(BrainProcessor):
    """
    Multi-step agentic loop over render-mcp, oxr-mcp, and vec-mcp tools.

    Uses Llama-Nemotron (port 8106) with OpenAI tool calling + LMFE for the
    reasoning loop — guaranteed syntactically valid tool calls every iteration.
    Uses Minitron (port 8101) for the parallel quick-ack (fast, cheap).

    On each utterance:
      1. Quick-ack fires immediately (parallel, max 25 tokens) → agent.progress
      2. Agentic loop: model calls tools via OpenAI tool_calls protocol until
         it returns a text response (finish_reason != "tool_calls")
      3. Progress messages sent before each tool execution → agent.progress
      4. Final response → agent.response + yielded to TTS
    """

    def __init__(
        self,
        *,
        transport:   XRMediaHubTransport,
        cfg:         WorkerConfig,
        render:      McpClient,
        oxr:         McpClient,
        vlm:         McpClient,
        video:       McpClient,
        vec:         McpClient,
        prompt_path: Path,
        tools:       list[ToolDef],
        llm:         LLMService,
        agent_llm:   LLMService,
        vlm_service: VLMService | None = None,
        frame_max_age_s: float = 2.0,
        frame_timeout_s: float = 5.0,
    ):
        super().__init__()
        self._transport      = transport
        self._cfg            = cfg
        self._render         = render
        self._oxr            = oxr
        self._vlm            = vlm
        self._video          = video
        self._vec            = vec
        # Worker-local VLM service for live-frame perception (look_at_current_frame).
        # None in unit tests / notice-only paths that never hit the perception tool.
        self._vlm_service    = vlm_service
        self._prompt_path      = prompt_path
        self._prompt_cache     = prompt_path.read_text(encoding="utf-8").strip()
        _prompts               = prompt_path.parent
        self._quick_ack_path   = _prompts / "quick_ack.txt"
        self._still_work_path  = _prompts / "still_working.txt"
        self._validate_path    = _prompts / "validate.txt"
        self._quick_ack_cache  = self._quick_ack_path.read_text(encoding="utf-8").strip()
        self._still_work_cache = self._still_work_path.read_text(encoding="utf-8").strip()
        self._validate_cache   = self._validate_path.read_text(encoding="utf-8").strip()
        self._tools            = tools
        self._llm              = llm
        self._agent_llm        = agent_llm

        # Rolling conversation buffer — last N turns of (user_text, agent_response).
        # Injected as context so the agent understands "fix that", "undo", "the one I just added".
        self._history:    list[tuple[str, str]] = []
        self._history_max = 4

        # Move log for "put it back" — (obj_id, prev, new), capped at N.
        self._recent_moves: list[tuple[str, tuple[float, float, float],
                                              tuple[float, float, float]]] = []
        self._recent_moves_max = 5
        # Per-turn snapshot used to compute prev→new pairs on update_primitive.
        self._pre_move_positions: dict[str, tuple[float, float, float]] = {}

        # Canned spoken notices the agent (XR lifecycle, outside handle_query)
        # asks us to deliver. Keyed by pid → list of exact strings. Drained in
        # handle_query, which short-circuits the LLM loop for a matching entry.
        # See enqueue_notice / _emit_notice.
        self._pending_notices: dict[str, list[str]] = {}

        # ── live-frame perception (look_at_current_frame) ────────────────────
        # VisionModule owns frame tracking and the VLM call — shared with
        # simple-vlm-example. Built only when a VLM service is wired
        # (None in unit tests / no-camera deployments).
        self._vision: VisionModule | None = None
        if vlm_service is not None:
            self._vision = VisionModule(
                transport.endpoint, vlm_service,
                system_prompt   = _PERCEPTION_SYSTEM_PROMPT,
                frame_max_age_s = frame_max_age_s,
                frame_timeout_s = frame_timeout_s,
            )
            # FrameSignal events aren't pipecat frames, so the module subscribes
            # on the endpoint directly. Guard the unit-test transport double.
            if getattr(transport, "endpoint", None) is not None:
                self._vision.register()

    # ── public: text-channel entry ────────────────────────────────────────────

    async def enqueue_text_query(self, pid: str, text: str) -> None:
        """Run a typed text query through the same path as a spoken utterance.

        The web client's "Send" button posts on the data channel; the brain
        needs to fire ``handle_query`` for it identically to a transcript
        that passed the voice gate. The base class' ``_spawn_query`` owns
        the per-pid in-flight task and cancellation semantics, so we route
        through a synthesized ``GatedQueryFrame``.
        """
        from xr_ai_pipecat import GatedQueryFrame
        await self._spawn_query(GatedQueryFrame(
            participant_id = pid,
            text           = text,
            fresh_match    = False,
            pts_us         = _now_us(),
        ))

    async def enqueue_notice(self, pid: str, text: str) -> None:
        """Speak a canned, agent-authored notice through the normal turn path.

        XR-lifecycle failures (start_xr error, LOVR never ready) happen in
        ``RenderDemoAgent``, outside ``handle_query``'s yield→TTS path. To
        surface them with voice *and* a panel line — the same delivery shape
        as a normal final answer (``_send(agent.response)`` + ``yield``) —
        the agent hands us the message here. We register it as pending for
        *pid* and inject a ``GatedQueryFrame`` through the same
        ``_spawn_query`` machinery a typed/spoken query uses, so the base
        class owns the per-pid in-flight task and its cancellation. The text
        is matched (and consumed) by ``handle_query`` below, which yields it
        verbatim instead of running the agentic loop. Matching on the exact
        string — not just pid — means a real query that interleaves before
        the notice task runs is never mistaken for the notice.
        """
        from xr_ai_pipecat import GatedQueryFrame
        self._pending_notices.setdefault(pid, []).append(text)
        await self._spawn_query(GatedQueryFrame(
            participant_id = pid,
            text           = text,
            fresh_match    = False,
            pts_us         = _now_us(),
        ))

    # ── BrainProcessor overrides ──────────────────────────────────────────────

    async def handle_query(
        self, pid: str, text: str, fresh_match: bool,
    ) -> AsyncIterator[str]:
        """Drive one full turn of the agentic loop for *text* from *pid*.

        Yields strings that should reach TTS:
          - the quick-ack, on EVERY turn — spoken first so the user always
            gets immediate audio feedback, especially before a tool-using
            turn that would otherwise be silent until the final reply
          - the final user-visible response.

        Per-tool progress and still-working ticks are sent to the data
        channel (``send_return_data``) only, NOT spoken: a long agentic loop
        can emit many of them and speaking each would stack the TTS queue and
        play after the real reply. The single spoken ack covers "I'm on it";
        the panel carries the detailed progress.
        """
        # Canned agent notice (XR-lifecycle failure) — speak it verbatim and
        # skip the LLM loop. Exact-text match guards against a real query
        # interleaving before the notice task runs. See enqueue_notice.
        pending = self._pending_notices.get(pid)
        if pending and text in pending:
            pending.remove(text)
            if not pending:
                self._pending_notices.pop(pid, None)
            return self._emit_notice(pid, text)
        return self._run_turn(pid, text)

    async def _emit_notice(self, pid: str, text: str) -> AsyncIterator[str]:
        """Deliver a canned notice with the same shape as a final answer:
        a panel line on ``agent.response`` plus a spoken (yielded) line."""
        send_pid = pid or self._transport.target_participant
        if send_pid:
            await self._send(send_pid, text, topic=_AGENT_RESPONSE_TOPIC)
        yield text

    async def _run_turn(self, pid: str, text: str) -> AsyncIterator[str]:
        text = text.strip()
        if not text:
            return

        send_pid = pid or self._transport.target_participant
        # Capture the moment the user finished speaking so visual tool calls
        # can be anchored to that timestamp (not to when the tool fires).
        ref_us = _now_us()
        t0     = time.monotonic()

        # Quick-ack: fast Minitron call that (a) speaks an immediate
        # acknowledgment and (b) classifies whether Nemotron needs
        # reasoning enabled.  Await it first so the think flag is ready
        # before the main loop starts — it takes ~1s so the delay is small.
        try:
            ack, needs_thinking = await self._quick_ack(text)
        except Exception:
            logger.exception("quick ack failed")
            ack, needs_thinking = "", False

        if ack and send_pid:
            # ACK-SPEAK POLICY (deliberate): speak the quick-ack on EVERY turn,
            # not just needs_thinking ones. It's yielded before the agentic
            # loop runs, so TTS plays it first — giving the user immediate
            # audio feedback at the start of every turn. This matters most for
            # tool-using turns (which may not be flagged needs_thinking yet
            # still take seconds): without a spoken ack the user hears nothing
            # until the final reply. Per-tool progress + still-working ticks
            # remain text-only (below) so they don't stack the TTS queue; the
            # single spoken ack is enough to signal "I'm on it". Also mirror
            # the ack to the panel.
            await self._send(send_pid, ack, topic=_AGENT_PROGRESS_TOPIC)
            yield ack

        # Start a "still working" timer — fires if reasoning takes >5s.
        # Cancelled as soon as the loop returns.
        # thinking_ctx is a one-element list shared with the agentic loop so
        # the still-working messages can reflect what the 30B is reasoning about.
        thinking_ctx: list[str] = [""]
        still_task = asyncio.create_task(
            self._still_working_loop(text, send_pid, thinking_ctx,
                                     enabled=needs_thinking),
            name="still-working",
        )

        response: str | None = None
        outcome = "done"
        try:
            response = await self._agentic_loop(
                text, pid, ref_us=ref_us, needs_thinking=needs_thinking,
                thinking_ctx=thinking_ctx,
            )
        except asyncio.CancelledError:
            outcome = "interrupted"
            logger.info("agentic loop interrupted by new utterance")
            if send_pid:
                try:
                    await self._transport.endpoint.flush_return_audio(send_pid)
                except Exception:
                    logger.opt(exception=True).debug(
                        "flush_return_audio failed during cancellation",
                    )
            raise
        except Exception:
            outcome = "error"
            logger.exception("agentic loop failed")
            response = "Something went wrong — please try again."
        finally:
            still_task.cancel()
            try:
                await still_task
            except asyncio.CancelledError:
                pass  # expected — we just cancelled it above
            print_task_done_banner(
                "xr-render-demo",
                status=outcome,
                detail=f"pid={pid!r}  utterance={text[:60]!r}",
                duration_s=time.monotonic() - t0,
            )

        # Strip leaked tool-call JSON from both the user-visible reply and
        # history; legit text starting with "{" passes through.
        #
        # Defensive guard only: _agentic_loop always returns a non-empty
        # string ("Done." fallbacks on lines ~697/741, the not-ready string,
        # or the error string set above), and the cancellation branch
        # re-raises before reaching here. So this never fires for a real
        # turn — we intentionally do NOT yield a spoken "Done." here, because
        # there is no audible-close gap to fill. Every reachable turn already
        # ends with a yielded final response below.
        if not response:
            return

        display = response
        if looks_like_leaked_tool_call(response):
            logger.warning(
                "response looks like a leaked tool call, sanitizing: {!r}",
                response[:120],
            )
            display = "Done."

        self._history.append((text, display))
        if len(self._history) > self._history_max:
            self._history.pop(0)

        if send_pid:
            await self._send(send_pid, display, topic=_AGENT_RESPONSE_TOPIC)
        yield display

    def _read_prompt(self, path: Path, cache_attr: str) -> str:
        try:
            text = path.read_text(encoding="utf-8").strip()
            setattr(self, cache_attr, text)
            return text
        except OSError:
            logger.warning("prompt file unreadable: {} — using cache", path.name)
            return getattr(self, cache_attr)

    # ── quick ack ─────────────────────────────────────────────────────────────

    async def _quick_ack(self, transcript: str) -> tuple[str, bool]:
        """Fast call: returns (ack_text, needs_thinking).

        Passes the last conversation turn as context so corrections like
        "try it again" or "that was wrong" produce sensible acks.
        """
        # Include the most recent agent action as context.
        context = ""
        if self._history:
            last_user, last_agent = self._history[-1]
            context = f"[Previous turn] User: {last_user} / Agent: {last_agent}\n"

        messages = [
            ChatMessage(role="system", content=self._read_prompt(
                self._quick_ack_path, "_quick_ack_cache")),
            ChatMessage(role="user", content=context + transcript),
        ]
        try:
            resp = await asyncio.wait_for(
                self._llm.chat(messages, max_tokens=40, temperature=0.0),
                timeout=8.0,
            )
            raw = resp.content.strip()
            obj_text = extract_json(raw)
            if obj_text:
                try:
                    obj = json.loads(obj_text)
                    ack   = str(obj.get("ack", "")).strip()
                    think = bool(obj.get("think", False))
                    logger.info("quick-ack: {!r}  think={}", ack, think)
                    _trace_log.info("ACK   {}  [think={}]", ack, think)
                    return ack, think
                except json.JSONDecodeError:
                    pass
            # Fallback: treat raw text as ack, no thinking
            return raw, False
        except Exception as exc:
            logger.warning("quick-ack failed: {}", exc)
        return "", False

    # ── agentic loop (OpenAI tool calling + LMFE) ────────────────────────────

    async def _still_working_msg(self, transcript: str, sent: list[str],
                                  thinking_ctx: list[str]) -> str:
        """Ask Minitron for a short contextual 'still working' sentence.

        `sent` is the list of messages already shown this turn.
        `thinking_ctx` is a one-element list holding the latest reasoning_content
        from the 30B model — used to make progress updates reflect what the model
        is actually working on rather than guessing from the transcript alone.
        """
        avoid = ""
        if sent:
            avoid = (
                " Do NOT repeat or paraphrase any of these already-sent messages: "
                + ", ".join(f'"{m}"' for m in sent[-3:])
                + "."
            )
        thinking = thinking_ctx[0].strip() if thinking_ctx[0] else ""
        # Truncate — we only need the last few lines of thinking to get the gist.
        if thinking:
            lines = [l.strip() for l in thinking.splitlines() if l.strip()]
            thinking = " ".join(lines[-6:])[-400:]

        user_content = f"User request: {transcript}"
        if thinking:
            user_content += f"\n\nWhat the AI is currently reasoning through:\n{thinking}"

        base = self._read_prompt(self._still_work_path, "_still_work_cache")
        messages = [
            ChatMessage(role="system", content=base + avoid),
            ChatMessage(role="user", content=user_content),
        ]
        try:
            resp = await asyncio.wait_for(
                self._llm.chat(messages, max_tokens=24, temperature=0.9),
                timeout=6.0,
            )
            return resp.content.strip()
        except Exception as exc:
            logger.debug("still-working message failed: {}", exc)
        return ""

    async def _still_working_loop(
        self, transcript: str, pid: str, thinking_ctx: list[str],
        *, first_after: float = 5.0, repeat_every: float = 10.0,
        enabled: bool = True,
    ) -> None:
        """Speak periodic contextual updates while the agentic loop runs.

        Only fires when enabled=True (i.e. needs_thinking).  Non-thinking turns
        resolve in under a second — a "still working" message would just be noise.
        """
        if not enabled:
            return
        sent: list[str] = []
        await asyncio.sleep(first_after)
        while True:
            msg = await self._still_working_msg(transcript, sent, thinking_ctx)
            if msg and pid:
                # PROGRESS POLICY (deliberate divergence from simple-vlm):
                # progress is panel-only, never spoken. render-demo's long
                # multi-tool loops emit many of these; speaking them would
                # stack in the TTS queue and play after the real response.
                # Only acks (on thinking turns), the final answer, and
                # failure notices are spoken — see ACK-SPEAK POLICY above.
                await self._send(pid, msg, topic=_AGENT_PROGRESS_TOPIC)
                sent.append(msg)
            await asyncio.sleep(repeat_every)

    async def _validate(self, transcript: str, post_scene: dict) -> tuple[bool, str]:
        """Ask Minitron whether the task was completed as requested.

        Returns (ok, issue). Defaults to ok=True on any failure so a broken
        validator never blocks the response.
        """
        messages = [
            ChatMessage(role="system", content=self._read_prompt(
                self._validate_path, "_validate_cache")),
            ChatMessage(role="user", content=(
                f"Request: {transcript}\n"
                f"Current scene: {json.dumps(post_scene or {})}"
            )),
        ]
        try:
            resp = await asyncio.wait_for(
                self._llm.chat(messages, max_tokens=60, temperature=0.0),
                timeout=8.0,
            )
            raw = resp.content.strip()
            obj_text = extract_json(raw)
            if obj_text:
                obj = json.loads(obj_text)
                ok    = bool(obj.get("ok", True))
                issue = str(obj.get("issue", ""))
                logger.debug("validation: ok={}  issue={!r}", ok, issue)
                return ok, issue
        except Exception:
            logger.opt(exception=True).debug(
                "validation call failed — defaulting to ok",
            )
        return True, ""

    async def _build_turn_context(self, pid: str, *, ref_us: int = 0) -> str:
        """Pre-fetch scene/pose and format the turn-context block.

        Fetches scene state, head pose, and the most common spatial position
        (1.5 m ahead) concurrently — saves 1-3 tool-call iterations per turn —
        and renders them, plus the move log and conversation history, into the
        text injected into the agentic loop's user message. Side effect: resets
        ``self._pre_move_positions`` to the current scene so update_primitive
        calls during this turn can be recorded as (prev → new) move-log entries.
        """
        scene, pose, ahead = await asyncio.gather(
            self._call_mcp(self._render, "get_scene_state",  {}, silent=True),
            self._call_mcp(self._oxr,    "get_head_pose",    {}, silent=True),
            self._call_mcp(self._oxr,    "position_ahead",   {"distance": 1.5}, silent=True),
        )

        ctx_parts: list[str] = []

        # ── Scene ──────────────────────────────────────────────────────────────
        self._pre_move_positions = {}
        if isinstance(scene, dict) and scene.get("objects"):
            objs = scene["objects"]
            lines = ["SCENE OBJECTS:"]
            for o in objs:
                pos = o.get("position", {})
                col = o.get("color", {})
                self._pre_move_positions[o["id"]] = (
                    float(pos.get("x", 0)),
                    float(pos.get("y", 0)),
                    float(pos.get("z", 0)),
                )
                lines.append(
                    f"  {o['id']} ({o['type']})  "
                    f"pos=({pos.get('x',0):.2f}, {pos.get('y',0):.2f}, {pos.get('z',0):.2f})  "
                    f"color=(r={col.get('r',0):.2f} g={col.get('g',0):.2f} b={col.get('b',0):.2f})  "
                    f"size={o.get('size',0.1):.3f}m"
                )
            ctx_parts.append("\n".join(lines))
        else:
            ctx_parts.append("SCENE OBJECTS: (empty)")

        # ── Head pose + derived spatial shortcuts ─────────────────────────────
        if isinstance(pose, dict) and pose.get("is_valid"):
            p  = pose["position"]
            fv = pose["forward"]
            rv = pose["right"]
            uv = pose.get("up", {"x": 0, "y": 1, "z": 0})

            # Compute common offsets directly — no extra tool calls needed.
            def _off(vec: dict, d: float) -> str:
                return (f"({p['x']+vec['x']*d:.2f}, "
                        f"{p['y']+vec['y']*d:.2f}, "
                        f"{p['z']+vec['z']*d:.2f})")

            ahead_str = (
                f"({ahead['x']:.2f}, {ahead['y']:.2f}, {ahead['z']:.2f})"
                if isinstance(ahead, dict) and "x" in ahead
                else _off(fv, 1.5)
            )

            ctx_parts.append(
                "HEAD POSE:\n"
                f"  position : ({p['x']:.2f}, {p['y']:.2f}, {p['z']:.2f})\n"
                f"  forward  : ({fv['x']:.3f}, {fv['y']:.3f}, {fv['z']:.3f})  ← 'ahead/forward'\n"
                f"  right    : ({rv['x']:.3f}, {rv['y']:.3f}, {rv['z']:.3f})  ← 'right'\n"
                f"  up       : ({uv['x']:.3f}, {uv['y']:.3f}, {uv['z']:.3f})  ← 'up'\n"
                f"  yaw={pose.get('yaw_deg',0):.1f}°  pitch={pose.get('pitch_deg',0):.1f}°\n"
                "SPATIAL SHORTCUTS (pre-computed — use directly, no tool call needed):\n"
                f"  1.5m ahead of you     : {ahead_str}\n"
                f"  1m to your right      : {_off(rv,  1.0)}\n"
                f"  1m to your left       : {_off(rv, -1.0)}\n"
                f"  0.5m above eye level  : {_off(uv,  0.5)}\n"
                f"  1m behind you         : {_off(fv, -1.0)}\n"
                "  For other distances: new_pos = obj.pos + direction_vec × distance (per component)"
            )
        else:
            ctx_parts.append("HEAD POSE: unavailable")

        if pid:
            ctx_parts.append(f"Participant: {pid}")
        if ref_us:
            ctx_parts.append(f"Reference time (when user spoke): {ref_us} µs")

        # Structured move log — machine-readable prior coords for "put it
        # back" / "undo" / "revert" so the model doesn't have to parse free
        # text out of the conversation history.
        if self._recent_moves:
            move_lines = []
            for obj_id, prev, new in self._recent_moves:
                move_lines.append(
                    f"  {obj_id}: ({prev[0]:.2f}, {prev[1]:.2f}, {prev[2]:.2f}) → "
                    f"({new[0]:.2f}, {new[1]:.2f}, {new[2]:.2f})"
                )
            ctx_parts.append("[Recent moves] (most recent last — prev → new)\n"
                             + "\n".join(move_lines))

        # Recent conversation history — lets the agent understand "fix that",
        # "undo", "the sphere I just added", etc.
        if self._history:
            hist_lines = []
            for u, a in self._history:
                hist_lines.append(f"  User: {u}")
                hist_lines.append(f"  Agent: {a}")
            ctx_parts.append("[Recent conversation]\n" + "\n".join(hist_lines))

        context = "\n".join(ctx_parts)
        logger.debug("pre-fetched context for turn")
        _trace_log.debug("CTX   {}", context.replace("\n", " | "))
        return context

    def _recover_text_tool_call(
        self, content: str, all_names: set[str],
    ) -> dict | None:
        """Recover a tool call the model emitted as plain text instead of via
        the tool_calls field. Two shapes seen in practice:
          (a) bare name:  "get_head_pose"
          (b) JSON obj:   {"name": "update_primitive", "arguments": {...}}
        Returns {"name", "arguments"} or None if *content* isn't a tool call.
        """
        if content in all_names:
            # Shape (a): bare tool name, no args.
            return {"name": content, "arguments": {}}

        obj_text = extract_json(content)
        if obj_text:
            try:
                obj = json.loads(obj_text)
                name = obj.get("name") or obj.get("tool") or obj.get("function")
                if isinstance(name, str) and name in all_names:
                    args = obj.get("arguments") or obj.get("args") or {}
                    return {"name": name, "arguments": args if isinstance(args, dict) else {}}
            except json.JSONDecodeError:
                # Best-effort recovery: the plain-text fragment is not valid JSON.
                logger.debug("failed to decode recovered tool-call JSON: {!r}", obj_text)
        return None

    async def _agentic_loop(self, transcript: str, pid: str, *,
                            ref_us: int = 0,
                            needs_thinking: bool = False,
                            thinking_ctx: list[str] | None = None) -> str:
        """
        Multi-turn tool-calling loop using the OpenAI tool calling protocol.

        Scene state and head pose are pre-fetched concurrently before the
        loop starts and included in the user message, so the model skips
        those tool calls and goes straight to the operation.  Tools remain
        available for refresh queries or follow-up calls.
        """
        context = await self._build_turn_context(pid, ref_us=ref_us)

        try:
            system_content = self._prompt_path.read_text(encoding="utf-8").strip()
            self._prompt_cache = system_content
        except OSError:
            logger.warning("prompt file unreadable — using cached version")
            system_content = self._prompt_cache
        if needs_thinking:
            system_content = (
                "Use your private <think> block to work through these steps. "
                "NEVER output these steps as your response — your only text output "
                "to the user is ONE SHORT sentence AFTER all tool calls are done.\n"
                "Be terse in <think>: use notation not prose. "
                "No full sentences, no restating the request. "
                "Example: 'obj=sphere-1 pos=(1,1.7,-1.5) above→y=1.8' not "
                "'We need to parse the request and find the sphere...'\n"
                "\n"
                "THINK STEP 1 — RESOLVE: Which object? "
                "Pronouns ('it', 'that') = most recently added/modified object. "
                "Named ('the blue sphere') = match by color/type in scene.\n"
                "\n"
                "THINK STEP 2 — LOCATE: Copy the exact x, y, z of the target object "
                "and the head pose right/forward/up vectors from the context.\n"
                "\n"
                "THINK STEP 3 — COMPUTE: Calculate new coordinates with explicit arithmetic. "
                "User-relative move: new = old + head_vec × distance (per component). "
                "Near object: new = obj.pos ± world_offset. "
                "Midpoint: new = (A + B) / 2 per component. "
                "Write out each component: x=…, y=…, z=…\n"
                "\n"
                "THINK STEP 4 — EXECUTE: call the tool with the computed values, "
                "then reply with ONE short sentence to the user.\n\n"
                + system_content
            )

        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=system_content),
            ChatMessage(
                role="user",
                content=(
                    f"[Pre-fetched context — do not call get_scene_state or "
                    f"get_head_pose unless you need to refresh after changes]\n"
                    f"{context}\n\n"
                    f"[Request]\n{transcript}"
                ),
            ),
        ]

        for iteration in range(_MAX_LOOP):
            try:
                # thinking off: 1024 covers any tool-call JSON.
                # thinking on: 4096 budget lets the model enumerate all scene
                # objects and work through multi-step coordinate arithmetic
                # without hitting the limit. 6144 total = 4096 thinking + 2048
                # for tool-call JSON / response text.
                resp = await self._agent_llm.chat(
                    messages,
                    tools=self._tools,
                    max_tokens=6144 if needs_thinking else 1024,
                    temperature=0.0,
                    enable_thinking=needs_thinking,
                    thinking_budget=4096 if needs_thinking else None,
                )
            except Exception:
                logger.exception("agent-llm call failed on iteration {}", iteration)
                break

            finish     = resp.finish_reason or ""
            tool_calls = resp.tool_calls or []
            content    = resp.content.strip()

            # Share the 30B's reasoning with the still-working loop so progress
            # updates reflect what the model is actually working on.
            reasoning = (resp.reasoning or "").strip()
            if reasoning and thinking_ctx is not None:
                thinking_ctx[0] = reasoning
            if reasoning:
                logger.debug("agent-llm thinking iter={}  {}", iteration, reasoning[:300])
                _trace_log.debug("THINK [{}]  {}", iteration, reasoning[:300])

            logger.debug(
                "agent-llm iter={}  finish={}  tool_calls={}  content={!r}",
                iteration, finish, len(tool_calls), content[:200],
            )

            if not tool_calls:
                # Thinking filled the token budget without emitting a tool call.
                # Turn off thinking and retry — `continue` in a for loop advances
                # the iteration counter, but messages is unchanged so the model
                # gets another chance with the same context.
                if finish == "length" and needs_thinking:
                    logger.warning(
                        "agent-llm iter={} hit length limit during thinking — "
                        "retrying without thinking", iteration,
                    )
                    needs_thinking = False
                    continue

                # Recover from off-script tool call output (model emitted a
                # tool call as plain text instead of via the tool_calls field).
                all_names = {t.name for t in self._tools}
                recovered = self._recover_text_tool_call(content, all_names)

                if recovered:
                    logger.warning("text-format tool call {!r} — recovering", recovered["name"])
                    tool_calls = [ToolCall(
                        id=f"call_{uuid.uuid4().hex[:12]}",
                        name=recovered["name"],
                        arguments=json.dumps(recovered["arguments"]),
                    )]
                else:
                    # Genuine final response.
                    _trace_log.info("RESP  {}", content or "Done.")
                    return content or "Done."

            # Add the assistant's tool-call message to the conversation.
            messages.append(ChatMessage(
                role="assistant",
                content=content or "",
                tool_calls=list(tool_calls),
            ))

            # The spatial thinking prompt helps plan the first action but
            # actively harms subsequent iterations: it steers the model to
            # re-anchor on the pre-fetched context (SCENE OBJECTS, HEAD POSE)
            # instead of reading the tool results, producing wrong answers like
            # "You're looking at empty space" when look_at_current_frame
            # returned a valid VLM description.
            needs_thinking = False

            # Execute each tool call and append results.
            for tc in tool_calls:
                name = tc.name
                try:
                    args = json.loads(tc.arguments)
                except json.JSONDecodeError:
                    args = {}

                progress = _TOOL_PROGRESS.get(name)
                if progress and pid:
                    # PROGRESS POLICY (deliberate divergence from simple-vlm):
                    # per-tool progress is panel-only. A turn can fire several
                    # tools; speaking each would stack in the TTS queue behind
                    # the real answer. Only acks/final/failures are spoken.
                    await self._send(pid, progress, topic=_AGENT_PROGRESS_TOPIC)

                logger.debug("tool call  iter={}  tool={}  args={}", iteration, name, args)
                _trace_log.debug(
                    "TOOL  [{}] {}({})", iteration, name,
                    ", ".join(f"{k}={v}" for k, v in args.items()),
                )
                try:
                    result = await self._execute_tool(name, args, pid=pid)
                except _SceneNotReadyError:
                    return (
                        "The XR scene isn't ready yet. "
                        "Please click 'Launch XR' to start the headset session first."
                    )
                except _PerceptionUnavailableError as exc:
                    # No camera feed / frame / VLM — end the turn with a short
                    # spoken+panel message rather than looping or going silent.
                    return exc.spoken
                result_str = json.dumps(result, default=str)
                logger.info("tool result  tool={}  {}", name, result_str[:200])
                _trace_log.info("RES   [{}] {} → {}", iteration, name, result_str[:300])

                messages.append(ChatMessage(
                    role="tool",
                    content=result_str,
                    tool_call_id=tc.id,
                ))

        return "Done."

    # ── live-frame perception (look_at_current_frame) ─────────────────────────

    async def _look_at_current_frame(self, pid: str, question: str) -> dict:
        """Run the VLM on the current live camera frame via the shared
        :class:`VisionModule` (frame tracking, camera-on-demand, and the VLM
        call all live there). Returns a dict the agentic loop relays:
          {"answer": "<vlm text>"}                  on success
          {"error":  "<reason>", "spoken": "..."}   on graceful failure

        On a failure the user should hear, the dict carries a short ``spoken``
        message; the loop relays the answer/error text and never hangs.
        """
        if not pid:
            return {"error": "no active participant", "spoken": _NO_FRAME_MSG}
        if self._vision is None:
            logger.error("look_at_current_frame called but no VLM service wired")
            return {"error": "vlm unavailable", "spoken": _NO_FRAME_MSG}

        _trace_log.info("LOOK  {}", question[:120])
        try:
            answer = await self._vision.perceive(
                pid, question, system_prompt=_PERCEPTION_SYSTEM_PROMPT,
            )
            _trace_log.info("VLM   {}", answer[:200])
            return {"answer": answer}
        except VisionUnavailable as exc:
            logger.info("perception unavailable: {}", exc)
            return {"error": str(exc), "spoken": _NO_FRAME_MSG}
        except Exception as exc:
            logger.exception("look_at_current_frame failed")
            return {"error": str(exc), "spoken": _NO_FRAME_MSG}

    # ── tool routing ──────────────────────────────────────────────────────────

    async def _execute_tool(
        self, tool: str, args: dict, *, pid: str = "",
    ) -> dict | str | None:
        """Route a tool call to render-mcp, oxr-mcp, vec-mcp, vlm-mcp,
        video-mcp, or the brain-local perception path."""
        # Brain-executed live-frame perception — not an MCP tool. Intercept
        # before _normalize_tool_args (which would strip the question text if
        # it ever produced an empty value) and before MCP routing.
        if tool == _PERCEPTION_TOOL:
            question = str(args.get("question") or "").strip()
            result = await self._look_at_current_frame(pid, question)
            # Deterministic graceful failure: end the turn with the spoken
            # message instead of feeding an error back to the model (which a
            # flaky 30B might silently swallow → the "hangs in thinking" bug).
            if isinstance(result, dict) and result.get("spoken"):
                raise _PerceptionUnavailableError(result["spoken"])
            return result

        # Normalize nested dicts that the LLM sometimes generates instead of
        # flat scalar args — e.g. {"position": {x,y,z}} → x=, y=, z=.
        args = normalize_tool_args(args)

        if tool in _OXR_TOOLS:
            return await self._call_mcp(self._oxr, tool, args)

        if tool in _VEC_TOOLS:
            return await self._call_mcp(self._vec, tool, args)

        # Intercept not_started before it reaches the model — it means LOVR
        # hasn't spawned yet and no render op will succeed.  Return a clear
        # explanation so the model can respond to the user instead of retrying.
        if tool in _VLM_TOOLS:
            # Guard against fabricated paths: the model must call
            # get_frame_from_time first and use the returned path.
            if tool == "ask_image":
                path = args.get("image_path", "")
                if path and not os.path.isfile(path):
                    return {
                        "error": (
                            f"File not found: {path!r}. "
                            "You must call get_frame_from_time first to get the "
                            "real image path, then pass that path to ask_image."
                        )
                    }
            return await self._call_mcp(self._vlm, tool, args)
        if tool in _VIDEO_TOOLS:
            return await self._call_mcp(self._video, tool, args)
        result = await self._call_mcp(self._render, tool, args)
        if isinstance(result, dict) and result.get("reason") == "not_started":
            raise _SceneNotReadyError()

        # Record (prev → new) for any update_primitive that touched x/y/z so
        # later turns can answer "put it back" by reading the move log.
        if (tool == "update_primitive" and isinstance(result, dict)
                and result.get("ok")):
            obj_id = args.get("obj_id")
            prev = self._pre_move_positions.get(obj_id) if obj_id else None
            if prev and ("x" in args or "y" in args or "z" in args):
                new = (
                    float(args.get("x", prev[0])),
                    float(args.get("y", prev[1])),
                    float(args.get("z", prev[2])),
                )
                if new != prev:
                    self._recent_moves.append((obj_id, prev, new))
                    if len(self._recent_moves) > self._recent_moves_max:
                        self._recent_moves.pop(0)
                    # Update the snapshot so a second update in the same turn
                    # records (latest → newer), not (original → newer).
                    self._pre_move_positions[obj_id] = new

        return result

    async def _call_mcp(
        self, client: McpClient, tool: str, args: dict, *, silent: bool = False
    ) -> dict | str | None:
        try:
            res  = await client.call_tool(tool, args)
            data = tool_payload(res)
            return data
        except Exception as exc:
            if not silent:
                logger.error("mcp {} failed: {}", tool, exc)
            return {"error": str(exc)}

    # ── helpers ───────────────────────────────────────────────────────────────

    async def _send(self, pid: str, text: str, *, topic: str) -> None:
        try:
            await self._transport.send_return_data(DataMessage(
                participant_id=pid,
                topic=topic,
                pts_us=_now_us(),
                data=text.encode(),
            ))
        except Exception:
            logger.exception("send failed  topic={}", topic)

    async def on_participant_left(self, pid: str) -> None:
        """Tear down per-pid live-frame / camera state. The base class cancels
        in-flight query tasks; the VisionModule owns the frame + camera state."""
        if self._vision is not None:
            self._vision.release(pid)

    async def close(self) -> None:
        await self._llm.close()
        await self._agent_llm.close()
        if self._vlm_service is not None:
            try:
                await self._vlm_service.close()
            except Exception:
                logger.opt(exception=True).debug("vlm_service close failed")
