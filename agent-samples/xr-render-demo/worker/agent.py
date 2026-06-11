# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
RenderDemoAgent — XR session lifecycle owner.

Wraps the ``RenderSceneProcessor`` brain with xr-render-demo-specific
behavior (``xr.session.started`` → ``start_xr`` → poll LOVR → ack;
typed-text input fed through the same path as STT).

The voice pipeline itself is assembled by ``xr_ai_pipecat.make_voice_pipeline``;
this class only owns the agent-to-hub bookkeeping that lives outside the
generic pipeline.
"""
from __future__ import annotations

import asyncio
import time

from fastmcp import Client as McpClient
from loguru import logger
from xr_ai_agent import AudioChunk, DataMessage, ParticipantEvent

from processors import RenderSceneProcessor
from tooling import tool_payload
from xr_ai_pipecat.transport import XRMediaHubTransport

_XR_SESSION_STARTED_TOPIC = "xr.session.started"
_RENDER_READY_TOPIC       = "render.ready"


def _now_us() -> int:
    return time.time_ns() // 1_000


class RenderDemoAgent:
    """Owns the XR session lifecycle on top of the unified voice pipeline.

    The voice pipeline (``transport.input() → VadStt → VoiceGate → brain
    → StreamingTts → transport.output()``) is built by
    :func:`xr_ai_pipecat.make_voice_pipeline`; this class subscribes the
    hub callbacks that the foundation does not handle (XR session start,
    typed-text input, target-participant tracking).
    """

    def __init__(
        self,
        *,
        transport: XRMediaHubTransport,
        brain:     RenderSceneProcessor,
        render:    McpClient,
    ) -> None:
        self._transport = transport
        self._brain     = brain
        self._render    = render

        self._xr_started = False

        # Subscribe to hub events the pipecat pipeline doesn't surface to us:
        # data messages (text input + xr.session.started), audio (lazy
        # target-pid set), and participant events.
        self._transport.endpoint.on_data(self._on_data)
        self._transport.endpoint.on_audio(self._on_audio)
        self._transport.endpoint.on_participant(self._on_participant)

    # ── XR session lifecycle ──────────────────────────────────────────────────

    async def _on_data(self, msg: DataMessage) -> None:
        if msg.topic != _XR_SESSION_STARTED_TOPIC:
            await self._handle_text_input(msg)
            return

        self._transport.set_target_participant(msg.participant_id)
        self._brain._history.clear()

        if self._xr_started:
            await self._transport.send_return_data(DataMessage(
                participant_id=msg.participant_id,
                topic=_RENDER_READY_TOPIC,
                pts_us=_now_us(), data=b"",
            ))
            return

        logger.info("{} from {} — calling start_xr", msg.topic, msg.participant_id)
        start_res = await self._call_render("start_xr", {})
        if start_res is None:
            logger.warning("start_xr failed")
            await self._notify_launch_failed(msg.participant_id)
            return
        if start_res.get("status") == "error":
            logger.error("start_xr error: {}", start_res.get("error"))
            await self._notify_launch_failed(msg.participant_id)
            return

        logger.info("start_xr status={} — polling lovr_started…", start_res.get("status"))
        if not await self._wait_lovr():
            await self._notify_launch_failed(msg.participant_id)
            return
        self._xr_started = True

        logger.info("render.ready — sending ack")
        await self._transport.send_return_data(DataMessage(
            participant_id=msg.participant_id,
            topic=_RENDER_READY_TOPIC,
            pts_us=_now_us(), data=b"",
        ))

    async def _notify_launch_failed(self, pid: str) -> None:
        """Surface an XR-launch failure to the user, spoken + on the panel.

        ``start_xr`` and the LOVR-spawn poll run here, outside the brain's
        ``handle_query``/yield→TTS path, so a bare ``logger.warning`` would
        leave the user staring at a "Launch XR" button that silently did
        nothing. Route a short, actionable message through the brain's
        ``enqueue_notice`` so it reaches TTS *and* the ``agent.response``
        panel exactly like a normal answer — the same delivery the in-loop
        "scene not ready" case already gets. One generic message covers
        both the start_xr-error and never-ready/spawn-error cases; the log
        lines above retain the specific cause for operators.
        """
        await self._brain.enqueue_notice(
            pid, "I couldn't start the XR session — try Launch XR again."
        )

    async def _handle_text_input(self, msg: DataMessage) -> None:
        """Feed a typed text message into the same path STT uses.

        The web client's "Send" button publishes typed text on the data
        channel with no topic, mirroring simple-vlm-example.  We hand the
        text to the brain via a synthesized ``GatedQueryFrame`` so the
        agentic loop fires identically to a spoken utterance, bypassing
        VAD/STT and the voice gate entirely.
        """
        text = (msg.data or b"").decode("utf-8", errors="replace").strip()
        if not text:
            return
        if not self._transport.target_participant:
            self._transport.set_target_participant(msg.participant_id)
        logger.info("text input  pid={!r}  {!r}", msg.participant_id, text[:80])
        await self._brain.enqueue_text_query(msg.participant_id, text)

    async def _wait_lovr(self, timeout_s: float = 120.0) -> bool:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            h = await self._call_render("get_health", {}, silent=True)
            if h:
                if h.get("lovr_started"):
                    return True
                if h.get("spawn_error"):
                    logger.error("spawn_error: {}", h["spawn_error"])
                    return False
            await asyncio.sleep(0.5)
        logger.warning("lovr_started never true within {:.0f}s", timeout_s)
        return False

    # ── participant tracking ───────────────────────────────────────────────────

    async def _on_audio(self, chunk: AudioChunk) -> None:
        # Lazily set target participant from the first audio chunk so TTS
        # can respond even if the xr.session.started message was missed.
        if chunk.participant_id and not self._transport.target_participant:
            self._transport.set_target_participant(chunk.participant_id)

    async def _on_participant(self, event: ParticipantEvent) -> None:
        if event.joined:
            self._transport.set_target_participant(event.participant_id)
        else:
            self._transport.cleanup_participant(event.participant_id)

    # ── render-mcp helper ─────────────────────────────────────────────────────

    async def _call_render(self, tool: str, args: dict, *, silent: bool = False) -> dict | None:
        try:
            res  = await self._render.call_tool(tool, args)
            data = tool_payload(res)
            if not isinstance(data, dict):
                if not silent:
                    logger.error("render-mcp {} non-dict: {!r}", tool, data)
                return None
            return data
        except Exception as exc:
            if not silent:
                logger.error("render-mcp {}: {}", tool, exc)
            return None
