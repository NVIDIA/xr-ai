# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
RenderDemoAgent — assembles the Pipecat pipeline and handles the XR session
lifecycle (start_xr, polling, render.ready ack).
"""
from __future__ import annotations

import asyncio
import time

from pathlib import Path

from fastmcp import Client as McpClient
from loguru import logger
from pipecat.frames.frames import TranscriptionFrame
from pipecat.pipeline.runner import PipelineRunner
from xr_ai_agent import AudioChunk, DataMessage, ParticipantEvent
from xr_ai_models import LLMService, STTService, TTSService, ToolDef

from config import WorkerConfig
from processors import RenderSceneProcessor, _tool_payload, build_pipeline
from xr_ai_pipecat.transport import XRMediaHubTransport

_XR_SESSION_STARTED_TOPIC = "xr.session.started"
_RENDER_READY_TOPIC       = "render.ready"


def _now_us() -> int:
    return time.time_ns() // 1_000


class RenderDemoAgent:
    def __init__(
        self,
        cfg:        WorkerConfig,
        render:     McpClient,
        oxr:        McpClient,
        vlm:        McpClient,
        video:      McpClient,
        vec:        McpClient,
        prompt_path: Path,
        tools:      list[ToolDef],
        llm:        LLMService,
        agent_llm:  LLMService,
        stt:        STTService,
        tts:        TTSService,
    ) -> None:
        self._cfg    = cfg
        self._render = render
        self._oxr    = oxr
        self._vlm    = vlm
        self._video  = video
        self._vec    = vec

        self._transport = XRMediaHubTransport()
        self._transport.endpoint.on_data(self._on_data)
        self._transport.endpoint.on_audio(self._on_audio)
        self._transport.endpoint.on_participant(self._on_participant)

        self._scene = RenderSceneProcessor(
            self._transport, cfg, render, oxr, vlm, video, vec,
            prompt_path, tools=tools, llm=llm, agent_llm=agent_llm,
        )
        self._pipeline, self._pipeline_task = build_pipeline(
            self._transport, stt, tts, self._scene,
        )
        self._runner_task: asyncio.Task | None = None
        self._xr_started = False

    # ── XR session lifecycle ──────────────────────────────────────────────────

    async def _on_data(self, msg: DataMessage) -> None:
        if msg.topic != _XR_SESSION_STARTED_TOPIC:
            await self._handle_text_input(msg)
            return

        self._transport.set_target_participant(msg.participant_id)
        self._scene._history.clear()

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
            return
        if start_res.get("status") == "error":
            logger.error("start_xr error: {}", start_res.get("error"))
            return

        logger.info("start_xr status={} — polling lovr_started…", start_res.get("status"))
        if not await self._wait_lovr():
            return
        self._xr_started = True

        logger.info("render.ready — sending ack")
        await self._transport.send_return_data(DataMessage(
            participant_id=msg.participant_id,
            topic=_RENDER_READY_TOPIC,
            pts_us=_now_us(), data=b"",
        ))

    async def _handle_text_input(self, msg: DataMessage) -> None:
        """Feed a typed text message into the same path STT uses.

        The web client's "Send" button publishes typed text on the data
        channel with no topic, mirroring simple-vlm-example. We synthesize
        a TranscriptionFrame so the agentic loop fires identically to a
        spoken utterance, bypassing VAD/STT entirely.
        """
        text = (msg.data or b"").decode("utf-8", errors="replace").strip()
        if not text:
            return
        if not self._transport.target_participant:
            self._transport.set_target_participant(msg.participant_id)
        logger.info("text input  pid={!r}  {!r}", msg.participant_id, text[:80])
        await self._scene._enqueue(TranscriptionFrame(
            text=text,
            user_id=msg.participant_id,
            timestamp=str(msg.pts_us or _now_us()),
        ))

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
            data = _tool_payload(res)
            if not isinstance(data, dict):
                if not silent:
                    logger.error("render-mcp {} non-dict: {!r}", tool, data)
                return None
            return data
        except Exception as exc:
            if not silent:
                logger.error("render-mcp {}: {}", tool, exc)
            return None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        runner = PipelineRunner()
        self._runner_task = asyncio.ensure_future(runner.run(self._pipeline_task))
        await self._runner_task

    def shutdown(self) -> None:
        if self._runner_task and not self._runner_task.done():
            self._runner_task.cancel()
        self._transport.shutdown()
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._scene.close())
        except RuntimeError:
            # No running event loop during synchronous shutdown — skip async close.
            pass
