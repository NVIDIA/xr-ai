# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
RenderDemoAgent — assembles the Pipecat pipeline and handles the XR session
lifecycle (start_xr, polling, render.ready ack).
"""
from __future__ import annotations

import asyncio
import logging
import time

from fastmcp import Client as McpClient
from pipecat.pipeline.runner import PipelineRunner

from xr_ai_agent import AudioChunk, DataMessage, ParticipantEvent

from config import WorkerConfig
from processors import RenderSceneProcessor, _tool_payload, build_pipeline
from xr_ai_pipecat.services import SttClient, TtsClient
from xr_ai_pipecat.transport import XRMediaHubTransport

log = logging.getLogger("xr_render_demo")

_XR_SESSION_STARTED_TOPIC = "xr.session.started"
_RENDER_READY_TOPIC       = "render.ready"


def _now_us() -> int:
    return time.time_ns() // 1_000


class RenderDemoAgent:
    def __init__(self, cfg: WorkerConfig, render: McpClient, oxr: McpClient,
                 vlm: McpClient, video: McpClient,
                 actions_prompt: str, tools_openai: list) -> None:
        self._cfg    = cfg
        self._render = render
        self._oxr    = oxr
        self._vlm    = vlm
        self._video  = video

        self._transport = XRMediaHubTransport()
        self._transport.endpoint.on_data(self._on_data)
        self._transport.endpoint.on_audio(self._on_audio)
        self._transport.endpoint.on_participant(self._on_participant)

        self._scene = RenderSceneProcessor(
            self._transport, cfg, render, oxr, vlm, video,
            actions_prompt, tools_openai=tools_openai,
        )
        stt = SttClient(cfg.stt_server)
        tts = TtsClient(cfg.tts_server)
        self._pipeline, self._pipeline_task = build_pipeline(
            self._transport, stt, tts, self._scene,
        )
        self._runner_task: asyncio.Task | None = None
        self._xr_started = False

    # ── XR session lifecycle ──────────────────────────────────────────────────

    async def _on_data(self, msg: DataMessage) -> None:
        if msg.topic != _XR_SESSION_STARTED_TOPIC:
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

        log.info("%s from %s — calling start_xr", msg.topic, msg.participant_id)
        start_res = await self._call_render("start_xr", {})
        if start_res is None:
            log.warning("start_xr failed")
            return
        if start_res.get("status") == "error":
            log.error("start_xr error: %s", start_res.get("error"))
            return

        log.info("start_xr status=%s — polling lovr_started…", start_res.get("status"))
        if not await self._wait_lovr():
            return
        self._xr_started = True

        log.info("render.ready — sending ack")
        await self._transport.send_return_data(DataMessage(
            participant_id=msg.participant_id,
            topic=_RENDER_READY_TOPIC,
            pts_us=_now_us(), data=b"",
        ))

    async def _wait_lovr(self, timeout_s: float = 120.0) -> bool:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            h = await self._call_render("get_health", {}, silent=True)
            if h:
                if h.get("lovr_started"):
                    return True
                if h.get("spawn_error"):
                    log.error("spawn_error: %s", h["spawn_error"])
                    return False
            await asyncio.sleep(0.5)
        log.warning("lovr_started never true within %.0fs", timeout_s)
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
                    log.error("render-mcp %s non-dict: %r", tool, data)
                return None
            return data
        except Exception as exc:
            if not silent:
                log.error("render-mcp %s: %s", tool, exc)
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
