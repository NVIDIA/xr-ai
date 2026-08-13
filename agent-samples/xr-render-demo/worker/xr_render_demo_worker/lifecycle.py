# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""XR launch and readiness lifecycle for xr-render-demo."""
from __future__ import annotations

import asyncio
import time
from contextlib import suppress

from loguru import logger
from xr_ai_hub import DataMessage
from xr_ai_runtime import AgentRuntime, RuntimeClosedError
from xr_ai_tools import Tool
from xr_ai_voice import HubVoiceTransport
from xr_render_scene import EmptyRequest

from .agent import RENDER_NOTICE_TOPIC, RenderNotice
from .scene_loop import SceneModelLoop

_XR_SESSION_STARTED_TOPIC = "xr.session.started"
_RENDER_READY_TOPIC       = "render.ready"


def _now_us() -> int:
    return time.time_ns() // 1_000


class XRSessionLifecycle:
    """Own XR launch and readiness while queries flow through the runtime."""

    def __init__(
        self,
        *,
        transport: HubVoiceTransport,
        scene_loop: SceneModelLoop,
        start_xr: Tool,
        get_health: Tool,
        runtime: AgentRuntime,
    ) -> None:
        self._transport = transport
        self._scene_loop = scene_loop
        self._start_xr = start_xr
        self._get_health = get_health
        self._runtime = runtime

        self._xr_started = False

        self._transport.endpoint.on_data(self._on_data)

    # ── XR session lifecycle ──────────────────────────────────────────────────

    async def _on_data(self, msg: DataMessage) -> None:
        if msg.topic != _XR_SESSION_STARTED_TOPIC:
            return

        self._transport.set_target_participant(msg.participant_id)
        self._scene_loop.reset_history(msg.participant_id)

        if self._xr_started:
            await self._transport.send_return_data(DataMessage(
                participant_id=msg.participant_id,
                topic=_RENDER_READY_TOPIC,
                pts_us=_now_us(), data=b"",
            ))
            return

        logger.info("{} from {} — calling start_xr", msg.topic, msg.participant_id)
        start_res = await self._call_render(self._start_xr)
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
        """Publish a launch failure through the same agent mailbox as queries."""

        with suppress(RuntimeClosedError):
            await self._runtime.publish(
                RENDER_NOTICE_TOPIC,
                RenderNotice(
                    text="I couldn't start the XR session — try Launch XR again.",
                    interrupt_output=False,
                ),
                participant_id=pid,
                source="xr-session",
            )

    async def _wait_lovr(self, timeout_s: float = 120.0) -> bool:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            h = await self._call_render(self._get_health, silent=True)
            if h:
                if h.get("lovr_started"):
                    return True
                if h.get("spawn_error"):
                    logger.error("spawn_error: {}", h["spawn_error"])
                    return False
            await asyncio.sleep(0.5)
        logger.warning("lovr_started never true within {:.0f}s", timeout_s)
        return False

    # ── native scene helper ───────────────────────────────────────────────────

    async def _call_render(self, tool: Tool, *, silent: bool = False) -> dict | None:
        try:
            result = await tool.execute(EmptyRequest())
            return result.model_dump(mode="python")
        except Exception as exc:
            if not silent:
                logger.error("scene tool {}: {}", tool.name, exc)
            return None
