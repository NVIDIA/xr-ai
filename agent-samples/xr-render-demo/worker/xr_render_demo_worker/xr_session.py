# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep XR lifecycle control outside the model-facing scene workflow."""

from __future__ import annotations

import asyncio
import time

from loguru import logger
from xr_ai_hub import DataMessage
from xr_ai_tools import Tool
from xr_ai_voice import HubVoiceTransport
from xr_render_scene import EmptyRequest

_START = "xr.session.started"
_READY = "render.ready"
_FAILED = "render.failed"


def _now_us() -> int:
    return time.time_ns() // 1_000


class XRSessionController:
    """Start LOVR on the client's session event and acknowledge readiness."""

    def __init__(
        self,
        *,
        transport: HubVoiceTransport,
        start_xr: Tool,
        get_render_health: Tool,
    ) -> None:
        self.transport = transport
        self.start_xr = start_xr
        self.get_render_health = get_render_health
        self.started = False
        self._start_lock = asyncio.Lock()

    def attach(self) -> None:
        self.transport.endpoint.on_data(self._on_data)

    async def _on_data(self, message: DataMessage) -> None:
        if message.topic != _START:
            return
        # The hub endpoint treats callback exceptions as fatal; a scene RPC
        # or launch failure must degrade to a failed session, not kill the
        # worker.
        try:
            await self._handle_start(message)
        except Exception:
            logger.exception("XR session start failed for {}", message.participant_id)

    async def _handle_start(self, message: DataMessage) -> None:
        self.transport.set_target_participant(message.participant_id)
        async with self._start_lock:
            if not self.started:
                try:
                    result = await self.start_xr.execute(EmptyRequest())
                except Exception:
                    logger.exception("start_xr RPC failed")
                else:
                    if result.error is not None:
                        logger.warning("start_xr failed: {} ({})", result.status, result.error)
                    else:
                        self.started = await self._wait_until_ready()
        if not self.started:
            logger.warning("XR session start failed; renderer never became ready")
        await self.transport.send_return_data(
            DataMessage(
                participant_id=message.participant_id,
                topic=_READY if self.started else _FAILED,
                pts_us=_now_us(),
                data=b"",
            )
        )

    async def _wait_until_ready(self, timeout_s: float = 120.0) -> bool:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            try:
                health = await self.get_render_health.execute(EmptyRequest())
            except Exception:
                logger.exception("render health poll failed")
                return False
            if health.lovr_started:
                return True
            if health.spawn_error:
                return False
            await asyncio.sleep(0.5)
        return False


__all__ = ["XRSessionController"]
