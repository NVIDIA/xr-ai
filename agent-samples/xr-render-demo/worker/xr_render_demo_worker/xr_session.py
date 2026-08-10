# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep XR lifecycle control outside the model-facing scene workflow."""

from __future__ import annotations

import asyncio
import time

from loguru import logger
from nat.plugin_api import Function
from xr_ai_hub import DataMessage
from xr_ai_voice import TextMessageInput, VoiceSession
from xr_render_scene import EmptyRequest

_START = "xr.session.started"
_READY = "render.ready"


def _now_us() -> int:
    return time.time_ns() // 1_000


class XRSessionController:
    """Start LOVR on the client's session event and acknowledge readiness."""

    def __init__(
        self,
        *,
        session: VoiceSession,
        start_xr: Function,
        get_render_health: Function,
    ) -> None:
        self.transport = session.transport
        self.start_xr = start_xr
        self.get_render_health = get_render_health
        self.started = False
        self._start_lock = asyncio.Lock()
        self.text = TextMessageInput(session=session, ignore_topics={_START})

    def attach(self) -> None:
        self.transport.endpoint.on_data(self._on_data)

    async def _on_data(self, message: DataMessage) -> None:
        if message.topic != _START:
            return
        self.transport.set_target_participant(message.participant_id)
        # Reconnects and multiple participants can race this event; only one
        # spawn attempt may run.
        async with self._start_lock:
            if not self.started:
                await self.start_xr.ainvoke(EmptyRequest())
                self.started = await self._wait_until_ready()
        if self.started:
            await self.transport.send_return_data(
                DataMessage(
                    participant_id=message.participant_id,
                    topic=_READY,
                    pts_us=_now_us(),
                    data=b"",
                )
            )
            return
        logger.warning("XR session start failed; renderer never became ready")

    async def _wait_until_ready(self, timeout_s: float = 120.0) -> bool:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            health = await self.get_render_health.ainvoke(EmptyRequest())
            if health.lovr_started:
                return True
            if health.spawn_error:
                return False
            await asyncio.sleep(0.5)
        return False


__all__ = ["XRSessionController"]
