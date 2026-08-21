# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private eval-only endpoint for injecting synthetic hub traffic.

This is deliberately NOT part of xr_ai_hub.ProcessorEndpoint. ProcessorEndpoint
is the worker-facing API; synthetic participant lifecycle and upstream data belong
in a connector-owned test fixture, not in any worker abstraction.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

import zmq
import zmq.asyncio
from xr_ai_hub._codec import encode
from xr_ai_hub._types import DataMessage, MsgType, ParticipantEvent


class LiveEvalEndpoint:
    """Minimal ZMQ connector for live eval drivers.

    Publishes synthetic participant events and text queries directly to the
    hub's input socket, simulating a connector (e.g. LiveKit bridge) without
    going through the worker-facing ProcessorEndpoint API.
    """

    def __init__(
        self,
        push_addr: str = "ipc:///tmp/xr_hub_in",
    ) -> None:
        ctx = zmq.asyncio.Context.instance()
        self._push: zmq.asyncio.Socket = ctx.socket(zmq.PUSH)
        self._push.connect(push_addr)

    async def inject_participant_event(self, event: ParticipantEvent) -> None:
        await self._push.send(encode(MsgType.PARTICIPANT_EVENT, event))

    async def inject_data(self, msg: DataMessage) -> None:
        await self._push.send(encode(MsgType.DATA_MESSAGE, msg))

    async def close(self) -> None:
        self._push.close(linger=0)


@asynccontextmanager
async def live_participant(endpoint: LiveEvalEndpoint, participant_id: str):
    """Join a synthetic participant for one case and always emit its leave."""
    await endpoint.inject_participant_event(ParticipantEvent(
        participant_id=participant_id, joined=True, pts_us=time.time_ns() // 1_000))
    await asyncio.sleep(1.0)
    try:
        yield participant_id
    finally:
        await endpoint.inject_participant_event(ParticipantEvent(
            participant_id=participant_id, joined=False, pts_us=time.time_ns() // 1_000))
