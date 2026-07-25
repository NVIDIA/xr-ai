# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Verify the agent-status loading/ready signal flow.

Three scenarios against a real HubEndpoint + ConnectorEndpoint +
ProcessorEndpoint stack (no GPU, no Docker):

  (a) Client joins before worker is ready → hub sends "loading".
  (b) Worker calls mark_ready() → "ready" broadcasts to connected clients.
  (c) Client joins after mark_ready() → auto-receives "ready".
"""
from __future__ import annotations

import asyncio
import json

import pytest
from xr_ai_agent import AGENT_STATUS_TOPIC, DataMessage

pytestmark = pytest.mark.asyncio

_LOADING = json.dumps({"status": "loading"}).encode()
_READY   = json.dumps({"status": "ready"}).encode()


async def _poll(condition, *, iters: int = 40) -> None:
    for _ in range(iters):
        if condition():
            return
        await asyncio.sleep(0.05)


async def _wire_connector(conn, *, participant_id: str) -> asyncio.Task:
    """Register connector, join participant, start its receive loop."""
    await conn.register()
    await asyncio.sleep(0.05)
    await conn.notify_participant_joined(participant_id, pts_us=1)
    await asyncio.sleep(0.05)
    return asyncio.create_task(conn.run(), name=f"conn_run_{participant_id}")


def _status_msgs(received: list) -> list:
    return [m for m in received if m.topic == AGENT_STATUS_TOPIC]


# ── (a) client joins before worker ready → receives "loading" ─────────────────

async def test_join_before_ready_receives_loading(hub, make_connector, settle):
    conn = make_connector()
    received: list[DataMessage] = []
    async def collect(msg: DataMessage) -> None: received.append(msg)
    conn.on_return_data(collect)

    conn_task = await _wire_connector(conn, participant_id="alice")
    try:
        await _poll(lambda: bool(_status_msgs(received)))
        assert _status_msgs(received)[0].data == _LOADING
    finally:
        conn_task.cancel()
        await asyncio.gather(conn_task, return_exceptions=True)


# ── (b) mark_ready() broadcasts "ready" to already-connected clients ──────────

async def test_mark_ready_broadcasts_to_connected_clients(
    hub, make_connector, make_processor, settle,
):
    conn = make_connector()
    received: list[DataMessage] = []
    async def collect(msg: DataMessage) -> None: received.append(msg)
    conn.on_return_data(collect)

    conn_task = await _wire_connector(conn, participant_id="bob")
    try:
        await _poll(lambda: bool(_status_msgs(received)))
        received.clear()

        proc = make_processor()
        await settle()
        await proc.mark_ready()

        await _poll(lambda: any(m.data == _READY for m in _status_msgs(received)))
        assert any(m.data == _READY for m in _status_msgs(received))
    finally:
        conn_task.cancel()
        await asyncio.gather(conn_task, return_exceptions=True)


# ── (c) client joins after mark_ready() → auto-receives "ready" ───────────────

async def test_late_join_receives_ready_after_mark_ready(
    hub, make_connector, make_processor, settle,
):
    proc = make_processor()
    await settle()
    await proc.mark_ready()
    await settle()

    conn = make_connector()
    received: list[DataMessage] = []
    async def collect(msg: DataMessage) -> None: received.append(msg)
    conn.on_return_data(collect)

    conn_task = await _wire_connector(conn, participant_id="carol")
    try:
        await _poll(lambda: any(m.data == _READY for m in _status_msgs(received)))
        assert any(m.data == _READY for m in _status_msgs(received))
    finally:
        conn_task.cancel()
        await asyncio.gather(conn_task, return_exceptions=True)
