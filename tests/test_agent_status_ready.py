# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Verify the agent-status ready/loading signal flow.

Three scenarios exercised against a real HubEndpoint + ConnectorEndpoint
+ ProcessorEndpoint stack (no GPU, no Docker):

  (a) mark_ready() broadcasts "ready" to participants already connected
      when the call is made.
  (b) A participant who joins after mark_ready() automatically receives
      "ready" via the PARTICIPANT_EVENT dispatch hook.
  (c) The hub's on_participant callback (as installed by __main__.py)
      sends "loading" the instant a participant joins, before any worker
      is present.
"""
from __future__ import annotations

import asyncio
import json
import time

import pytest

from xr_ai_agent import AGENT_STATUS_TOPIC, DataMessage
from xr_media_hub.ipc import ParticipantEvent

pytestmark = pytest.mark.asyncio

_LOADING = json.dumps({"status": "loading"}).encode()
_READY   = json.dumps({"status": "ready"}).encode()


def _install_loading_hook(hub) -> None:
    """Replicate the __main__.py on_participant hook that sends 'loading'."""
    async def _on_participant(event: ParticipantEvent) -> None:
        if event.joined:
            await hub.send_return_data(DataMessage(
                participant_id=event.participant_id,
                topic=AGENT_STATUS_TOPIC,
                pts_us=int(time.time() * 1_000_000),
                data=_LOADING,
            ))
    hub.on_participant(_on_participant)


async def _poll(condition, *, iters: int = 40) -> None:
    for _ in range(iters):
        if condition():
            return
        await asyncio.sleep(0.05)


async def _wire_connector(conn, *, participant_id: str) -> None:
    await conn.register()
    await asyncio.sleep(0.05)
    await conn.notify_participant_joined(participant_id, pts_us=1)
    await asyncio.sleep(0.05)


# ── (a) mark_ready() broadcasts "ready" to already-connected participants ─────

async def test_mark_ready_sends_ready_to_current_participants(
    hub, make_connector, make_processor, settle,
):
    conn = make_connector()
    await _wire_connector(conn, participant_id="alice")

    received: list[DataMessage] = []
    async def collect(msg: DataMessage) -> None: received.append(msg)
    conn.on_return_data(collect)

    conn_task = asyncio.create_task(conn.run(), name="conn_run")
    try:
        proc = make_processor()
        await settle()

        await proc.mark_ready()

        await _poll(lambda: any(
            m.topic == AGENT_STATUS_TOPIC and m.data == _READY for m in received
        ))

        status = [m for m in received if m.topic == AGENT_STATUS_TOPIC]
        assert any(m.data == _READY for m in status), (
            f"Expected 'ready' for pre-existing participant; got {[m.data for m in status]}"
        )
    finally:
        conn_task.cancel()
        await asyncio.gather(conn_task, return_exceptions=True)


# ── (b) participant joining after mark_ready() auto-receives "ready" ──────────

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

    conn_task = asyncio.create_task(conn.run(), name="conn_run")
    try:
        await _wire_connector(conn, participant_id="bob")

        await _poll(lambda: any(
            m.topic == AGENT_STATUS_TOPIC and m.data == _READY for m in received
        ))

        status = [m for m in received if m.topic == AGENT_STATUS_TOPIC]
        assert any(m.data == _READY for m in status), (
            f"Expected auto-ready for late joiner; got {[m.data for m in status]}"
        )
    finally:
        conn_task.cancel()
        await asyncio.gather(conn_task, return_exceptions=True)


# ── (c) hub sends "loading" on participant join ───────────────────────────────

async def test_hub_sends_loading_on_participant_join(
    hub, make_connector, settle,
):
    _install_loading_hook(hub)

    conn = make_connector()
    received: list[DataMessage] = []
    async def collect(msg: DataMessage) -> None: received.append(msg)
    conn.on_return_data(collect)

    conn_task = asyncio.create_task(conn.run(), name="conn_run")
    try:
        await _wire_connector(conn, participant_id="carol")

        await _poll(lambda: any(
            m.topic == AGENT_STATUS_TOPIC and m.data == _LOADING for m in received
        ))

        status = [m for m in received if m.topic == AGENT_STATUS_TOPIC]
        assert any(m.data == _LOADING for m in status), (
            f"Expected 'loading' on join; got {[m.data for m in status]}"
        )
    finally:
        conn_task.cancel()
        await asyncio.gather(conn_task, return_exceptions=True)
