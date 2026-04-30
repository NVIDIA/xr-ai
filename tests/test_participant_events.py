"""
Verify that participant join/leave events from connectors are observed by
every subscribed processor and that ``ProcessorEndpoint.connected_participants``
stays in sync automatically.
"""
from __future__ import annotations

import asyncio

import pytest

from xr_ai_agent import ParticipantEvent

pytestmark = pytest.mark.asyncio


async def test_connected_participants_auto_maintained(hub, make_connector, make_processor, settle):
    proc = make_processor()
    await settle()

    conn_a = make_connector()
    conn_b = make_connector()
    await conn_a.register()
    await conn_b.register()
    await settle()

    await conn_a.notify_participant_joined("alice", pts_us=1)
    await conn_b.notify_participant_joined("bob",   pts_us=2)

    # Wait for participant events to land at the processor.
    for _ in range(20):
        if proc.connected_participants == {"alice", "bob"}:
            break
        await asyncio.sleep(0.05)

    assert proc.connected_participants == {"alice", "bob"}

    await conn_a.notify_participant_left("alice", pts_us=3)
    for _ in range(20):
        if proc.connected_participants == {"bob"}:
            break
        await asyncio.sleep(0.05)
    assert proc.connected_participants == {"bob"}


async def test_participant_event_seen_by_every_processor(hub, make_connector, make_processor, settle):
    """Multi-agent: every processor receives the same join/leave event."""
    events_a: list[ParticipantEvent] = []
    events_b: list[ParticipantEvent] = []

    async def cb_a(e): events_a.append(e)
    async def cb_b(e): events_b.append(e)

    p_a = make_processor()
    p_b = make_processor()
    p_a.on_participant(cb_a)
    p_b.on_participant(cb_b)
    await settle()

    conn = make_connector()
    await conn.register()
    await settle()

    await conn.notify_participant_joined("alice", pts_us=1)
    await conn.notify_participant_left  ("alice", pts_us=2)

    for _ in range(20):
        if len(events_a) >= 2 and len(events_b) >= 2:
            break
        await asyncio.sleep(0.05)

    assert [e.joined for e in events_a] == [True, False]
    assert [e.joined for e in events_b] == [True, False]
    assert all(e.participant_id == "alice" for e in events_a + events_b)
