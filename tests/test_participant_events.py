# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Verify that participant join/leave events from connectors are observed by
every subscribed processor and that ``ProcessorEndpoint.connected_participants``
stays in sync automatically.
"""
from __future__ import annotations

import asyncio

import pytest

from xr_ai_agent import ParticipantEvent, PixelFormat

pytestmark = pytest.mark.asyncio


_W, _H = 4, 4  # tiny synthetic frames — content irrelevant for slot-accounting
_FRAME = b"\x00" * (_W * _H * 3 // 2)  # I420: 1.5 bytes/pixel


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


async def test_participant_leave_releases_held_slots(hub, make_connector, settle):
    """Regression for #143: ring slots held by ended tracks must be released
    on participant disconnect, or the hub eventually drops 100% of frames
    from fresh participants once the ring fills with abandoned slots."""
    # make_connector uses num_slots=4; run more than that many connect/publish/
    # leave cycles so the ring would saturate without the fix.
    conn = make_connector()
    await conn.register()
    await settle()

    cycles = 6  # > num_slots (4)
    for i in range(cycles):
        pid = f"churn_{i}"
        await conn.notify_participant_joined(pid, pts_us=i)
        await settle()
        await conn.push_frame(
            data=_FRAME, width=_W, height=_H, fmt=PixelFormat.I420,
            pts_us=i, participant_id=pid, track_id=f"track_{i}",
        )
        await settle()
        await conn.notify_participant_left(pid, pts_us=i)
        await settle()

    # Hub's _latest_slots map must be empty — every held slot was released
    # when its participant left. Without the fix, this dict would carry
    # one stale (pid, track_id) per cycle.
    assert hub._latest_slots == {}

    # And — the user-visible symptom — a brand-new participant can still
    # publish frames. Pre-fix, push_frame on the connector raises
    # RuntimeError("ShmRingBuffer: all slots occupied") after enough cycles.
    await conn.notify_participant_joined("fresh", pts_us=99)
    await settle()
    for seq in range(3):
        await conn.push_frame(
            data=_FRAME, width=_W, height=_H, fmt=PixelFormat.I420,
            pts_us=100 + seq, participant_id="fresh", track_id="cam",
        )
        await settle()

    # The most recent frame for ("fresh", "cam") should be the one held.
    assert ("fresh", "cam") in hub._latest_slots
