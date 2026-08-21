# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Verify that participant join/leave events from connectors are observed by
every subscribed processor and that ``ProcessorEndpoint.connected_participants``
stays in sync automatically.
"""
from __future__ import annotations

import array
import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
import device_io_hub.ipc._hub as hub_module

from xr_ai_hub import (
    ConnectorRegistration,
    FrameSignal,
    MsgType,
    ParticipantEvent,
    PixelFormat,
)

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


async def test_connector_reregistration_releases_held_slots(hub, make_connector, settle):
    """Regression for #197: when a connector re-registers while a frame is
    still held, the hub must release the slots backed by the old ring BEFORE
    closing it. Otherwise the still-exported SlotView memoryview makes
    ``ShmRingBuffer.close()``'s ``_buf.release()`` raise (leaving the ring
    half-closed) and ``_latest_slots`` keeps pointing at it — a use-after-close
    on the next frame for that participant."""
    conn = make_connector()
    await conn.register()
    await settle()

    await conn.notify_participant_joined("alice", pts_us=1)
    await settle()
    await conn.push_frame(
        data=_FRAME, width=_W, height=_H, fmt=PixelFormat.I420,
        pts_us=1, participant_id="alice", track_id="cam",
    )
    await settle()
    assert ("alice", "cam") in hub._latest_slots  # frame held in the ring

    # Re-register the same connector (crash/reconnect while a frame is held).
    # Pre-fix: the held slot is left dangling and close() raises BufferError
    # (swallowed by the run loop), so the stale entry survives. Post-fix: the
    # held slot is released and dropped before the old ring is closed.
    await conn.register()
    await settle()
    assert ("alice", "cam") not in hub._latest_slots

    # And the hub is still healthy — a fresh frame after re-registration lands.
    await conn.push_frame(
        data=_FRAME, width=_W, height=_H, fmt=PixelFormat.I420,
        pts_us=2, participant_id="alice", track_id="cam",
    )
    await settle()
    assert ("alice", "cam") in hub._latest_slots


async def test_connector_reregistration_open_failure_drops_stale_ring(monkeypatch):
    """A failed replacement must not retain the closed old ring."""

    class CloseTrackingRing:
        def __init__(self) -> None:
            self.closed = False
            self.released_slots = []

        def release_slot(self, slot: int) -> None:
            self.released_slots.append(slot)

        def close(self) -> None:
            self.closed = True

    def fail_open(*, name: str, create: bool):
        raise RuntimeError(f"cannot open {name} create={create}")

    hub = hub_module.HubEndpoint.__new__(hub_module.HubEndpoint)
    old_ring = CloseTrackingRing()
    held_data = memoryview(b"held")
    held_view = SimpleNamespace(
        data=held_data,
        signal=SimpleNamespace(slot=2),
    )
    hub._latest_slots = {("alice", "cam"): (old_ring, held_view)}
    hub._ring_registry = {"conn": old_ring}
    monkeypatch.setattr(hub_module, "ShmRingBuffer", fail_open)

    hub._handle_registration(
        ConnectorRegistration(connector_id="conn", shm_name="missing")
    )

    assert old_ring.closed is True
    assert old_ring.released_slots == [2]
    assert hub._latest_slots == {}
    with pytest.raises(ValueError, match="released memoryview"):
        held_data.tobytes()
    assert "conn" not in hub._ring_registry


async def test_rejected_frame_signal_releases_new_ring_slot(
    hub,
    make_connector,
    settle,
):
    connector = make_connector()
    await connector.register()
    await settle()
    await connector.notify_participant_joined("alice", pts_us=0)
    await settle()

    # More rejected signals than the four-slot test ring must not exhaust it.
    for seq in range(1, 7):
        slot = connector._ring.write_frame(
            _FRAME,
            width=_W,
            height=_H,
            fmt=PixelFormat.I420,
            pts_us=seq,
            seq=seq,
        )
        bad_signal = FrameSignal(
            slot=slot,
            seq=seq,
            pts_us=seq,
            width=_W,
            height=_H,
            fmt=PixelFormat.I420,
            data_sz=len(_FRAME) + 1,
            participant_id="alice",
            track_id="cam",
        )

        await hub._dispatch(MsgType.FRAME_SIGNAL, bad_signal)

    assert ("alice", "cam") not in hub._latest_slots

    slot = connector._ring.write_frame(
        _FRAME,
        width=_W,
        height=_H,
        fmt=PixelFormat.I420,
        pts_us=10,
        seq=10,
    )
    valid_signal = FrameSignal(
        slot=slot,
        seq=10,
        pts_us=10,
        width=_W,
        height=_H,
        fmt=PixelFormat.I420,
        data_sz=len(_FRAME),
        participant_id="alice",
        track_id="cam",
    )
    await hub._dispatch(MsgType.FRAME_SIGNAL, valid_signal)

    assert ("alice", "cam") in hub._latest_slots


async def test_non_integer_frame_size_is_rejected_and_slot_reused(
    hub,
    make_connector,
    settle,
):
    connector = make_connector()
    await connector.register()
    await settle()
    await connector.notify_participant_joined("alice", pts_us=0)
    await settle()

    slot = connector._ring.write_frame(
        _FRAME,
        width=_W,
        height=_H,
        fmt=PixelFormat.I420,
        pts_us=1,
        seq=1,
    )
    malformed = FrameSignal(
        slot=slot,
        seq=1,
        pts_us=1,
        width=_W,
        height=_H,
        fmt=PixelFormat.I420,
        data_sz=str(len(_FRAME)),
        participant_id="alice",
        track_id="cam",
    )

    await hub._dispatch(MsgType.FRAME_SIGNAL, malformed)

    assert ("alice", "cam") not in hub._latest_slots
    reused_slots = [
        connector._ring.write_frame(
            _FRAME,
            width=_W,
            height=_H,
            fmt=PixelFormat.I420,
            pts_us=seq,
            seq=seq,
        )
        for seq in range(2, 6)
    ]
    assert slot in reused_slots


async def test_duplicate_frame_signal_keeps_held_slot_occupied(
    hub,
    make_connector,
    settle,
):
    connector = make_connector()
    await connector.register()
    await settle()
    await connector.notify_participant_joined("alice", pts_us=0)
    await settle()

    slot = connector._ring.write_frame(
        _FRAME,
        width=_W,
        height=_H,
        fmt=PixelFormat.I420,
        pts_us=1,
        seq=1,
    )
    signal = FrameSignal(
        slot=slot,
        seq=1,
        pts_us=1,
        width=_W,
        height=_H,
        fmt=PixelFormat.I420,
        data_sz=len(_FRAME),
        participant_id="alice",
        track_id="cam",
    )
    await hub._dispatch(MsgType.FRAME_SIGNAL, signal)
    held_before = hub._latest_slots[("alice", "cam")]

    await hub._dispatch(MsgType.FRAME_SIGNAL, signal)

    assert hub._latest_slots[("alice", "cam")] is held_before
    for seq in range(2, 5):
        connector._ring.write_frame(
            _FRAME,
            width=_W,
            height=_H,
            fmt=PixelFormat.I420,
            pts_us=seq,
            seq=seq,
        )
    with pytest.raises(RuntimeError, match="all slots occupied"):
        connector._ring.write_frame(
            b"WXYZ",
            width=2,
            height=2,
            fmt=PixelFormat.RGBA,
            pts_us=5,
            seq=5,
        )
    assert bytes(held_before[1].data) == _FRAME


async def test_duplicate_frame_signal_for_another_track_is_not_held(
    hub,
    make_connector,
    settle,
):
    connector = make_connector()
    await connector.register()
    await settle()
    await connector.notify_participant_joined("alice", pts_us=0)
    await settle()

    slot = connector._ring.write_frame(
        b"AAAA",
        width=1,
        height=1,
        fmt=PixelFormat.RGBA,
        pts_us=1,
        seq=1,
    )
    signal = FrameSignal(
        slot=slot,
        seq=1,
        pts_us=1,
        width=1,
        height=1,
        fmt=PixelFormat.RGBA,
        data_sz=4,
        participant_id="alice",
        track_id="camA",
    )
    await hub._dispatch(MsgType.FRAME_SIGNAL, signal)

    # A replay under another track key must not create a second view of the
    # same slot. Otherwise advancing camA releases the slot while camB still
    # aliases it, and a later producer write mutates camB's held frame.
    await hub._dispatch(
        MsgType.FRAME_SIGNAL,
        replace(signal, track_id="camB"),
    )

    assert ("alice", "camB") not in hub._latest_slots
    assert bytes(hub._latest_slots[("alice", "camA")][1].data) == b"AAAA"

    next_slot = connector._ring.write_frame(
        b"BBBB",
        width=1,
        height=1,
        fmt=PixelFormat.RGBA,
        pts_us=2,
        seq=2,
    )
    await hub._dispatch(
        MsgType.FRAME_SIGNAL,
        replace(signal, slot=next_slot, seq=2, pts_us=2),
    )

    # Fill the remaining slots and prove the original slot can be reused
    # without leaving any stale cross-track view behind.
    reused_slot = -1
    for seq, data in ((3, b"CCCC"), (4, b"DDDD"), (5, b"WXYZ")):
        reused_slot = connector._ring.write_frame(
            data,
            width=1,
            height=1,
            fmt=PixelFormat.RGBA,
            pts_us=seq,
            seq=seq,
        )
    assert reused_slot == slot
    assert ("alice", "camB") not in hub._latest_slots
    assert bytes(hub._latest_slots[("alice", "camA")][1].data) == b"BBBB"


async def test_participant_leave_continues_when_held_slot_release_fails():
    class FailingRing:
        def release_slot(self, _slot: int) -> None:
            raise RuntimeError("corrupt slot header")

    class FakePublisher:
        def __init__(self) -> None:
            self.messages = []

        async def send_multipart(self, parts) -> None:
            self.messages.append(parts)

    hub = hub_module.HubEndpoint.__new__(hub_module.HubEndpoint)
    view_data = memoryview(b"held")
    view = SimpleNamespace(
        data=view_data,
        signal=SimpleNamespace(slot=0),
    )
    hub._participant_connector = {"alice": "conn"}
    hub._published_status = {"alice": "ready"}
    hub._agent_status = {"agent": {"alice": "ready"}}
    hub._latest_slots = {("alice", "cam"): (FailingRing(), view)}
    hub._participant_cbs = []
    hub._pub = FakePublisher()
    event = ParticipantEvent(
        participant_id="alice",
        joined=False,
        pts_us=1,
        connector_id="conn",
    )

    await hub._dispatch(MsgType.PARTICIPANT_EVENT, event)

    assert hub._latest_slots == {}
    assert hub._pub.messages
    with pytest.raises(ValueError, match="released memoryview"):
        view_data.tobytes()


async def test_connector_signals_memoryview_byte_size(hub, make_connector, settle):
    connector = make_connector()
    await connector.register()
    await settle()
    await connector.notify_participant_joined("alice", pts_us=0)
    await settle()

    payload = memoryview(array.array("H", range(5)))
    await connector.push_frame(
        payload,
        width=5,
        height=1,
        fmt=PixelFormat.RGB24,
        pts_us=1,
        participant_id="alice",
        track_id="cam",
    )
    await settle()

    _, view = hub._latest_slots[("alice", "cam")]
    assert view.signal.data_sz == payload.nbytes
    assert len(view.data) == payload.nbytes
