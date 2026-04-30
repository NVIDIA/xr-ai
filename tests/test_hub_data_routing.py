# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Verify that the hub fans data messages out to subscribed processors with
the correct ``DataMessage`` payload, including the topic field that used
to be swallowed at the StreamKit layer.

Each scenario runs against a real :class:`HubEndpoint` driven by one or
more :class:`ConnectorEndpoint`s and observed by one or more
:class:`ProcessorEndpoint`s — exactly the production wiring minus the
LiveKit transport.
"""
from __future__ import annotations

import asyncio

import pytest

from xr_ai_agent import DataMessage


pytestmark = pytest.mark.asyncio


async def _wire_connector(connector, *, participant_id: str = "alice") -> None:
    """Register the connector and announce one participant joined."""
    await connector.register()
    await asyncio.sleep(0.05)  # let registration land at the hub
    await connector.notify_participant_joined(participant_id, pts_us=1)


async def test_data_topic_propagates_to_processor(hub, make_connector, make_processor, settle):
    """A data message pushed by the connector reaches the processor with
    the original topic intact."""
    received: list[DataMessage] = []

    async def collect(msg: DataMessage) -> None:
        received.append(msg)

    proc = make_processor()
    proc.on_data(collect)
    await settle()

    conn = make_connector()
    await _wire_connector(conn, participant_id="alice")
    await settle()

    msg = DataMessage(participant_id="alice", topic="chat", pts_us=42, data=b"hello")
    await conn.push_data(msg)

    # Wait for delivery (PUB/SUB hop).
    for _ in range(20):
        if received:
            break
        await asyncio.sleep(0.05)

    assert len(received) == 1
    got = received[0]
    assert got.participant_id == "alice"
    assert got.topic          == "chat"
    assert got.data           == b"hello"


async def test_data_fanout_to_multiple_processors(hub, make_connector, make_processor, settle):
    """A single inbound data message reaches every subscribed processor."""
    seen_a: list[DataMessage] = []
    seen_b: list[DataMessage] = []

    async def cb_a(msg): seen_a.append(msg)
    async def cb_b(msg): seen_b.append(msg)

    p_a = make_processor()
    p_b = make_processor()
    p_a.on_data(cb_a)
    p_b.on_data(cb_b)
    await settle()

    conn = make_connector()
    await _wire_connector(conn, participant_id="alice")
    await settle()

    await conn.push_data(DataMessage(
        participant_id="alice", topic="vlm.query", pts_us=1, data=b"what is this?",
    ))

    for _ in range(20):
        if seen_a and seen_b:
            break
        await asyncio.sleep(0.05)

    assert [m.topic for m in seen_a] == ["vlm.query"]
    assert [m.topic for m in seen_b] == ["vlm.query"]
    assert seen_a[0].data == b"what is this?"
    assert seen_b[0].data == b"what is this?"


async def test_multi_client_data_is_attributed_to_origin(hub, make_connector, make_processor, settle):
    """Two clients pushing on different connectors arrive at the same
    processor with the right ``participant_id`` on each message."""
    received: list[DataMessage] = []

    async def cb(msg): received.append(msg)

    proc = make_processor()
    proc.on_data(cb)
    await settle()

    alice_conn = make_connector(connector_id="alice_conn")
    bob_conn   = make_connector(connector_id="bob_conn")
    await _wire_connector(alice_conn, participant_id="alice")
    await _wire_connector(bob_conn,   participant_id="bob")
    await settle()

    await alice_conn.push_data(DataMessage("alice", "chat", 1, b"hi from alice"))
    await bob_conn  .push_data(DataMessage("bob",   "chat", 2, b"hi from bob"))

    for _ in range(20):
        if len(received) >= 2:
            break
        await asyncio.sleep(0.05)

    by_pid = {m.participant_id: m.data for m in received}
    assert by_pid == {"alice": b"hi from alice", "bob": b"hi from bob"}
