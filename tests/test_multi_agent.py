# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end multi-agent scenarios.

Multiple :class:`ProcessorEndpoint` instances may share the same hub. They
all observe the full inbound stream (data, audio, participants) and may
each produce return traffic for any connected participant. The hub
arbitrates: every return message is targeted at exactly one participant
via its connector, regardless of which agent emitted it.
"""
from __future__ import annotations

import asyncio

import pytest

from xr_ai_agent import AudioChunk, DataMessage

from _helpers import wait_for_subscribed

pytestmark = pytest.mark.asyncio


async def _bring_up(make_connector):
    alice_conn = make_connector(connector_id="alice_conn")
    bob_conn   = make_connector(connector_id="bob_conn")
    await alice_conn.register()
    await bob_conn.register()
    await asyncio.sleep(0.1)
    await alice_conn.notify_participant_joined("alice", pts_us=1)
    await bob_conn  .notify_participant_joined("bob",   pts_us=2)
    await asyncio.sleep(0.1)
    return alice_conn, bob_conn


async def test_two_agents_observe_the_same_inbound_stream(hub, make_connector, make_processor, settle):
    alice_conn, _bob_conn = await _bring_up(make_connector)

    seen_a: list[DataMessage] = []
    seen_b: list[DataMessage] = []

    async def cb_a(m): seen_a.append(m)
    async def cb_b(m): seen_b.append(m)

    p_a = make_processor()
    p_b = make_processor()
    p_a.on_data(cb_a)
    p_b.on_data(cb_b)
    await wait_for_subscribed(p_a, p_b, pids=["alice", "bob"])

    for i in range(3):
        await alice_conn.push_data(DataMessage(
            participant_id="alice", topic="chat", pts_us=10 + i,
            data=f"msg-{i}".encode(),
        ))

    for _ in range(20):
        if len(seen_a) == 3 and len(seen_b) == 3:
            break
        await asyncio.sleep(0.05)

    assert [m.data for m in seen_a] == [b"msg-0", b"msg-1", b"msg-2"]
    assert [m.data for m in seen_b] == [b"msg-0", b"msg-1", b"msg-2"]


async def test_two_agents_two_clients_isolated_return_paths(hub, make_connector, make_processor, settle):
    """Two agents reply to two different clients in parallel; each client
    only ever sees the reply addressed to it."""
    alice_conn, bob_conn = await _bring_up(make_connector)

    alice_seen: list[DataMessage] = []
    bob_seen:   list[DataMessage] = []

    async def cb_alice(m): alice_seen.append(m)
    async def cb_bob(m):   bob_seen.append(m)

    alice_conn.on_return_data(cb_alice)
    bob_conn  .on_return_data(cb_bob)

    a_task = asyncio.create_task(alice_conn.run())
    b_task = asyncio.create_task(bob_conn.run())

    try:
        # Two independent agents in the same hub.
        agent_alice = make_processor()
        agent_bob   = make_processor()
        await wait_for_subscribed(agent_alice, agent_bob, pids=["alice", "bob"])

        # agent_alice replies to alice; agent_bob replies to bob.
        await agent_alice.send_return_data(DataMessage(
            "alice", "vlm.response", 100, b"alice-answer",
        ))
        await agent_bob.send_return_data(DataMessage(
            "bob",   "vlm.response", 101, b"bob-answer",
        ))

        for _ in range(20):
            if alice_seen and bob_seen:
                break
            await asyncio.sleep(0.05)

        # Soak time so any cross-leak would manifest.
        await asyncio.sleep(0.1)

        assert [m.data for m in alice_seen] == [b"alice-answer"]
        assert [m.data for m in bob_seen]   == [b"bob-answer"]
    finally:
        alice_conn.stop()
        bob_conn.stop()
        a_task.cancel()
        b_task.cancel()
        await asyncio.gather(a_task, b_task, return_exceptions=True)


async def test_agent_with_auto_subscribe_off_sees_participant_events_only(
    hub, make_connector, make_processor, settle,
):
    """``auto_subscribe=False`` and no explicit subscribe(pid) call →
    the processor wakes only for participant events; data, audio, and
    video are filtered at the ZMQ kernel."""
    seen_data:        list[DataMessage] = []
    seen_participant: list[str]         = []

    async def cb_data(m):        seen_data.append(m)
    async def cb_participant(e): seen_participant.append(e.participant_id)

    proc = make_processor(auto_subscribe=False)
    proc.on_data(cb_data)
    proc.on_participant(cb_participant)
    await settle()

    conn = make_connector()
    await conn.register()
    await asyncio.sleep(0.1)
    await conn.notify_participant_joined("alice", pts_us=1)
    await asyncio.sleep(0.1)
    await conn.push_data(DataMessage("alice", "chat", 10, b"hello"))
    await asyncio.sleep(0.2)

    assert seen_participant == ["alice"]
    assert seen_data == []  # not subscribed
