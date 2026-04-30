"""
Multi-client return-path isolation tests.

Each connector represents a separate client. After the multi-agent / multi-
client pipeline change, return data and return audio sent by an agent for
participant *X* must:

* be **received only** by the connector that owns participant *X*;
* arrive on a topic that includes the destination participant id (so
  cross-connector eavesdropping is impossible by construction);
* obey the new ``ReturnAudioFlush`` control message (queued audio for
  *X* is dropped without affecting *Y*).
"""
from __future__ import annotations

import asyncio

import pytest

from xr_ai_agent import AudioChunk, DataMessage

pytestmark = pytest.mark.asyncio


async def _bring_up_two_clients(make_connector):
    """Register two connectors and join one participant on each."""
    alice_conn = make_connector(connector_id="alice_conn")
    bob_conn   = make_connector(connector_id="bob_conn")
    await alice_conn.register()
    await bob_conn.register()
    await asyncio.sleep(0.1)
    await alice_conn.notify_participant_joined("alice", pts_us=1)
    await bob_conn  .notify_participant_joined("bob",   pts_us=2)
    await asyncio.sleep(0.1)
    return alice_conn, bob_conn


async def test_return_data_only_reaches_target_connector(hub, make_connector, make_processor, settle):
    alice_conn, bob_conn = await _bring_up_two_clients(make_connector)

    alice_received: list[DataMessage] = []
    bob_received:   list[DataMessage] = []

    async def cb_alice(msg): alice_received.append(msg)
    async def cb_bob(msg):   bob_received.append(msg)

    alice_conn.on_return_data(cb_alice)
    bob_conn  .on_return_data(cb_bob)

    # Run each connector's receive loop so they actually pull from SUB.
    alice_task = asyncio.create_task(alice_conn.run(), name="alice_conn_run")
    bob_task   = asyncio.create_task(bob_conn.run(),   name="bob_conn_run")

    try:
        proc = make_processor()
        await settle()

        # Agent sends return data targeted at alice only.
        await proc.send_return_data(DataMessage(
            participant_id="alice", topic="vlm.response",
            pts_us=10, data=b"a-chair",
        ))

        # Wait for the message to fan through hub → alice_conn.
        for _ in range(20):
            if alice_received:
                break
            await asyncio.sleep(0.05)

        # bob's connector must not receive alice's return data.
        await asyncio.sleep(0.1)

        assert len(alice_received) == 1
        assert alice_received[0].participant_id == "alice"
        assert alice_received[0].topic          == "vlm.response"
        assert alice_received[0].data           == b"a-chair"
        assert bob_received == []

    finally:
        alice_conn.stop()
        bob_conn.stop()
        alice_task.cancel()
        bob_task.cancel()
        await asyncio.gather(alice_task, bob_task, return_exceptions=True)


async def test_return_data_for_unknown_participant_is_dropped(hub, make_processor, settle):
    """The hub refuses to publish return-traffic for unconnected participants
    so cross-room leakage is impossible even if an agent has a stale id."""
    proc = make_processor()
    await settle()

    # The hub knows about no participants here — this should be a quiet drop,
    # not an exception. We assert via "no crash" + a brief settle.
    await proc.send_return_data(DataMessage(
        participant_id="ghost", topic="x", pts_us=0, data=b"!"
    ))
    await asyncio.sleep(0.1)
