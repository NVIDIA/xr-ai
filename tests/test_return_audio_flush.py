"""
Verify the new ``ReturnAudioFlush`` control message round-trips end-to-end.

Flow:
    processor.flush_return_audio("alice")
    → hub publishes on  return_audio_flush.alice
    → alice's connector receives it via on_return_audio_flush(...)
    → bob's connector never sees it.
"""
from __future__ import annotations

import asyncio

import pytest

from xr_ai_agent import ReturnAudioFlush

pytestmark = pytest.mark.asyncio


async def test_flush_reaches_only_target_connector(hub, make_connector, make_processor, settle):
    alice_conn = make_connector(connector_id="alice_conn")
    bob_conn   = make_connector(connector_id="bob_conn")
    await alice_conn.register()
    await bob_conn.register()
    await asyncio.sleep(0.1)
    await alice_conn.notify_participant_joined("alice", pts_us=1)
    await bob_conn  .notify_participant_joined("bob",   pts_us=2)
    await asyncio.sleep(0.1)

    alice_flushes: list[ReturnAudioFlush] = []
    bob_flushes:   list[ReturnAudioFlush] = []

    async def cb_alice(msg): alice_flushes.append(msg)
    async def cb_bob(msg):   bob_flushes.append(msg)

    alice_conn.on_return_audio_flush(cb_alice)
    bob_conn  .on_return_audio_flush(cb_bob)

    alice_task = asyncio.create_task(alice_conn.run())
    bob_task   = asyncio.create_task(bob_conn.run())

    try:
        proc = make_processor()
        await settle()

        await proc.flush_return_audio("alice")

        for _ in range(20):
            if alice_flushes:
                break
            await asyncio.sleep(0.05)

        # Give the bus a chance to (incorrectly) leak the flush to bob.
        await asyncio.sleep(0.1)

        assert len(alice_flushes) == 1
        assert alice_flushes[0].participant_id == "alice"
        assert bob_flushes == []
    finally:
        alice_conn.stop()
        bob_conn.stop()
        alice_task.cancel()
        bob_task.cancel()
        await asyncio.gather(alice_task, bob_task, return_exceptions=True)


async def test_flush_for_unknown_participant_is_silently_dropped(hub, make_processor):
    """Mirror of ``test_return_data_for_unknown_participant_is_dropped`` —
    flushes for participants the hub doesn't know about must not raise."""
    proc = make_processor()
    await asyncio.sleep(0.05)
    await proc.flush_return_audio("ghost")
    await asyncio.sleep(0.1)
