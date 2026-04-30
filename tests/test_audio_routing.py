# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Audio chunk and return-audio routing tests.

Inbound audio (client → hub → every processor) and return audio (processor
→ hub → only the target connector) must keep their per-participant
attribution and never leak across clients.
"""
from __future__ import annotations

import asyncio

import pytest

from xr_ai_agent import AudioChunk

pytestmark = pytest.mark.asyncio

_PCM_SILENCE = b"\x00" * 1920  # 480 samples @ 16-bit ≈ 10 ms


def _silence(pid: str, *, pts_us: int = 0, sample_rate: int = 48_000) -> AudioChunk:
    return AudioChunk(
        pts_us=pts_us, sample_rate=sample_rate, channels=1,
        samples=480, data=_PCM_SILENCE, participant_id=pid, track_id="mic",
    )


async def test_inbound_audio_is_attributed_to_participant(hub, make_connector, make_processor, settle):
    received: list[AudioChunk] = []

    async def cb(c): received.append(c)

    proc = make_processor()
    proc.on_audio(cb)
    await settle()

    alice_conn = make_connector(connector_id="alice_conn")
    bob_conn   = make_connector(connector_id="bob_conn")
    await alice_conn.register()
    await bob_conn.register()
    await asyncio.sleep(0.1)
    await alice_conn.notify_participant_joined("alice", pts_us=1)
    await bob_conn  .notify_participant_joined("bob",   pts_us=2)
    await asyncio.sleep(0.1)

    await alice_conn.push_audio(_silence("alice", pts_us=10))
    await bob_conn  .push_audio(_silence("bob",   pts_us=11))

    for _ in range(20):
        if len(received) >= 2:
            break
        await asyncio.sleep(0.05)

    by_pid = {c.participant_id for c in received}
    assert by_pid == {"alice", "bob"}


async def test_return_audio_only_reaches_target_connector(hub, make_connector, make_processor, settle):
    alice_conn = make_connector(connector_id="alice_conn")
    bob_conn   = make_connector(connector_id="bob_conn")
    await alice_conn.register()
    await bob_conn.register()
    await asyncio.sleep(0.1)
    await alice_conn.notify_participant_joined("alice", pts_us=1)
    await bob_conn  .notify_participant_joined("bob",   pts_us=2)
    await asyncio.sleep(0.1)

    alice_audio: list[AudioChunk] = []
    bob_audio:   list[AudioChunk] = []

    async def cb_alice(c): alice_audio.append(c)
    async def cb_bob(c):   bob_audio.append(c)

    alice_conn.on_return_audio(cb_alice)
    bob_conn  .on_return_audio(cb_bob)

    a_task = asyncio.create_task(alice_conn.run())
    b_task = asyncio.create_task(bob_conn.run())

    try:
        proc = make_processor()
        await settle()

        await proc.send_return_audio(_silence("alice", pts_us=200))

        for _ in range(20):
            if alice_audio:
                break
            await asyncio.sleep(0.05)

        await asyncio.sleep(0.1)  # leak window

        assert len(alice_audio) == 1
        assert alice_audio[0].participant_id == "alice"
        assert bob_audio == []
    finally:
        alice_conn.stop()
        bob_conn.stop()
        a_task.cancel()
        b_task.cancel()
        await asyncio.gather(a_task, b_task, return_exceptions=True)
