# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Prefix-collision isolation tests for the return path.

The hub publishes return traffic on pid-scoped ZMQ topics and each connector
subscribes only for the participants it owns. ZMQ ``SUBSCRIBE`` is a *byte
prefix* match, so a subscription for participant ``alice`` will also match a
topic addressed to ``alice2`` unless a delimiter terminates the pid segment.

These tests pin that boundary using two identities where one is a byte prefix
of the other (``alice`` ⊂ ``alice2``). The connector owning ``alice`` must
*never* receive traffic addressed to ``alice2`` — on any of the three return
families (data, audio, audio-flush).

This is the same hazard the processor subscription path already guards against
(see ``_PREFIXES_BY_FLAG`` / ``_prefixes`` in ``xr_ai_agent._processor``); these
tests lock the connector return path to the same guarantee. The existing
``test_return_routing`` suite uses non-prefix identities (``alice``/``bob``) and
so cannot detect this case.
"""
from __future__ import annotations

import asyncio

import pytest

from xr_ai_agent import AudioChunk, DataMessage, ReturnAudioFlush

pytestmark = pytest.mark.asyncio

# Identities where one is a byte prefix of the other. The connector owning the
# shorter id is the one that over-matches if the topic boundary is wrong.
_SHORT = "alice"
_LONG = "alice2"


async def _bring_up_prefix_pair(make_connector):
    """One connector per identity, each joining its own prefix-related pid."""
    short_conn = make_connector(connector_id="short_conn")
    long_conn = make_connector(connector_id="long_conn")
    await short_conn.register()
    await long_conn.register()
    await asyncio.sleep(0.1)
    await short_conn.notify_participant_joined(_SHORT, pts_us=1)
    await long_conn.notify_participant_joined(_LONG, pts_us=2)
    await asyncio.sleep(0.1)
    return short_conn, long_conn


async def test_return_data_does_not_leak_to_prefix_participant(hub, make_connector, make_processor, settle):
    short_conn, long_conn = await _bring_up_prefix_pair(make_connector)

    short_received: list[DataMessage] = []
    long_received: list[DataMessage] = []

    async def cb_short(msg): short_received.append(msg)
    async def cb_long(msg): long_received.append(msg)

    short_conn.on_return_data(cb_short)
    long_conn.on_return_data(cb_long)

    short_task = asyncio.create_task(short_conn.run(), name="short_run")
    long_task = asyncio.create_task(long_conn.run(), name="long_run")
    try:
        proc = make_processor()
        await settle()

        # Targeted at alice2 only.
        await proc.send_return_data(DataMessage(
            participant_id=_LONG, topic="vlm.response", pts_us=10, data=b"for-long",
        ))

        for _ in range(20):
            if long_received:
                break
            await asyncio.sleep(0.05)
        # Give the bus a chance to (incorrectly) leak to alice's connector.
        await asyncio.sleep(0.1)

        assert len(long_received) == 1
        assert long_received[0].participant_id == _LONG
        # alice ⊂ alice2: alice's connector must NOT over-match alice2's data.
        assert short_received == []
    finally:
        await _teardown(short_conn, long_conn, short_task, long_task)


async def test_return_audio_does_not_leak_to_prefix_participant(hub, make_connector, make_processor, settle):
    short_conn, long_conn = await _bring_up_prefix_pair(make_connector)

    short_received: list[AudioChunk] = []
    long_received: list[AudioChunk] = []

    async def cb_short(msg): short_received.append(msg)
    async def cb_long(msg): long_received.append(msg)

    short_conn.on_return_audio(cb_short)
    long_conn.on_return_audio(cb_long)

    short_task = asyncio.create_task(short_conn.run(), name="short_run")
    long_task = asyncio.create_task(long_conn.run(), name="long_run")
    try:
        proc = make_processor()
        await settle()

        await proc.send_return_audio(AudioChunk(
            pts_us=10, sample_rate=48_000, channels=1, samples=480,
            data=b"\x00" * (480 * 4), participant_id=_LONG, track_id="ret",
        ))

        for _ in range(20):
            if long_received:
                break
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.1)

        assert len(long_received) == 1
        assert long_received[0].participant_id == _LONG
        assert short_received == []
    finally:
        await _teardown(short_conn, long_conn, short_task, long_task)


async def test_return_audio_flush_does_not_leak_to_prefix_participant(hub, make_connector, make_processor, settle):
    short_conn, long_conn = await _bring_up_prefix_pair(make_connector)

    short_flushes: list[ReturnAudioFlush] = []
    long_flushes: list[ReturnAudioFlush] = []

    async def cb_short(msg): short_flushes.append(msg)
    async def cb_long(msg): long_flushes.append(msg)

    short_conn.on_return_audio_flush(cb_short)
    long_conn.on_return_audio_flush(cb_long)

    short_task = asyncio.create_task(short_conn.run(), name="short_run")
    long_task = asyncio.create_task(long_conn.run(), name="long_run")
    try:
        proc = make_processor()
        await settle()

        await proc.flush_return_audio(_LONG)

        for _ in range(20):
            if long_flushes:
                break
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.1)

        assert len(long_flushes) == 1
        assert long_flushes[0].participant_id == _LONG
        assert short_flushes == []
    finally:
        await _teardown(short_conn, long_conn, short_task, long_task)


async def _teardown(short_conn, long_conn, short_task, long_task) -> None:
    short_conn.stop()
    long_conn.stop()
    short_task.cancel()
    long_task.cancel()
    await asyncio.gather(short_task, long_task, return_exceptions=True)
