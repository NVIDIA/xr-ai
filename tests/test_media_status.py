# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Connector-owned media observations reach agents without client telemetry."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import zmq
import zmq.asyncio
from device_io_hub.transport.livekit._room_client import RoomClient
from livekit import rtc
from xr_ai_hub import DataMessage, MediaStatus, MsgType, Subscribe, decode, encode

from _helpers import wait_for


def publication(*, muted=False, kind=rtc.TrackKind.KIND_VIDEO):
    return SimpleNamespace(kind=kind, muted=muted)


def room_client(endpoint):
    participant = SimpleNamespace(
        identity="alice", connection_quality=rtc.ConnectionQuality.QUALITY_POOR,
        track_publications={"camera": publication()},
    )
    client = RoomClient.__new__(RoomClient)
    client._ep = endpoint
    client._room = SimpleNamespace(
        isconnected=lambda: True, remote_participants={"alice": participant},
    )
    client._participant_sessions = {"alice": "session"}
    client._stop = asyncio.Event()
    return client, participant


def test_media_status_codec_and_grade_validation():
    status = MediaStatus("alice", "session", 123, "poor", True, "connector")
    assert decode(encode(MsgType.MEDIA_STATUS, status)) == (MsgType.MEDIA_STATUS, status)
    with pytest.raises(ValueError, match="grade"):
        replace(status, quality="arbitrary")
    with pytest.raises(ValueError, match="boolean"):
        replace(status, video_expected="false")


async def test_connector_normalizes_native_or_web_participant_without_client_messages():
    ep = SimpleNamespace(push_media_status=AsyncMock())
    client, participant = room_client(ep)
    for grade, expected in [
        (rtc.ConnectionQuality.QUALITY_EXCELLENT, "excellent"),
        (rtc.ConnectionQuality.QUALITY_GOOD, "good"),
        (rtc.ConnectionQuality.QUALITY_POOR, "poor"),
        (rtc.ConnectionQuality.QUALITY_LOST, "lost"),
        (rtc.ConnectionQuality.QUALITY_UNKNOWN, "unknown"),
    ]:
        participant.connection_quality = grade
        await client._report_media_status()
        status = ep.push_media_status.call_args.args[0]
        assert status.quality == expected and status.video_expected
        assert status.participant_session_id == "session"
    participant.track_publications["camera"].muted = True
    await client._report_media_status()
    assert not ep.push_media_status.call_args.args[0].video_expected
    participant.track_publications = {"microphone": publication(kind=rtc.TrackKind.KIND_AUDIO)}
    await client._report_media_status()
    assert not ep.push_media_status.call_args.args[0].video_expected
    ep.push_media_status.reset_mock()
    client._room.isconnected = lambda: False
    await client._report_media_status()
    ep.push_media_status.assert_not_awaited()
    client._room.isconnected = lambda: True
    client._participant_sessions.clear()
    await client._report_media_status()
    ep.push_media_status.assert_not_awaited()


async def test_periodic_status_survives_unchanged_grade_and_stops_with_connector():
    ep = SimpleNamespace(push_media_status=AsyncMock())
    client, _ = room_client(ep)
    task = asyncio.create_task(client.run())
    try:
        await wait_for(lambda: ep.push_media_status.await_count >= 2, timeout=3)
        assert ep.push_media_status.call_args_list[0].args[0].quality == "poor"
    finally:
        client.stop()
        await asyncio.wait_for(task, 1)
    count = ep.push_media_status.await_count
    await asyncio.sleep(0.02)
    assert ep.push_media_status.await_count == count


async def test_connector_hub_sdk_routing_and_session_isolation(hub, make_connector, make_processor, settle):
    connector = make_connector()
    processor = make_processor(filter=Subscribe.DATA)
    audio_only = make_processor(filter=Subscribe.AUDIO)
    seen, data_seen, audio_seen = [], [], []

    async def receive(msg):
        seen.append(msg)

    async def receive_data(msg):
        data_seen.append(msg)

    async def receive_audio_status(msg):
        audio_seen.append(msg)

    unsubscribe = processor.on_media_status(receive)
    processor.on_data(receive_data)
    audio_only.on_media_status(receive_audio_status)
    await settle()
    await connector.notify_participant_joined("alice", 1, "session")
    await wait_for(lambda: "alice" in processor.subscribed_participants)
    await processor.wait_for_subscriptions()
    client, participant = room_client(connector)
    await client._report_media_status()
    await wait_for(lambda: len(seen) == 1)
    assert seen[0].quality == "poor" and seen[0].video_expected
    assert not data_seen and not audio_seen
    # A data-channel message cannot impersonate a connector status, even with
    # the reserved routing topic and a valid encoded MediaStatus payload.
    await connector.push_data(DataMessage("alice", "_media_status", 2, encode(MsgType.MEDIA_STATUS, seen[0])))
    await wait_for(lambda: bool(data_seen))
    assert len(seen) == 1
    old = seen[0]
    await connector.notify_participant_joined("alice", 3, "replacement")
    await wait_for(lambda: processor._participant_sessions.get("alice") == "replacement")
    # Validate at the hub as well as the connector: bypass the connector guard.
    await connector._push.send(encode(MsgType.MEDIA_STATUS, old))
    await connector.push_media_status(old)
    await connector._push.send(encode(
        MsgType.MEDIA_STATUS, replace(old, participant_session_id="replacement", connector_id="other"),
    ))
    client._participant_sessions["alice"] = "replacement"
    participant.connection_quality = rtc.ConnectionQuality.QUALITY_GOOD
    await client._report_media_status()
    await wait_for(lambda: len(seen) == 2)
    assert seen[-1].quality == "good" and seen[-1].participant_session_id == "replacement"
    assert not audio_seen
    unsubscribe()
    unsubscribe()
    await client._report_media_status()
    await settle()
    assert len(seen) == 2


async def test_late_worker_gets_unchanged_grade_without_client_opt_in(hub, make_connector, make_processor, settle):
    connector = make_connector()
    await settle()
    await connector.notify_participant_joined("alice", 1, "session")
    client, _ = room_client(connector)
    await client._report_media_status()
    processor = make_processor(filter=Subscribe.DATA)
    received = asyncio.Event()
    statuses = []

    async def on_status(message):
        statuses.append(message)
        received.set()

    processor.on_media_status(on_status)
    await wait_for(lambda: "alice" in processor.subscribed_participants)
    await processor.wait_for_subscriptions()
    task = asyncio.create_task(client.run())
    try:
        await asyncio.wait_for(received.wait(), 2)
        assert statuses[0].quality == "poor"
        assert statuses[0].video_expected
    finally:
        client.stop()
        await asyncio.wait_for(task, 1)


async def test_existing_data_subscribers_do_not_receive_new_wire_type(hub, hub_addrs, make_connector, settle):
    subscriber = zmq.asyncio.Context.instance().socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"data.alice.")
    subscriber.connect(hub_addrs[1])
    connector = make_connector()
    try:
        await settle()
        await connector.notify_participant_joined("alice", 1, "session")
        client, _ = room_client(connector)
        await client._report_media_status()
        original = DataMessage("alice", "chat", 2, b"hello")
        await connector.push_data(original)
        topic, payload = await asyncio.wait_for(subscriber.recv_multipart(), 2)
        assert topic == b"data.alice.chat"
        assert decode(payload) == (MsgType.DATA_MESSAGE, original)
    finally:
        subscriber.close(linger=0)


@pytest.mark.parametrize("subscribe_before_start", [False, True])
async def test_late_manual_subscriber_initializes_sessions_without_subscribing_everyone(
    hub, make_connector, make_processor, settle, subscribe_before_start,
):
    connector = make_connector()
    await settle()
    await connector.notify_participant_joined("alice", 1, "session")
    await connector.notify_participant_joined("bob", 1, "bob-session")
    await settle()
    processor = make_processor(auto_subscribe=False, filter=Subscribe.DATA)
    statuses, data = [], []

    async def status_received(message):
        statuses.append(message)

    async def data_received(message):
        data.append(message)

    processor.on_media_status(status_received)
    processor.on_data(data_received)
    if not subscribe_before_start:
        await processor.wait_until_running()
        await settle()
    processor.subscribe("alice")
    await processor.wait_until_running()
    assert await processor.wait_for_subscriptions()
    client, _ = room_client(connector)
    for i in range(3):
        await client._report_media_status()
        await connector.push_data(DataMessage("alice", "chat", i, b"hello"))
        await settle()
    assert len(data) == 3
    assert statuses, "manual subscriptions must receive current-session media status without a roster request"
    assert processor.subscribed_participants == {"alice"}
    assert all(msg.participant_session_id == "session" for msg in statuses)
    old_count = len(statuses)
    await connector.notify_participant_joined("alice", 5, "replacement")
    await wait_for(lambda: processor._participant_sessions.get("alice") == "replacement")
    await client._report_media_status()  # The old session must still be rejected.
    client._participant_sessions["alice"] = "replacement"
    await client._report_media_status()
    await wait_for(lambda: len(statuses) > old_count)
    assert len(statuses) == old_count + 1
    assert statuses[-1].participant_session_id == "replacement"
