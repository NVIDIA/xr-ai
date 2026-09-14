# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import uuid
from dataclasses import replace
from types import SimpleNamespace

import pytest
import zmq
from device_io_hub.ipc import ConnectorEndpoint, HubEndpoint
from device_io_hub.ipc._hub import _file_prefix as hub_file_prefix
from device_io_hub.transport.livekit._room_client import RoomClient
from device_io_hub.transport.livekit.config import LiveKitConnectorConfig
from xr_ai_hub import FileMessage, MsgType, ProcessorEndpoint, Subscribe, decode, encode
from xr_ai_hub._processor import _file_prefix as processor_file_prefix


def test_file_message_codec_round_trip() -> None:
    message = FileMessage(
        participant_id="alice",
        topic="image.response",
        pts_us=123,
        transfer_id="stream-1",
        name="capture.png",
        mime_type="image/png",
        attributes={"request_id": "request-1", "image_index": "0"},
        data=b"png",
        participant_session_id="session-1",
    )

    type_id, decoded = decode(encode(MsgType.FILE_MESSAGE, message))

    assert type_id == MsgType.FILE_MESSAGE
    assert decoded == message


def test_file_participant_prefix_is_exact_and_shared() -> None:
    alice_prefix = processor_file_prefix("alice")

    assert alice_prefix == hub_file_prefix("alice")
    assert not hub_file_prefix("alice.foo").startswith(alice_prefix)


@pytest.mark.asyncio
async def test_room_client_delivers_only_a_complete_validated_file(monkeypatch) -> None:
    class Reader:
        def __init__(self) -> None:
            self.info = SimpleNamespace(
                stream_id="stream-1",
                size=6,
                name="capture.png",
                mime_type="image/png",
                attributes={
                    "_streamkit.topic": "image.response",
                    "request_id": "request-1",
                },
            )
            self._chunks = iter((b"png", b"123"))
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self) -> bytes:
            try:
                return next(self._chunks)
            except StopIteration:
                raise StopAsyncIteration from None

        def close(self) -> None:
            self.closed = True

    class Endpoint:
        def __init__(self) -> None:
            self.files: list[FileMessage] = []

        async def push_file(self, message: FileMessage) -> bool:
            self.files.append(message)
            return True

    client = RoomClient.__new__(RoomClient)
    client._cfg = LiveKitConnectorConfig(api_key="key", api_secret="secret")
    client._ep = Endpoint()
    reader = Reader()
    header = client._snapshot_file_header(reader)

    monkeypatch.setattr(
        "device_io_hub.transport.livekit._room_client._now_us",
        lambda: 456,
    )
    await client._consume_file(reader, "alice", "session-1", header)

    assert reader.closed
    assert client._ep.files == [
        FileMessage(
            participant_id="alice",
            topic="image.response",
            pts_us=456,
            transfer_id="stream-1",
            name="capture.png",
            mime_type="image/png",
            attributes={"request_id": "request-1"},
            data=b"png123",
            participant_session_id="session-1",
        )
    ]


@pytest.mark.asyncio
async def test_room_client_rejects_declared_size_mismatch() -> None:
    class Reader:
        def __init__(self) -> None:
            self.info = SimpleNamespace(
                stream_id="stream-bad-size",
                size=5,
                name="capture.png",
                mime_type="image/png",
                attributes={"_streamkit.topic": "image.response"},
            )
            self._chunks = iter((b"png", b"123"))
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self) -> bytes:
            try:
                return next(self._chunks)
            except StopIteration:
                raise StopAsyncIteration from None

        def close(self) -> None:
            self.closed = True

    class Endpoint:
        def __init__(self) -> None:
            self.files: list[FileMessage] = []

        async def push_file(self, message: FileMessage) -> bool:
            self.files.append(message)
            return True

    client = RoomClient.__new__(RoomClient)
    client._cfg = LiveKitConnectorConfig(api_key="key", api_secret="secret")
    client._ep = Endpoint()
    reader = Reader()

    await client._consume_file(
        reader,
        "alice",
        "session-1",
        client._snapshot_file_header(reader),
    )

    assert reader.closed
    assert client._ep.files == []


def test_file_ipc_rejects_unbounded_hwm() -> None:
    with pytest.raises(ValueError, match="file_hwm"):
        HubEndpoint("inproc://hwm-hub-in", "inproc://hwm-hub-pub", file_hwm=0)
    with pytest.raises(ValueError, match="file_hwm"):
        ConnectorEndpoint("inproc://hwm-connector-in", "inproc://hwm-connector-pub", file_hwm=0)
    with pytest.raises(ValueError, match="file_hwm"):
        ProcessorEndpoint("inproc://hwm-processor-pub", "inproc://hwm-processor-in", file_hwm=0)


@pytest.mark.asyncio
async def test_connector_rejects_file_larger_than_ipc_limit() -> None:
    connector = ConnectorEndpoint(
        "inproc://oversize-file-in",
        "inproc://oversize-file-pub",
        file_push_addr="inproc://oversize-file-lane",
        file_max_bytes=3,
    )
    connector._participant_sessions["alice"] = "session-1"
    try:
        accepted = await connector.push_file(FileMessage(
            participant_id="alice",
            topic="image.response",
            pts_us=123,
            transfer_id="stream-too-large",
            name="capture.png",
            mime_type="image/png",
            attributes={},
            data=b"four",
            participant_session_id="session-1",
        ))
        assert not accepted
    finally:
        connector.close()


@pytest.mark.asyncio
async def test_room_client_enforces_per_participant_admission() -> None:
    class Reader:
        def __init__(self, stream_id: str) -> None:
            self.info = SimpleNamespace(
                stream_id=stream_id,
                size=1,
                name="capture.png",
                mime_type="image/png",
                attributes={"_streamkit.topic": "image.response"},
            )
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self) -> bytes:
            await asyncio.Future()

        def close(self) -> None:
            self.closed = True

    client = RoomClient.__new__(RoomClient)
    client._cfg = LiveKitConnectorConfig(
        api_key="key",
        api_secret="secret",
        incoming_file_max_concurrent_per_participant=1,
    )
    client._accepting_files = True
    client._participant_sessions = {"alice": "session-1", "bob": "session-2"}
    client._file_tasks = {}
    client._ep = SimpleNamespace()
    first = Reader("stream-1")
    rejected = Reader("stream-2")
    other_participant = Reader("stream-3")

    client._on_file_stream(first, "alice")
    client._on_file_stream(rejected, "alice")
    client._on_file_stream(other_participant, "bob")

    assert not first.closed
    assert rejected.closed
    assert not other_participant.closed
    tasks = list(client._file_tasks)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    assert first.closed
    assert other_participant.closed


@pytest.mark.asyncio
async def test_completed_file_routes_on_bounded_file_lane(tmp_path) -> None:
    suffix = uuid.uuid4().hex[:8]
    pull = f"ipc://{tmp_path}/in-{suffix}"
    pub = f"ipc://{tmp_path}/pub-{suffix}"
    file_pull = f"ipc://{tmp_path}/file-in-{suffix}"
    file_pub = f"ipc://{tmp_path}/file-pub-{suffix}"
    hub = HubEndpoint(
        pull,
        pub,
        file_pull_addr=file_pull,
        file_pub_addr=file_pub,
    )
    connector = ConnectorEndpoint(
        pull,
        pub,
        file_push_addr=file_pull,
        num_slots=1,
        max_frame_bytes=1024,
    )
    processor = ProcessorEndpoint(
        pub,
        pull,
        file_sub_addr=file_pub,
        filter=Subscribe.DEFAULT | Subscribe.FILE,
    )
    received: list[FileMessage] = []
    subscription_confirmed: list[bool] = []
    delivered = asyncio.Event()

    async def on_file(message: FileMessage) -> None:
        received.append(message)
        processor.subscribe("bob")
        subscription_confirmed.append(
            await processor.wait_for_subscriptions(timeout=0.5)
        )
        delivered.set()

    processor.on_file(on_file)
    hub_task = asyncio.create_task(hub.run())
    processor_task = asyncio.create_task(processor.run())
    try:
        await connector.register()
        participant_session_id = await connector.notify_participant_joined("alice")
        await asyncio.wait_for(processor.wait_until_running(), 1)
        async with asyncio.timeout(1):
            while "alice" not in processor.connected_participants:
                await asyncio.sleep(0)
        assert await processor.wait_for_subscriptions(timeout=1)

        message = FileMessage(
            participant_id="alice",
            topic="image.response",
            pts_us=123,
            transfer_id="stream-1",
            name="capture.png",
            mime_type="image/png",
            attributes={"request_id": "request-1"},
            data=b"png",
            participant_session_id=participant_session_id,
        )
        # Bypass ConnectorEndpoint's first-line check to model an old file
        # already queued when this participant identity reconnects. The hub
        # must discard it before publishing the current session's file.
        stale_message = replace(message, participant_session_id="stale-session")
        await connector._file_push.send(encode(MsgType.FILE_MESSAGE, stale_message))
        assert await connector.push_file(message)
        await asyncio.wait_for(delivered.wait(), 1)

        assert received == [message]
        assert subscription_confirmed == [True]
    finally:
        processor.stop()
        processor.close()
        connector.stop()
        connector.close()
        hub.stop()
        hub.close()
        hub_task.cancel()
        processor_task.cancel()
        await asyncio.gather(hub_task, processor_task, return_exceptions=True)


def test_file_fanout_uses_per_subscriber_drop_policy() -> None:
    hub = HubEndpoint(
        "inproc://fanout-policy-in",
        "inproc://fanout-policy-pub",
        file_pull_addr="inproc://fanout-policy-file-in",
        file_pub_addr="inproc://fanout-policy-file-pub",
        file_hwm=1,
    )
    try:
        assert hub._file_pub.getsockopt(zmq.TYPE) == zmq.PUB
        assert hub._file_pub.getsockopt(zmq.SNDHWM) == 1
    finally:
        hub.close()
