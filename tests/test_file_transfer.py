# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import uuid
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import msgpack
import pytest
import zmq
from device_io_hub.ipc import ConnectorEndpoint, HubEndpoint
from device_io_hub.ipc._hub import _file_prefix as hub_file_prefix
from device_io_hub.transport.livekit._room_client import RoomClient
from device_io_hub.transport.livekit.config import LiveKitConnectorConfig
from xr_ai_hub import (
    FileMessage,
    MsgType,
    ParticipantEvent,
    ProcessorEndpoint,
    Subscribe,
    decode,
    encode,
)
from xr_ai_hub._file_ordering import FileRoute, FileSessionOrderer
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


def test_file_and_participant_decoders_accept_pre_session_payloads() -> None:
    file_payload = [
        "alice", "image.response", 123, "stream-1", "capture.png",
        "image/png", {}, b"png",
    ]
    type_id, file_message = decode(
        bytes([MsgType.FILE_MESSAGE]) + msgpack.packb(file_payload, use_bin_type=True),
    )
    assert type_id == MsgType.FILE_MESSAGE
    assert file_message.participant_session_id == ""

    participant_payload = ["alice", True, 123, "connector-1"]
    type_id, participant_event = decode(
        bytes([MsgType.PARTICIPANT_EVENT])
        + msgpack.packb(participant_payload, use_bin_type=True),
    )
    assert type_id == MsgType.PARTICIPANT_EVENT
    assert participant_event.participant_session_id == ""


def test_file_participant_prefix_is_exact_and_shared() -> None:
    alice_prefix = processor_file_prefix("alice")

    assert alice_prefix == hub_file_prefix("alice")
    assert not hub_file_prefix("alice.foo").startswith(alice_prefix)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("size", None, "nonnegative total size"),
        ("size", -1, "nonnegative total size"),
        ("size", 16 * 1024 * 1024 + 1, "declared size exceeds"),
        ("name", "bad\x00name", "cannot contain NUL"),
        ("name", "€" * 86, "exceeds 255 UTF-8 bytes"),
        ("mime_type", "", "must be a nonempty string"),
        ("attributes", [], "string key/value pairs"),
        ("attributes", {}, "file topic must be a nonempty string"),
        (
            "attributes",
            {"_streamkit.topic": "_streamkit.private"},
            "file topic cannot begin",
        ),
        (
            "attributes",
            {"_streamkit.topic": "image.response", "_streamkit.other": "x"},
            "reserved file attribute",
        ),
        (
            "attributes",
            {"_streamkit.topic": "image.response", **{f"k{i}": "v" for i in range(33)}},
            "exceed 32 application entries",
        ),
        (
            "attributes",
            {"_streamkit.topic": "image.response", **{f"k{i}": "v" * 1024 for i in range(9)}},
            "exceed 8192 UTF-8 bytes",
        ),
        (
            "attributes",
            {"_streamkit.topic": "image.response", "k" * 129: "v"},
            "key exceeds 128 UTF-8 bytes",
        ),
        (
            "attributes",
            {"_streamkit.topic": "image.response", "key": "v" * 1025},
            "exceeds 1024 UTF-8 bytes",
        ),
        (
            "attributes",
            {"_streamkit.topic": "image.response", "key": 1},
            "must be a nonempty string",
        ),
    ],
)
def test_room_client_rejects_invalid_file_headers(field, value, message) -> None:
    info = {
        "stream_id": "stream-invalid",
        "size": 1,
        "name": "capture.png",
        "mime_type": "image/png",
        "attributes": {"_streamkit.topic": "image.response"},
    }
    info[field] = value
    client = RoomClient.__new__(RoomClient)
    client._cfg = LiveKitConnectorConfig(
        api_key="key",
        api_secret="test-secret-at-least-32-bytes-long",
    )
    reader = SimpleNamespace(info=SimpleNamespace(**info))

    with pytest.raises(ValueError, match=message):
        client._snapshot_file_header(reader)


def test_room_client_accepts_maximum_application_attribute_count() -> None:
    attributes = {
        "_streamkit.topic": "image.response",
        **{f"k{i}": "v" for i in range(32)},
    }
    client = RoomClient.__new__(RoomClient)
    client._cfg = LiveKitConnectorConfig(
        api_key="key",
        api_secret="test-secret-at-least-32-bytes-long",
    )
    reader = SimpleNamespace(
        info=SimpleNamespace(
            stream_id="stream-valid",
            size=1,
            name="capture.png",
            mime_type="image/png",
            attributes=attributes,
        ),
    )

    header = client._snapshot_file_header(reader)

    assert header["attributes"] == attributes


def test_prejoin_file_expires_after_departure_churn() -> None:
    now = [0.0]
    orderer = FileSessionOrderer(
        1,
        clock=lambda: now[0],
        pending_ttl_s=1.0,
        max_departures=2,
    )
    stale = FileMessage(
        participant_id="alice",
        topic="image.response",
        pts_us=1,
        transfer_id="stale",
        name="stale.png",
        mime_type="image/png",
        attributes={},
        data=b"stale",
        participant_session_id="alice-old",
    )
    for participant_id in ("alice", "bob", "carol"):
        orderer.participant_joined(participant_id, f"{participant_id}-old")
        orderer.participant_left(participant_id, f"{participant_id}-old")

    assert orderer.route(stale) is FileRoute.BUFFERED
    assert orderer.pending_count == 1

    now[0] = 2.0
    current = replace(
        stale,
        transfer_id="current",
        participant_session_id="alice-current",
    )
    assert orderer.route(current) is FileRoute.BUFFERED
    assert orderer.pending_count == 1
    ready, inactive = orderer.participant_joined("alice", "alice-current")
    assert ready == [current]
    assert inactive == []
    assert orderer.pending_count == 0


def test_file_session_orderer_routes_departures_capacity_and_expiry() -> None:
    now = [0.0]
    orderer = FileSessionOrderer(
        1,
        clock=lambda: now[0],
        pending_ttl_s=1.0,
        departure_ttl_s=1.0,
    )
    message = FileMessage(
        participant_id="alice",
        topic="image.response",
        pts_us=1,
        transfer_id="old",
        name="old.png",
        mime_type="image/png",
        attributes={},
        data=b"old",
        participant_session_id="session-1",
    )

    assert orderer.seconds_until_expiry() is None
    orderer.participant_joined("alice", "session-1")
    assert orderer.participant_left("alice", "session-1") == []
    assert orderer.route(message) is FileRoute.INACTIVE

    now[0] = 2.0
    assert orderer.route(message) is FileRoute.BUFFERED
    assert orderer.seconds_until_expiry() == 1.0
    assert orderer.route(replace(message, transfer_id="full")) is FileRoute.FULL

    now[0] = 4.0
    assert orderer.seconds_until_expiry() == 0.0
    assert orderer.expire() == [message]
    assert orderer.seconds_until_expiry() is None


def test_file_session_orderer_handles_displacement_discard_and_legacy_session() -> None:
    orderer = FileSessionOrderer(2)
    old = FileMessage(
        participant_id="alice",
        topic="image.response",
        pts_us=1,
        transfer_id="old",
        name="old.png",
        mime_type="image/png",
        attributes={},
        data=b"old",
        participant_session_id="session-1",
    )

    orderer.participant_joined("alice", "session-1")
    orderer.participant_joined("alice", "session-2")
    assert orderer.route(old) is FileRoute.INACTIVE
    assert orderer.route(replace(old, participant_session_id="")) is FileRoute.READY

    pending = replace(
        old,
        participant_id="bob",
        participant_session_id="session-b",
    )
    assert orderer.route(pending) is FileRoute.BUFFERED
    assert orderer.participant_left("bob", "session-b") == [pending]


@pytest.mark.parametrize("endpoint_type", [HubEndpoint, ProcessorEndpoint])
async def test_file_lane_drains_all_lifecycle_events_before_routing(
    endpoint_type,
) -> None:
    endpoint = endpoint_type.__new__(endpoint_type)
    endpoint._file_orderer = FileSessionOrderer(1)
    endpoint._file_session_events = asyncio.Queue()
    joined = ParticipantEvent("alice", True, 1, "connector", "session-1")
    left = replace(joined, joined=False, pts_us=2)
    endpoint._file_session_events.put_nowait(joined)
    endpoint._file_session_events.put_nowait(left)
    lifecycle = asyncio.create_task(endpoint._file_session_events.get())

    try:
        lifecycle = await endpoint._drain_file_session_events(lifecycle)
        stale = FileMessage(
            participant_id="alice",
            topic="image.response",
            pts_us=3,
            transfer_id="stale",
            name="stale.png",
            mime_type="image/png",
            attributes={},
            data=b"stale",
            participant_session_id="session-1",
        )
        assert endpoint._file_orderer.route(stale) is FileRoute.INACTIVE
    finally:
        lifecycle.cancel()
        await asyncio.gather(lifecycle, return_exceptions=True)


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


async def test_room_client_rejects_topic_changed_during_transfer(monkeypatch) -> None:
    reader = SimpleNamespace(
        info=SimpleNamespace(
            stream_id="stream-changed-topic",
            size=3,
            name="capture.png",
            mime_type="image/png",
            attributes={"_streamkit.topic": "image.response"},
        ),
    )
    endpoint = SimpleNamespace(push_file=AsyncMock(return_value=True))
    client = RoomClient.__new__(RoomClient)
    client._cfg = LiveKitConnectorConfig(api_key="key", api_secret="secret")
    client._ep = endpoint
    header = client._snapshot_file_header(reader)
    reader.info.attributes = {"_streamkit.topic": "other.response"}
    monkeypatch.setattr(
        "device_io_hub.transport.livekit._room_client.read_byte_stream",
        AsyncMock(return_value=b"png"),
    )

    await client._consume_file(reader, "alice", "session-1", header)

    endpoint.push_file.assert_not_awaited()


@pytest.mark.parametrize(
    ("final_attributes", "read_error"),
    [
        (
            {
                "_streamkit.topic": "image.response",
                "_streamkit.late": "reserved",
            },
            None,
        ),
        ({"_streamkit.topic": "image.response"}, TimeoutError()),
    ],
)
async def test_room_client_rejects_late_reserved_metadata_and_timeouts(
    monkeypatch,
    final_attributes,
    read_error,
) -> None:
    reader = SimpleNamespace(
        info=SimpleNamespace(
            stream_id="stream-rejected",
            size=3,
            name="capture.png",
            mime_type="image/png",
            attributes={"_streamkit.topic": "image.response"},
        ),
    )
    endpoint = SimpleNamespace(push_file=AsyncMock(return_value=True))
    client = RoomClient.__new__(RoomClient)
    client._cfg = LiveKitConnectorConfig(api_key="key", api_secret="secret")
    client._ep = endpoint
    header = client._snapshot_file_header(reader)
    reader.info.attributes = final_attributes
    read = AsyncMock(return_value=b"png")
    if read_error is not None:
        read.side_effect = read_error
    monkeypatch.setattr(
        "device_io_hub.transport.livekit._room_client.read_byte_stream",
        read,
    )

    await client._consume_file(reader, "alice", "session-1", header)

    endpoint.push_file.assert_not_awaited()


def test_room_client_rejects_file_before_task_creation() -> None:
    class Reader:
        def __init__(self, stream_id: str, attributes=None) -> None:
            self.info = SimpleNamespace(
                stream_id=stream_id,
                size=1,
                name="capture.png",
                mime_type="image/png",
                attributes=(
                    {"_streamkit.topic": "image.response"}
                    if attributes is None
                    else attributes
                ),
            )
            self.closed = False

        def close(self) -> None:
            self.closed = True

    client = RoomClient.__new__(RoomClient)
    client._cfg = LiveKitConnectorConfig(
        api_key="key",
        api_secret="secret",
        incoming_file_max_concurrent=1,
        incoming_file_max_concurrent_per_participant=1,
    )
    client._accepting_files = False
    client._participant_sessions = {"alice": "session-1"}
    client._file_tasks = {}

    disabled = Reader("disabled")
    client._on_file_stream(disabled, "alice")
    assert disabled.closed

    client._accepting_files = True
    client._file_tasks = {object(): ("alice", "session-1", object())}
    unknown = Reader("unknown")
    client._on_file_stream(unknown, "bob")
    assert unknown.closed

    full = Reader("full")
    client._on_file_stream(full, "alice")
    assert full.closed

    client._file_tasks.clear()
    invalid = Reader("invalid", attributes={})
    client._on_file_stream(invalid, "alice")
    assert invalid.closed
    assert client._file_tasks == {}


async def test_room_connect_assigns_all_sessions_before_join_notifications() -> None:
    alice = SimpleNamespace(identity="alice", track_publications={})
    bob = SimpleNamespace(identity="bob", track_publications={})
    client = RoomClient.__new__(RoomClient)
    client._cfg = LiveKitConnectorConfig(
        api_key="key",
        api_secret="test-secret-at-least-32-bytes-long",
    )
    client._room = SimpleNamespace(
        connect=AsyncMock(),
        remote_participants={"alice": alice, "bob": bob},
    )
    client._participant_sessions = {}
    client._maybe_start_track = lambda _track, _identity: None
    observed_sessions: list[set[str]] = []
    observed_file_admission: list[bool] = []

    async def handle_joined(_participant, _session_id) -> None:
        observed_sessions.append(set(client._participant_sessions))
        observed_file_admission.append(client._accepting_files)

    client._handle_joined = handle_joined

    await client.connect()

    assert observed_sessions == [{"alice", "bob"}, {"alice", "bob"}]
    assert observed_file_admission == [False, False]
    assert client._accepting_files


def test_file_ipc_rejects_unbounded_hwm() -> None:
    with pytest.raises(ValueError, match="file_hwm"):
        HubEndpoint("inproc://hwm-hub-in", "inproc://hwm-hub-pub", file_hwm=0)
    with pytest.raises(ValueError, match="file_hwm"):
        ConnectorEndpoint("inproc://hwm-connector-in", "inproc://hwm-connector-pub", file_hwm=0)
    with pytest.raises(ValueError, match="file_hwm"):
        ProcessorEndpoint("inproc://hwm-processor-pub", "inproc://hwm-processor-in", file_hwm=0)


async def test_queued_file_is_dropped_after_unsubscribe() -> None:
    processor = ProcessorEndpoint(
        "inproc://queued-file-pub",
        "inproc://queued-file-in",
        file_sub_addr="inproc://queued-file-lane",
        file_hwm=1,
    )
    message = FileMessage(
        participant_id="alice",
        topic="image.response",
        pts_us=123,
        transfer_id="stream-queued",
        name="capture.png",
        mime_type="image/png",
        attributes={},
        data=b"png",
        participant_session_id="session-1",
    )
    delivered: list[FileMessage] = []

    async def on_file(message: FileMessage) -> None:
        delivered.append(message)

    processor.on_file(on_file)
    processor._participants.add("alice")
    processor._participant_sessions["alice"] = "session-1"
    processor.subscribe("alice", filter=Subscribe.FILE)
    processor._apply_file_session_event(ParticipantEvent(
        participant_id="alice",
        joined=True,
        pts_us=1,
        connector_id="connector-1",
        participant_session_id="session-1",
    ))
    processor._route_file(message)
    processor._route_file(replace(message, transfer_id="stream-full"))
    assert processor._file_queue.qsize() == 1
    processor.unsubscribe("alice")
    processor._route_file(replace(message, transfer_id="stream-unsubscribed"))
    assert processor._file_queue.qsize() == 1
    processor._running = True
    worker = asyncio.create_task(processor._run_file_callbacks())
    try:
        await asyncio.wait_for(processor._file_queue.join(), 1)
        assert delivered == []
        processor.subscribe("alice", filter=Subscribe.FILE)
        accepted = replace(message, transfer_id="stream-accepted")
        processor._route_file(accepted)
        await asyncio.wait_for(processor._file_queue.join(), 1)
        assert delivered == [accepted]
    finally:
        processor._running = False
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        processor.close()


def test_file_callback_unsubscriber_is_idempotent() -> None:
    processor = ProcessorEndpoint(
        "inproc://file-callback-pub",
        "inproc://file-callback-in",
    )

    async def callback(_message: FileMessage) -> None:
        pass

    try:
        unsubscribe = processor.on_file(callback)
        assert callback in processor._file_cbs
        unsubscribe()
        unsubscribe()
        assert callback not in processor._file_cbs
    finally:
        processor.close()


async def test_file_callback_failure_uses_fatal_callback_policy(monkeypatch) -> None:
    processor = ProcessorEndpoint(
        "inproc://file-callback-error-pub",
        "inproc://file-callback-error-in",
        file_hwm=1,
    )
    message = FileMessage(
        participant_id="alice",
        topic="image.response",
        pts_us=1,
        transfer_id="stream-error",
        name="capture.png",
        mime_type="image/png",
        attributes={},
        data=b"png",
        participant_session_id="session-1",
    )

    async def fail(_message: FileMessage) -> None:
        raise RuntimeError("callback failed")

    exit_process = Mock()
    monkeypatch.setattr("xr_ai_hub._processor.os._exit", exit_process)
    processor._participants.add("alice")
    processor._participant_sessions["alice"] = "session-1"
    processor._subscribed["alice"] = Subscribe.FILE
    processor.on_file(fail)
    processor._running = True
    worker = asyncio.create_task(processor._run_file_callbacks())
    processor._file_queue.put_nowait(message)
    try:
        await asyncio.wait_for(processor._file_queue.join(), 1)
        exit_process.assert_called_once_with(1)
    finally:
        processor._running = False
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        processor.close()


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


async def test_connector_file_admission_guards_and_session_fill() -> None:
    message = FileMessage(
        participant_id="alice",
        topic="image.response",
        pts_us=123,
        transfer_id="stream-guarded",
        name="capture.png",
        mime_type="image/png",
        attributes={},
        data=b"png",
    )
    connector = ConnectorEndpoint(
        "inproc://guarded-file-in",
        "inproc://guarded-file-pub",
    )
    try:
        assert not await connector.push_file(message)
    finally:
        connector.close()

    sent: list[bytes] = []

    class FileSocket:
        async def send(self, raw: bytes, *, flags: int) -> None:
            assert flags == zmq.NOBLOCK
            sent.append(raw)

    connector = ConnectorEndpoint(
        "inproc://session-file-in",
        "inproc://session-file-pub",
        file_push_addr="inproc://session-file-lane",
    )
    connector._file_push.close(linger=0)
    connector._file_push = FileSocket()
    try:
        assert not await connector.push_file(message)
        connector._participant_sessions["alice"] = "session-1"
        assert not await connector.push_file(
            replace(message, participant_session_id="stale-session"),
        )
        assert await connector.push_file(message)
        type_id, queued = decode(sent[0])
        assert type_id == MsgType.FILE_MESSAGE
        assert queued.participant_session_id == "session-1"
    finally:
        connector._file_push = None
        connector.close()


async def test_connector_reports_full_file_queue() -> None:
    class FullFileSocket:
        async def send(self, _raw: bytes, *, flags: int) -> None:
            raise zmq.Again()

    connector = ConnectorEndpoint(
        "inproc://full-file-in",
        "inproc://full-file-pub",
        file_push_addr="inproc://full-file-lane",
    )
    connector._file_push.close(linger=0)
    connector._file_push = FullFileSocket()
    connector._participant_sessions["alice"] = "session-1"
    try:
        assert not await connector.push_file(FileMessage(
            participant_id="alice",
            topic="image.response",
            pts_us=123,
            transfer_id="stream-full",
            name="capture.png",
            mime_type="image/png",
            attributes={},
            data=b"png",
            participant_session_id="session-1",
        ))
    finally:
        connector._file_push = None
        connector.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("incoming_file_max_bytes", 0),
        ("incoming_file_max_concurrent", True),
        ("incoming_file_max_concurrent_per_participant", 1.5),
        ("incoming_file_idle_timeout_s", float("inf")),
        ("incoming_file_total_timeout_s", 0),
        ("incoming_file_ipc_hwm", "bad"),
    ],
)
def test_file_config_rejects_invalid_limits(field, value) -> None:
    with pytest.raises(ValueError, match=field):
        LiveKitConnectorConfig(
            api_key="key",
            api_secret="secret",
            **{field: value},
        )


def test_file_config_rejects_inconsistent_limits() -> None:
    with pytest.raises(ValueError, match="idle_timeout"):
        LiveKitConnectorConfig(
            api_key="key",
            api_secret="secret",
            incoming_file_idle_timeout_s=2,
            incoming_file_total_timeout_s=1,
        )
    with pytest.raises(ValueError, match="per_participant"):
        LiveKitConnectorConfig(
            api_key="key",
            api_secret="secret",
            incoming_file_max_concurrent=1,
            incoming_file_max_concurrent_per_participant=2,
        )


def test_file_subscription_requires_file_address_before_socket_creation() -> None:
    with pytest.raises(ValueError, match="file_sub_addr"):
        ProcessorEndpoint(
            "inproc://missing-file-sub-pub",
            "inproc://missing-file-sub-in",
            filter=Subscribe.FILE,
        )


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


async def test_room_client_leave_cancels_only_departed_session_files() -> None:
    async def wait_forever() -> None:
        await asyncio.Future()

    prior = asyncio.create_task(wait_forever())
    current = asyncio.create_task(wait_forever())
    client = RoomClient.__new__(RoomClient)
    client._return_audio = {}
    client._file_tasks = {
        prior: ("alice", "session-1", object()),
        current: ("alice", "session-2", object()),
    }
    client._ep = SimpleNamespace(notify_participant_left=AsyncMock())

    try:
        await client._handle_left(
            SimpleNamespace(identity="alice"),
            "session-1",
        )

        assert prior.cancelled()
        assert not current.done()
        client._ep.notify_participant_left.assert_awaited_once()
    finally:
        current.cancel()
        await asyncio.gather(current, return_exceptions=True)


async def test_completed_file_routes_on_bounded_file_lane(monkeypatch, tmp_path) -> None:
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
    published: list[str] = []
    delivered = asyncio.Event()

    publish_file = hub._publish_file

    async def record_publish(message: FileMessage) -> None:
        published.append(message.transfer_id)
        await publish_file(message)

    monkeypatch.setattr(hub, "_publish_file", record_publish)

    async def on_file(message: FileMessage) -> None:
        received.append(message)
        delivered.set()

    processor.on_file(on_file)
    hub_task = asyncio.create_task(hub.run())
    processor_task = asyncio.create_task(processor.run())
    try:
        await connector.register()
        prior_session_id = "prior-session"
        await connector.notify_participant_joined(
            "alice",
            participant_session_id=prior_session_id,
        )
        await asyncio.wait_for(processor.wait_until_running(), 1)
        async with asyncio.timeout(1):
            while "alice" not in processor.connected_participants:
                await asyncio.sleep(0.001)
        await connector.notify_participant_left(
            "alice",
            participant_session_id=prior_session_id,
        )
        async with asyncio.timeout(1):
            while "alice" in processor.connected_participants:
                await asyncio.sleep(0.001)
        participant_session_id = "current-session"
        await connector.notify_participant_joined(
            "alice",
            participant_session_id=participant_session_id,
        )
        async with asyncio.timeout(1):
            while processor._participant_sessions.get("alice") != participant_session_id:
                await asyncio.sleep(0.001)
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
        stale_message = replace(message, participant_session_id=prior_session_id)
        await connector._file_push.send(encode(MsgType.FILE_MESSAGE, stale_message))
        assert await connector.push_file(message)
        await asyncio.wait_for(delivered.wait(), 1)
        await asyncio.sleep(0.05)

        assert received == [message]
        assert published == ["stream-1"]
        assert hub._file_orderer.pending_count == 0
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


async def test_slow_file_subscriber_does_not_block_healthy_subscriber(tmp_path) -> None:
    suffix = uuid.uuid4().hex[:8]
    pull = f"ipc://{tmp_path}/{suffix}-a"
    pub = f"ipc://{tmp_path}/{suffix}-b"
    file_pull = f"ipc://{tmp_path}/{suffix}-c"
    file_pub = f"ipc://{tmp_path}/{suffix}-d"
    hub = HubEndpoint(
        pull,
        pub,
        file_pull_addr=file_pull,
        file_pub_addr=file_pub,
        file_hwm=1,
    )
    connector = ConnectorEndpoint(
        pull,
        pub,
        file_push_addr=file_pull,
        file_hwm=1,
        num_slots=1,
        max_frame_bytes=1024,
    )
    healthy = ProcessorEndpoint(
        pub,
        pull,
        file_sub_addr=file_pub,
        file_hwm=1,
        filter=Subscribe.FILE,
    )
    slow = ProcessorEndpoint(
        pub,
        pull,
        file_sub_addr=file_pub,
        file_hwm=1,
        filter=Subscribe.FILE,
    )
    received: list[str] = []

    async def on_file(message: FileMessage) -> None:
        received.append(message.transfer_id)

    healthy.on_file(on_file)
    hub_task = asyncio.create_task(hub.run())
    healthy_task = asyncio.create_task(healthy.run())
    try:
        await connector.register()
        await connector.notify_participant_joined(
            "alice",
            participant_session_id="session-1",
        )
        await asyncio.wait_for(healthy.wait_until_running(), 1)
        async with asyncio.timeout(1):
            while "alice" not in healthy.connected_participants:
                await asyncio.sleep(0.001)
        assert await healthy.wait_for_subscriptions(timeout=1)
        await asyncio.sleep(0.05)
        assert slow._file_sub is not None

        for index in range(3):
            message = FileMessage(
                participant_id="alice",
                topic="image.response",
                pts_us=index,
                transfer_id=f"stream-{index}",
                name=f"capture-{index}.png",
                mime_type="image/png",
                attributes={},
                data=b"png",
                participant_session_id="session-1",
            )
            async with asyncio.timeout(1):
                while not await connector.push_file(message):
                    await asyncio.sleep(0.001)
            if index == 0:
                assert await slow._file_sub.poll(timeout=1000) & zmq.POLLIN
            async with asyncio.timeout(1):
                while len(received) <= index:
                    await asyncio.sleep(0.001)

        assert received == ["stream-0", "stream-1", "stream-2"]
    finally:
        healthy.stop()
        healthy.close()
        slow.close()
        connector.stop()
        connector.close()
        hub.stop()
        hub.close()
        hub_task.cancel()
        healthy_task.cancel()
        await asyncio.gather(hub_task, healthy_task, return_exceptions=True)


@pytest.mark.parametrize("prior_session", [None, "session-0"])
async def test_hub_buffers_file_that_arrives_before_participant_join(
    monkeypatch,
    prior_session,
    tmp_path,
) -> None:
    suffix = uuid.uuid4().hex[:8]
    hub = HubEndpoint(
        f"ipc://{tmp_path}/{suffix}-a",
        f"ipc://{tmp_path}/{suffix}-b",
        file_pull_addr=f"ipc://{tmp_path}/{suffix}-c",
        file_pub_addr=f"ipc://{tmp_path}/{suffix}-d",
    )
    message = FileMessage(
        participant_id="alice",
        topic="image.response",
        pts_us=123,
        transfer_id="stream-prejoin-hub",
        name="capture.png",
        mime_type="image/png",
        attributes={},
        data=b"png",
        participant_session_id="session-1",
    )
    published: list[FileMessage] = []

    async def record_file(msg: FileMessage) -> None:
        published.append(msg)

    monkeypatch.setattr(hub, "_publish_file", record_file)
    try:
        await hub._route_file(message)
        assert published == []

        if prior_session is not None:
            prior_join = ParticipantEvent(
                participant_id="alice",
                joined=True,
                pts_us=122,
                connector_id="connector-1",
                participant_session_id=prior_session,
            )
            await hub._dispatch(MsgType.PARTICIPANT_EVENT, prior_join)
            await hub._apply_file_session_event(prior_join)
            assert published == []

        if prior_session is not None:
            prior_leave = ParticipantEvent(
                participant_id="alice",
                joined=False,
                pts_us=123,
                connector_id="connector-1",
                participant_session_id=prior_session,
            )
            await hub._dispatch(MsgType.PARTICIPANT_EVENT, prior_leave)
            await hub._apply_file_session_event(prior_leave)

        current_join = ParticipantEvent(
            participant_id="alice",
            joined=True,
            pts_us=124,
            connector_id="connector-1",
            participant_session_id="session-1",
        )
        await hub._dispatch(MsgType.PARTICIPANT_EVENT, current_join)
        await hub._apply_file_session_event(current_join)

        assert published == [message]
    finally:
        hub.close()


async def test_hub_drops_buffered_file_when_newer_session_is_already_active() -> None:
    hub = HubEndpoint.__new__(HubEndpoint)
    hub._file_orderer = FileSessionOrderer(1)
    hub._file_session_events = asyncio.Queue()
    hub._participant_sessions = {"alice": "session-b"}
    hub._file_pub = SimpleNamespace(send_multipart=AsyncMock())
    message = FileMessage(
        participant_id="alice",
        topic="image.response",
        pts_us=123,
        transfer_id="stream-session-a",
        name="capture.png",
        mime_type="image/png",
        attributes={},
        data=b"png",
        participant_session_id="session-a",
    )
    lifecycle = None
    try:
        await hub._route_file(message)
        hub._file_pub.send_multipart.assert_not_awaited()

        session_a_joined = ParticipantEvent(
            participant_id="alice",
            joined=True,
            pts_us=124,
            connector_id="connector-1",
            participant_session_id="session-a",
        )
        session_a_left = replace(session_a_joined, joined=False, pts_us=125)
        session_b_joined = replace(
            session_a_joined,
            pts_us=126,
            participant_session_id="session-b",
        )
        for event in (session_a_joined, session_a_left, session_b_joined):
            hub._file_session_events.put_nowait(event)

        lifecycle = asyncio.create_task(hub._file_session_events.get())
        lifecycle = await hub._drain_file_session_events(lifecycle)

        assert hub._participant_sessions["alice"] == "session-b"
        hub._file_pub.send_multipart.assert_not_awaited()
    finally:
        if lifecycle is not None:
            lifecycle.cancel()
            await asyncio.gather(lifecycle, return_exceptions=True)


@pytest.mark.parametrize("prior_session", [None, "session-0"])
async def test_processor_buffers_file_that_arrives_before_participant_join(
    prior_session,
    tmp_path,
) -> None:
    suffix = uuid.uuid4().hex[:8]
    processor = ProcessorEndpoint(
        f"ipc://{tmp_path}/{suffix}-a",
        f"ipc://{tmp_path}/{suffix}-b",
        file_sub_addr=f"ipc://{tmp_path}/{suffix}-c",
        filter=Subscribe.DEFAULT | Subscribe.FILE,
    )
    message = FileMessage(
        participant_id="alice",
        topic="image.response",
        pts_us=123,
        transfer_id="stream-prejoin-processor",
        name="capture.png",
        mime_type="image/png",
        attributes={},
        data=b"png",
        participant_session_id="session-1",
    )
    try:
        processor._route_file(message)
        assert processor._file_queue.empty()

        if prior_session is not None:
            prior_join = ParticipantEvent(
                participant_id="alice",
                joined=True,
                pts_us=122,
                connector_id="connector-1",
                participant_session_id=prior_session,
            )
            await processor._dispatch(MsgType.PARTICIPANT_EVENT, prior_join)
            processor._apply_file_session_event(prior_join)
            assert processor._file_queue.empty()

        if prior_session is not None:
            prior_leave = ParticipantEvent(
                participant_id="alice",
                joined=False,
                pts_us=123,
                connector_id="connector-1",
                participant_session_id=prior_session,
            )
            await processor._dispatch(MsgType.PARTICIPANT_EVENT, prior_leave)
            processor._apply_file_session_event(prior_leave)

        current_join = ParticipantEvent(
            participant_id="alice",
            joined=True,
            pts_us=124,
            connector_id="connector-1",
            participant_session_id="session-1",
        )
        await processor._dispatch(MsgType.PARTICIPANT_EVENT, current_join)
        processor._apply_file_session_event(current_join)

        assert processor._file_queue.get_nowait() == message
        processor._file_queue.task_done()
    finally:
        processor.close()
