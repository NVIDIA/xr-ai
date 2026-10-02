# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera lifecycle over real hub IPC, independent of client data messages."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from xr_ai_hub import MsgType, VideoTrackEvent, decode, encode


def test_video_event_codec():
    event = VideoTrackEvent("alice", "camera", True, 123, "connection")
    assert decode(encode(MsgType.VIDEO_TRACK_EVENT, event)) == (MsgType.VIDEO_TRACK_EVENT, event)


async def test_live_events_roster_replay_and_stale_session_isolation(
    hub,
    make_connector,
    make_processor,
    settle,
):
    connector = make_connector()
    await connector.register()
    await connector.notify_participant_joined("alice", participant_session_id="first")
    await connector.notify_video_track(VideoTrackEvent("alice", "cam", True, 1, "first"))
    await settle()
    # A worker starting after the camera catches up through the roster.
    events = []

    async def collect(event):
        events.append(event)

    processor = make_processor()
    processor.on_video_track(collect)
    async with asyncio.timeout(2):
        while not events:
            await asyncio.sleep(0.01)
    assert events[0] == VideoTrackEvent("alice", "cam", True, 1, "first")
    await connector.notify_video_track(VideoTrackEvent("alice", "cam", False, 2, "first"))
    await settle()
    assert events[-1].active is False
    events.clear()
    await processor.request_roster()
    await settle()
    assert events == []
    await connector.notify_participant_left("alice", participant_session_id="first")
    await connector.notify_participant_joined("alice", participant_session_id="second")
    await settle()
    await connector.notify_video_track(VideoTrackEvent("alice", "cam", True, 3, "first"))
    await settle()
    assert events == []
    await connector.notify_video_track(VideoTrackEvent("alice", "cam2", True, 4, "second"))
    await settle()
    assert len(events) == 1
    await connector.notify_participant_left("alice", participant_session_id="second")
    await settle()
    assert not hub._video_tracks


async def test_unsubscribed_participant_video_is_not_delivered(
    hub,
    make_connector,
    make_processor,
    settle,
):
    processor = make_processor(auto_subscribe=False)
    callback = AsyncMock()
    processor.on_video_track(callback)
    connector = make_connector()
    await connector.register()
    await settle()
    await connector.notify_participant_joined("alice", participant_session_id="session")
    await connector.notify_video_track(VideoTrackEvent("alice", "cam", True, 1, "session"))
    await settle()
    callback.assert_not_awaited()


@pytest.mark.parametrize(
    "camera,muted,expected",
    [
        (True, False, [True, False]),
        (False, False, []),
        (True, True, [False]),
    ],
)
async def test_room_reports_first_live_frame_and_stream_end(monkeypatch, camera, muted, expected):
    from device_io_hub.transport.livekit import _room_client as module

    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if frames:
                return frames.pop()
            raise StopAsyncIteration

        aclose = AsyncMock()

    frames = [SimpleNamespace(frame=SimpleNamespace(data=b"image", width=2, height=2))]
    stream = Stream()
    monkeypatch.setattr(module.rtc, "VideoStream", lambda *a, **kw: stream)
    client = module.RoomClient.__new__(module.RoomClient)
    client._participant_sessions = {"alice": "session"}
    client._muted_camera_tracks = {"camera"} if muted else set()
    client._ep = SimpleNamespace(push_frame=AsyncMock(), notify_video_track=AsyncMock())
    await client._stream_video(object(), "alice", "camera", camera=camera)
    events = [call.args[0] for call in client._ep.notify_video_track.await_args_list]
    assert [event.active for event in events] == expected
    assert all(event.participant_session_id == "session" for event in events)
    stream.aclose.assert_awaited_once()


async def test_muting_camera_emits_boundary_without_disconnect(monkeypatch):
    from device_io_hub.transport.livekit import _room_client as module
    from device_io_hub.transport.livekit.config import LiveKitConnectorConfig

    handlers = {}

    class Room:
        def register_byte_stream_handler(self, *args):
            pass

        def on(self, name):
            def register(fn):
                handlers[name] = fn
                return fn

            return register

    monkeypatch.setattr(module.rtc, "Room", Room)
    endpoint = SimpleNamespace(notify_video_track=AsyncMock())
    client = module.RoomClient(LiveKitConnectorConfig(api_key="test", api_secret="test"), endpoint)
    client._participant_sessions = {"alice": "session"}
    publication = SimpleNamespace(
        kind=module.rtc.TrackKind.KIND_VIDEO, source=module.rtc.TrackSource.SOURCE_CAMERA, track=object(), sid="cam"
    )
    participant = SimpleNamespace(identity="alice")
    handlers["track_muted"](publication, participant)
    handlers["track_unmuted"](publication, participant)
    await asyncio.gather(*client._pending_tasks)
    assert [call.args[0].active for call in endpoint.notify_video_track.await_args_list] == [False, True]
    # Unsubscribe may clear publication.track before the streaming task ever
    # starts, so it must still send an explicit inactive boundary.
    publication.track = None
    handlers["track_unsubscribed"](SimpleNamespace(sid="cam"), publication, participant)
    await asyncio.gather(*client._pending_tasks)
    assert endpoint.notify_video_track.await_args_list[-1].args[0].active is False
