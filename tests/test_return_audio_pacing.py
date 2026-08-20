# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit test for _ReturnAudioPipe — ensures flood + flush drops everything fast."""
from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

import pytest

from device_io_hub.transport.livekit import _room_client as room_client_module
from device_io_hub.transport.livekit._room_client import (
    RoomClient,
    _ReturnAudioPipe,
)
from device_io_hub.transport.livekit.config import LiveKitConnectorConfig

pytestmark = pytest.mark.asyncio


class _FakeFrame:
    def __init__(
        self,
        name: str,
        *,
        samples_per_channel: int = 480,
        sample_rate: int = 48_000,
    ) -> None:
        self.name = name
        self.samples_per_channel = samples_per_channel
        self.sample_rate = sample_rate

    def __repr__(self) -> str:
        return self.name


class _FakeSource:
    """Mock AudioSource that paces capture_frame at audio rate (10 ms/frame)."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.captured: list[object] = []
        self.cleared: int = 0

    async def capture_frame(self, frame) -> None:
        await asyncio.sleep(0.01)
        self.captured.append(frame)

    def clear_queue(self) -> None:
        self.cleared += 1


async def test_flood_then_flush_drops_unflushed_frames():
    src  = _FakeSource()
    pipe = _ReturnAudioPipe(src)
    try:
        # Flood 50 frames as fast as possible (no awaits between push calls).
        for i in range(50):
            pipe.push(_FakeFrame(f"frame_{i}"))

        # A few have been captured by now.
        await asyncio.sleep(0.025)
        captured_before_flush = len(src.captured)
        assert 1 <= captured_before_flush < 50, (
            f"expected partial drain before flush, got {captured_before_flush}"
        )

        pipe.flush()
        # capture_frame already in flight may finish, but no new frames picked up.
        await asyncio.sleep(0.1)
        captured_after_flush = len(src.captured)

        # After flush, queue should be empty and no further frames captured.
        assert captured_after_flush <= captured_before_flush + 1, (
            f"expected flush to halt drain, before={captured_before_flush} "
            f"after={captured_after_flush}"
        )
        assert pipe.queued_frames == 0
        assert pipe.queued_duration_s == 0.0
        assert src.cleared == 1
    finally:
        await pipe.close()


async def test_normal_flow_drains_all_frames():
    src  = _FakeSource()
    pipe = _ReturnAudioPipe(src)
    frames = [_FakeFrame(f"frame_{i}") for i in range(5)]
    for f in frames:
        pipe.push(f)
    # Wait for full drain (5 frames * 10 ms + slack).
    await asyncio.sleep(0.15)
    assert src.captured == frames
    await pipe.close()


async def test_overflow_drops_oldest_frames_by_audio_duration():
    src = _FakeSource()
    pipe = _ReturnAudioPipe(src, participant_id="alice", max_buffer_s=0.03)
    frames = [_FakeFrame(f"frame_{i}") for i in range(5)]
    try:
        for frame in frames:
            pipe.push(frame)

        assert pipe.queued_frames == 3
        assert pipe.queued_duration_s == pytest.approx(0.03)
        assert pipe.dropped_frames == 2
        assert pipe.dropped_duration_s == pytest.approx(0.02)

        await asyncio.sleep(0.06)
        assert src.captured == frames[2:]
    finally:
        await pipe.close()


async def test_frame_larger_than_limit_is_dropped():
    src = _FakeSource()
    pipe = _ReturnAudioPipe(src, max_buffer_s=0.005)
    try:
        pipe.push(_FakeFrame("oversized"))

        assert pipe.queued_frames == 0
        assert pipe.dropped_frames == 1
        assert pipe.dropped_duration_s == pytest.approx(0.01)
    finally:
        await pipe.close()


@pytest.mark.parametrize("limit", [0, -1, math.inf, math.nan, "invalid", True])
async def test_return_audio_buffer_limit_must_be_positive_and_finite(limit):
    with pytest.raises(ValueError, match="return_audio_max_buffer_s"):
        _ReturnAudioPipe(_FakeSource(), max_buffer_s=limit)


async def test_room_client_applies_buffer_limit_per_participant(monkeypatch):
    class _FakeLocalAudioTrack:
        @staticmethod
        def create_audio_track(name, source):
            return SimpleNamespace(name=name, source=source)

    class _FakeLocalParticipant:
        async def publish_track(self, track):
            return SimpleNamespace(sid=f"pub-{track.name}")

    monkeypatch.setattr(
        room_client_module,
        "rtc",
        SimpleNamespace(
            AudioSource=_FakeSource,
            LocalAudioTrack=_FakeLocalAudioTrack,
        ),
    )
    client = RoomClient.__new__(RoomClient)
    client._cfg = LiveKitConnectorConfig(
        api_key="test-key",
        api_secret="test-secret",
        return_audio_max_buffer_s=0.02,
    )
    client._room = SimpleNamespace(local_participant=_FakeLocalParticipant())

    alice_source, _alice_pub, alice_pipe = await client._publish_return_track(
        "alice", 48_000, 1
    )
    bob_source, _bob_pub, bob_pipe = await client._publish_return_track(
        "bob", 48_000, 1
    )
    try:
        for i in range(3):
            alice_pipe.push(_FakeFrame(f"alice-{i}"))
        bob_pipe.push(_FakeFrame("bob-0"))

        assert alice_pipe.queued_frames == 2
        assert alice_pipe.dropped_frames == 1
        assert bob_pipe.queued_frames == 1
        assert bob_pipe.dropped_frames == 0
        assert alice_source is not bob_source
    finally:
        await asyncio.gather(alice_pipe.close(), bob_pipe.close())
