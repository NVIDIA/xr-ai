# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit test for _ReturnAudioPipe — ensures flood + flush drops everything fast."""
from __future__ import annotations

import asyncio

import pytest

from xr_media_hub.transport.livekit._room_client import _ReturnAudioPipe

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

    def __init__(self) -> None:
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
    src  = _FakeSource()
    pipe = _ReturnAudioPipe(src, participant_id="alice", max_buffer_s=0.03)
    frames = [_FakeFrame(f"frame_{i}") for i in range(5)]
    try:
        for f in frames:
            pipe.push(f)

        assert pipe.queued_frames == 3
        assert pipe.queued_duration_s == pytest.approx(0.03)
        assert pipe.dropped_frames == 2
        assert pipe.dropped_duration_s == pytest.approx(0.02)

        await asyncio.sleep(0.06)
        assert src.captured == frames[2:]
    finally:
        await pipe.close()


async def test_return_audio_buffer_limit_must_be_positive():
    with pytest.raises(ValueError, match="return_audio_max_buffer_s must be > 0"):
        _ReturnAudioPipe(_FakeSource(), max_buffer_s=0)
