"""Unit test for _ReturnAudioPipe — ensures flood + flush drops everything fast."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from xr_media_hub.transport.livekit._room_client import _ReturnAudioPipe

pytestmark = pytest.mark.asyncio


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
            pipe.push(MagicMock(name=f"frame_{i}"))

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
        assert src.cleared == 1
    finally:
        await pipe.close()


async def test_normal_flow_drains_all_frames():
    src  = _FakeSource()
    pipe = _ReturnAudioPipe(src)
    frames = [MagicMock(name=f"frame_{i}") for i in range(5)]
    for f in frames:
        pipe.push(f)
    # Wait for full drain (5 frames * 10 ms + slack).
    await asyncio.sleep(0.15)
    assert src.captured == frames
    await pipe.close()
