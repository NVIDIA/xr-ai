# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import array
import uuid

import pytest
from xr_ai_agent._shm import _GH_SIZE, _SH, _SH_SIZE, ShmRingBuffer
from xr_ai_agent._types import FrameSignal, PixelFormat


@pytest.fixture
def ring():
    name = f"xr_test_{uuid.uuid4().hex[:10]}"
    rb = ShmRingBuffer(name=name, num_slots=2, max_frame_bytes=8, create=True)
    try:
        yield rb
    finally:
        rb.close()
        rb.unlink()


def test_write_frame_rejects_oversized_frame_before_claiming_slot(ring):
    with pytest.raises(ValueError, match=r"bytes=9, max_frame_bytes=8"):
        ring.write_frame(
            b"A" * 9,
            width=3,
            height=3,
            fmt=PixelFormat.RGBA,
            pts_us=1,
            seq=1,
        )

    assert ring._write_pos == 0
    assert ring.write_frame(b"B" * 8, 2, 2, PixelFormat.RGBA, 2, 2) == 0


def test_write_frame_rejects_oversized_frame_without_corrupting_next_slot_header(ring):
    slot1_hdr_off = _GH_SIZE + ring._slot_stride
    before = bytes(ring._buf[slot1_hdr_off : slot1_hdr_off + _SH_SIZE])

    with pytest.raises(ValueError, match=r"width=4, height=4, format=RGBA"):
        ring.write_frame(
            b"A" * 12,
            width=4,
            height=4,
            fmt=PixelFormat.RGBA,
            pts_us=1,
            seq=1,
        )

    after = bytes(ring._buf[slot1_hdr_off : slot1_hdr_off + _SH_SIZE])
    assert after == before
    assert _SH.unpack_from(ring._buf, slot1_hdr_off)[0] == _SH.unpack(before)[0]


def test_write_frame_checks_memoryview_byte_size(ring):
    payload = memoryview(array.array("H", range(5)))

    with pytest.raises(ValueError, match=r"bytes=10, max_frame_bytes=8"):
        ring.write_frame(
            payload,
            width=5,
            height=1,
            fmt=PixelFormat.RGB24,
            pts_us=1,
            seq=1,
        )

    assert ring._write_pos == 0


def test_read_slot_rejects_signal_larger_than_slot(ring):
    slot = ring.write_frame(b"A" * 8, 2, 2, PixelFormat.RGBA, 1, 1)
    signal = FrameSignal(
        slot=slot,
        seq=1,
        pts_us=1,
        width=2,
        height=2,
        fmt=PixelFormat.RGBA,
        data_sz=9,
    )

    with pytest.raises(ValueError, match=r"bytes=9, max_frame_bytes=8"):
        ring.read_slot(signal)
