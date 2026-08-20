# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared-memory ring-buffer lifecycle regressions."""
from __future__ import annotations

import array
import uuid
from dataclasses import replace

import pytest
from xr_ai_hub import FrameSignal, PixelFormat, ShmRingBuffer
from xr_ai_hub._shm import _GH_SIZE, _SH, _SH_SIZE


@pytest.fixture
def ring():
    name = f"xr_test_{uuid.uuid4().hex[:12]}"
    buffer = ShmRingBuffer(
        name=name,
        num_slots=2,
        max_frame_bytes=8,
        create=True,
    )
    try:
        yield buffer
    finally:
        buffer.close()
        buffer.unlink()


def _write_test_signal(ring) -> FrameSignal:
    slot = ring.write_frame(
        b"ABCD",
        width=2,
        height=2,
        fmt=PixelFormat.RGBA,
        pts_us=10,
        seq=7,
    )
    return FrameSignal(
        slot=slot,
        seq=7,
        pts_us=10,
        width=2,
        height=2,
        fmt=PixelFormat.RGBA,
        data_sz=4,
    )


def test_unlink_tolerates_repeated_same_process_cleanup():
    with ShmRingBuffer(
        name=f"xr_test_{uuid.uuid4().hex[:12]}",
        num_slots=1,
        max_frame_bytes=64,
        create=True,
    ) as ring:
        ring.unlink()
        ring.unlink()


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


def test_write_frame_rejects_oversized_frame_without_corrupting_header(ring):
    slot1_header_offset = _GH_SIZE + ring._slot_stride
    before = bytes(
        ring._buf[slot1_header_offset : slot1_header_offset + _SH_SIZE]
    )

    with pytest.raises(ValueError, match=r"width=4, height=4, format=RGBA"):
        ring.write_frame(
            b"A" * 12,
            width=4,
            height=4,
            fmt=PixelFormat.RGBA,
            pts_us=1,
            seq=1,
        )

    after = bytes(
        ring._buf[slot1_header_offset : slot1_header_offset + _SH_SIZE]
    )
    assert after == before
    assert _SH.unpack_from(ring._buf, slot1_header_offset)[0] == _SH.unpack(before)[0]


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


def test_write_frame_rejects_noncontiguous_memoryview(ring):
    payload = memoryview(bytearray(range(8)))[::2]

    with pytest.raises(ValueError, match="C-contiguous"):
        ring.write_frame(
            payload,
            width=2,
            height=2,
            fmt=PixelFormat.RGBA,
            pts_us=1,
            seq=1,
        )

    assert ring._write_pos == 0


@pytest.mark.parametrize("data_size", [-1, 9])
def test_read_slot_rejects_signal_outside_slot_capacity(ring, data_size):
    slot = ring.write_frame(b"A" * 8, 2, 2, PixelFormat.RGBA, 1, 1)
    signal = FrameSignal(
        slot=slot,
        seq=1,
        pts_us=1,
        width=2,
        height=2,
        fmt=PixelFormat.RGBA,
        data_sz=data_size,
    )

    with pytest.raises(ValueError, match=rf"bytes={data_size}, max_frame_bytes=8"):
        ring.read_slot(signal)


def test_read_slot_rejects_in_capacity_size_mismatch(ring):
    signal = replace(_write_test_signal(ring), data_sz=8)

    with pytest.raises(ValueError, match="does not match shared-memory slot header"):
        ring.read_slot(signal)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seq", 8),
        ("pts_us", 11),
        ("width", 3),
        ("height", 3),
        ("fmt", PixelFormat.RGB24),
    ],
)
def test_read_slot_rejects_signal_metadata_mismatch(ring, field, value):
    signal = replace(_write_test_signal(ring), **{field: value})

    with pytest.raises(ValueError, match="does not match shared-memory slot header"):
        ring.read_slot(signal)


@pytest.mark.parametrize("slot", [-1, 2])
def test_read_slot_rejects_out_of_range_slot_index(ring, slot):
    signal = replace(_write_test_signal(ring), slot=slot)

    with pytest.raises(ValueError, match="invalid shared-memory slot index"):
        ring.read_slot(signal)


def test_read_slot_rejects_invalid_header_magic(ring):
    signal = _write_test_signal(ring)
    header_offset = _GH_SIZE + signal.slot * ring._slot_stride
    header = list(_SH.unpack_from(ring._buf, header_offset))
    header[0] = 0
    _SH.pack_into(ring._buf, header_offset, *header)

    with pytest.raises(RuntimeError, match="invalid shared-memory header magic"):
        ring.read_slot(signal)


def test_read_slot_rejects_header_size_outside_slot_capacity(ring):
    signal = _write_test_signal(ring)
    header_offset = _GH_SIZE + signal.slot * ring._slot_stride
    header = list(_SH.unpack_from(ring._buf, header_offset))
    header[8] = 9
    _SH.pack_into(ring._buf, header_offset, *header)

    with pytest.raises(ValueError, match=r"bytes=9, max_frame_bytes=8"):
        ring.read_slot(signal)


def test_read_slot_rejects_non_ready_slot(ring):
    signal = _write_test_signal(ring)
    ring.release_slot(signal.slot)

    with pytest.raises(RuntimeError, match="not READY"):
        ring.read_slot(signal)
