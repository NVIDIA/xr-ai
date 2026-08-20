# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fixed-slot shared-memory ring buffer for raw video frames.

Memory layout
─────────────
  [GlobalHeader  24 B ]
  [SlotHeader    64 B | frame_data  max_frame_bytes] × num_slots

GlobalHeader  struct "=IIQQ"
  magic          u32   0xC0FFEE01
  num_slots      u32
  max_frame_bytes u64
  slot_stride    u64   = SLOT_HDR_SIZE + max_frame_bytes

SlotHeader  struct "=IBBHQqIII28x"
  magic    u32   0xF4A3E501
  state    u8    0=FREE  1=WRITING  2=READY     ← byte offset 4
  fmt      u8    PixelFormat
  _pad     u16
  seq      u64
  pts_us   i64   (signed — allows negative PTS)
  width    u32
  height   u32
  data_sz  u32
  [28 B padding to reach 64 B]

Single-producer / single-consumer. The ZMQ signal carries the slot index so the
consumer never polls shared memory — it only reads after being signalled.
"""
from __future__ import annotations

import struct
from multiprocessing.shared_memory import SharedMemory
from typing import NamedTuple

from ._types import FrameSignal, PixelFormat

# ── struct definitions ────────────────────────────────────────────────────────

_GH = struct.Struct("=IIQQ")   # GlobalHeader — 24 bytes
_SH = struct.Struct("=IBBHQqIII28x")  # SlotHeader — 64 bytes

_GH_SIZE = _GH.size   # 24
_SH_SIZE = _SH.size   # 64

_MAGIC_GLOBAL = 0xC0FFEE01
_MAGIC_SLOT   = 0xF4A3E501

_STATE_FREE    = 0
_STATE_WRITING = 1
_STATE_READY   = 2

# Byte offset of the state field within a SlotHeader (after the 4-byte magic).
_STATE_OFFSET = 4


def _frame_nbytes(data: bytes | memoryview) -> int:
    return data.nbytes if isinstance(data, memoryview) else len(data)


def _frame_size_error(
    width: int,
    height: int,
    fmt: PixelFormat,
    size: int,
    capacity: int,
) -> str:
    fmt_name = getattr(fmt, "name", str(fmt))
    return (
        "invalid shared-memory frame size "
        f"(width={width}, height={height}, format={fmt_name}, "
        f"bytes={size}, max_frame_bytes={capacity})"
    )


class SlotView(NamedTuple):
    """Zero-copy view into one ring-buffer slot's pixel data."""
    data:   memoryview
    """Pixel bytes backed directly by the shared-memory segment."""

    signal: FrameSignal
    """Metadata identifying and describing the occupied slot."""


class ShmRingBuffer:
    """
    Shared-memory ring buffer for raw video frames.

    Hub creates the buffer (create=True). Connector opens it (create=False) and
    reads num_slots / max_frame_bytes from the global header automatically.

    The caller that uses read_slot() MUST call release_slot() before the next
    write_frame() for that slot can succeed. Both operations are O(1).

    Parameters
    ----------
    name :
        Operating-system name of the shared-memory segment.
    num_slots :
        Number of fixed-capacity slots to allocate when ``create`` is true.
    max_frame_bytes :
        Maximum pixel payload per slot when ``create`` is true.
    create :
        Create and initialize the segment instead of attaching to an existing
        one. The creator is responsible for eventually calling :meth:`unlink`.
    """

    def __init__(
        self,
        name:            str,
        num_slots:       int       = 0,
        max_frame_bytes: int       = 0,
        create:          bool      = False,
    ) -> None:
        if create:
            slot_stride = _SH_SIZE + max_frame_bytes
            total       = _GH_SIZE + num_slots * slot_stride
            self._shm   = SharedMemory(name=name, create=True, size=total)
            buffer = self._shm.buf
            assert buffer is not None
            _GH.pack_into(buffer, 0, _MAGIC_GLOBAL, num_slots, max_frame_bytes, slot_stride)
            for i in range(num_slots):
                off = _GH_SIZE + i * slot_stride
                _SH.pack_into(buffer, off, _MAGIC_SLOT, _STATE_FREE, 0, 0, 0, 0, 0, 0, 0)
        else:
            self._shm                              = SharedMemory(name=name, create=False)
            buffer = self._shm.buf
            assert buffer is not None
            _, num_slots, max_frame_bytes, slot_stride = _GH.unpack_from(buffer, 0)

        self._buf            = buffer
        self._num_slots      = num_slots
        self._max_frame_bytes = max_frame_bytes
        self._slot_stride    = slot_stride
        self._write_pos      = 0  # local to producer; never shared

    # ── producer ──────────────────────────────────────────────────────────────

    def write_frame(
        self,
        data:    bytes | memoryview,
        width:   int,
        height:  int,
        fmt:     PixelFormat,
        pts_us:  int,
        seq:     int,
    ) -> int:
        """Write a frame into the next free slot and return its index.

        Raises
        ------
        RuntimeError
            If every slot is occupied. This is the producer's back-pressure
            signal; consumers release occupied slots with :meth:`release_slot`.
        ValueError
            If the frame is larger than the configured slot capacity.
        """
        size = _frame_nbytes(data)
        if size > self._max_frame_bytes:
            raise ValueError(
                _frame_size_error(
                    width, height, fmt, size, self._max_frame_bytes
                )
            )

        if isinstance(data, memoryview):
            try:
                payload = data.cast("B")
            except TypeError as exc:
                raise ValueError("frame data must be C-contiguous") from exc
        else:
            payload = data
        slot    = self._claim_slot()
        hdr_off = _GH_SIZE + slot * self._slot_stride
        dat_off = hdr_off + _SH_SIZE

        # Mark WRITING so consumer won't touch this slot.
        _SH.pack_into(self._buf, hdr_off, _MAGIC_SLOT, _STATE_WRITING, int(fmt), 0, seq, pts_us, width, height, 0)
        self._buf[dat_off : dat_off + size] = payload
        # Mark READY — consumer may read after receiving the ZMQ signal.
        _SH.pack_into(self._buf, hdr_off, _MAGIC_SLOT, _STATE_READY,   int(fmt), 0, seq, pts_us, width, height, size)

        return slot

    # ── consumer ──────────────────────────────────────────────────────────────

    def read_slot(self, signal: FrameSignal) -> SlotView:
        """Return a zero-copy view of a ready slot's pixel data.

        The view remains valid until :meth:`release_slot` is called for the
        slot and must not be retained afterward.

        Raises
        ------
        RuntimeError
            If the indicated slot header is invalid or not ready for consumption.
        ValueError
            If the signal identifies an invalid slot or disagrees with the
            canonical metadata stored in the slot header.
        """
        slot = signal.slot
        if (
            not isinstance(slot, int)
            or isinstance(slot, bool)
            or not 0 <= slot < self._num_slots
        ):
            raise ValueError(
                f"invalid shared-memory slot index {slot!r} "
                f"(num_slots={self._num_slots})"
            )

        hdr_off = _GH_SIZE + slot * self._slot_stride
        (
            magic,
            state,
            header_fmt,
            _pad,
            header_seq,
            header_pts_us,
            header_width,
            header_height,
            header_data_sz,
        ) = _SH.unpack_from(self._buf, hdr_off)
        if magic != _MAGIC_SLOT:
            raise RuntimeError(
                f"slot {slot} has invalid shared-memory header magic "
                f"0x{magic:08x}"
            )
        if state != _STATE_READY:
            raise RuntimeError(f"slot {slot} not READY (state={state})")
        if not 0 <= header_data_sz <= self._max_frame_bytes:
            raise ValueError(
                _frame_size_error(
                    header_width,
                    header_height,
                    header_fmt,
                    header_data_sz,
                    self._max_frame_bytes,
                )
            )
        if not 0 <= signal.data_sz <= self._max_frame_bytes:
            raise ValueError(
                _frame_size_error(
                    signal.width,
                    signal.height,
                    signal.fmt,
                    signal.data_sz,
                    self._max_frame_bytes,
                )
            )

        signal_metadata = (
            signal.seq,
            signal.pts_us,
            signal.width,
            signal.height,
            signal.fmt,
            signal.data_sz,
        )
        header_metadata = (
            header_seq,
            header_pts_us,
            header_width,
            header_height,
            header_fmt,
            header_data_sz,
        )
        if signal_metadata != header_metadata:
            raise ValueError(
                "frame signal does not match shared-memory slot header "
                f"(slot={slot}, signal={signal_metadata!r}, "
                f"header={header_metadata!r})"
            )

        dat_off = hdr_off + _SH_SIZE
        return SlotView(
            data=self._buf[dat_off : dat_off + header_data_sz],
            signal=signal,
        )

    def release_slot(self, slot: int) -> None:
        """Mark a consumed slot as free so the producer can reuse it."""
        if (
            not isinstance(slot, int)
            or isinstance(slot, bool)
            or not 0 <= slot < self._num_slots
        ):
            raise ValueError(
                f"invalid shared-memory slot index {slot!r} "
                f"(num_slots={self._num_slots})"
            )
        hdr_off = _GH_SIZE + slot * self._slot_stride
        hdr     = _SH.unpack_from(self._buf, hdr_off)
        if hdr[0] != _MAGIC_SLOT:
            raise RuntimeError(
                f"slot {slot} has invalid shared-memory header magic "
                f"0x{hdr[0]:08x}"
            )
        if hdr[1] != _STATE_READY:
            raise RuntimeError(f"slot {slot} not READY (state={hdr[1]})")
        _SH.pack_into(self._buf, hdr_off, _MAGIC_SLOT, _STATE_FREE, hdr[2], 0, hdr[4], hdr[5], hdr[6], hdr[7], 0)

    # ── internal ──────────────────────────────────────────────────────────────

    def _claim_slot(self) -> int:
        for _ in range(self._num_slots):
            slot    = self._write_pos % self._num_slots
            hdr_off = _GH_SIZE + slot * self._slot_stride
            # Read only the state byte — avoids unpacking the full 64-byte header.
            if self._buf[hdr_off + _STATE_OFFSET] == _STATE_FREE:
                self._write_pos += 1
                return slot
            self._write_pos += 1
        raise RuntimeError("ShmRingBuffer: all slots occupied — consumer is too slow")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release this process's view and close its shared-memory handle."""
        self._buf.release()
        try:
            self._shm.close()
        except BufferError:
            # A SlotView memoryview is still live (unreleased slot during abnormal
            # shutdown).  The OS reclaims the segment once all views are GC'd.
            pass

    def unlink(self) -> None:
        """Remove the segment, tolerating repeated cleanup by this process."""
        try:
            self._shm.unlink()
        except FileNotFoundError:
            # The creating owner can reach cleanup more than once. Unrelated
            # processes must not unlink a segment they did not create.
            pass

    def __enter__(self):
        """Return this ring buffer from a context manager."""
        return self

    def __exit__(self, *_):
        """Close this process's handle when leaving a context manager."""
        self.close()
