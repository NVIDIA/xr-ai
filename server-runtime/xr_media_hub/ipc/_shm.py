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


class SlotView(NamedTuple):
    """Zero-copy view into one ring-buffer slot's pixel data."""
    data:   memoryview
    signal: FrameSignal


class ShmRingBuffer:
    """
    Shared-memory ring buffer for raw video frames.

    Hub creates the buffer (create=True). Connector opens it (create=False) and
    reads num_slots / max_frame_bytes from the global header automatically.

    The caller that uses read_slot() MUST call release_slot() before the next
    write_frame() for that slot can succeed. Both operations are O(1).
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
            _GH.pack_into(self._shm.buf, 0, _MAGIC_GLOBAL, num_slots, max_frame_bytes, slot_stride)
            for i in range(num_slots):
                off = _GH_SIZE + i * slot_stride
                _SH.pack_into(self._shm.buf, off, _MAGIC_SLOT, _STATE_FREE, 0, 0, 0, 0, 0, 0, 0)
        else:
            self._shm                              = SharedMemory(name=name, create=False)
            _, num_slots, max_frame_bytes, slot_stride = _GH.unpack_from(self._shm.buf, 0)

        self._buf            = self._shm.buf
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
        """
        Write frame into the next free slot. Returns slot index.
        Raises RuntimeError if all slots are occupied (back-pressure signal).
        """
        slot    = self._claim_slot()
        hdr_off = _GH_SIZE + slot * self._slot_stride
        dat_off = hdr_off + _SH_SIZE
        n       = len(data)

        # Mark WRITING so consumer won't touch this slot.
        _SH.pack_into(self._buf, hdr_off, _MAGIC_SLOT, _STATE_WRITING, int(fmt), 0, seq, pts_us, width, height, 0)
        self._buf[dat_off : dat_off + n] = data
        # Mark READY — consumer may read after receiving the ZMQ signal.
        _SH.pack_into(self._buf, hdr_off, _MAGIC_SLOT, _STATE_READY,   int(fmt), 0, seq, pts_us, width, height, n)

        return slot

    # ── consumer ──────────────────────────────────────────────────────────────

    def read_slot(self, signal: FrameSignal) -> SlotView:
        """
        Return a zero-copy memoryview of a ready slot's pixel data.
        The view is valid until release_slot() is called — do not hold it longer.
        """
        hdr_off = _GH_SIZE + signal.slot * self._slot_stride
        hdr     = _SH.unpack_from(self._buf, hdr_off)
        if hdr[1] != _STATE_READY:
            raise RuntimeError(f"slot {signal.slot} not READY (state={hdr[1]})")
        dat_off = hdr_off + _SH_SIZE
        return SlotView(
            data=self._buf[dat_off : dat_off + signal.data_sz],
            signal=signal,
        )

    def release_slot(self, slot: int) -> None:
        """Mark slot FREE so the producer can reuse it."""
        hdr_off = _GH_SIZE + slot * self._slot_stride
        hdr     = _SH.unpack_from(self._buf, hdr_off)
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
        self._buf.release()
        self._shm.close()

    def unlink(self) -> None:
        """Remove the underlying shared memory segment. Call once from the owner (Hub)."""
        self._shm.unlink()

    def __enter__(self):    return self
    def __exit__(self, *_): self.close()
