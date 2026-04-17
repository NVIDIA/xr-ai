"""Core data types for the XR-Media-Hub IPC layer. No external dependencies."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class PixelFormat(IntEnum):
    I420  = 0
    NV12  = 1
    RGB24 = 2
    RGBA  = 3
    BGRA  = 4


class MsgType(IntEnum):
    FRAME_SIGNAL = 1
    AUDIO_CHUNK  = 2
    CONTROL      = 3
    # Add new types here; existing code is unaffected.


@dataclass(slots=True)
class FrameSignal:
    """Signals that a decoded frame has been written into the shared-memory ring buffer."""
    slot:    int
    seq:     int          # monotonically increasing producer sequence number
    pts_us:  int          # presentation timestamp, microseconds (signed)
    width:   int
    height:  int
    fmt:     PixelFormat
    data_sz: int          # bytes actually written into the slot


@dataclass(slots=True)
class AudioChunk:
    """Raw PCM audio chunk from the connector."""
    pts_us:      int
    sample_rate: int
    channels:    int
    samples:     int   # frames per channel
    data:        bytes # float32 LE, interleaved


@dataclass(slots=True)
class ControlMessage:
    """Extensible key/value control message."""
    topic:   str
    payload: dict[str, Any] = field(default_factory=dict)
