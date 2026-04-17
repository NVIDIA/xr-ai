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
    FRAME_SIGNAL  = 1
    AUDIO_CHUNK   = 2
    CONTROL       = 3
    DATA_MESSAGE  = 4
    # Add new types here; existing code is unaffected.


@dataclass(slots=True)
class FrameSignal:
    """Signals that a decoded frame has been written into the shared-memory ring buffer."""
    slot:     int
    seq:      int          # per-track monotonically increasing sequence number
    pts_us:   int          # presentation timestamp, microseconds (signed)
    width:    int
    height:   int
    fmt:      PixelFormat
    data_sz:  int          # bytes actually written into the slot
    track_id: str = "default"


@dataclass(slots=True)
class AudioChunk:
    """Raw PCM audio chunk from the connector."""
    pts_us:      int
    sample_rate: int
    channels:    int
    samples:     int    # frames per channel
    data:        bytes  # float32 LE, interleaved
    track_id:    str = "default"


@dataclass(slots=True)
class DataMessage:
    """Arbitrary binary/text payload from a LiveKit data channel."""
    track_id: str
    pts_us:   int
    data:     bytes


@dataclass(slots=True)
class ControlMessage:
    """Extensible key/value control message (hub-internal, no track concept)."""
    topic:   str
    payload: dict[str, Any] = field(default_factory=dict)
