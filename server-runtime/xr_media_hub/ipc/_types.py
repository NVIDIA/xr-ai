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
    # Inbound  (connector → hub)
    FRAME_SIGNAL  = 1
    AUDIO_CHUNK   = 2
    CONTROL       = 3
    DATA_MESSAGE  = 4
    # Outbound (hub → connector)
    RETURN_AUDIO     = 5   # agent/TTS audio destined for a specific client
    RETURN_DATA      = 6   # agent text/binary destined for a specific client
    # Bidirectional lifecycle events
    PARTICIPANT_EVENT   = 7  # participant joined or left the LiveKit room
    CONNECTOR_REGISTER  = 8  # connector announces itself + its shm name to the hub
    # Add new types here; existing code is unaffected.


@dataclass(slots=True)
class FrameSignal:
    """Signals that a decoded frame has been written into the shared-memory ring buffer."""
    slot:           int
    seq:            int          # per-(participant, track) monotonically increasing sequence
    pts_us:         int          # presentation timestamp, microseconds (signed)
    width:          int
    height:         int
    fmt:            PixelFormat
    data_sz:        int          # bytes actually written into the slot
    participant_id: str = "default"  # LiveKit participant identity
    track_id:       str = "default"  # LiveKit track SID


@dataclass(slots=True)
class AudioChunk:
    """Raw PCM audio chunk from the connector."""
    pts_us:         int
    sample_rate:    int
    channels:       int
    samples:        int    # frames per channel
    data:           bytes  # float32 LE, interleaved
    participant_id: str = "default"  # LiveKit participant identity
    track_id:       str = "default"  # LiveKit track SID


@dataclass(slots=True)
class DataMessage:
    """
    Arbitrary binary/text payload from a LiveKit data channel.

    LiveKit data channels are per-participant and routed by topic string —
    there is no track SID for data.
    """
    participant_id: str
    topic:          str    # LiveKit data channel topic
    pts_us:         int
    data:           bytes


@dataclass(slots=True)
class ParticipantEvent:
    """A LiveKit participant has joined or left the room."""
    participant_id: str
    joined:         bool   # True = joined, False = left
    pts_us:         int
    connector_id:   str = ""  # which connector this participant arrived on


@dataclass(slots=True)
class ConnectorRegistration:
    """Sent by a connector on startup so the hub can open its ring buffer."""
    connector_id: str
    shm_name:     str


@dataclass(slots=True)
class ControlMessage:
    """Extensible key/value control message (hub-internal, no track concept)."""
    topic:   str
    payload: dict[str, Any] = field(default_factory=dict)
