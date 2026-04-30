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
    # Frame pixel request/response (processor → hub → processor)
    FRAME_REQUEST = 9   # processor requests pixel data for a specific frame by seq
    FRAME_DATA    = 10  # hub delivers pixel data to requesting processor
    # Return-audio control (processor → hub → connector)
    RETURN_AUDIO_FLUSH = 11  # drop any audio queued for a participant's return track
    # Roster (processor → hub → processor): used by an endpoint started
    # mid-session to learn about participants who joined before it did.
    ROSTER_REQUEST = 12
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
class FrameRequest:
    """Sent by a processor to request a copy of the current latest frame."""
    participant_id: str
    track_id: str


@dataclass(slots=True)
class FrameData:
    """
    Pixel data for the latest frame, published by the hub on `video_data.<pid>.<track>`.

    The hub holds one SHM slot per (participant, track) — always the most recent
    frame. Processors receive FrameSignal metadata at full rate via on_frame(),
    then call ProcessorEndpoint.request_frame() to get a pixel copy at their own
    sampling rate. The hub only copies pixels when a request arrives.
    """
    seq:            int
    pts_us:         int
    width:          int
    height:         int
    fmt:            PixelFormat
    data:           bytes          # raw pixels in the format specified by fmt
    participant_id: str = "default"
    track_id:       str = "default"


@dataclass(slots=True)
class ControlMessage:
    """Extensible key/value control message (hub-internal, no track concept)."""
    topic:   str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReturnAudioFlush:
    """Drop any audio queued for *participant_id*'s return track."""
    participant_id: str


@dataclass(slots=True)
class RosterRequest:
    """
    Ask the hub to re-publish ``PARTICIPANT_EVENT(joined=True)`` on the
    ``participant`` topic for every currently-connected participant.

    Used by a :class:`ProcessorEndpoint` started mid-session to learn
    about clients that joined before it did. Replays go on the regular
    participant topic, so other endpoints will see them too — keep
    ``on_participant`` callbacks idempotent.
    """
    pass
