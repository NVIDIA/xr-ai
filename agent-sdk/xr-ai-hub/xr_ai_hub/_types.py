# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core data types for the XR-Media-Hub IPC layer. No external dependencies."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class PixelFormat(IntEnum):
    """Pixel layouts supported by shared-memory video frames."""

    I420  = 0
    """Planar YUV 4:2:0 with Y, U, and V planes."""

    NV12  = 1
    """YUV 4:2:0 with a Y plane followed by interleaved UV samples."""

    RGB24 = 2
    """Packed 24-bit RGB pixels."""

    RGBA  = 3
    """Packed 32-bit RGB pixels with an alpha channel."""

    BGRA  = 4
    """Packed 32-bit BGR pixels with an alpha channel."""


class MsgType(IntEnum):
    """Wire-level message identifiers used by the hub IPC protocol."""

    # Inbound  (connector → hub)
    FRAME_SIGNAL  = 1
    """Metadata indicating that a video frame is ready in shared memory."""

    AUDIO_CHUNK   = 2
    """Inbound PCM audio from a connector."""

    CONTROL       = 3
    """Hub-internal control data."""

    DATA_MESSAGE  = 4
    """Inbound application data from a connector."""

    # Outbound (hub → connector)
    RETURN_AUDIO     = 5   # agent/TTS audio destined for a specific client
    """PCM audio returned to a client."""

    RETURN_DATA      = 6   # agent text/binary destined for a specific client
    """Application data returned to a client."""

    # Bidirectional lifecycle events
    PARTICIPANT_EVENT   = 7  # participant joined or left the LiveKit room
    """Participant join or leave notification."""

    CONNECTOR_REGISTER  = 8  # connector announces itself + its shm name to the hub
    """Connector registration and shared-memory discovery."""

    # Frame pixel request/response (processor → hub → processor)
    FRAME_REQUEST = 9   # processor requests pixel data for a specific frame by seq
    """Request for the latest pixels from a participant track."""

    FRAME_DATA    = 10  # hub delivers pixel data to requesting processor
    """Pixel data returned in response to a frame request."""

    # Return-audio control (processor → hub → connector)
    RETURN_AUDIO_FLUSH = 11  # drop any audio queued for a participant's return track
    """Request to discard queued return audio for a participant."""

    # Roster (processor → hub → processor): used by an endpoint started
    # mid-session to learn about participants who joined before it did.
    ROSTER_REQUEST = 12
    """Request to replay join events for the current participant roster."""

    # Subscription barrier (processor → hub → processor): round-trip that
    # proves the processor's pending SUBSCRIBEs have been applied by the hub.
    SUBSCRIPTION_PROBE = 13
    """Subscription-barrier token echoed by the hub."""

    # Agent presence (processor → hub): tells the hub an agent exists so it
    # can be counted as not-yet-available when aggregating agent status.
    AGENT_PRESENCE = 14
    """Readiness-participating agent attachment or detachment."""

    # Add new types here; existing code is unaffected.


@dataclass(slots=True)
class FrameSignal:
    """Signals that a decoded frame has been written into the shared-memory ring buffer."""
    slot:           int
    """Index of the shared-memory slot containing the frame."""

    seq:            int          # per-(participant, track) monotonically increasing sequence
    """Sequence number that increases for each participant and track pair."""

    pts_us:         int          # presentation timestamp, microseconds (signed)
    """Presentation timestamp in microseconds."""

    width:          int
    """Frame width in pixels."""

    height:         int
    """Frame height in pixels."""

    fmt:            PixelFormat
    """Pixel layout used by the frame data."""

    data_sz:        int          # bytes actually written into the slot
    """Number of frame-data bytes written into the slot."""

    participant_id: str = "default"  # LiveKit participant identity
    """Identity of the participant that produced the frame."""

    track_id:       str = "default"  # LiveKit track SID
    """Identity of the participant's video track."""


@dataclass(slots=True)
class AudioChunk:
    """Raw PCM audio chunk from the connector."""
    pts_us:         int
    """Presentation timestamp in microseconds."""

    sample_rate:    int
    """Sample rate in hertz."""

    channels:       int
    """Number of interleaved audio channels."""

    samples:        int    # frames per channel
    """Number of sample frames per channel."""

    data:           bytes  # float32 LE, interleaved
    """Little-endian, interleaved float32 PCM samples."""

    participant_id: str = "default"  # LiveKit participant identity
    """Identity of the participant that produced the audio."""

    track_id:       str = "default"  # LiveKit track SID
    """Identity of the participant's audio track."""


@dataclass(slots=True)
class DataMessage:
    """
    Arbitrary binary/text payload from a LiveKit data channel.

    LiveKit data channels are per-participant and routed by topic string —
    there is no track SID for data.
    """
    participant_id: str
    """Identity of the participant that sent or should receive the payload."""

    topic:          str    # LiveKit data channel topic
    """Application-defined data-channel topic."""

    pts_us:         int
    """Presentation timestamp in microseconds."""

    data:           bytes
    """Opaque message payload."""


@dataclass(slots=True)
class ParticipantEvent:
    """A LiveKit participant has joined or left the room."""
    participant_id: str
    """Identity of the participant whose state changed."""

    joined:         bool   # True = joined, False = left
    """Whether the participant joined; ``False`` indicates departure."""

    pts_us:         int
    """Timestamp of the event in microseconds."""

    connector_id:   str = ""  # which connector this participant arrived on
    """Identity of the connector that reported the participant."""


@dataclass(slots=True)
class ConnectorRegistration:
    """Sent by a connector on startup so the hub can open its ring buffer."""
    connector_id: str
    """Identity assigned to the connector."""

    shm_name:     str
    """Name of the connector's shared-memory ring buffer."""


@dataclass(slots=True)
class FrameRequest:
    """Sent by a processor to request a copy of the current latest frame."""
    participant_id: str
    """Identity of the participant whose frame is requested."""

    track_id: str
    """Identity of the video track whose frame is requested."""


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
    """Sequence number of the copied frame."""

    pts_us:         int
    """Presentation timestamp in microseconds."""

    width:          int
    """Frame width in pixels."""

    height:         int
    """Frame height in pixels."""

    fmt:            PixelFormat
    """Pixel layout used by :attr:`data`."""

    data:           bytes          # raw pixels in the format specified by fmt
    """Raw pixel bytes encoded in :attr:`fmt`."""

    participant_id: str = "default"
    """Identity of the participant that produced the frame."""

    track_id:       str = "default"
    """Identity of the participant's video track."""


@dataclass(slots=True)
class ControlMessage:
    """Extensible key/value control message (hub-internal, no track concept)."""
    topic:   str
    """Control-message topic."""

    payload: dict[str, Any] = field(default_factory=dict)
    """Topic-specific control values."""


@dataclass(slots=True)
class ReturnAudioFlush:
    """Drop any audio queued for *participant_id*'s return track."""
    participant_id: str
    """Identity of the participant whose queued return audio is discarded."""


@dataclass(slots=True)
class RosterRequest:
    """
    Ask the hub to re-publish ``PARTICIPANT_EVENT(joined=True)`` on the
    ``participant`` topic for every currently-connected participant.

    Used by a :class:`ProcessorEndpoint` started mid-session to learn
    about clients that joined before it did. Replays go on the regular
    participant topic, so other endpoints will see them too.
    """
    pass


@dataclass(slots=True)
class SubscriptionProbe:
    """Round-trip token echoed by the hub on ``_probe.<token>``.

    ZMQ applies SUBSCRIBEs from one socket in order, so receiving the echo
    proves every subscription issued before the probe is live on the hub.
    """
    token: str
    """Opaque correlation token echoed by the hub."""


@dataclass(slots=True)
class AgentPresence:
    """An agent endpoint has attached to (or detached from) the hub.

    Only endpoints that opt into readiness send this. The hub counts every
    attached agent as unavailable until it publishes an availability status,
    so one ready agent cannot make the room look ready.

    ``scope`` names the participants this agent answers for — *None* means
    every participant. A participant's readiness aggregates only over the
    agents whose scope covers it.
    """
    agent_id: str
    """Identity of the agent endpoint."""

    attached: bool
    """Whether the endpoint is attaching to or detaching from the hub."""

    scope:    list[str] | None = None
    """Participant IDs served by the agent, or ``None`` for every participant."""
