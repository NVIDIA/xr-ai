# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
device_io_hub.ipc — extensible IPC layer for DeviceIOHub.

Endpoints
---------
ConnectorEndpoint   — producer (LiveKit connector process)
HubEndpoint         — server  (DeviceIOHub process)
ProcessorEndpoint   — subscriber + publisher (agents, analytics, downstream processors)

Agent code should import from `xr_ai_hub` directly rather than this module —
it avoids pulling in the full DeviceIOHub dependency tree.

Extensibility
-------------
Register new message types at import time:

    from device_io_hub.ipc import register_encoder, register_decoder, MsgType
    from enum import IntEnum

    class MyMsgType(IntEnum):
        MY_MSG = 10          # pick an ID outside 1-9 (built-ins)

    register_encoder(MyMsgType.MY_MSG, lambda m: [m.field_a, m.field_b])
    register_decoder(MyMsgType.MY_MSG, lambda p: MyMsg(p[0], p[1]))
"""

# Agent-facing types and endpoint — re-exported from xr_ai_hub for
# backwards compatibility with code that imports from device_io_hub.ipc.
from xr_ai_hub import (
    AGENT_STATUS_TOPIC,
    AudioChunk,
    ConnectorRegistration,
    ControlMessage,
    DataMessage,
    FrameData,
    FrameRequest,
    FrameSignal,
    MsgType,
    ParticipantEvent,
    PixelFormat,
    ProcessorEndpoint,
    ReturnAudioFlush,
    RosterRequest,
    ShmRingBuffer,
    SlotView,
    Subscribe,
    decode,
    encode,
    register_decoder,
    register_encoder,
)

# Server-side endpoints — only available when device-io-hub is installed.
from ._connector import ConnectorEndpoint
from ._hub import (
    TOPIC_AUDIO,
    TOPIC_CONTROL,
    TOPIC_DATA,
    TOPIC_RETURN_AUDIO,
    TOPIC_RETURN_AUDIO_FLUSH,
    TOPIC_RETURN_DATA,
    TOPIC_VIDEO,
    TOPIC_VIDEO_DATA,
    HubEndpoint,
)

__all__ = [
    # endpoints
    "ConnectorEndpoint",
    "HubEndpoint",
    "ProcessorEndpoint",
    "Subscribe",
    # shared memory
    "ShmRingBuffer",
    "SlotView",
    # codec extension points
    "encode",
    "decode",
    "register_encoder",
    "register_decoder",
    # data types
    "AudioChunk",
    "ConnectorRegistration",
    "ControlMessage",
    "DataMessage",
    "FrameData",
    "FrameRequest",
    "FrameSignal",
    "MsgType",
    "ParticipantEvent",
    "PixelFormat",
    "ReturnAudioFlush",
    "RosterRequest",
    # well-known topic prefixes
    "TOPIC_VIDEO",
    "TOPIC_VIDEO_DATA",
    "TOPIC_AUDIO",
    "TOPIC_DATA",
    "TOPIC_CONTROL",
    "TOPIC_RETURN_AUDIO",
    "TOPIC_RETURN_AUDIO_FLUSH",
    "TOPIC_RETURN_DATA",
    # internal SDK channel topic
    "AGENT_STATUS_TOPIC",
]
