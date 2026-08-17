# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
xr_ai_hub — lightweight agent-side SDK for XR-Media-Hub.

Agents only need this package (pyzmq + msgpack). The hub implementation and
its LiveKit, FastAPI, and uvicorn dependencies are not included.

Typical usage::

    from xr_ai_hub import ProcessorEndpoint, DataMessage, FrameSignal

    ep = ProcessorEndpoint(sub_addr="ipc:///tmp/xr_hub_pub",
                           push_addr="ipc:///tmp/xr_hub_in")
    ep.on_frame(my_frame_handler)
    ep.on_data(my_data_handler)
    await ep.run()
"""

from ._codec import decode, encode, register_decoder, register_encoder
from ._live_frames import FrameUnavailable, LiveFrameSource
from ._processor import AGENT_STATUS_TOPIC, ProcessorEndpoint, Subscribe
from ._shm import ShmRingBuffer, SlotView
from ._types import (
    AgentPresence,
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
    ReturnAudioFlush,
    RosterRequest,
    SubscriptionProbe,
)

__all__ = [
    # endpoint
    "ProcessorEndpoint",
    "Subscribe",
    "AGENT_STATUS_TOPIC",
    # shared memory (for agents that read raw pixels)
    "ShmRingBuffer",
    "SlotView",
    # codec extension points
    "encode",
    "decode",
    "register_encoder",
    "register_decoder",
    # data types
    "AgentPresence",
    "AudioChunk",
    "ConnectorRegistration",
    "ControlMessage",
    "DataMessage",
    "FrameData",
    "FrameUnavailable",
    "FrameRequest",
    "FrameSignal",
    "LiveFrameSource",
    "MsgType",
    "ParticipantEvent",
    "PixelFormat",
    "ReturnAudioFlush",
    "RosterRequest",
    "SubscriptionProbe",
]
