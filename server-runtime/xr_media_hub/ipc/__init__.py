"""
xr_media_hub.ipc — extensible IPC layer for XR-Media-Hub.

Endpoints
---------
ConnectorEndpoint   — producer (LiveKit connector process)
HubEndpoint         — server  (XR-Media-Hub process)
ProcessorEndpoint   — subscriber + publisher (downstream processors, agents, analytics)

Extensibility
-------------
Register new message types at import time:

    from xr_media_hub.ipc import register_encoder, register_decoder, MsgType
    from enum import IntEnum

    class MyMsgType(IntEnum):
        MY_MSG = 10          # pick an ID outside 1-9 (built-ins)

    register_encoder(MyMsgType.MY_MSG, lambda m: [m.field_a, m.field_b])
    register_decoder(MyMsgType.MY_MSG, lambda p: MyMsg(p[0], p[1]))
"""

from ._codec import decode, encode, register_decoder, register_encoder
from ._connector import ConnectorEndpoint
from ._hub import (HubEndpoint,
                   TOPIC_VIDEO, TOPIC_VIDEO_DATA,
                   TOPIC_AUDIO, TOPIC_DATA, TOPIC_CONTROL,
                   TOPIC_RETURN_AUDIO, TOPIC_RETURN_DATA)
from ._processor import ProcessorEndpoint
from ._shm import ShmRingBuffer, SlotView
from ._types import (AudioChunk, ConnectorRegistration, ControlMessage, DataMessage,
                     FrameData, FrameRequest, FrameSignal, MsgType, ParticipantEvent, PixelFormat)

__all__ = [
    # endpoints
    "ConnectorEndpoint",
    "HubEndpoint",
    "ProcessorEndpoint",
    # shared memory
    "ShmRingBuffer",
    "SlotView",
    # codec extension points
    "encode",
    "decode",
    "register_encoder",
    "register_decoder",
    # data types
    "PixelFormat",
    "FrameSignal",
    "AudioChunk",
    "DataMessage",
    "ParticipantEvent",
    "ConnectorRegistration",
    "ControlMessage",
    "MsgType",
    # data types
    "FrameData",
    "FrameRequest",
    # well-known topic prefixes
    "TOPIC_VIDEO",
    "TOPIC_VIDEO_DATA",
    "TOPIC_AUDIO",
    "TOPIC_DATA",
    "TOPIC_CONTROL",
    "TOPIC_RETURN_AUDIO",
    "TOPIC_RETURN_DATA",
]
