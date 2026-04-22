"""
xr_ai_agent — lightweight agent-side SDK for XR-Media-Hub.

Agents only need this package (pyzmq + msgpack). The heavy server-runtime
(LiveKit, FastAPI, uvicorn) is not a dependency.

Typical usage
-------------
    from xr_ai_agent import ProcessorEndpoint, DataMessage, FrameSignal

    ep = ProcessorEndpoint(sub_addr="ipc:///tmp/xr_hub_pub",
                           push_addr="ipc:///tmp/xr_hub_in")
    ep.on_frame(my_frame_handler)
    ep.on_data(my_data_handler)
    await ep.run()
"""

from ._codec import decode, encode, register_decoder, register_encoder
from ._processor import AGENT_STATUS_TOPIC, ProcessorEndpoint
from ._shm import ShmRingBuffer, SlotView
from ._types import (
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
)

__all__ = [
    # endpoint
    "ProcessorEndpoint",
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
]
