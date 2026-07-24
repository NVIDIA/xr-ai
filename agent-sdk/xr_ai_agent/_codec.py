# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal extensible wire codec.

Format: [u8 type_id] [msgpack payload]

Register new message types with register_encoder / register_decoder without
touching existing code.
"""
from __future__ import annotations

import struct
from typing import Any, Callable

import msgpack

from ._types import (AudioChunk, ConnectorRegistration, ControlMessage, DataMessage,
                     FrameData, FrameRequest, FrameSignal, MsgType, ParticipantEvent,
                     PixelFormat, ReturnAudioFlush, RosterRequest)

_TYPE_HDR = struct.Struct("=B")

_encoders: dict[int, Callable[[Any], list]] = {}
_decoders: dict[int, Callable[[list], Any]] = {}


def register_encoder(type_id: int, fn: Callable[[Any], list]) -> None:
    """Register a serializer for type_id. fn must return a msgpack-serialisable list."""
    _encoders[type_id] = fn


def register_decoder(type_id: int, fn: Callable[[list], Any]) -> None:
    """Register a deserializer for type_id. fn receives the decoded list."""
    _decoders[type_id] = fn


def encode(type_id: int, msg: Any) -> bytes:
    payload = msgpack.packb(_encoders[type_id](msg), use_bin_type=True)
    return _TYPE_HDR.pack(type_id) + payload


def decode(raw: bytes) -> tuple[int, Any]:
    (type_id,) = _TYPE_HDR.unpack_from(raw, 0)
    payload = msgpack.unpackb(raw[1:], raw=False)
    return type_id, _decoders[type_id](payload)


# ── built-in codecs ────────────────────────────────────────────────────────────

register_encoder(
    MsgType.FRAME_SIGNAL,
    lambda m: [m.slot, m.seq, m.pts_us, m.width, m.height, int(m.fmt), m.data_sz,
               m.participant_id, m.track_id],
)
register_decoder(
    MsgType.FRAME_SIGNAL,
    lambda p: FrameSignal(p[0], p[1], p[2], p[3], p[4], PixelFormat(p[5]), p[6], p[7], p[8]),
)

register_encoder(
    MsgType.AUDIO_CHUNK,
    lambda m: [m.pts_us, m.sample_rate, m.channels, m.samples, m.data,
               m.participant_id, m.track_id],
)
register_decoder(
    MsgType.AUDIO_CHUNK,
    lambda p: AudioChunk(p[0], p[1], p[2], p[3], bytes(p[4]), p[5], p[6]),
)

register_encoder(MsgType.CONTROL,      lambda m: [m.topic, m.payload])
register_decoder(MsgType.CONTROL,      lambda p: ControlMessage(p[0], p[1]))

register_encoder(MsgType.DATA_MESSAGE, lambda m: [m.participant_id, m.topic, m.pts_us, m.data])
register_decoder(MsgType.DATA_MESSAGE, lambda p: DataMessage(p[0], p[1], p[2], bytes(p[3])))

# Return-path types reuse the same wire layout as their inbound counterparts.
register_encoder(MsgType.RETURN_AUDIO, lambda m: [m.pts_us, m.sample_rate, m.channels, m.samples, m.data, m.participant_id, m.track_id])
register_decoder(MsgType.RETURN_AUDIO, lambda p: AudioChunk(p[0], p[1], p[2], p[3], bytes(p[4]), p[5], p[6]))

register_encoder(MsgType.RETURN_DATA,  lambda m: [m.participant_id, m.topic, m.pts_us, m.data])
register_decoder(MsgType.RETURN_DATA,  lambda p: DataMessage(p[0], p[1], p[2], bytes(p[3])))

register_encoder(MsgType.PARTICIPANT_EVENT,  lambda m: [m.participant_id, m.joined, m.pts_us, m.connector_id])
register_decoder(MsgType.PARTICIPANT_EVENT,  lambda p: ParticipantEvent(p[0], p[1], p[2], p[3]))

register_encoder(MsgType.CONNECTOR_REGISTER, lambda m: [m.connector_id, m.shm_name])
register_decoder(MsgType.CONNECTOR_REGISTER, lambda p: ConnectorRegistration(p[0], p[1]))

register_encoder(MsgType.FRAME_REQUEST, lambda m: [m.participant_id, m.track_id])
register_decoder(MsgType.FRAME_REQUEST, lambda p: FrameRequest(p[0], p[1]))

register_encoder(MsgType.FRAME_DATA,
                 lambda m: [m.seq, m.pts_us, m.width, m.height, int(m.fmt), m.data,
                            m.participant_id, m.track_id])
register_decoder(MsgType.FRAME_DATA,
                 lambda p: FrameData(p[0], p[1], p[2], p[3], PixelFormat(p[4]),
                                     bytes(p[5]), p[6], p[7]))

register_encoder(MsgType.RETURN_AUDIO_FLUSH, lambda m: [m.participant_id])
register_decoder(MsgType.RETURN_AUDIO_FLUSH, lambda p: ReturnAudioFlush(p[0]))

register_encoder(MsgType.ROSTER_REQUEST, lambda _m: [])
register_decoder(MsgType.ROSTER_REQUEST, lambda _p: RosterRequest())
