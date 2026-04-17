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

from ._types import AudioChunk, ControlMessage, FrameSignal, MsgType, PixelFormat

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
    lambda m: [m.slot, m.seq, m.pts_us, m.width, m.height, int(m.fmt), m.data_sz],
)
register_decoder(
    MsgType.FRAME_SIGNAL,
    lambda p: FrameSignal(p[0], p[1], p[2], p[3], p[4], PixelFormat(p[5]), p[6]),
)

register_encoder(
    MsgType.AUDIO_CHUNK,
    lambda m: [m.pts_us, m.sample_rate, m.channels, m.samples, m.data],
)
register_decoder(
    MsgType.AUDIO_CHUNK,
    lambda p: AudioChunk(p[0], p[1], p[2], p[3], bytes(p[4])),
)

register_encoder(MsgType.CONTROL, lambda m: [m.topic, m.payload])
register_decoder(MsgType.CONTROL, lambda p: ControlMessage(p[0], p[1]))
