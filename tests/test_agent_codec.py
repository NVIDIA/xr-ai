# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for xr_ai_agent._codec (encode/decode round-trips) and _types."""
from __future__ import annotations

import pytest

from xr_ai_agent._codec import decode, encode
from xr_ai_agent._types import (
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
)


# ── helpers ────────────────────────────────────────────────────────────────────

def rt(msg_type: int, obj) -> object:
    """Encode *obj* as *msg_type*, then decode the wire bytes. Returns the decoded object."""
    return decode(encode(msg_type, obj))[1]


def rt_type_id(msg_type: int, obj) -> int:
    """Return the decoded type_id from the wire bytes."""
    return decode(encode(msg_type, obj))[0]


# ── type_id round-trips ────────────────────────────────────────────────────────

class TestTypeIdPreservation:
    """The type byte written by encode must come back from decode unchanged."""

    def test_frame_signal_type_id(self):
        msg = FrameSignal(0, 1, 1000, 640, 480, PixelFormat.NV12, 460800)
        assert rt_type_id(MsgType.FRAME_SIGNAL, msg) == MsgType.FRAME_SIGNAL

    def test_audio_chunk_type_id(self):
        msg = AudioChunk(0, 48000, 1, 480, b"\x00" * 1920)
        assert rt_type_id(MsgType.AUDIO_CHUNK, msg) == MsgType.AUDIO_CHUNK

    def test_return_audio_type_id(self):
        msg = AudioChunk(0, 22050, 1, 220, b"\x00" * 440)
        assert rt_type_id(MsgType.RETURN_AUDIO, msg) == MsgType.RETURN_AUDIO

    def test_data_message_type_id(self):
        msg = DataMessage("pid1", "chat", 5000, b"hello")
        assert rt_type_id(MsgType.DATA_MESSAGE, msg) == MsgType.DATA_MESSAGE

    def test_return_data_type_id(self):
        msg = DataMessage("pid1", "reply", 6000, b"hi")
        assert rt_type_id(MsgType.RETURN_DATA, msg) == MsgType.RETURN_DATA

    def test_control_type_id(self):
        msg = ControlMessage("hub.status", {"state": "ready"})
        assert rt_type_id(MsgType.CONTROL, msg) == MsgType.CONTROL

    def test_participant_event_type_id(self):
        msg = ParticipantEvent("p1", True, 100)
        assert rt_type_id(MsgType.PARTICIPANT_EVENT, msg) == MsgType.PARTICIPANT_EVENT

    def test_connector_register_type_id(self):
        msg = ConnectorRegistration("conn1", "xr_test_shm")
        assert rt_type_id(MsgType.CONNECTOR_REGISTER, msg) == MsgType.CONNECTOR_REGISTER

    def test_frame_request_type_id(self):
        msg = FrameRequest("pid1", "track1")
        assert rt_type_id(MsgType.FRAME_REQUEST, msg) == MsgType.FRAME_REQUEST

    def test_frame_data_type_id(self):
        msg = FrameData(7, 2000, 320, 240, PixelFormat.RGB24, b"\xff" * (320 * 240 * 3))
        assert rt_type_id(MsgType.FRAME_DATA, msg) == MsgType.FRAME_DATA

    def test_return_audio_flush_type_id(self):
        msg = ReturnAudioFlush("pid1")
        assert rt_type_id(MsgType.RETURN_AUDIO_FLUSH, msg) == MsgType.RETURN_AUDIO_FLUSH

    def test_roster_request_type_id(self):
        msg = RosterRequest()
        assert rt_type_id(MsgType.ROSTER_REQUEST, msg) == MsgType.ROSTER_REQUEST


# ── payload field round-trips ──────────────────────────────────────────────────

class TestFrameSignalCodec:
    def test_fields_preserved(self):
        orig = FrameSignal(
            slot=3, seq=42, pts_us=999_000, width=1280, height=720,
            fmt=PixelFormat.I420, data_sz=1382400,
            participant_id="alice", track_id="camera0",
        )
        out = rt(MsgType.FRAME_SIGNAL, orig)
        assert isinstance(out, FrameSignal)
        assert out.slot           == orig.slot
        assert out.seq            == orig.seq
        assert out.pts_us         == orig.pts_us
        assert out.width          == orig.width
        assert out.height         == orig.height
        assert out.fmt            == orig.fmt
        assert out.data_sz        == orig.data_sz
        assert out.participant_id == orig.participant_id
        assert out.track_id       == orig.track_id

    @pytest.mark.parametrize("fmt", list(PixelFormat))
    def test_pixel_format_preserved(self, fmt):
        orig = FrameSignal(0, 0, 0, 1, 1, fmt, 0)
        out = rt(MsgType.FRAME_SIGNAL, orig)
        assert out.fmt == fmt


class TestAudioChunkCodec:
    def test_inbound_fields_preserved(self):
        payload = bytes(range(256)) * 3
        orig = AudioChunk(
            pts_us=12345, sample_rate=48000, channels=1,
            samples=384, data=payload,
            participant_id="bob", track_id="mic",
        )
        out = rt(MsgType.AUDIO_CHUNK, orig)
        assert isinstance(out, AudioChunk)
        assert out.pts_us         == orig.pts_us
        assert out.sample_rate    == orig.sample_rate
        assert out.channels       == orig.channels
        assert out.samples        == orig.samples
        assert out.data           == orig.data
        assert out.participant_id == orig.participant_id
        assert out.track_id       == orig.track_id

    def test_return_audio_uses_same_layout(self):
        """RETURN_AUDIO reuses the AudioChunk wire layout (same fields)."""
        orig = AudioChunk(500, 22050, 1, 220, b"\xab" * 440)
        out = rt(MsgType.RETURN_AUDIO, orig)
        assert isinstance(out, AudioChunk)
        assert out.data == orig.data

    def test_binary_data_roundtrip(self):
        """Arbitrary binary payloads must survive msgpack serialisation."""
        data = bytes(range(256))
        orig = AudioChunk(0, 16000, 1, 128, data)
        out = rt(MsgType.AUDIO_CHUNK, orig)
        assert out.data == data


class TestDataMessageCodec:
    def test_fields_preserved(self):
        orig = DataMessage(
            participant_id="carol", topic="chat",
            pts_us=7777, data=b"hello world",
        )
        out = rt(MsgType.DATA_MESSAGE, orig)
        assert isinstance(out, DataMessage)
        assert out.participant_id == orig.participant_id
        assert out.topic          == orig.topic
        assert out.pts_us         == orig.pts_us
        assert out.data           == orig.data

    def test_return_data_same_layout(self):
        orig = DataMessage("dave", "reply", 8888, b"ack")
        out = rt(MsgType.RETURN_DATA, orig)
        assert isinstance(out, DataMessage)
        assert out.data == orig.data

    def test_empty_binary_data(self):
        orig = DataMessage("p1", "t", 0, b"")
        out = rt(MsgType.DATA_MESSAGE, orig)
        assert out.data == b""


class TestControlMessageCodec:
    def test_fields_preserved(self):
        orig = ControlMessage("hub.status", {"running": True, "connections": 3})
        out = rt(MsgType.CONTROL, orig)
        assert isinstance(out, ControlMessage)
        assert out.topic   == orig.topic
        assert out.payload == orig.payload

    def test_empty_payload(self):
        orig = ControlMessage("ping", {})
        out = rt(MsgType.CONTROL, orig)
        assert out.payload == {}


class TestParticipantEventCodec:
    def test_joined_event(self):
        orig = ParticipantEvent("eve", joined=True, pts_us=1000, connector_id="conn0")
        out = rt(MsgType.PARTICIPANT_EVENT, orig)
        assert isinstance(out, ParticipantEvent)
        assert out.participant_id == "eve"
        assert out.joined         is True
        assert out.connector_id   == "conn0"

    def test_left_event(self):
        orig = ParticipantEvent("frank", joined=False, pts_us=2000)
        out = rt(MsgType.PARTICIPANT_EVENT, orig)
        assert out.joined is False


class TestConnectorRegistrationCodec:
    def test_fields_preserved(self):
        orig = ConnectorRegistration("conn42", "xr_shm_abcdef")
        out = rt(MsgType.CONNECTOR_REGISTER, orig)
        assert isinstance(out, ConnectorRegistration)
        assert out.connector_id == "conn42"
        assert out.shm_name     == "xr_shm_abcdef"


class TestFrameRequestCodec:
    def test_fields_preserved(self):
        orig = FrameRequest(participant_id="grace", track_id="camera1")
        out = rt(MsgType.FRAME_REQUEST, orig)
        assert isinstance(out, FrameRequest)
        assert out.participant_id == "grace"
        assert out.track_id       == "camera1"


class TestFrameDataCodec:
    def test_fields_preserved(self):
        pixels = b"\xde\xad\xbe\xef" * 1000
        orig = FrameData(
            seq=99, pts_us=55555, width=160, height=90,
            fmt=PixelFormat.RGBA, data=pixels,
            participant_id="hank", track_id="cam0",
        )
        out = rt(MsgType.FRAME_DATA, orig)
        assert isinstance(out, FrameData)
        assert out.seq            == orig.seq
        assert out.pts_us         == orig.pts_us
        assert out.width          == orig.width
        assert out.height         == orig.height
        assert out.fmt            == PixelFormat.RGBA
        assert out.data           == orig.data
        assert out.participant_id == orig.participant_id
        assert out.track_id       == orig.track_id


class TestReturnAudioFlushCodec:
    def test_participant_id_preserved(self):
        orig = ReturnAudioFlush("iris")
        out = rt(MsgType.RETURN_AUDIO_FLUSH, orig)
        assert isinstance(out, ReturnAudioFlush)
        assert out.participant_id == "iris"


class TestRosterRequestCodec:
    def test_roundtrip_produces_instance(self):
        orig = RosterRequest()
        out = rt(MsgType.ROSTER_REQUEST, orig)
        assert isinstance(out, RosterRequest)


# ── wire format sanity ──────────────────────────────────────────────────────────

class TestWireFormat:
    def test_first_byte_is_type_id(self):
        """The wire format starts with the type byte (u8)."""
        msg = FrameRequest("pid", "track")
        wire = encode(MsgType.FRAME_REQUEST, msg)
        # The type byte is u8 — every registered MsgType value must stay in 0..255
        # or `_TYPE_HDR.pack(type_id)` in _codec.py will raise.
        assert wire[0] == int(MsgType.FRAME_REQUEST)

    def test_encode_is_bytes(self):
        msg = RosterRequest()
        assert isinstance(encode(MsgType.ROSTER_REQUEST, msg), bytes)

    def test_minimum_wire_length(self):
        """Even the smallest message (RosterRequest, empty payload) must have
        at least 1 byte for the type header."""
        wire = encode(MsgType.ROSTER_REQUEST, RosterRequest())
        assert len(wire) >= 1

    def test_unknown_type_id_raises_on_decode(self):
        """A byte sequence with an unregistered type_id must raise KeyError."""
        import msgpack

        from xr_ai_agent._codec import _decoders

        # Pick the first unused type_id dynamically so this test stays valid
        # if 255 is ever registered.
        free_id = next(i for i in range(256) if i not in _decoders)
        fake = bytes([free_id]) + msgpack.packb([], use_bin_type=True)
        with pytest.raises(KeyError, match=str(free_id)):
            decode(fake)
