# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private timestamp carrier for H.264 MP4 finalization."""

from __future__ import annotations

import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_TIMECODE_SCALE_NS = 1_000_000
_CLUSTER_MS = 5_000


@dataclass(frozen=True, slots=True)
class VideoPacket:
    offset: int
    size: int
    pts_us: int
    key_frame: bool


def mux_h264(
    *,
    output_path: Path,
    h264_path: Path,
    packets: Iterable[VideoPacket],
    width: int,
    height: int,
    fps: float,
) -> None:
    """Wrap H.264 packets with their recorded timestamps for FFmpeg input."""

    if fps <= 0:
        raise ValueError("Matroska frame rate must be positive")
    packet_list = sorted(packets, key=lambda packet: packet.pts_us)
    if not packet_list:
        raise ValueError("cannot mux a video stream without packets")

    with h264_path.open("rb") as video_stream:
        parameter_sets: dict[int, bytes] = {}
        for packet in packet_list:
            payload = _read_packet(video_stream, packet, h264_path)
            for nal in _split_annex_b(payload):
                nal_type = nal[0] & 0x1F
                if nal_type in (7, 8):
                    parameter_sets[nal_type] = nal
    try:
        codec_private = _avc_decoder_configuration_record(
            parameter_sets[7],
            parameter_sets[8],
        )
    except KeyError as exc:
        raise ValueError(f"H.264 stream {h264_path} has no SPS/PPS") from exc

    first_pts_us = packet_list[0].pts_us
    final_video_ms = max(
        0,
        round((packet_list[-1].pts_us - first_pts_us) / 1_000),
    )
    duration_ms = final_video_ms + round(1_000 / fps)

    with h264_path.open("rb") as video_stream, output_path.open("wb") as output:
        output.write(_ebml_header())
        output.write(_id(0x18538067) + b"\x01\xff\xff\xff\xff\xff\xff\xff")
        output.write(_segment_info(duration_ms))
        output.write(
            _tracks(
                width=width,
                height=height,
                fps=fps,
                codec_private=codec_private,
            )
        )

        packet_index = 0
        for cluster_ms in range(0, max(1, duration_ms), _CLUSTER_MS):
            cluster_end_ms = cluster_ms + _CLUSTER_MS
            blocks: list[bytes] = []
            while packet_index < len(packet_list):
                packet = packet_list[packet_index]
                pts_ms = max(0, round((packet.pts_us - first_pts_us) / 1_000))
                if pts_ms >= cluster_end_ms:
                    break
                flags = 0x80 if packet.key_frame else 0
                payload = _length_prefix(
                    _split_annex_b(_read_packet(video_stream, packet, h264_path))
                )
                blocks.append(
                    _simple_block(1, pts_ms - cluster_ms, flags, payload)
                )
                packet_index += 1
            if blocks:
                body = _uint_element(0xE7, cluster_ms) + b"".join(blocks)
                output.write(_element(0x1F43B675, body))


def _read_packet(stream, packet: VideoPacket, path: Path) -> bytes:
    stream.seek(packet.offset)
    payload = stream.read(packet.size)
    if len(payload) != packet.size:
        raise ValueError(f"truncated H.264 packet in {path}")
    return payload


def _ebml_header() -> bytes:
    return _element(
        0x1A45DFA3,
        b"".join((
            _uint_element(0x4286, 1),
            _uint_element(0x42F7, 1),
            _uint_element(0x42F2, 4),
            _uint_element(0x42F3, 8),
            _string_element(0x4282, "matroska"),
            _uint_element(0x4287, 4),
            _uint_element(0x4285, 2),
        )),
    )


def _segment_info(duration_ms: int) -> bytes:
    return _element(
        0x1549A966,
        b"".join((
            _uint_element(0x2AD7B1, _TIMECODE_SCALE_NS),
            _element(0x4489, struct.pack(">d", float(duration_ms))),
            _string_element(0x4D80, "xr-ai"),
            _string_element(0x5741, "xr-ai"),
        )),
    )


def _tracks(
    *,
    width: int,
    height: int,
    fps: float,
    codec_private: bytes,
) -> bytes:
    return _element(
        0x1654AE6B,
        _element(
            0xAE,
            b"".join((
                _uint_element(0xD7, 1),
                _uint_element(0x73C5, 1),
                _uint_element(0x83, 1),
                _uint_element(0x23E383, round(1_000_000_000 / fps)),
                _string_element(0x86, "V_MPEG4/ISO/AVC"),
                _element(0x63A2, codec_private),
                _element(
                    0xE0,
                    _uint_element(0xB0, width) + _uint_element(0xBA, height),
                ),
            )),
        ),
    )


def _split_annex_b(payload: bytes) -> list[bytes]:
    starts: list[tuple[int, int]] = []
    index = 0
    while index + 3 <= len(payload):
        if payload[index : index + 4] == b"\x00\x00\x00\x01":
            starts.append((index, 4))
            index += 4
        elif payload[index : index + 3] == b"\x00\x00\x01":
            starts.append((index, 3))
            index += 3
        else:
            index += 1
    if not starts:
        raise ValueError("H.264 packet is not Annex B")
    units = []
    for item_index, (start, prefix_size) in enumerate(starts):
        end = starts[item_index + 1][0] if item_index + 1 < len(starts) else len(payload)
        unit = payload[start + prefix_size : end]
        if unit:
            units.append(unit)
    return units


def _length_prefix(nal_units: Iterable[bytes]) -> bytes:
    return b"".join(struct.pack(">I", len(nal)) + nal for nal in nal_units)


def _avc_decoder_configuration_record(sps: bytes, pps: bytes) -> bytes:
    if len(sps) < 4:
        raise ValueError("invalid H.264 SPS")
    return b"".join((
        bytes((1, sps[1], sps[2], sps[3], 0xFF, 0xE1)),
        struct.pack(">H", len(sps)),
        sps,
        b"\x01",
        struct.pack(">H", len(pps)),
        pps,
    ))


def _simple_block(track: int, relative_ms: int, flags: int, payload: bytes) -> bytes:
    if not -32_768 <= relative_ms <= 32_767:
        raise ValueError("Matroska block is outside its cluster")
    return _element(0xA3, _vint(track) + struct.pack(">hB", relative_ms, flags) + payload)


def _element(element_id: int, payload: bytes) -> bytes:
    return _id(element_id) + _vint(len(payload)) + payload


def _uint_element(element_id: int, value: int) -> bytes:
    size = max(1, (value.bit_length() + 7) // 8)
    return _element(element_id, value.to_bytes(size, "big"))


def _string_element(element_id: int, value: str) -> bytes:
    return _element(element_id, value.encode("utf-8"))


def _id(value: int) -> bytes:
    return value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big")


def _vint(value: int) -> bytes:
    for size in range(1, 9):
        if value < (1 << (7 * size)) - 1:
            return ((1 << (7 * size)) | value).to_bytes(size, "big")
    raise ValueError("EBML value is too large")


__all__: list[str] = []
