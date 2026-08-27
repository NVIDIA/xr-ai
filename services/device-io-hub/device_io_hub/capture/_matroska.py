# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal H.264/PCM Matroska muxing for finalized capture segments."""

from __future__ import annotations

import struct
import wave
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_TIMECODE_SCALE_NS = 1_000_000
_CLUSTER_MS = 5_000
_PCM_BLOCK_FRAMES = 960


@dataclass(frozen=True, slots=True)
class VideoPacket:
    offset: int
    size: int
    pts_us: int
    key_frame: bool


def mux_h264_pcm(
    *,
    output_path: Path,
    h264_path: Path,
    packets: Iterable[VideoPacket],
    wave_path: Path,
    audio_start_frame: int,
    audio_end_frame: int,
    width: int,
    height: int,
    fps: float,
) -> None:
    """Mux one H.264 segment and its matching stereo PCM window."""
    packet_list = list(packets)
    if not packet_list:
        raise ValueError("cannot mux a video segment without packets")
    with h264_path.open("rb") as video_stream:
        parameter_sets: dict[int, bytes] = {}
        first_pts_us = packet_list[0].pts_us
        for packet in packet_list:
            nal_units = _split_annex_b(_read_packet(video_stream, packet, h264_path))
            for nal in nal_units:
                nal_type = nal[0] & 0x1F
                if nal_type in (7, 8):
                    parameter_sets[nal_type] = nal
            if 7 in parameter_sets and 8 in parameter_sets:
                break
    try:
        codec_private = _avc_decoder_configuration_record(parameter_sets[7], parameter_sets[8])
    except KeyError as exc:
        raise ValueError(f"H.264 segment {h264_path} has no SPS/PPS") from exc

    with h264_path.open("rb") as video_stream, wave.open(str(wave_path), "rb") as audio_stream:
        if audio_stream.getnchannels() != 2 or audio_stream.getsampwidth() != 2:
            raise ValueError("capture conversation WAV must be stereo 16-bit PCM")
        sample_rate = audio_stream.getframerate()
        start_frame = min(max(0, audio_start_frame), audio_stream.getnframes())
        end_frame = min(max(start_frame, audio_end_frame), audio_stream.getnframes())
        audio_stream.setpos(start_frame)
        final_video_ms = max(
            0,
            round((packet_list[-1].pts_us - first_pts_us) / 1_000),
        )
        final_audio_ms = round((end_frame - start_frame) * 1_000 / sample_rate)
        duration_ms = max(final_video_ms + round(1_000 / fps), final_audio_ms)

        with output_path.open("wb") as output:
            output.write(_ebml_header())
            output.write(_id(0x18538067) + b"\x01\xff\xff\xff\xff\xff\xff\xff")
            output.write(_segment_info(duration_ms))
            output.write(
                _tracks(
                    width=width,
                    height=height,
                    fps=fps,
                    sample_rate=sample_rate,
                    codec_private=codec_private,
                )
            )

            video_index = 0
            audio_frame = start_frame
            for cluster_ms in range(0, max(1, duration_ms), _CLUSTER_MS):
                cluster_end_ms = cluster_ms + _CLUSTER_MS
                blocks: list[tuple[int, int, bytes]] = []
                while video_index < len(packet_list):
                    packet = packet_list[video_index]
                    pts_ms = max(0, round((packet.pts_us - first_pts_us) / 1_000))
                    if pts_ms >= cluster_end_ms:
                        break
                    flags = 0x80 if packet.key_frame else 0
                    payload = _length_prefix(
                        _split_annex_b(_read_packet(video_stream, packet, h264_path))
                    )
                    blocks.append((pts_ms, 1, _simple_block(1, pts_ms - cluster_ms, flags, payload)))
                    video_index += 1
                cluster_end_frame = min(
                    end_frame,
                    start_frame + round(cluster_end_ms * sample_rate / 1_000),
                )
                while audio_frame < cluster_end_frame:
                    frames = min(_PCM_BLOCK_FRAMES, cluster_end_frame - audio_frame)
                    payload = audio_stream.readframes(frames)
                    if not payload:
                        audio_frame = cluster_end_frame
                        break
                    pts_ms = round((audio_frame - start_frame) * 1_000 / sample_rate)
                    blocks.append((pts_ms, 2, _simple_block(2, pts_ms - cluster_ms, 0, payload)))
                    audio_frame += len(payload) // 4
                blocks.sort(key=lambda item: (item[0], item[1]))
                if blocks:
                    body = _uint_element(0xE7, cluster_ms) + b"".join(block for _, _, block in blocks)
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
        b"".join(
            (
                _uint_element(0x4286, 1),
                _uint_element(0x42F7, 1),
                _uint_element(0x42F2, 4),
                _uint_element(0x42F3, 8),
                _string_element(0x4282, "matroska"),
                _uint_element(0x4287, 4),
                _uint_element(0x4285, 2),
            )
        ),
    )


def _segment_info(duration_ms: int) -> bytes:
    return _element(
        0x1549A966,
        b"".join(
            (
                _uint_element(0x2AD7B1, _TIMECODE_SCALE_NS),
                _element(0x4489, struct.pack(">d", float(duration_ms))),
                _string_element(0x4D80, "xr-ai"),
                _string_element(0x5741, "xr-ai"),
            )
        ),
    )


def _tracks(
    *,
    width: int,
    height: int,
    fps: float,
    sample_rate: int,
    codec_private: bytes,
) -> bytes:
    video = _element(
        0xAE,
        b"".join(
            (
                _uint_element(0xD7, 1),
                _uint_element(0x73C5, 1),
                _uint_element(0x83, 1),
                _uint_element(0x23E383, round(1_000_000_000 / fps)),
                _string_element(0x86, "V_MPEG4/ISO/AVC"),
                _element(0x63A2, codec_private),
                _element(0xE0, _uint_element(0xB0, width) + _uint_element(0xBA, height)),
            )
        ),
    )
    audio = _element(
        0xAE,
        b"".join(
            (
                _uint_element(0xD7, 2),
                _uint_element(0x73C5, 2),
                _uint_element(0x83, 2),
                _string_element(0x86, "A_PCM/INT/LIT"),
                _element(
                    0xE1,
                    b"".join(
                        (
                            _element(0xB5, struct.pack(">d", float(sample_rate))),
                            _uint_element(0x9F, 2),
                            _uint_element(0x6264, 16),
                        )
                    ),
                ),
            )
        ),
    )
    return _element(0x1654AE6B, video + audio)


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
    return b"".join(
        (
            bytes((1, sps[1], sps[2], sps[3], 0xFF, 0xE1)),
            struct.pack(">H", len(sps)),
            sps,
            b"\x01",
            struct.pack(">H", len(pps)),
            pps,
        )
    )


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
