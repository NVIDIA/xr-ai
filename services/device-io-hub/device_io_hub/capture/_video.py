# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared H.264 encoding and bundle assembly primitives."""

from __future__ import annotations

import hashlib
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from xr_ai_hub import FrameData

from ._frontends import _CaptureFrontend
from ._matroska import VideoPacket
from .config import CaptureConfig

_MAX_SAFE_NAME = 96


def _safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_." else "_" for char in value)
    if cleaned == value and 0 < len(cleaned) <= _MAX_SAFE_NAME:
        return cleaned
    digest = hashlib.sha256(value.encode()).hexdigest()[:8]
    prefix = (cleaned or "unnamed")[:_MAX_SAFE_NAME - len(digest) - 1]
    return f"{prefix}_{digest}"


def _encoded_packets(packets: object) -> list[dict]:
    if not isinstance(packets, list):
        raise TypeError(f"unexpected PyNvVideoCodec packet collection: {type(packets).__name__}")
    output: list[dict] = []
    for packet in packets:
        if not isinstance(packet, dict) or not isinstance(packet.get("data"), bytes):
            raise TypeError(f"unexpected PyNvVideoCodec encoded packet: {packet!r}")
        output.append(packet)
    return output


@dataclass(frozen=True, slots=True)
class _TrackPacket:
    track_id: str
    packet: VideoPacket


class _H264TrackWriter:
    """One NVENC stream for one source or rendering track."""

    def __init__(
        self,
        *,
        nvc: object,
        root: Path,
        track_id: str,
        stream_name: str,
        config: CaptureConfig,
        frontend: _CaptureFrontend,
    ) -> None:
        self._nvc = nvc
        self._root = root
        self._track_id = track_id
        self._stream_name = stream_name
        self._config = config
        self._frontend = frontend
        self._encoder = None
        self._stream = None
        self._width = 0
        self._height = 0
        self._last_pts_us = 0
        self._segment_index = 0
        self._active: dict | None = None
        self._packets: list[VideoPacket] = []
        self._submitted_pts: deque[int] = deque()
        self.segments: list[dict] = []

    def write(
        self,
        frame: FrameData,
        caption: str,
        data_feed: tuple[str, ...],
    ) -> tuple[int, int, str] | None:
        min_interval_us = round(1_000_000 / self._config.sample_fps)
        if self._last_pts_us and frame.pts_us - self._last_pts_us < min_interval_us:
            return None
        rendered = self._frontend.render_frame(
            frame,
            caption=caption,
            data_feed=data_feed,
        )
        width = rendered.width
        height = rendered.height
        if self._encoder is None or (width, height) != (self._width, self._height):
            self._start_segment(width, height, frame.pts_us)
        picture_params = self._nvc.NV_ENC_PIC_PARAMS()
        picture_params.inputTimeStamp = frame.pts_us
        self._submitted_pts.append(frame.pts_us)
        for packet in _encoded_packets(self._encoder.Encode(rendered.pixels, picture_params)):
            self._write_packet(packet)
        self._last_pts_us = frame.pts_us
        self._active["end_us"] = frame.pts_us
        self._active["num_frames"] += 1
        return width, height, str(self._active["path"])

    def _start_segment(self, width: int, height: int, pts_us: int) -> None:
        self._finish_segment()
        self._width = width
        self._height = height
        name = f"{self._stream_name}_{_safe_name(self._track_id)}_{self._segment_index:03d}.264"
        self._segment_index += 1
        self._stream = (self._root / name).open("wb")
        try:
            self._encoder = self._nvc.CreateEncoder(
                width,
                height,
                "NV12",
                True,
                gpu_id=self._config.gpu_id,
                codec="h264",
                preset="P4",
                tuning_info="high_quality",
                rc="vbr",
                fps=int(round(self._config.sample_fps)),
                bitrate=self._config.bitrate,
                maxbitrate=self._config.bitrate,
                bf=0,
                repeat_sps_pps=1,
            )
        except Exception:
            self._stream.close()
            (self._root / name).unlink(missing_ok=True)
            self._stream = None
            raise
        self._active = {
            "path": f"video/{name}",
            "start_us": pts_us,
            "end_us": pts_us,
            "num_frames": 0,
            "width": width,
            "height": height,
            "fps": self._config.sample_fps,
        }
        self._packets = []
        self._submitted_pts.clear()

    def _write_packet(self, packet: dict) -> None:
        payload = packet["data"]
        offset = self._stream.tell()
        self._stream.write(payload)
        picture_type = int(packet.get("picture_type", 0))
        if not self._submitted_pts:
            raise ValueError("NVENC emitted more packets than submitted frames")
        self._packets.append(
            VideoPacket(
                offset=offset,
                size=len(payload),
                pts_us=self._submitted_pts.popleft(),
                key_frame=picture_type in (2, 3),
            )
        )

    def _finish_segment(self) -> None:
        if self._encoder is None:
            return
        try:
            for packet in _encoded_packets(self._encoder.EndEncode()):
                self._write_packet(packet)
            if self._submitted_pts:
                raise ValueError(
                    f"NVENC omitted {len(self._submitted_pts)} submitted frames"
                )
        except Exception as exc:
            logger.warning("media capture NVENC flush failed track={!r}: {}", self._track_id, exc)
        finally:
            self._stream.close()
            self._encoder = None
            self._stream = None
        if self._active is not None:
            path = self._root.parent / self._active["path"]
            self._active["size_bytes"] = path.stat().st_size
            self._active["_packets"] = self._packets
            self.segments.append(self._active)
            self._active = None
            self._packets = []
            self._submitted_pts.clear()

    def close(self) -> None:
        self._finish_segment()


def _video_sources(
    writers: dict[str, _H264TrackWriter],
) -> list[tuple[str, dict]]:
    return sorted(
        (
            (track_id, segment)
            for track_id, writer in writers.items()
            for segment in writer.segments
        ),
        key=lambda item: (item[1]["start_us"], item[0]),
    )


def _join_sources(
    session_root: Path,
    sources: list[tuple[str, dict]],
    output_path: Path,
) -> list[_TrackPacket]:
    pending_path = output_path.with_suffix(".264.pending")
    packets: list[_TrackPacket] = []
    try:
        with pending_path.open("wb") as output:
            for track_id, segment in sources:
                source_path = session_root / segment["path"]
                offset = output.tell()
                with source_path.open("rb") as source:
                    shutil.copyfileobj(source, output)
                packets.extend(
                    _TrackPacket(
                        track_id=track_id,
                        packet=VideoPacket(
                            offset=offset + packet.offset,
                            size=packet.size,
                            pts_us=packet.pts_us,
                            key_frame=packet.key_frame,
                        ),
                    )
                    for packet in segment["_packets"]
                )
        pending_path.replace(output_path)
    except Exception:
        pending_path.unlink(missing_ok=True)
        raise
    for _, segment in sources:
        (session_root / segment["path"]).unlink(missing_ok=True)
    return sorted(packets, key=lambda indexed: indexed.packet.pts_us)


__all__: list[str] = []
