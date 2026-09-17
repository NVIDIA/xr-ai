# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a presentation-ready video from a completed raw capture bundle."""

from __future__ import annotations

import ctypes
import json
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from xr_ai_hub import FrameData, PixelFormat

from ._frontends import _DemoCaptureFrontend
from ._video import _H264TrackWriter, _join_sources, _video_sources
from .config import CaptureConfig

_MAX_DATA_FEED_TEXT = 1_024


def _load_nvc() -> object:
    try:
        import PyNvVideoCodec as nvc
    except (ImportError, RuntimeError, OSError) as exc:
        raise RuntimeError(f"PyNvVideoCodec is required for capture rendering: {exc}") from exc
    return nvc


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    values = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {path.name} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path.name} line {line_number} must be an object")
            values.append(value)
    return values


def _bundle_file(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"capture manifest {label} must be a relative path")
    root = root.resolve()
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"capture manifest {label} escapes the bundle")
    if not path.is_file():
        raise ValueError(f"capture manifest {label} does not exist: {relative}")
    return path


def _copy_decoded_frame(frame: object) -> np.ndarray:
    if isinstance(frame, np.ndarray):
        return np.ascontiguousarray(frame)
    shape = getattr(frame, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise TypeError("NVDEC returned a frame without a two-dimensional shape")
    size = int(shape[0]) * int(shape[1])
    view = (ctypes.c_uint8 * size).from_address(frame.GetPtrToPlane(0))
    return np.ctypeslib.as_array(view).reshape(shape).copy()


class _OverlayTimeline:
    def __init__(self, events: list[dict[str, Any]], *, duration_us: int) -> None:
        self._events = sorted(
            events,
            key=lambda event: (int(event.get("pts_us", 0)), int(event.get("event_id", 0))),
        )
        self._duration_us = duration_us
        self._position = 0
        self._caption = ""
        self._caption_expires_us = 0
        self._data_feed: deque[str] = deque(maxlen=64)

    def at(self, pts_us: int) -> tuple[str, tuple[str, ...]]:
        while self._position < len(self._events):
            event = self._events[self._position]
            event_pts = event.get("pts_us")
            if not isinstance(event_pts, int) or event_pts > pts_us:
                break
            self._position += 1
            if event.get("kind") == "voice_caption":
                source = str(event.get("source", "")).upper()
                text = str(event.get("text", "")).strip()
                self._caption = f"{source}: {text}" if source and text else text
                self._caption_expires_us = event_pts + self._duration_us
            elif event.get("kind") == "data" and isinstance(event.get("text"), str):
                text = " ".join(event["text"].split())[:_MAX_DATA_FEED_TEXT]
                if text:
                    direction = str(event.get("direction", "data")).upper()
                    topic = str(event.get("topic", ""))
                    self._data_feed.append(f"{direction} {topic}: {text}")
        caption = self._caption if pts_us <= self._caption_expires_us else ""
        return caption, tuple(self._data_feed)


class CaptureRenderer:
    """Create a captioned H.264/AAC MP4 from a completed capture bundle."""

    def __init__(
        self,
        *,
        gpu_id: int = 0,
        bitrate: int = 6_000_000,
        overlay_seconds: float = 12.0,
        overlay_lines: int = 4,
    ) -> None:
        self._config = CaptureConfig(
            profile="demo",
            gpu_id=gpu_id,
            bitrate=bitrate,
            overlay_seconds=overlay_seconds,
            overlay_lines=overlay_lines,
            max_total_bytes=0,
        )
        self._nvc = _load_nvc()
        self._frontend = _DemoCaptureFrontend(overlay_lines=overlay_lines)

    def render(self, bundle: Path) -> Path:
        """Render ``video/session.mp4`` and record it in the bundle manifest."""

        root = bundle.expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"capture bundle has no manifest: {root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("capture manifest must be an object")
        tracks = manifest.get("video_tracks")
        if not isinstance(tracks, dict):
            raise ValueError("capture manifest has no video tracks")
        segments = [
            segment
            for track in tracks.values()
            if isinstance(track, list)
            for segment in track
            if isinstance(segment, dict)
        ]
        if len(segments) != 1:
            raise ValueError("capture renderer requires one combined raw video segment")
        segment = segments[0]
        raw_path = _bundle_file(root, segment.get("raw_path"), label="raw video")
        packet_index_path = _bundle_file(
            root,
            segment.get("packet_index"),
            label="video packet index",
        )
        audio = manifest.get("audio")
        if not isinstance(audio, dict):
            raise ValueError("capture manifest has no audio description")
        wave_path = _bundle_file(root, audio.get("conversation"), label="conversation audio")
        events_path = _bundle_file(root, manifest.get("events"), label="event timeline")
        packet_rows = _read_json_lines(packet_index_path)
        if not packet_rows:
            raise ValueError("capture video packet index is empty")

        fps = float(segment.get("fps", 0))
        if fps <= 0:
            raise ValueError("capture video frame rate must be positive")
        self._config = CaptureConfig(
            profile="demo",
            sample_fps=fps,
            bitrate=self._config.bitrate,
            gpu_id=self._config.gpu_id,
            overlay_seconds=self._config.overlay_seconds,
            overlay_lines=self._config.overlay_lines,
            max_total_bytes=0,
        )
        dimensions = segment.get("source_dimensions")
        if not isinstance(dimensions, list) or not dimensions:
            raise ValueError("capture manifest has no source video dimensions")
        max_width = max(int(item["width"]) for item in dimensions)
        max_height = max(int(item["height"]) for item in dimensions)
        timeline = _OverlayTimeline(
            _read_json_lines(events_path),
            duration_us=round(self._config.overlay_seconds * 1_000_000),
        )
        writers: dict[str, _H264TrackWriter] = {}
        video_root = root / "video"
        existing_temporary_segments = set(video_root.glob("render_*.264"))
        try:
            for sequence, (row, pixels) in enumerate(
                self._decode_packets(
                    raw_path,
                    packet_rows,
                    max_width=max_width,
                    max_height=max_height,
                ),
                start=1,
            ):
                pts_us = int(row["pts_us"])
                track_id = str(row.get("track_id", "session"))
                writer = writers.get(track_id)
                if writer is None:
                    writer = _H264TrackWriter(
                        nvc=self._nvc,
                        root=video_root,
                        track_id=track_id,
                        stream_name="render",
                        config=self._config,
                        frontend=self._frontend,
                    )
                    writers[track_id] = writer
                height = pixels.shape[0] * 2 // 3
                width = pixels.shape[1]
                caption, data_feed = timeline.at(pts_us)
                writer.write(
                    FrameData(
                        seq=sequence,
                        pts_us=pts_us,
                        width=width,
                        height=height,
                        fmt=PixelFormat.NV12,
                        data=pixels.tobytes(),
                        participant_id=str(manifest.get("participant_id", "capture")),
                        track_id=track_id,
                    ),
                    caption,
                    data_feed,
                )
            for writer in writers.values():
                writer.close()
            sources = _video_sources(writers)
            if not sources:
                raise RuntimeError("capture renderer decoded no video frames")
            projection_path = video_root / "rendering.264"
            indexed_packets = _join_sources(root, sources, projection_path)
            packets = [indexed.packet for indexed in indexed_packets]
            display = max(
                (source for _, source in sources),
                key=lambda source: source["width"] * source["height"],
            )
            sample_rate = int(audio.get("sample_rate", 48_000))
            audio_start_frame = round(
                (int(segment["start_us"]) - int(manifest["start_us"]))
                * sample_rate
                / 1_000_000
            )
            audio_end_frame = round(
                (int(manifest["end_us"]) - int(manifest["start_us"]))
                * sample_rate
                / 1_000_000
            )
            artifact = self._frontend.finalize_video(
                session_root=root,
                raw_path=projection_path,
                wave_path=wave_path,
                audio_start_frame=audio_start_frame,
                audio_end_frame=audio_end_frame,
                fps=fps,
                packets=packets,
                width=int(display["width"]),
                height=int(display["height"]),
            )
        finally:
            for writer in writers.values():
                writer.close()
            for path in set(video_root.glob("render_*.264")) - existing_temporary_segments:
                path.unlink(missing_ok=True)
            (video_root / "rendering.264").unlink(missing_ok=True)

        renderings = manifest.setdefault("renderings", {})
        if not isinstance(renderings, dict):
            raise ValueError("capture manifest renderings must be an object")
        renderings["captioned_mp4"] = {
            "path": artifact.path,
            "size_bytes": artifact.size_bytes,
            "video_codec": "h264",
            "audio_codec": "aac_lc",
            "audio_embedded": artifact.audio_embedded,
            "width": int(display["width"]),
            "height": int(display["height"]),
            "source": segment["raw_path"],
            "overlay_seconds": self._config.overlay_seconds,
            "overlay_lines": self._config.overlay_lines,
        }
        pending_manifest = manifest_path.with_suffix(".json.pending")
        try:
            pending_manifest.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            pending_manifest.replace(manifest_path)
        except Exception:
            pending_manifest.unlink(missing_ok=True)
            raise
        return root / artifact.path

    def _decode_packets(
        self,
        raw_path: Path,
        rows: list[dict[str, Any]],
        *,
        max_width: int,
        max_height: int,
    ) -> Iterator[tuple[dict[str, Any], np.ndarray]]:
        decoder = self._nvc.CreateDecoder(
            gpuid=self._config.gpu_id,
            codec=self._nvc.cudaVideoCodec.H264,
            cudacontext=0,
            cudastream=0,
            usedevicememory=False,
            maxwidth=max_width,
            maxheight=max_height,
        )
        pending: deque[dict[str, Any]] = deque()
        file_size = raw_path.stat().st_size
        with raw_path.open("rb") as stream:
            for row in rows:
                offset = int(row.get("offset", -1))
                size = int(row.get("size_bytes", 0))
                if offset < 0 or size <= 0 or offset + size > file_size:
                    raise ValueError("capture video packet index contains an invalid byte range")
                stream.seek(offset)
                source = np.frombuffer(stream.read(size), dtype=np.uint8)
                packet = self._nvc.PacketData()
                packet.bsl = int(source.size)
                packet.bsl_data = int(source.ctypes.data)
                pending.append(row)
                for frame in decoder.Decode(packet):
                    if not pending:
                        raise RuntimeError("NVDEC emitted more frames than indexed packets")
                    yield pending.popleft(), _copy_decoded_frame(frame)
        end = self._nvc.PacketData()
        end.bsl = 0
        end.bsl_data = 0
        end.decode_flag = int(self._nvc.VideoPacketFlag.ENDOFSTREAM)
        for frame in decoder.Decode(end):
            if not pending:
                raise RuntimeError("NVDEC emitted more frames than indexed packets")
            yield pending.popleft(), _copy_decoded_frame(frame)
        if pending:
            raise RuntimeError(f"NVDEC omitted {len(pending)} indexed video frames")


__all__ = ["CaptureRenderer"]
