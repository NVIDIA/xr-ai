# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private video projections over the shared timestamped capture session."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from xr_ai_hub import FrameData

from device_io_hub.video._recorder import _to_nv12

from ._compositor import compose_caption
from ._matroska import VideoPacket, mux_h264
from ._mp4 import find_ffmpeg, mux_h264_aac


@dataclass(frozen=True, slots=True)
class _RenderedFrame:
    pixels: np.ndarray
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class _VideoArtifact:
    path: str
    size_bytes: int
    audio_embedded: bool


class _CaptureFrontend(Protocol):
    name: str

    def render_frame(
        self,
        frame: FrameData,
        *,
        caption: str,
        data_feed: tuple[str, ...],
    ) -> _RenderedFrame: ...

    def finalize_video(
        self,
        *,
        session_root: Path,
        raw_path: Path,
        wave_path: Path,
        audio_start_frame: int,
        audio_end_frame: int,
        fps: float,
        packets: Sequence[VideoPacket],
        width: int,
        height: int,
    ) -> _VideoArtifact: ...


class _DemoCaptureFrontend:
    """Render captions and produce the human-viewable session MP4."""

    name = "demo"

    def __init__(self, *, overlay_lines: int) -> None:
        self._overlay_lines = overlay_lines
        self._ffmpeg_path = find_ffmpeg()

    def render_frame(
        self,
        frame: FrameData,
        *,
        caption: str,
        data_feed: tuple[str, ...],
    ) -> _RenderedFrame:
        pixels, width, height = compose_caption(
            frame,
            caption,
            data_feed=data_feed,
            max_lines=self._overlay_lines,
        )
        return _RenderedFrame(pixels=pixels, width=width, height=height)

    def finalize_video(
        self,
        *,
        session_root: Path,
        raw_path: Path,
        wave_path: Path,
        audio_start_frame: int,
        audio_end_frame: int,
        fps: float,
        packets: Sequence[VideoPacket],
        width: int,
        height: int,
    ) -> _VideoArtifact:
        relative_path = "video/session.mp4"
        output_path = session_root / relative_path
        pending_path = output_path.with_suffix(".mp4.pending")
        timeline_path = session_root / "video" / "session.timeline.mkv.pending"
        try:
            mux_h264(
                output_path=timeline_path,
                h264_path=raw_path,
                packets=packets,
                width=width,
                height=height,
                fps=fps,
            )
            mux_h264_aac(
                ffmpeg_path=self._ffmpeg_path,
                output_path=pending_path,
                video_path=timeline_path,
                wave_path=wave_path,
                audio_start_frame=audio_start_frame,
                audio_end_frame=audio_end_frame,
                fps=fps,
            )
            pending_path.replace(output_path)
        except Exception:
            pending_path.unlink(missing_ok=True)
            raise
        finally:
            timeline_path.unlink(missing_ok=True)
        return _VideoArtifact(
            path=relative_path,
            size_bytes=output_path.stat().st_size,
            audio_embedded=True,
        )


class _RawCaptureFrontend:
    """Preserve camera pixels and leave media as separately indexed artifacts."""

    name = "raw"

    def render_frame(
        self,
        frame: FrameData,
        *,
        caption: str,
        data_feed: tuple[str, ...],
    ) -> _RenderedFrame:
        del caption, data_feed
        if frame.width % 2 or frame.height % 2:
            raise ValueError("NV12 capture requires even video dimensions")
        pixels = _to_nv12(frame.data, frame.width, frame.height, frame.fmt)
        if pixels is None:
            raise ValueError(f"unsupported capture pixel format: {frame.fmt!r}")
        return _RenderedFrame(
            pixels=np.ascontiguousarray(pixels),
            width=frame.width,
            height=frame.height,
        )

    def finalize_video(
        self,
        *,
        session_root: Path,
        raw_path: Path,
        wave_path: Path,
        audio_start_frame: int,
        audio_end_frame: int,
        fps: float,
        packets: Sequence[VideoPacket],
        width: int,
        height: int,
    ) -> _VideoArtifact:
        del (
            session_root,
            wave_path,
            audio_start_frame,
            audio_end_frame,
            fps,
            packets,
            width,
            height,
        )
        return _VideoArtifact(
            path="video/session.264",
            size_bytes=raw_path.stat().st_size,
            audio_embedded=False,
        )


def _make_frontend(*, profile: str, overlay_lines: int) -> _CaptureFrontend:
    if profile == "demo":
        return _DemoCaptureFrontend(overlay_lines=overlay_lines)
    if profile == "raw":
        return _RawCaptureFrontend()
    raise ValueError(f"unsupported capture profile: {profile!r}")


__all__: list[str] = []
