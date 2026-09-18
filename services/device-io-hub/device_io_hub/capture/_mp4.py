# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""H.264/AAC MP4 finalization for participant capture bundles."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

_MP4_AUDIO_SAMPLE_RATE = 48_000
_MP4_AUDIO_CHANNELS = 2


def find_ffmpeg() -> str:
    """Return the FFmpeg executable required to finalize capture MP4s."""

    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError(
            "FFmpeg is required for media capture finalization. Install FFmpeg "
            "with the native AAC encoder and make `ffmpeg` available on PATH."
        )
    result = subprocess.run(
        [executable, "-hide_banner", "-loglevel", "quiet", "-encoders"],
        check=False,
        capture_output=True,
        text=True,
    )
    has_aac = any(
        len(fields) >= 2 and fields[0].startswith("A") and fields[1] == "aac"
        for line in result.stdout.splitlines()
        if (fields := line.split())
    )
    if result.returncode != 0 or not has_aac:
        raise RuntimeError(
            "FFmpeg media capture finalization requires the native AAC encoder."
        )
    return executable


def mux_h264_aac(
    *,
    ffmpeg_path: str,
    output_path: Path,
    video_path: Path,
    wave_path: Path,
    audio_start_frame: int,
    audio_end_frame: int,
    fps: float,
) -> None:
    """Write normalized H.264/AAC-LC MP4 with fast-start metadata."""

    if fps <= 0:
        raise ValueError("MP4 frame rate must be positive")
    if audio_start_frame < 0 or audio_end_frame < audio_start_frame:
        raise ValueError("invalid MP4 audio frame window")

    audio_filter = (
        f"[1:a:0]atrim=start_sample={audio_start_frame}:end_sample={audio_end_frame},"
        "asetpts=PTS-STARTPTS,"
        f"aresample={_MP4_AUDIO_SAMPLE_RATE}:async=1:first_pts=0,"
        "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
        "asetpts=N/SR/TB[audio]"
    )
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(wave_path),
        "-filter_complex",
        audio_filter,
        "-map",
        "0:v:0",
        "-map",
        "[audio]",
        "-c:v",
        "copy",
        "-tag:v",
        "avc1",
        "-vsync",
        "passthrough",
        "-copytb",
        "1",
        "-c:a",
        "aac",
        "-profile:a",
        "aac_low",
        "-b:a",
        "192k",
        "-ar:a",
        str(_MP4_AUDIO_SAMPLE_RATE),
        "-ac:a",
        str(_MP4_AUDIO_CHANNELS),
        "-video_track_timescale",
        "90000",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(output_path),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit status {result.returncode}"
        raise RuntimeError(f"FFmpeg MP4 finalization failed: {detail}")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("FFmpeg MP4 finalization produced no output")


__all__: list[str] = []
