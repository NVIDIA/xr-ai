# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy MCP PNG export over a caller-owned live frame source."""

import asyncio
from pathlib import Path

import numpy as np
from PIL import Image
from xr_ai_agent import FrameData, LiveFrameSource, PixelFormat


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in value)


def _yuv_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32) - 16
    cb = cb.astype(np.float32) - 128
    cr = cr.astype(np.float32) - 128
    rgb = np.stack(
        (
            1.164 * y + 1.596 * cr,
            1.164 * y - 0.392 * cb - 0.813 * cr,
            1.164 * y + 2.017 * cb,
        ),
        axis=-1,
    )
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _frame_to_rgb(frame: FrameData) -> np.ndarray:
    array = np.frombuffer(frame.data, dtype=np.uint8)
    if frame.fmt == PixelFormat.RGB24:
        return array.reshape(frame.height, frame.width, 3).copy()
    if frame.fmt == PixelFormat.RGBA:
        return array.reshape(frame.height, frame.width, 4)[:, :, :3].copy()
    if frame.fmt == PixelFormat.BGRA:
        return array.reshape(frame.height, frame.width, 4)[:, :, [2, 1, 0]].copy()
    if frame.fmt == PixelFormat.NV12:
        y = array[: frame.width * frame.height].reshape(frame.height, frame.width)
        uv = array[frame.width * frame.height :].reshape(
            frame.height // 2,
            frame.width // 2,
            2,
        )
        cb = np.repeat(np.repeat(uv[:, :, 0], 2, axis=0), 2, axis=1)
        cr = np.repeat(np.repeat(uv[:, :, 1], 2, axis=0), 2, axis=1)
        return _yuv_to_rgb(y, cb, cr)
    if frame.fmt == PixelFormat.I420:
        y_size = frame.width * frame.height
        uv_size = (frame.width // 2) * (frame.height // 2)
        y = array[:y_size].reshape(frame.height, frame.width)
        cb = array[y_size : y_size + uv_size].reshape(frame.height // 2, frame.width // 2)
        cr = array[y_size + uv_size :].reshape(frame.height // 2, frame.width // 2)
        return _yuv_to_rgb(
            y,
            np.repeat(np.repeat(cb, 2, axis=0), 2, axis=1),
            np.repeat(np.repeat(cr, 2, axis=0), 2, axis=1),
        )
    raise ValueError(f"Unsupported PixelFormat for PNG export: {frame.fmt!r}")


def _write_png(frame: FrameData, path: Path) -> None:
    Image.fromarray(_frame_to_rgb(frame), "RGB").save(path, "PNG")


class LiveFrameExporter:
    """Save a fresh raw hub frame in the legacy MCP PNG result shape."""

    def __init__(self, frames: LiveFrameSource, out_dir: Path) -> None:
        self._frames = frames
        self._out_dir = out_dir
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def participants(self) -> list[str]:
        """Return participants with a fresh camera frame."""
        return self._frames.participants()

    async def get_latest(self, participant_id: str) -> dict:
        """Write a fresh frame to PNG and return the legacy metadata shape."""
        frame = await self._frames.get(participant_id)
        path = self._out_dir / f"{_safe_name(participant_id)}_live_{frame.pts_us}.png"
        await asyncio.to_thread(_write_png, frame, path)
        return {
            "path": str(path),
            "width": frame.width,
            "height": frame.height,
            "timestamp_us": frame.pts_us,
        }


__all__ = ["LiveFrameExporter"]
