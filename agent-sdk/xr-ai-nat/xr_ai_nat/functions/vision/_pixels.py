# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert hub frames into JPEG data URLs accepted by VLM services."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image
from xr_ai_agent import FrameData, PixelFormat


def load_jpeg_data_url(image_path: str | Path, quality: int = 85) -> str:
    """Convert a local image to an RGB JPEG data URL."""

    with Image.open(image_path) as image:
        return encode_image(image.convert("RGB"), quality=quality)


def _yuv_to_rgb(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> Image.Image:
    y = y.astype(np.float32) - 16.0
    u = u.astype(np.float32) - 128.0
    v = v.astype(np.float32) - 128.0
    red = np.clip(1.164 * y + 1.596 * v, 0, 255)
    green = np.clip(1.164 * y - 0.392 * u - 0.813 * v, 0, 255)
    blue = np.clip(1.164 * y + 2.017 * u, 0, 255)
    return Image.fromarray(np.stack([red, green, blue], axis=-1).astype(np.uint8), "RGB")


def frame_to_pil(frame: FrameData) -> Image.Image:
    width, height = frame.width, frame.height
    data = np.frombuffer(frame.data, dtype=np.uint8)
    if frame.fmt == PixelFormat.RGB24:
        return Image.fromarray(data.reshape(height, width, 3), "RGB")
    if frame.fmt == PixelFormat.RGBA:
        return Image.fromarray(data.reshape(height, width, 4), "RGBA").convert("RGB")
    if frame.fmt == PixelFormat.BGRA:
        bgra = data.reshape(height, width, 4)
        return Image.fromarray(bgra[:, :, [2, 1, 0]], "RGB")
    y_end = width * height
    y = data[:y_end].reshape(height, width)
    if frame.fmt == PixelFormat.I420:
        uv_size = (width // 2) * (height // 2)
        u = data[y_end:y_end + uv_size].reshape(height // 2, width // 2).repeat(2, 0).repeat(2, 1)
        v = data[y_end + uv_size:].reshape(height // 2, width // 2).repeat(2, 0).repeat(2, 1)
        return _yuv_to_rgb(y, u, v)
    if frame.fmt == PixelFormat.NV12:
        uv = data[y_end:].reshape(height // 2, width)
        return _yuv_to_rgb(y, uv[:, 0::2].repeat(2, 0).repeat(2, 1), uv[:, 1::2].repeat(2, 0).repeat(2, 1))
    raise ValueError(f"Unsupported pixel format: {frame.fmt!r}")


def encode_image(image: Image.Image, quality: int = 90) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"
