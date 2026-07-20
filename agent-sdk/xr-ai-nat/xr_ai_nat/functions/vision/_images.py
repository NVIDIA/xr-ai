# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image normalization for VLM inputs."""

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np


def load_jpeg_data_url(image_path: str | Path, quality: int = 85) -> str:
    """Convert a local image to an RGB JPEG data URL."""

    from PIL import Image

    with Image.open(image_path) as image:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def frame_jpeg_data_url(frame: Any, quality: int = 90) -> str:
    """Convert one hub frame to a JPEG data URL accepted by VLM services."""

    from PIL import Image
    from xr_ai_agent import PixelFormat

    width, height = frame.width, frame.height
    pixels = np.frombuffer(frame.data, dtype=np.uint8)

    if frame.fmt == PixelFormat.RGB24:
        image = Image.fromarray(pixels.reshape(height, width, 3), "RGB")
    elif frame.fmt == PixelFormat.RGBA:
        image = Image.fromarray(pixels.reshape(height, width, 4), "RGBA").convert("RGB")
    elif frame.fmt == PixelFormat.BGRA:
        bgra = pixels.reshape(height, width, 4)
        image = Image.fromarray(bgra[:, :, [2, 1, 0]], "RGB")
    elif frame.fmt in {PixelFormat.I420, PixelFormat.NV12}:
        image = _yuv_frame_to_rgb(frame, pixels)
    else:
        raise ValueError(f"Unsupported pixel format: {frame.fmt!r}")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _yuv_frame_to_rgb(frame: Any, pixels: np.ndarray):
    from PIL import Image
    from xr_ai_agent import PixelFormat

    width, height = frame.width, frame.height
    y_end = width * height
    luminance = pixels[:y_end].reshape(height, width)
    if frame.fmt == PixelFormat.I420:
        chroma_size = (width // 2) * (height // 2)
        blue = pixels[y_end : y_end + chroma_size].reshape(height // 2, width // 2)
        red = pixels[y_end + chroma_size :].reshape(height // 2, width // 2)
    else:
        chroma = pixels[y_end:].reshape(height // 2, width)
        blue = chroma[:, 0::2]
        red = chroma[:, 1::2]

    blue = blue.repeat(2, axis=0).repeat(2, axis=1)
    red = red.repeat(2, axis=0).repeat(2, axis=1)
    y = luminance.astype(np.float32) - 16.0
    u = blue.astype(np.float32) - 128.0
    v = red.astype(np.float32) - 128.0
    rgb = np.stack(
        [
            np.clip(1.164 * y + 1.596 * v, 0, 255),
            np.clip(1.164 * y - 0.392 * u - 0.813 * v, 0, 255),
            np.clip(1.164 * y + 2.017 * u, 0, 255),
        ],
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")
