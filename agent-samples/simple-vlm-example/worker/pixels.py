"""Pixel-format conversion: hub FrameData → PIL Image → JPEG data URL.

The hub may deliver frames in any of several pixel formats depending on the
client and codec.  We convert them all to RGB PIL Images for the VLM API.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from xr_ai_agent import FrameData, PixelFormat


def _yuv_to_rgb(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> Image.Image:
    """BT.601 limited-range YCbCr → RGB.  U/V must already be full-size (upsampled)."""
    Y = Y.astype(np.float32) - 16.0
    U = U.astype(np.float32) - 128.0
    V = V.astype(np.float32) - 128.0
    R = np.clip(1.164 * Y               + 1.596 * V, 0, 255)
    G = np.clip(1.164 * Y - 0.392 * U  - 0.813 * V, 0, 255)
    B = np.clip(1.164 * Y + 2.017 * U,              0, 255)
    return Image.fromarray(np.stack([R, G, B], axis=-1).astype(np.uint8), "RGB")


def frame_to_pil(frame: FrameData) -> Image.Image:
    w, h = frame.width, frame.height
    arr  = np.frombuffer(frame.data, dtype=np.uint8)

    if frame.fmt == PixelFormat.RGB24:
        return Image.fromarray(arr.reshape(h, w, 3), "RGB")

    if frame.fmt == PixelFormat.RGBA:
        return Image.fromarray(arr.reshape(h, w, 4), "RGBA").convert("RGB")

    if frame.fmt == PixelFormat.BGRA:
        a = arr.reshape(h, w, 4)
        return Image.fromarray(a[:, :, [2, 1, 0]], "RGB")

    if frame.fmt == PixelFormat.I420:
        y_end = w * h
        uv_sz = (w // 2) * (h // 2)
        Y = arr[:y_end].reshape(h, w)
        U = arr[y_end : y_end + uv_sz].reshape(h // 2, w // 2).repeat(2, 0).repeat(2, 1)
        V = arr[y_end + uv_sz :].reshape(h // 2, w // 2).repeat(2, 0).repeat(2, 1)
        return _yuv_to_rgb(Y, U, V)

    if frame.fmt == PixelFormat.NV12:
        y_end = w * h
        Y  = arr[:y_end].reshape(h, w)
        uv = arr[y_end:].reshape(h // 2, w)
        U  = uv[:, 0::2].repeat(2, 0).repeat(2, 1)
        V  = uv[:, 1::2].repeat(2, 0).repeat(2, 1)
        return _yuv_to_rgb(Y, U, V)

    raise ValueError(f"Unsupported pixel format: {frame.fmt!r}")


def encode_image(image: Image.Image) -> str:
    """PIL Image → JPEG data URL for the vlm-server API."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"
