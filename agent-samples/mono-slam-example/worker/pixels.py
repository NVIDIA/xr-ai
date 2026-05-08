# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hub FrameData → numpy pixel arrays for visual odometry.

NV12 and I420 frames use only the luma (Y) plane for grayscale output.
For RGB output (required by DPVO), NV12/I420 are converted via YUV→BGR→RGB.
"""
from __future__ import annotations

import cv2
import numpy as np

from xr_ai_agent import FrameData, PixelFormat


def frame_to_rgb(frame: FrameData) -> np.ndarray:
    """Convert a FrameData to a (H, W, 3) uint8 RGB array.

    Required by DPVO, which expects a 3-channel RGB tensor.

    Args:
        frame: FrameData from ProcessorEndpoint.request_frame().

    Returns:
        RGB image as (height, width, 3) uint8 ndarray.

    Raises:
        ValueError: for unrecognised pixel formats.
    """
    w, h = frame.width, frame.height
    arr = np.frombuffer(frame.data, dtype=np.uint8)

    if frame.fmt == PixelFormat.NV12:
        yuv = arr[: w * h * 3 // 2].reshape(h * 3 // 2, w)
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if frame.fmt == PixelFormat.I420:
        yuv = arr[: w * h * 3 // 2].reshape(h * 3 // 2, w)
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if frame.fmt == PixelFormat.RGB24:
        return arr.reshape(h, w, 3).copy()

    if frame.fmt == PixelFormat.RGBA:
        rgba = arr.reshape(h, w, 4)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)

    if frame.fmt == PixelFormat.BGRA:
        bgra = arr.reshape(h, w, 4)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)

    raise ValueError(f"Unsupported pixel format: {frame.fmt!r}")


def frame_to_gray(frame: FrameData) -> np.ndarray:
    """Convert a FrameData to a 2D uint8 grayscale array (H, W).

    Args:
        frame: FrameData from ProcessorEndpoint.request_frame().

    Returns:
        Grayscale image as a (height, width) uint8 ndarray.

    Raises:
        ValueError: for unrecognised pixel formats.
    """
    w, h = frame.width, frame.height
    arr = np.frombuffer(frame.data, dtype=np.uint8)

    if frame.fmt in (PixelFormat.NV12, PixelFormat.I420):
        # Luma plane is the first w*h bytes — already grayscale.
        return arr[: w * h].reshape(h, w).copy()

    if frame.fmt == PixelFormat.RGB24:
        rgb = arr.reshape(h, w, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if frame.fmt == PixelFormat.RGBA:
        rgba = arr.reshape(h, w, 4)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)

    if frame.fmt == PixelFormat.BGRA:
        bgra = arr.reshape(h, w, 4)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2GRAY)

    raise ValueError(f"Unsupported pixel format: {frame.fmt!r}")
