# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hub FrameData → grayscale numpy array for visual odometry.

NV12 and I420 frames use only the luma (Y) plane — no color conversion
needed.  RGB/RGBA/BGRA frames are converted with cv2.cvtColor.
"""
from __future__ import annotations

import cv2
import numpy as np

from xr_ai_agent import FrameData, PixelFormat


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
