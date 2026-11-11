# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic but non-trivial test images, so vision latency can be swept without a camera."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw


def make_jpeg(width: int, height: int, seed: int = 0, quality: int = 85) -> bytes:
    """A gradient-plus-noise scene with labelled boxes, encoded as JPEG."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 255, width, dtype=np.float32)[None, :, None]
    y = np.linspace(0, 255, height, dtype=np.float32)[:, None, None]
    base = np.concatenate([np.broadcast_to(x, (height, width, 1)), np.broadcast_to(y, (height, width, 1)),
                           np.full((height, width, 1), 128.0, np.float32)], axis=2)
    base += rng.normal(0, 12, base.shape).astype(np.float32)
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for i in range(6):
        x0, y0 = int(rng.integers(0, width * 3 // 4)), int(rng.integers(0, height * 3 // 4))
        draw.rectangle([x0, y0, x0 + width // 6, y0 + height // 6], outline=(255, 255, 255), width=3)
        draw.text((x0 + 4, y0 + 4), f"gauge {i}", fill=(255, 255, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()
