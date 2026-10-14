# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode inline images without granting the service URL or filesystem access."""

from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image

# The pinned Qwen3.5 processor uses 16-pixel patches and a 2x2 spatial merge.
_PIXELS_PER_TOKEN = (16 * 2) ** 2
_MIN_IMAGE_TOKENS = 65536 // _PIXELS_PER_TOKEN
_PREFIXES = {"data:image/jpeg;base64,": "JPEG", "data:image/png;base64,": "PNG"}


def validate_images(images: Any) -> None:
    if not isinstance(images, list):
        raise ValueError("images must be a list of inline JPEG or PNG data URLs")
    for image in images:
        if not isinstance(image, str) or not any(image.startswith(prefix) for prefix in _PREFIXES):
            raise ValueError("images must contain only inline JPEG or PNG base64 data URLs")


def decode_images(images: list[str], *, max_length: int) -> list[Image.Image]:
    """Bound aggregate source pixels by the pinned processor's context budget.

    This conservative limit applies before resizing or allocating RGB buffers;
    the subsequent multimodal token check includes text and question tokens.
    The caller owns and closes the returned detached RGB images.
    """
    # Tiny images are resized up to the pinned processor's 65,536-pixel minimum.
    if len(images) > max_length // _MIN_IMAGE_TOKENS:
        raise ValueError("too many images for max_length")
    decoded: list[Image.Image] = []
    remaining_pixels = max_length * _PIXELS_PER_TOKEN
    try:
        for image in images:
            prefix, payload = image.split(",", 1)
            expected_format = _PREFIXES[prefix + ","]
            raw = base64.b64decode(payload, validate=True)
            with Image.open(io.BytesIO(raw), formats=[expected_format]) as source:
                remaining_pixels -= source.width * source.height
                if remaining_pixels < 0:
                    raise ValueError("images exceed the aggregate source-pixel budget for max_length")
                source.load()
                decoded.append(source.convert("RGB"))
        return decoded
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        for image in decoded:
            image.close()
        raise ValueError(f"invalid images: {exc}") from exc
