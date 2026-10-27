# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode inline images without granting the service URL or filesystem access."""

from __future__ import annotations

import base64
import io
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

from PIL import Image


class InputTooLongError(ValueError):
    """A request needs more model input than the configured ``max_length`` allows."""


_PREFIXES = {"data:image/jpeg;base64,": "JPEG", "data:image/png;base64,": "PNG"}


@dataclass(frozen=True)
class _ImageLimits:
    pixels_per_token: int
    min_image_tokens: int

    @classmethod
    def from_processor(cls, processor: Any) -> _ImageLimits:
        """Read admission geometry from the processor that will resize images."""
        image_processor = processor.image_processor
        values = (
            image_processor.patch_size,
            image_processor.merge_size,
            image_processor.size["shortest_edge"],
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError("Clef image processor requires positive integer patch, merge, and minimum-pixel sizes")
        patch, merge, min_pixels = values
        pixels_per_token = (patch * merge) ** 2
        return cls(pixels_per_token, max(1, min_pixels // pixels_per_token))


def validate_images(images: Any) -> None:
    if not isinstance(images, list):
        raise ValueError("images must be a list of inline JPEG or PNG data URLs")
    for image in images:
        if not isinstance(image, str) or not any(image.startswith(prefix) for prefix in _PREFIXES):
            raise ValueError("images must contain only inline JPEG or PNG base64 data URLs")


def decode_images(images: list[str], *, max_length: int, limits: _ImageLimits) -> list[Image.Image]:
    """Bound aggregate source pixels by the loaded processor's context budget.

    This conservative limit applies before resizing or allocating RGB buffers;
    the subsequent multimodal token check includes text and question tokens.
    The caller owns and closes the returned detached RGB images.
    """
    # Even tiny images consume the processor's minimum resized pixel area.
    if len(images) > max_length // limits.min_image_tokens:
        raise InputTooLongError("too many images for max_length")
    decoded: list[Image.Image] = []
    remaining_pixels = max_length * limits.pixels_per_token
    try:
        with ExitStack() as cleanup:
            for image in images:
                prefix, payload = image.split(",", 1)
                expected_format = _PREFIXES[prefix + ","]
                raw = base64.b64decode(payload, validate=True)
                with Image.open(io.BytesIO(raw), formats=[expected_format]) as source:
                    remaining_pixels -= source.width * source.height
                    if remaining_pixels < 0:
                        raise InputTooLongError("images exceed the aggregate source-pixel budget for max_length")
                    source.load()
                    rgb = source.convert("RGB")
                    cleanup.callback(rgb.close)
                    decoded.append(rgb)
            # Ownership transfers only after the complete list has decoded.
            cleanup.pop_all()
        return decoded
    except InputTooLongError:
        raise
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError(f"invalid images: {exc}") from exc
