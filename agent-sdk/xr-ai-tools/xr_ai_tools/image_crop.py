# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministically crop registered images with normalized rectangles."""

from __future__ import annotations

import asyncio
import io
import logging
import math
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image
from pydantic import BaseModel, Field

from .image import ImageInput, ImageReference, ImageRegistry, NormalizedImageBox
from .tools import Tool
from .types import StrictRequest

_LOGGER = logging.getLogger(__name__)


class ImageCropRequest(StrictRequest):
    """Select a normalized rectangular region from one registered image."""

    image: ImageReference = Field(
        description="Source image returned by an image tool or supplied by the caller."
    )
    """Source image returned by an image tool or supplied by the caller."""

    box: NormalizedImageBox
    """Normalized region to crop from the source image."""

    padding: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        allow_inf_nan=False,
        description="Optional normalized padding added on every side.",
    )
    """Normalized image fraction added on every side before clamping."""


class ImageCropResult(BaseModel):
    """A derived lossless PNG and the exact source region used to create it."""

    image: ImageReference | None = None
    """Opaque reference to the cropped PNG when available."""

    box: NormalizedImageBox | None = None
    """Pixel-aligned normalized source bounds actually used for the crop."""

    width: int | None = Field(default=None, gt=0)
    """Width of the cropped image in pixels, when available."""

    height: int | None = Field(default=None, gt=0)
    """Height of the cropped image in pixels, when available."""

    available: bool = True
    """Whether the supplied image and box produced a derived image."""

    message: str | None = None
    """Recoverable failure detail when no crop was produced."""


def crop_image(
    source: ImageInput,
    box: NormalizedImageBox,
    *,
    padding: float = 0.0,
) -> tuple[bytes, int, int, NormalizedImageBox]:
    """Return lossless PNG bytes cropped to deterministic pixel-aligned bounds."""

    if not math.isfinite(padding) or not 0.0 <= padding <= 0.5:
        raise ValueError("padding must be finite and between 0 and 0.5")
    image = _open_image(source)
    left = max(0.0, box.left - padding)
    top = max(0.0, box.top - padding)
    right = min(1.0, box.right + padding)
    bottom = min(1.0, box.bottom + padding)

    left_px = min(math.floor(left * image.width), image.width - 1)
    top_px = min(math.floor(top * image.height), image.height - 1)
    right_px = max(left_px + 1, min(math.ceil(right * image.width), image.width))
    bottom_px = max(top_px + 1, min(math.ceil(bottom * image.height), image.height))

    cropped = image.crop((left_px, top_px, right_px, bottom_px))
    output = io.BytesIO()
    cropped.save(output, format="PNG")
    applied_box = NormalizedImageBox(
        left=left_px / image.width,
        top=top_px / image.height,
        right=right_px / image.width,
        bottom=bottom_px / image.height,
    )
    return output.getvalue(), cropped.width, cropped.height, applied_box


class ImageCropTool(Tool[ImageCropRequest, ImageCropResult]):
    """Create a derived image from a normalized rectangular selection."""

    def __init__(self, *, images: ImageRegistry) -> None:
        self.images = images
        super().__init__(
            "crop_image",
            (
                "Crop an image reference to a normalized rectangle and return a "
                "derived image reference for focused inspection or OCR."
            ),
            ImageCropRequest,
            ImageCropResult,
            self._crop,
        )

    async def _crop(self, request: ImageCropRequest) -> ImageCropResult:
        try:
            source = self.images.resolve(request.image)
        except (LookupError, ValueError) as error:
            _LOGGER.warning("Image input could not be resolved: %s", error)
            return ImageCropResult(
                available=False,
                message="Image input unavailable — please select it again.",
            )

        try:
            output, width, height, box = await asyncio.to_thread(
                crop_image,
                source,
                request.box,
                padding=request.padding,
            )
        except (OSError, ValueError) as error:
            _LOGGER.warning("Image could not be cropped: %s", error)
            return ImageCropResult(
                available=False,
                message=f"Image crop unavailable — {error}",
            )

        try:
            image = self.images.put_derived(output, source=request.image)
        except (LookupError, ValueError) as error:
            _LOGGER.warning("Source image was released during crop: %s", error)
            return ImageCropResult(
                available=False,
                message="Image input unavailable — please select it again.",
            )
        return ImageCropResult(image=image, box=box, width=width, height=height)


def _open_image(source: ImageInput) -> Image.Image:
    if isinstance(source, bytes):
        opened = Image.open(io.BytesIO(source))
    else:
        if isinstance(source, str):
            parsed = urlsplit(source)
            if parsed.scheme:
                raise ValueError("image crop requires bytes or a local image path")
        opened = Image.open(Path(source))

    with opened:
        opened.load()
        mode = "RGBA" if "A" in opened.getbands() or "transparency" in opened.info else "RGB"
        return opened.convert(mode)


__all__ = ["ImageCropRequest", "ImageCropResult", "ImageCropTool", "crop_image"]
