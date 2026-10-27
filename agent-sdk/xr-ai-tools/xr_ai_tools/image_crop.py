# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crop selected images without interpreting their contents."""

from __future__ import annotations

import asyncio
import io
import math
from pathlib import Path

from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .image import ImageInput, ImageReference, ImageRegistry
from .tools import Tool
from .types import StrictRequest


class ImageRegion(BaseModel):
    """Axis-aligned rectangle in unmirrored, normalized image coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    left: float = Field(ge=0, le=1, allow_inf_nan=False)
    """Left edge as a fraction of source width."""
    top: float = Field(ge=0, le=1, allow_inf_nan=False)
    """Top edge as a fraction of source height."""
    right: float = Field(ge=0, le=1, allow_inf_nan=False)
    """Right edge as a fraction of source width."""
    bottom: float = Field(ge=0, le=1, allow_inf_nan=False)
    """Bottom edge as a fraction of source height."""

    @model_validator(mode="after")
    def _validate_area(self) -> ImageRegion:
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("image region must have positive area")
        return self


class ImageCropRequest(StrictRequest):
    """Select a rectangular part of an existing image."""

    image: ImageReference
    """Image to crop."""
    region: ImageRegion
    """Bounds in this image's coordinate system."""


class ImageCropResult(BaseModel):
    """A lossless cropped image and its pixel dimensions."""

    image: ImageReference
    """New image reference retaining the source owner."""
    width: int = Field(gt=0)
    """Cropped width in pixels."""
    height: int = Field(gt=0)
    """Cropped height in pixels."""


def _crop(source: ImageInput, region: ImageRegion) -> tuple[bytes, int, int]:
    if isinstance(source, str) and "://" in source:
        raise ValueError("image crop requires bytes or a local path")
    with Image.open(io.BytesIO(source) if isinstance(source, bytes) else Path(source)) as image:
        image = ImageOps.exif_transpose(image)
        box = (math.floor(region.left * image.width), math.floor(region.top * image.height),
               math.ceil(region.right * image.width), math.ceil(region.bottom * image.height))
        cropped = image.crop(box)
        output = io.BytesIO()
        cropped.save(output, "PNG")
        return output.getvalue(), cropped.width, cropped.height


class ImageCropTool(Tool[ImageCropRequest, ImageCropResult]):
    """Crop an image in its own coordinates; install the ``image-editing`` extra.

    Mapping another camera's region into this image requires calibration by the
    caller. This tool performs no alignment or inference.
    """

    def __init__(self, *, images: ImageRegistry) -> None:
        self._images = images
        super().__init__("crop_image", "Crop a normalized rectangle from a selected image.",
                         ImageCropRequest, ImageCropResult, self._crop_image)

    async def _crop_image(self, request: ImageCropRequest) -> ImageCropResult:
        source = self._images.resolve(request.image)
        data, width, height = await asyncio.to_thread(_crop, source, request.region)
        return ImageCropResult(
            image=self._images.put_derived(data, source=request.image), width=width, height=height,
        )


__all__ = ["ImageRegion", "ImageCropRequest", "ImageCropResult", "ImageCropTool"]
