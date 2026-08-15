# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fill an image-space polygon with opaque magenta pixels."""

from __future__ import annotations

import asyncio
import io
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field

from .image import ImageInput, ImageReference, ImageRegistry
from .tools import Tool
from .types import StrictRequest

_MAGENTA_RGB = (255, 0, 255)
_LOGGER = logging.getLogger(__name__)


class ImagePoint(BaseModel):
    """One non-negative source-image pixel coordinate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float = Field(
        ge=0,
        allow_inf_nan=False,
        description="Horizontal pixel coordinate from the left edge.",
    )
    y: float = Field(
        ge=0,
        allow_inf_nan=False,
        description="Vertical pixel coordinate from the top edge.",
    )


class ImagePolygonFillRequest(StrictRequest):
    """Fill the polygon defined by ordered image-space vertices."""

    image: ImageReference = Field(description="Source image returned by an image tool or supplied by the caller.")
    coordinates: list[ImagePoint] = Field(
        min_length=3,
        description="Ordered polygon vertices; the final vertex is connected to the first.",
    )


class ImagePolygonFillResult(BaseModel):
    """A lossless PNG copy containing the filled polygon."""

    image: ImageReference | None = Field(
        default=None,
        description="Opaque reference to the edited PNG image when available.",
    )
    available: bool = Field(
        default=True,
        description="Whether the supplied image and polygon produced an edited image.",
    )
    message: str | None = Field(
        default=None,
        description="Recoverable failure detail when no edited image was produced.",
    )


def _open_image(source: ImageInput) -> Image.Image:
    if isinstance(source, bytes):
        stream = io.BytesIO(source)
        opened = Image.open(stream)
    else:
        if isinstance(source, str):
            parsed = urlsplit(source)
            if parsed.scheme:
                raise ValueError("image polygon fill requires bytes or a local image path")
        opened = Image.open(Path(source))

    with opened:
        opened.load()
        mode = "RGBA" if "A" in opened.getbands() or "transparency" in opened.info else "RGB"
        return opened.convert(mode)


def _validate_polygon(image: Image.Image, coordinates: Sequence[ImagePoint]) -> None:
    if len(coordinates) < 3:
        raise ValueError("at least three polygon coordinates are required")

    for point in coordinates:
        if point.x >= image.width or point.y >= image.height:
            raise ValueError(
                f"polygon coordinate ({point.x}, {point.y}) is outside "
                f"the {image.width}x{image.height} image"
            )

    area_twice = sum(
        point.x * following.y - following.x * point.y
        for point, following in zip(
            coordinates,
            (*coordinates[1:], coordinates[0]),
            strict=True,
        )
    )
    if math.isclose(area_twice, 0.0, abs_tol=1e-9):
        raise ValueError("polygon coordinates must enclose a non-zero area")


def fill_polygon_magenta(
    source: ImageInput,
    coordinates: Sequence[ImagePoint],
) -> bytes:
    """Return lossless PNG bytes with the supplied polygon filled ``#FF00FF``."""

    image = _open_image(source)
    _validate_polygon(image, coordinates)
    fill = (*_MAGENTA_RGB, 255) if image.mode == "RGBA" else _MAGENTA_RGB
    ImageDraw.Draw(image).polygon(
        [(point.x, point.y) for point in coordinates],
        fill=fill,
    )
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class ImagePolygonFillTool(Tool[ImagePolygonFillRequest, ImagePolygonFillResult]):
    """Create a magenta polygon overlay without changing the source image."""

    def __init__(self, *, images: ImageRegistry) -> None:
        self.images = images
        super().__init__(
            "fill_image_polygon",
            "Fill a polygon of at least three ordered pixel coordinates with magenta and return a new image.",
            ImagePolygonFillRequest,
            ImagePolygonFillResult,
            self._fill,
        )

    async def _fill(self, request: ImagePolygonFillRequest) -> ImagePolygonFillResult:
        try:
            source = self.images.resolve(request.image)
        except (LookupError, ValueError) as error:
            _LOGGER.warning("Image input could not be resolved: %s", error)
            return ImagePolygonFillResult(
                available=False,
                message="Image input unavailable — please select it again.",
            )

        try:
            output = await asyncio.to_thread(
                fill_polygon_magenta,
                source,
                request.coordinates,
            )
        except (OSError, ValueError) as error:
            _LOGGER.warning("Image polygon could not be filled: %s", error)
            return ImagePolygonFillResult(
                available=False,
                message=f"Image polygon invalid — {error}",
            )

        try:
            image = self.images.put_derived(output, source=request.image)
        except (LookupError, ValueError) as error:
            _LOGGER.warning("Source image was released during polygon fill: %s", error)
            return ImagePolygonFillResult(
                available=False,
                message="Image input unavailable — please select it again.",
            )
        return ImagePolygonFillResult(image=image)


__all__ = [
    "ImagePoint",
    "ImagePolygonFillRequest",
    "ImagePolygonFillResult",
    "ImagePolygonFillTool",
    "fill_polygon_magenta",
]
