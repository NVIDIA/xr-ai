# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pixel-accurate coverage for the magenta polygon-fill image tool."""

import io

import pytest
from PIL import Image
from pydantic import ValidationError
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.image_polygon import (
    ImagePoint,
    ImagePolygonFillRequest,
    ImagePolygonFillTool,
    fill_polygon_magenta,
)


def _png_bytes(*, mode: str = "RGB") -> bytes:
    color = (10, 20, 30, 120) if mode == "RGBA" else (10, 20, 30)
    image = Image.new(mode, (8, 8), color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _read_pixel(image: bytes, coordinate: tuple[int, int]):
    with Image.open(io.BytesIO(image)) as opened:
        return opened.getpixel(coordinate)


def test_fill_polygon_magenta_preserves_pixels_outside_the_polygon() -> None:
    source = _png_bytes()
    output = fill_polygon_magenta(
        source,
        [
            ImagePoint(x=2, y=2),
            ImagePoint(x=5, y=2),
            ImagePoint(x=5, y=5),
            ImagePoint(x=2, y=5),
        ],
    )

    assert _read_pixel(output, (3, 3)) == (255, 0, 255)
    assert _read_pixel(output, (0, 0)) == (10, 20, 30)
    assert _read_pixel(source, (3, 3)) == (10, 20, 30)


def test_fill_polygon_magenta_makes_rgba_fill_opaque() -> None:
    output = fill_polygon_magenta(
        _png_bytes(mode="RGBA"),
        [
            ImagePoint(x=1, y=1),
            ImagePoint(x=6, y=1),
            ImagePoint(x=3, y=6),
        ],
    )

    assert _read_pixel(output, (3, 3)) == (255, 0, 255, 255)
    assert _read_pixel(output, (0, 0)) == (10, 20, 30, 120)


def test_fill_polygon_magenta_accepts_fractional_detector_coordinates() -> None:
    output = fill_polygon_magenta(
        _png_bytes(),
        [
            ImagePoint(x=1.25, y=1.5),
            ImagePoint(x=6.5, y=1.25),
            ImagePoint(x=3.75, y=6.5),
        ],
    )

    assert _read_pixel(output, (3, 3)) == (255, 0, 255)


@pytest.mark.parametrize(
    "coordinates",
    [
        [ImagePoint(x=0, y=0), ImagePoint(x=1, y=1)],
        [ImagePoint(x=0, y=0), ImagePoint(x=1, y=1), ImagePoint(x=2, y=2)],
        [ImagePoint(x=0, y=0), ImagePoint(x=7, y=0), ImagePoint(x=8, y=7)],
    ],
)
def test_polygon_validation_rejects_invalid_coordinates(coordinates) -> None:
    if len(coordinates) < 3:
        with pytest.raises(ValidationError, match="at least 3"):
            ImagePolygonFillRequest(
                image={"uri": "xr-image://source"},
                coordinates=coordinates,
            )
        return

    with pytest.raises(ValueError, match="non-zero area|outside"):
        fill_polygon_magenta(_png_bytes(), coordinates)


def test_image_point_rejects_negative_or_extra_values() -> None:
    with pytest.raises(ValidationError):
        ImagePoint(x=-1, y=0)
    with pytest.raises(ValidationError):
        ImagePoint(x=1, y=2, z=3)


async def test_image_polygon_fill_tool_returns_a_new_registry_image() -> None:
    images = ImageRegistry()
    source_bytes = _png_bytes()
    source = images.put(source_bytes)
    tool = ImagePolygonFillTool(images=images)

    result = await tool.execute(
        ImagePolygonFillRequest(
            image=source,
            coordinates=[
                ImagePoint(x=1, y=1),
                ImagePoint(x=6, y=1),
                ImagePoint(x=6, y=6),
                ImagePoint(x=1, y=6),
            ],
        )
    )

    assert result.image != source
    edited = images.resolve(result.image)
    assert isinstance(edited, bytes)
    assert _read_pixel(edited, (4, 4)) == (255, 0, 255)
    assert images.resolve(source) == source_bytes
