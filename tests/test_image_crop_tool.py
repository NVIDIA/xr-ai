# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pixel-accurate coverage for normalized image cropping."""

import io

import pytest
from PIL import Image
from pydantic import ValidationError
from xr_ai_tools.image import ImageRegistry, NormalizedImageBox
from xr_ai_tools.image_crop import (
    ImageCropRequest,
    ImageCropTool,
    crop_image,
)


def _coordinate_png() -> bytes:
    image = Image.new("RGB", (8, 8))
    for y in range(image.height):
        for x in range(image.width):
            image.putpixel((x, y), (x, y, 0))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _open_png(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    image.load()
    return image


def test_crop_image_uses_deterministic_pixel_aligned_bounds() -> None:
    output, width, height, box = crop_image(
        _coordinate_png(),
        NormalizedImageBox(left=0.25, top=0.25, right=0.75, bottom=0.75),
    )

    with _open_png(output) as image:
        assert image.size == (4, 4)
        assert image.getpixel((0, 0)) == (2, 2, 0)
        assert image.getpixel((3, 3)) == (5, 5, 0)
    assert (width, height) == (4, 4)
    assert box == NormalizedImageBox(left=0.25, top=0.25, right=0.75, bottom=0.75)


def test_crop_image_clamps_padding_and_reports_applied_bounds() -> None:
    output, width, height, box = crop_image(
        _coordinate_png(),
        NormalizedImageBox(left=0.0, top=0.0, right=0.25, bottom=0.25),
        padding=0.25,
    )

    with _open_png(output) as image:
        assert image.size == (4, 4)
        assert image.getpixel((3, 3)) == (3, 3, 0)
    assert (width, height) == (4, 4)
    assert box == NormalizedImageBox(left=0.0, top=0.0, right=0.5, bottom=0.5)


def test_normalized_image_box_rejects_empty_non_finite_or_extra_values() -> None:
    with pytest.raises(ValidationError, match="left must be less than right"):
        NormalizedImageBox(left=0.5, top=0.1, right=0.5, bottom=0.9)
    with pytest.raises(ValidationError):
        NormalizedImageBox(left=0.1, top=0.1, right=float("nan"), bottom=0.9)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        NormalizedImageBox(left=0.1, top=0.1, right=0.9, bottom=0.9, units="px")


async def test_crop_tool_returns_a_derived_image_with_source_ownership() -> None:
    images = ImageRegistry()
    source = images.put(_coordinate_png(), owner="alice")
    tool = ImageCropTool(images=images)

    result = await tool.execute(
        ImageCropRequest(
            image=source,
            box=NormalizedImageBox(left=0.25, top=0.25, right=0.75, bottom=0.75),
        )
    )

    assert result.available is True
    assert result.image is not None
    assert result.image != source
    assert (result.width, result.height) == (4, 4)
    cropped = images.resolve(result.image)
    assert isinstance(cropped, bytes)

    images.release_owner("alice")
    with pytest.raises(LookupError, match="unavailable"):
        images.resolve(result.image)


async def test_crop_tool_returns_unavailable_for_a_stale_image() -> None:
    images = ImageRegistry(capacity=1)
    stale = images.put(_coordinate_png())
    images.put(_coordinate_png())
    tool = ImageCropTool(images=images)

    result = await tool.execute(
        ImageCropRequest(
            image=stale,
            box=NormalizedImageBox(left=0.1, top=0.1, right=0.9, bottom=0.9),
        )
    )

    assert result.available is False
    assert result.image is None
    assert result.message == "Image input unavailable — please select it again."
