# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit still-image acquisition, independent of streaming video."""

from __future__ import annotations

import asyncio
import io

from PIL import Image, ImageOps
from pydantic import Field
from xr_ai_hub import ClientImageCaptureSource, ImageCaptureUnavailable, ProcessorEndpoint

from .image import ImageRegistry, TimedImage
from .tools import Tool
from .types import StrictRequest


class StillCaptureRequest(StrictRequest):
    """Request a new photograph from one participant's opted-in camera."""

    participant_id: str = Field(min_length=1)
    """Participant whose device should capture the still image."""


class StillImage(TimedImage):
    """An encoded still image with its actual dimensions."""

    width: int = Field(gt=0)
    """Width in pixels after normalizing EXIF orientation."""

    height: int = Field(gt=0)
    """Height in pixels after normalizing EXIF orientation."""

    participant_id: str = Field(min_length=1)
    """Participant that supplied the image."""


def _prepare_image(data: bytes) -> tuple[bytes, int, int]:
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        upright = ImageOps.exif_transpose(image)
        if image.getexif().get(274, 1) != 1:
            output = io.BytesIO()
            upright.save(output, "PNG")
            data = output.getvalue()
        return data, upright.width, upright.height


class StillCaptureTool(Tool[StillCaptureRequest, StillImage]):
    """Request a new photo even while video is streaming.

    The device chooses its photo resolution and must install a capture handler.
    A photo can exceed video resolution; neither 4K nor a matching field of view
    is guaranteed. Failure raises ImageCaptureUnavailable; video is never used
    as a fallback. Install the ``frames`` extra for image validation.
    """

    def __init__(self, *, endpoint: ProcessorEndpoint, images: ImageRegistry,
                 timeout_s: float = 10.0) -> None:
        self._images = images
        self._source = ClientImageCaptureSource(endpoint, timeout_s=timeout_s)
        super().__init__(
            "capture_still_image", "Request a new device photograph independently of video.",
            StillCaptureRequest, StillImage, self._capture,
        )

    async def _capture(self, request: StillCaptureRequest) -> StillImage:
        captured = await self._source.capture(request.participant_id)
        try:
            data, width, height = await asyncio.to_thread(_prepare_image, captured.data)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            raise ImageCaptureUnavailable("Device returned an invalid image.") from error
        return StillImage(
            image=self._images.put(data, owner=request.participant_id),
            width=width, height=height, timestamp_us=captured.pts_us,
            participant_id=captured.participant_id,
        )

    def close(self) -> None:
        """Fail pending requests and detach the capture response handler."""
        self._source.close()


__all__ = ["StillCaptureRequest", "StillCaptureTool", "StillImage"]
