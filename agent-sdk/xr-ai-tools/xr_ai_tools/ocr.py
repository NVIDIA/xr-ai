# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed OCR tool for text and numeric values in images."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from xr_ai_models import OCRMergeLevel, OCRService

from .tools import Tool
from .types import StrictRequest


class OCRRequest(StrictRequest):
    image: str = Field(
        min_length=1,
        description=(
            "PNG or JPEG as a base64 data URL, or an application-enabled "
            "local path or HTTP(S) URL."
        ),
    )
    merge_level: OCRMergeLevel = Field(
        default="paragraph",
        description="Granularity of recognized text regions.",
    )


class OCRPointResult(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class OCRDetectionResult(BaseModel):
    text: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    bounding_box: list[OCRPointResult] = Field(default_factory=list)


class OCRResult(BaseModel):
    text: str = Field(description="Recognized text and numbers in reading order.")
    detections: list[OCRDetectionResult]
    model: str | None = None


class OCRTool(Tool[OCRRequest, OCRResult]):
    """Read visible text from one image with an injected OCR service."""

    def __init__(
        self,
        *,
        ocr: OCRService,
        allow_local_paths: bool = False,
        allow_remote_urls: bool = False,
    ) -> None:
        self.ocr = ocr
        self._allow_local_paths = allow_local_paths
        self._allow_remote_urls = allow_remote_urls
        super().__init__(
            "read_image_text",
            "Read visible text and numeric equipment values from a PNG or JPEG image.",
            OCRRequest,
            OCRResult,
            self._read,
            render_result=lambda result: result.text,
        )

    async def _read(self, request: OCRRequest) -> OCRResult:
        image = _image_input(
            request.image,
            allow_local_paths=self._allow_local_paths,
            allow_remote_urls=self._allow_remote_urls,
        )
        response = await self.ocr.recognize(
            image,
            merge_level=request.merge_level,
        )
        return OCRResult(
            text=response.text,
            detections=[
                OCRDetectionResult(
                    text=detection.text,
                    confidence=detection.confidence,
                    bounding_box=[
                        OCRPointResult(x=point.x, y=point.y)
                        for point in detection.bounding_box
                    ],
                )
                for detection in response.detections
            ],
            model=response.model,
        )


def _image_input(
    image: str,
    *,
    allow_local_paths: bool,
    allow_remote_urls: bool,
) -> str | Path:
    if image.startswith("data:"):
        return image
    if image.startswith(("http://", "https://")):
        if not allow_remote_urls:
            raise ValueError("OCR remote URLs are disabled")
        return image
    if not allow_local_paths:
        raise ValueError("OCR local paths are disabled")
    return Path(image)


__all__ = [
    "OCRDetectionResult",
    "OCRPointResult",
    "OCRRequest",
    "OCRResult",
    "OCRTool",
]
