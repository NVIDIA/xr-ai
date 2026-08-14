# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ZXing-C++-backed QR-code extraction from live participant frames."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping
from inspect import isawaitable, iscoroutinefunction
from typing import Any, TypeAlias

import numpy as np
import zxingcpp
from PIL import Image
from pydantic import BaseModel, Field
from xr_ai_hub import FrameUnavailable, LiveFrameSource, ProcessorEndpoint

from ._pixels import frame_to_pil
from .tools import Tool
from .types import StrictRequest

_LOGGER = logging.getLogger(__name__)


class QRCodeRequest(StrictRequest):
    """Read QR codes from one participant's current live camera frame."""

    participant_id: str = Field(
        min_length=1,
        description="Participant whose current camera frame should be scanned.",
    )


class QRCodePoint(BaseModel):
    """One image-space corner of a decoded QR code."""

    x: float = Field(description="Horizontal pixel coordinate.")
    y: float = Field(description="Vertical pixel coordinate.")


class DecodedQRCode(BaseModel):
    """A decoded QR payload and its quadrilateral in the source frame."""

    data: str = Field(description="UTF-8 text extracted from the QR code.")
    corners: tuple[QRCodePoint, QRCodePoint, QRCodePoint, QRCodePoint] | None = Field(
        default=None,
        description="Four image-space corners, or null if the extractor does not localize.",
    )


class QRCodeResult(BaseModel):
    """All readable QR codes found in one current camera frame."""

    codes: list[DecodedQRCode] = Field(
        default_factory=list,
        description="Decoded QR codes in extractor-provided order.",
    )
    available: bool = Field(
        default=True,
        description="Whether a current frame was available and could be scanned.",
    )
    message: str | None = Field(
        default=None,
        description="Human-readable detail when no QR payload was returned.",
    )


QRCodeExtraction: TypeAlias = DecodedQRCode | Mapping[str, Any]
QRCodeExtractor: TypeAlias = Callable[
    [Image.Image],
    Iterable[QRCodeExtraction] | Awaitable[Iterable[QRCodeExtraction]],
]


def extract_qr_codes_zxing(
    image: Image.Image | np.ndarray,
) -> list[DecodedQRCode]:
    """Decode every readable QR code with the ZXing-C++ Python bindings."""

    pixels = np.asarray(image)
    if pixels.dtype != np.uint8:
        raise TypeError("QR-code image must use uint8 pixels")
    if pixels.ndim not in (2, 3):
        raise ValueError("QR-code image must be grayscale or color")

    barcodes = zxingcpp.read_barcodes(
        pixels,
        formats=zxingcpp.BarcodeFormat.QRCode,
    )
    return [
        DecodedQRCode(
            data=barcode.text,
            corners=tuple(
                QRCodePoint(x=float(point.x), y=float(point.y))
                for point in (
                    barcode.position.top_left,
                    barcode.position.top_right,
                    barcode.position.bottom_right,
                    barcode.position.bottom_left,
                )
            ),
        )
        for barcode in barcodes
    ]


class QRCodeTool(Tool[QRCodeRequest, QRCodeResult]):
    """A finite tool that reads QR payloads from a current camera frame."""

    def __init__(
        self,
        *,
        endpoint: ProcessorEndpoint,
        frame_max_age_s: float = 2.0,
        frame_timeout_s: float = 5.0,
        manage_status: bool = True,
        extractor: QRCodeExtractor | None = None,
    ) -> None:
        if frame_max_age_s <= 0.0:
            raise ValueError("frame_max_age_s must be positive")
        if frame_timeout_s <= 0.0:
            raise ValueError("frame_timeout_s must be positive")
        self.endpoint = endpoint
        self.manage_status = manage_status
        self.extractor = (
            extractor if extractor is not None else extract_qr_codes_zxing
        )
        self.frames = LiveFrameSource(
            endpoint,
            max_age_s=frame_max_age_s,
            timeout_s=frame_timeout_s,
        )
        super().__init__(
            "read_qr_codes",
            "Read all QR codes in a participant's current live camera frame "
            "and return their decoded text and available corner coordinates.",
            QRCodeRequest,
            QRCodeResult,
            self._read_current,
        )

    def release(self, participant_id: str) -> None:
        """Forget cached frame state after a participant disconnects."""

        self.frames.release(participant_id)

    async def _read_current(self, request: QRCodeRequest) -> QRCodeResult:
        try:
            frame = await self.frames.get(request.participant_id)
        except FrameUnavailable as exc:
            return QRCodeResult(available=False, message=str(exc))

        if self.manage_status:
            await self.endpoint.set_status("processing", request.participant_id)
        try:
            image = await asyncio.to_thread(frame_to_pil, frame)
            codes = await self._extract(image)
        except Exception:
            _LOGGER.exception("QR-code extraction failed")
            return QRCodeResult(
                available=False,
                message="QR-code extraction failed — please retry.",
            )
        finally:
            if self.manage_status:
                await self.endpoint.set_status("idle", request.participant_id)

        if not codes:
            return QRCodeResult(
                message="No readable QR code was found in the current camera frame.",
            )
        return QRCodeResult(codes=codes)

    async def _extract(self, image: Image.Image) -> list[DecodedQRCode]:
        call = self.extractor
        if iscoroutinefunction(call) or iscoroutinefunction(getattr(call, "__call__", None)):
            extracted = call(image)
        else:
            extracted = await asyncio.to_thread(call, image)
        if isawaitable(extracted):
            extracted = await extracted
        return [DecodedQRCode.model_validate(code) for code in extracted]


__all__ = [
    "DecodedQRCode",
    "QRCodeExtraction",
    "QRCodeExtractor",
    "QRCodePoint",
    "QRCodeRequest",
    "QRCodeResult",
    "QRCodeTool",
    "extract_qr_codes_zxing",
]
