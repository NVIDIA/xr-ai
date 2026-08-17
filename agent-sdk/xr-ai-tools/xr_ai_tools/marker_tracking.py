# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""QR and ArUco marker detection from live participant frames."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from enum import StrEnum

import cv2
import numpy as np
import zxingcpp
from PIL import Image
from pydantic import BaseModel, Field
from xr_ai_hub import FrameUnavailable, LiveFrameSource, ProcessorEndpoint

from ._pixels import frame_to_pil
from .tools import Tool
from .types import StrictRequest

_LOGGER = logging.getLogger(__name__)
_QR_SCAN_LONG_EDGE = 3840
_QR_SCAN_MAX_SCALE = 6


class MarkerType(StrEnum):
    """Marker families supported by :class:`MarkerTrackingTool`."""

    QR_CODE = "qr_code"
    ARUCO = "aruco"


class MarkerTrackingRequest(StrictRequest):
    """Track markers in one participant's current live camera frame."""

    participant_id: str = Field(
        min_length=1,
        description="Participant whose current camera frame should be scanned.",
    )


class MarkerPoint(BaseModel):
    """One image-space corner of a tracked marker."""

    x: float = Field(description="Horizontal pixel coordinate.")
    y: float = Field(description="Vertical pixel coordinate.")


class TrackedMarker(BaseModel):
    """One detected marker and its quadrilateral in the source frame."""

    marker_type: MarkerType = Field(description="Detected marker family.")
    value: str = Field(
        description="Decoded QR text or decimal ArUco marker identifier."
    )
    corners: tuple[MarkerPoint, MarkerPoint, MarkerPoint, MarkerPoint] = Field(
        description="Four clockwise image-space corners."
    )


class MarkerTrackingResult(BaseModel):
    """All enabled markers found in one current camera frame."""

    markers: list[TrackedMarker] = Field(
        default_factory=list,
        description="Detected markers in detector-provided order.",
    )
    available: bool = Field(
        default=True,
        description="Whether a current frame was available and could be scanned.",
    )
    message: str | None = Field(
        default=None,
        description="Human-readable detail when no marker was returned.",
    )


def _image_pixels(image: Image.Image | np.ndarray) -> np.ndarray:
    pixels = np.asarray(image)
    if pixels.dtype != np.uint8:
        raise TypeError("marker image must use uint8 pixels")
    if pixels.ndim not in (2, 3):
        raise ValueError("marker image must be grayscale or color")
    return pixels


def _corners(points: np.ndarray) -> tuple[MarkerPoint, MarkerPoint, MarkerPoint, MarkerPoint]:
    flattened = np.asarray(points, dtype=np.float32).reshape(4, 2)
    return tuple(
        MarkerPoint(x=float(point[0]), y=float(point[1]))
        for point in flattened
    )


def _extract_qr_markers(pixels: np.ndarray) -> list[TrackedMarker]:
    height, width = pixels.shape[:2]
    grayscale = Image.fromarray(pixels).convert("L")
    scans = [(np.asarray(grayscale), 1)]
    scale = min(
        _QR_SCAN_MAX_SCALE,
        max(
            1,
            (_QR_SCAN_LONG_EDGE + max(width, height) - 1)
            // max(width, height),
        ),
    )
    if scale > 1:
        enlarged = grayscale.resize(
            (width * scale, height * scale),
            Image.Resampling.LANCZOS,
        )
        scans.append((np.asarray(enlarged), scale))

    markers: list[TrackedMarker] = []
    for scan, scan_scale in scans:
        barcodes = zxingcpp.read_barcodes(
            np.ascontiguousarray(scan),
            formats=zxingcpp.BarcodeFormat.QRCode,
        )
        for barcode in barcodes:
            marker = TrackedMarker(
                marker_type=MarkerType.QR_CODE,
                value=barcode.text,
                corners=tuple(
                    MarkerPoint(
                        x=float(point.x / scan_scale),
                        y=float(point.y / scan_scale),
                    )
                    for point in (
                        barcode.position.top_left,
                        barcode.position.top_right,
                        barcode.position.bottom_right,
                        barcode.position.bottom_left,
                    )
                ),
            )
            if not any(_same_marker(marker, existing) for existing in markers):
                markers.append(marker)
    return markers


def _same_marker(first: TrackedMarker, second: TrackedMarker) -> bool:
    if first.marker_type is not second.marker_type or first.value != second.value:
        return False

    first_center = (
        sum(point.x for point in first.corners) / len(first.corners),
        sum(point.y for point in first.corners) / len(first.corners),
    )
    second_center = (
        sum(point.x for point in second.corners) / len(second.corners),
        sum(point.y for point in second.corners) / len(second.corners),
    )
    return (
        abs(first_center[0] - second_center[0]) <= 24
        and abs(first_center[1] - second_center[1]) <= 24
    )


def _make_aruco_detector(dictionary_name: str) -> cv2.aruco.ArucoDetector:
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if not dictionary_name.startswith("DICT_") or not isinstance(dictionary_id, int):
        raise ValueError(f"unknown ArUco dictionary: {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    return cv2.aruco.ArucoDetector(dictionary)


def _extract_aruco_markers(
    pixels: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
) -> list[TrackedMarker]:
    if pixels.ndim == 3:
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    corners, identifiers, _rejected = detector.detectMarkers(pixels)
    if identifiers is None:
        return []
    return [
        TrackedMarker(
            marker_type=MarkerType.ARUCO,
            value=str(int(identifier)),
            corners=_corners(marker_corners),
        )
        for marker_corners, identifier in zip(corners, identifiers.reshape(-1), strict=True)
    ]


class MarkerTrackingTool(Tool[MarkerTrackingRequest, MarkerTrackingResult]):
    """A finite tool that tracks enabled marker families in a current frame."""

    def __init__(
        self,
        *,
        endpoint: ProcessorEndpoint,
        marker_types: Iterable[MarkerType | str] = (
            MarkerType.QR_CODE,
            MarkerType.ARUCO,
        ),
        aruco_dictionary: str = "DICT_4X4_50",
        frame_max_age_s: float = 2.0,
        frame_timeout_s: float = 5.0,
        manage_status: bool = True,
    ) -> None:
        if frame_max_age_s <= 0.0:
            raise ValueError("frame_max_age_s must be positive")
        if frame_timeout_s <= 0.0:
            raise ValueError("frame_timeout_s must be positive")
        enabled = tuple(dict.fromkeys(MarkerType(marker_type) for marker_type in marker_types))
        if not enabled:
            raise ValueError("marker_types must enable at least one marker family")

        self.endpoint = endpoint
        self.marker_types = enabled
        self.manage_status = manage_status
        self._aruco_detector = (
            _make_aruco_detector(aruco_dictionary)
            if MarkerType.ARUCO in enabled
            else None
        )
        self.frames = LiveFrameSource(
            endpoint,
            max_age_s=frame_max_age_s,
            timeout_s=frame_timeout_s,
        )
        enabled_names = ", ".join(marker_type.value for marker_type in enabled)
        super().__init__(
            "track_markers",
            "Track enabled marker families in a participant's current live camera "
            f"frame ({enabled_names}) and return their identifiers and corners.",
            MarkerTrackingRequest,
            MarkerTrackingResult,
            self._read_current,
        )

    def release(self, participant_id: str) -> None:
        """Forget cached frame state after a participant disconnects."""

        self.frames.release(participant_id)

    async def _read_current(
        self,
        request: MarkerTrackingRequest,
    ) -> MarkerTrackingResult:
        try:
            frame = await self.frames.get(request.participant_id)
        except FrameUnavailable as exc:
            return MarkerTrackingResult(available=False, message=str(exc))

        if self.manage_status:
            await self.endpoint.set_status("processing", request.participant_id)
        try:
            image = await asyncio.to_thread(frame_to_pil, frame)
            markers = await asyncio.to_thread(self._extract, image)
        except Exception:
            _LOGGER.exception("marker extraction failed")
            return MarkerTrackingResult(
                available=False,
                message="Marker extraction failed — please retry.",
            )
        finally:
            if self.manage_status:
                await self.endpoint.set_status("idle", request.participant_id)

        if not markers:
            return MarkerTrackingResult(
                message="No enabled marker was found in the current camera frame.",
            )
        return MarkerTrackingResult(markers=markers)

    def _extract(self, image: Image.Image) -> list[TrackedMarker]:
        pixels = _image_pixels(image)
        markers: list[TrackedMarker] = []
        if MarkerType.QR_CODE in self.marker_types:
            markers.extend(_extract_qr_markers(pixels))
        if self._aruco_detector is not None:
            markers.extend(_extract_aruco_markers(pixels, self._aruco_detector))
        return markers


__all__ = [
    "MarkerPoint",
    "MarkerTrackingRequest",
    "MarkerTrackingResult",
    "MarkerTrackingTool",
    "MarkerType",
    "TrackedMarker",
]
