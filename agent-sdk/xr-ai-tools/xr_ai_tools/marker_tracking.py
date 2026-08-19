# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""QR and ArUco marker detection from live participant frames."""

from __future__ import annotations

import asyncio
import io
import logging
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
import zxingcpp
from PIL import Image
from pydantic import BaseModel, Field
from xr_ai_hub import FrameUnavailable, LiveFrameSource, ProcessorEndpoint

from ._pixels import frame_to_pil
from .image import ImageInput, ImageReference, ImageRegistry
from .tools import Tool
from .types import StrictRequest

_LOGGER = logging.getLogger(__name__)
_MARKER_SCAN_LONG_EDGE = 3840
_MARKER_SCAN_MAX_SCALE = 6


class MarkerType(StrEnum):
    """Marker families supported by :class:`MarkerTrackingTool`."""

    QR_CODE = "qr_code"
    """QR codes containing arbitrary text."""

    ARUCO = "aruco"
    """ArUco fiducial markers containing numeric identifiers."""


class MarkerTrackingRequest(StrictRequest):
    """Track markers in one participant's current live camera frame."""

    participant_id: str = Field(
        min_length=1,
        description="Participant whose current camera frame should be scanned.",
    )
    """Participant whose current camera frame should be scanned."""

    image: ImageReference | None = Field(
        default=None,
        description="Previously selected image to scan instead of acquiring another frame.",
    )
    """Previously selected image to scan instead of acquiring another frame."""


class MarkerPoint(BaseModel):
    """One image-space corner of a tracked marker."""

    x: float = Field(description="Horizontal pixel coordinate.")
    """Horizontal pixel coordinate."""

    y: float = Field(description="Vertical pixel coordinate.")
    """Vertical pixel coordinate."""


class TrackedMarker(BaseModel):
    """One detected marker and its quadrilateral in the source frame."""

    marker_type: MarkerType = Field(description="Detected marker family.")
    """Detected marker family."""

    value: str = Field(
        description="Decoded QR text or decimal ArUco marker identifier."
    )
    """Decoded QR text or decimal ArUco marker identifier."""

    corners: tuple[MarkerPoint, MarkerPoint, MarkerPoint, MarkerPoint] = Field(
        description="Four clockwise image-space corners."
    )
    """Four clockwise image-space corners."""


class MarkerTrackingResult(BaseModel):
    """All enabled markers found in one current camera frame."""

    markers: list[TrackedMarker] = Field(
        default_factory=list,
        description="Detected markers in detector-provided order.",
    )
    """Detected markers in detector-provided order."""

    available: bool = Field(
        default=True,
        description="Whether a current frame was available and could be scanned.",
    )
    """Whether a current frame was available and could be scanned."""

    message: str | None = Field(
        default=None,
        description="Human-readable detail when no marker was returned.",
    )
    """Human-readable detail when no marker was returned."""


def _image_pixels(image: Image.Image | np.ndarray) -> np.ndarray:
    pixels = np.asarray(image)
    if pixels.dtype != np.uint8:
        raise TypeError("marker image must use uint8 pixels")
    if pixels.ndim not in (2, 3):
        raise ValueError("marker image must be grayscale or color")
    return pixels


def _open_image(source: ImageInput) -> Image.Image:
    if isinstance(source, bytes):
        opened = Image.open(io.BytesIO(source))
    else:
        if isinstance(source, str) and urlsplit(source).scheme:
            raise ValueError("marker tracking requires bytes or a local image path")
        opened = Image.open(Path(source))
    with opened:
        opened.load()
        return opened.convert("RGB")


def _corners(points: np.ndarray) -> tuple[MarkerPoint, MarkerPoint, MarkerPoint, MarkerPoint]:
    flattened = np.asarray(points, dtype=np.float32).reshape(4, 2)
    return tuple(
        MarkerPoint(x=float(point[0]), y=float(point[1]))
        for point in flattened
    )


def _full_frame_scans(
    pixels: np.ndarray,
) -> list[tuple[np.ndarray, float, float]]:
    height, width = pixels.shape[:2]
    grayscale = Image.fromarray(pixels).convert("L")
    scans = [(np.asarray(grayscale), 1.0, 1.0)]
    long_edge = max(width, height)
    target_long_edge = min(
        _MARKER_SCAN_LONG_EDGE,
        long_edge * _MARKER_SCAN_MAX_SCALE,
    )
    if target_long_edge <= long_edge:
        return scans

    if width >= height:
        resized_width = target_long_edge
        resized_height = max(1, round(height * target_long_edge / width))
    else:
        resized_width = max(1, round(width * target_long_edge / height))
        resized_height = target_long_edge
    enlarged = grayscale.resize(
        (resized_width, resized_height),
        Image.Resampling.LANCZOS,
    )
    scans.append(
        (
            np.asarray(enlarged),
            resized_width / width,
            resized_height / height,
        )
    )
    return scans


def _extract_qr_markers(
    scans: Iterable[tuple[np.ndarray, float, float]],
) -> list[TrackedMarker]:
    scan_markers: list[list[TrackedMarker]] = []
    for scan, scale_x, scale_y in scans:
        barcodes = zxingcpp.read_barcodes(
            np.ascontiguousarray(scan),
            formats=zxingcpp.BarcodeFormat.QRCode,
        )
        scan_markers.append(
            [
                TrackedMarker(
                    marker_type=MarkerType.QR_CODE,
                    value=barcode.text,
                    corners=tuple(
                        MarkerPoint(
                            x=float(point.x / scale_x),
                            y=float(point.y / scale_y),
                        )
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
        )
    return _merge_marker_scans(scan_markers)


def _marker_overlap_area(first: TrackedMarker, second: TrackedMarker) -> float:
    first_left = min(point.x for point in first.corners)
    first_top = min(point.y for point in first.corners)
    first_right = max(point.x for point in first.corners)
    first_bottom = max(point.y for point in first.corners)
    second_left = min(point.x for point in second.corners)
    second_top = min(point.y for point in second.corners)
    second_right = max(point.x for point in second.corners)
    second_bottom = max(point.y for point in second.corners)
    return max(0.0, min(first_right, second_right) - max(first_left, second_left)) * max(
        0.0,
        min(first_bottom, second_bottom) - max(first_top, second_top),
    )


def _merge_marker_scans(
    marker_scans: Iterable[list[TrackedMarker]],
) -> list[TrackedMarker]:
    merged: list[TrackedMarker] = []
    for scan_markers in marker_scans:
        unmatched_previous = set(range(len(merged)))
        for marker in scan_markers:
            best_index: int | None = None
            best_overlap = 0.0
            for index in unmatched_previous:
                existing = merged[index]
                if (
                    marker.marker_type != existing.marker_type
                    or marker.value != existing.value
                ):
                    continue
                overlap = _marker_overlap_area(marker, existing)
                if overlap > best_overlap:
                    best_index = index
                    best_overlap = overlap
            if best_index is None:
                merged.append(marker)
            else:
                unmatched_previous.remove(best_index)
    return merged


def _make_aruco_detector(dictionary_name: str) -> cv2.aruco.ArucoDetector:
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if not dictionary_name.startswith("DICT_") or not isinstance(dictionary_id, int):
        raise ValueError(f"unknown ArUco dictionary: {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    return cv2.aruco.ArucoDetector(dictionary)


def _extract_aruco_markers(
    scans: Iterable[tuple[np.ndarray, float, float]],
    detector: cv2.aruco.ArucoDetector,
) -> list[TrackedMarker]:
    marker_scans: list[list[TrackedMarker]] = []
    for scan, scale_x, scale_y in scans:
        corners, identifiers, _rejected = detector.detectMarkers(scan)
        if identifiers is None:
            marker_scans.append([])
            continue
        scan_markers: list[TrackedMarker] = []
        for marker_corners, identifier in zip(
            corners,
            identifiers.reshape(-1),
            strict=True,
        ):
            source_corners = np.asarray(marker_corners, dtype=np.float32) / np.array(
                [scale_x, scale_y],
                dtype=np.float32,
            )
            scan_markers.append(
                TrackedMarker(
                    marker_type=MarkerType.ARUCO,
                    value=str(int(identifier)),
                    corners=_corners(source_corners),
                )
            )
        marker_scans.append(scan_markers)
    return _merge_marker_scans(marker_scans)


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
        images: ImageRegistry | None = None,
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
        self.images = images
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
        frame = None
        source: ImageInput | None = None
        if request.image is None:
            try:
                frame = await self.frames.get(request.participant_id)
            except FrameUnavailable as exc:
                return MarkerTrackingResult(available=False, message=str(exc))
        else:
            if self.images is None:
                return MarkerTrackingResult(
                    available=False,
                    message="Image input is unsupported by this marker tracker.",
                )
            try:
                source = self.images.resolve(request.image)
            except (LookupError, ValueError) as exc:
                return MarkerTrackingResult(available=False, message=str(exc))

        if self.manage_status:
            await self.endpoint.set_status("processing", request.participant_id)
        try:
            if source is None:
                assert frame is not None
                image = await asyncio.to_thread(frame_to_pil, frame)
            else:
                image = await asyncio.to_thread(_open_image, source)
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
        scans = _full_frame_scans(pixels)
        markers: list[TrackedMarker] = []
        if MarkerType.QR_CODE in self.marker_types:
            markers.extend(_extract_qr_markers(scans))
        if self._aruco_detector is not None:
            markers.extend(_extract_aruco_markers(scans, self._aruco_detector))
        return markers


__all__ = [
    "MarkerPoint",
    "MarkerTrackingRequest",
    "MarkerTrackingResult",
    "MarkerTrackingTool",
    "MarkerType",
    "TrackedMarker",
]
