# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for live QR and ArUco marker tracking."""

from __future__ import annotations

import io
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError
from xr_ai_hub import FrameData, FrameSignal, PixelFormat
from xr_ai_tools import marker_tracking as marker_tracking_module
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.marker_tracking import (
    MarkerTrackingRequest,
    MarkerTrackingTool,
    MarkerType,
)

_QR_MODULES = (
    "0000000000000000000000000",
    "0000000000000000000000000",
    "0011111110010110111111100",
    "0010000010110100100000100",
    "0010111010110010101110100",
    "0010111010010100101110100",
    "0010111010100010101110100",
    "0010000010100110100000100",
    "0011111110101010111111100",
    "0000000000111110000000000",
    "0011010011011000111011000",
    "0010110100000101000100000",
    "0001111110000011010000100",
    "0000111101110001100001100",
    "0001100010001100001000000",
    "0000000000110110001101000",
    "0011111110111010011111000",
    "0010000010011010010000000",
    "0010111010011100000010000",
    "0010111010111010111001100",
    "0010111010010110101110100",
    "0010000010110011001000000",
    "0011111110100000100111000",
    "0000000000000000000000000",
    "0000000000000000000000000",
)

_SECOND_QR_MODULES = (
    "00000000000000000000000000000",
    "00000000000000000000000000000",
    "00000000000000000000000000000",
    "00000000000000000000000000000",
    "00001111111001000011111110000",
    "00001000001010011010000010000",
    "00001011101000010010111010000",
    "00001011101011110010111010000",
    "00001011101000000010111010000",
    "00001000001011011010000010000",
    "00001111111010101011111110000",
    "00000000000001001000000000000",
    "00001111101111010101010100000",
    "00000011000000100000011010000",
    "00000010011101011100011100000",
    "00001110000110011101011110000",
    "00000000111100100101000000000",
    "00000000000011010001101110000",
    "00001111111010111011010100000",
    "00001000001001011001011100000",
    "00001011101011001010000000000",
    "00001011101010100001110100000",
    "00001011101011011000010000000",
    "00001000001011001101111000000",
    "00001111111011101110100100000",
    "00000000000000000000000000000",
    "00000000000000000000000000000",
    "00000000000000000000000000000",
    "00000000000000000000000000000",
)


def _qr_image(
    modules: tuple[str, ...] = _QR_MODULES,
    scale: int = 6,
) -> np.ndarray:
    pixels = np.array(
        [[0 if value == "1" else 255 for value in row] for row in modules],
        dtype=np.uint8,
    )
    return pixels.repeat(scale, axis=0).repeat(scale, axis=1)


def _aruco_image(
    *,
    dictionary_name: str = "DICT_4X4_50",
    marker_id: int = 23,
    size: int = 160,
) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, dictionary_name)
    )
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
    canvas = np.full((size + 40, size + 40), 255, dtype=np.uint8)
    canvas[20 : 20 + size, 20 : 20 + size] = marker
    return canvas


def _mixed_image() -> np.ndarray:
    qr = _qr_image()
    aruco = _aruco_image()
    canvas = np.full(
        (max(qr.shape[0], aruco.shape[0]) + 40, qr.shape[1] + aruco.shape[1] + 60),
        255,
        dtype=np.uint8,
    )
    canvas[20 : 20 + qr.shape[0], 20 : 20 + qr.shape[1]] = qr
    aruco_x = 40 + qr.shape[1]
    canvas[20 : 20 + aruco.shape[0], aruco_x : aruco_x + aruco.shape[1]] = aruco
    return canvas


def _downsampled_aruco_scene() -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    source = np.full((3840, 2880), 220, dtype=np.uint8)
    large = cv2.aruco.generateImageMarker(dictionary, 23, 600)
    small = cv2.aruco.generateImageMarker(dictionary, 42, 48)
    source[500:1100, 300:900] = large
    source[1733:1781, 1279:1327] = small
    return np.asarray(
        Image.fromarray(source).resize((480, 640), Image.Resampling.LANCZOS)
    )


class _Endpoint:
    def __init__(self, image: np.ndarray) -> None:
        rgb = np.repeat(image[:, :, None], 3, axis=2)
        self.frame = FrameData(
            seq=1,
            pts_us=time.time_ns() // 1_000,
            width=rgb.shape[1],
            height=rgb.shape[0],
            fmt=PixelFormat.RGB24,
            data=rgb.tobytes(),
            participant_id="alice",
            track_id="camera",
        )
        self.frame_callback = None
        self.statuses: list[tuple[str, str]] = []

    def on_frame(self, callback) -> None:
        self.frame_callback = callback

    def on_participant(self, _callback) -> None:
        pass

    async def request_frame(self, _signal: FrameSignal) -> FrameData:
        return self.frame

    async def set_status(self, status: str, participant_id: str) -> None:
        self.statuses.append((status, participant_id))

    async def seed(self) -> None:
        await self.frame_callback(
            FrameSignal(
                slot=0,
                seq=self.frame.seq,
                pts_us=self.frame.pts_us,
                width=self.frame.width,
                height=self.frame.height,
                fmt=self.frame.fmt,
                data_sz=len(self.frame.data),
                participant_id=self.frame.participant_id,
                track_id=self.frame.track_id,
            )
        )


def test_marker_scan_enlargement_respects_dimension_and_scale_limits() -> None:
    below_long_edge_limit = np.full((2160, 3839), 255, dtype=np.uint8)

    scans = marker_tracking_module._full_frame_scans(below_long_edge_limit)

    assert [scan.shape for scan, _scale_x, _scale_y in scans] == [
        (2160, 3839),
        (2161, 3840),
    ]
    assert scans[-1][1] == pytest.approx(3840 / 3839)
    assert scans[-1][2] == pytest.approx(2161 / 2160)

    at_long_edge_limit = np.full((2, 3840), 255, dtype=np.uint8)
    assert len(marker_tracking_module._full_frame_scans(at_long_edge_limit)) == 1

    below_scale_limit = np.full((2, 100), 255, dtype=np.uint8)
    limited_scans = marker_tracking_module._full_frame_scans(below_scale_limit)
    assert limited_scans[-1][0].shape == (12, 600)
    assert limited_scans[-1][1:] == (6.0, 6.0)


@pytest.mark.asyncio
async def test_marker_tool_detects_qr_and_aruco_with_uniform_schema() -> None:
    endpoint = _Endpoint(_mixed_image())
    tool = MarkerTrackingTool(endpoint=endpoint)
    await endpoint.seed()

    result = await tool.execute(MarkerTrackingRequest(participant_id="alice"))

    assert result.available is True
    assert result.message is None
    assert {(marker.marker_type, marker.value) for marker in result.markers} == {
        (MarkerType.QR_CODE, "XR AI QR tool"),
        (MarkerType.ARUCO, "23"),
    }
    assert all(len(marker.corners) == 4 for marker in result.markers)
    assert all(
        point.x >= 0 and point.y >= 0
        for marker in result.markers
        for point in marker.corners
    )
    assert endpoint.statuses == [("processing", "alice"), ("idle", "alice")]
    assert tool.frames.participants() == ["alice"]
    tool.release("alice")
    assert tool.frames.participants() == []


@pytest.mark.asyncio
async def test_marker_tool_scans_selected_image_instead_of_newer_live_frame() -> None:
    endpoint = _Endpoint(np.full((200, 200), 255, dtype=np.uint8))
    images = ImageRegistry()
    encoded = io.BytesIO()
    Image.fromarray(_qr_image()).save(encoded, format="PNG")
    selected = images.put(encoded.getvalue(), owner="alice")
    tool = MarkerTrackingTool(
        endpoint=endpoint,
        images=images,
        marker_types=(MarkerType.QR_CODE,),
    )
    await endpoint.seed()

    result = await tool.execute(
        MarkerTrackingRequest(participant_id="alice", image=selected)
    )

    assert [marker.value for marker in result.markers] == ["XR AI QR tool"]


@pytest.mark.asyncio
async def test_marker_tool_returns_every_qr_marker_in_frame() -> None:
    first = _qr_image()
    second = _qr_image(_SECOND_QR_MODULES)
    canvas = np.full(
        (max(first.shape[0], second.shape[0]) + 40, first.shape[1] + second.shape[1] + 80),
        255,
        dtype=np.uint8,
    )
    canvas[20 : 20 + first.shape[0], 20 : 20 + first.shape[1]] = first
    second_x = 60 + first.shape[1]
    canvas[20 : 20 + second.shape[0], second_x : second_x + second.shape[1]] = second
    endpoint = _Endpoint(canvas)

    tool = MarkerTrackingTool(
        endpoint=endpoint,
        marker_types=(MarkerType.QR_CODE,),
    )
    await endpoint.seed()

    result = await tool.execute(MarkerTrackingRequest(participant_id="alice"))

    assert {(marker.marker_type, marker.value) for marker in result.markers} == {
        (MarkerType.QR_CODE, "XR AI QR tool"),
        (MarkerType.QR_CODE, "second QR payload"),
    }
    assert len(result.markers) == 2


def test_marker_tool_preserves_repeated_qr_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def barcode(left: int) -> SimpleNamespace:
        return SimpleNamespace(
            text="XR AI QR tool",
            position=SimpleNamespace(
                top_left=SimpleNamespace(x=left, y=2),
                top_right=SimpleNamespace(x=left + 6, y=2),
                bottom_right=SimpleNamespace(x=left + 6, y=8),
                bottom_left=SimpleNamespace(x=left, y=8),
            ),
        )

    detections = iter(
        [
            [barcode(2), barcode(10)],
            [barcode(4), barcode(20)],
        ]
    )
    monkeypatch.setattr(
        marker_tracking_module.zxingcpp,
        "read_barcodes",
        lambda *_args, **_kwargs: next(detections),
    )

    markers = marker_tracking_module._extract_qr_markers(
        [
            (np.full((12, 20), 255, dtype=np.uint8), 1.0, 1.0),
            (np.full((24, 40), 255, dtype=np.uint8), 2.0, 2.0),
        ]
    )

    assert [marker.value for marker in markers] == [
        "XR AI QR tool",
        "XR AI QR tool",
    ]


@pytest.mark.asyncio
async def test_marker_tool_recovers_multiple_small_qr_codes() -> None:
    first = _qr_image()
    second = _qr_image(_SECOND_QR_MODULES)
    source = np.full((3840, 2880), 220, dtype=np.uint8)
    source[2100 : 2100 + first.shape[0], 120 : 120 + first.shape[1]] = first
    source[
        2200 : 2200 + second.shape[0],
        2100 : 2100 + second.shape[1],
    ] = second
    frame = np.asarray(
        Image.fromarray(source).resize((480, 640), Image.Resampling.LANCZOS)
    )
    endpoint = _Endpoint(frame)
    tool = MarkerTrackingTool(
        endpoint=endpoint,
        marker_types=(MarkerType.QR_CODE,),
    )
    await endpoint.seed()

    result = await tool.execute(MarkerTrackingRequest(participant_id="alice"))

    assert {marker.value for marker in result.markers} == {
        "XR AI QR tool",
        "second QR payload",
    }
    assert len(result.markers) == 2
    assert all(
        0 <= point.x < frame.shape[1] and 0 <= point.y < frame.shape[0]
        for marker in result.markers
        for point in marker.corners
    )


@pytest.mark.asyncio
async def test_marker_type_selection_does_not_change_agent_api() -> None:
    request = MarkerTrackingRequest(participant_id="alice")
    observed: dict[MarkerType, set[MarkerType]] = {}

    for marker_type in MarkerType:
        endpoint = _Endpoint(_mixed_image())
        tool = MarkerTrackingTool(endpoint=endpoint, marker_types=(marker_type,))
        await endpoint.seed()

        result = await tool.execute(request)
        observed[marker_type] = {marker.marker_type for marker in result.markers}
        assert tool.name == "track_markers"
        assert tool.request_model is MarkerTrackingRequest

    assert observed == {
        MarkerType.QR_CODE: {MarkerType.QR_CODE},
        MarkerType.ARUCO: {MarkerType.ARUCO},
    }


@pytest.mark.asyncio
async def test_marker_tool_uses_configured_aruco_dictionary() -> None:
    endpoint = _Endpoint(
        _aruco_image(dictionary_name="DICT_6X6_250", marker_id=42)
    )
    tool = MarkerTrackingTool(
        endpoint=endpoint,
        marker_types=(MarkerType.ARUCO,),
        aruco_dictionary="DICT_6X6_250",
    )
    await endpoint.seed()

    result = await tool.execute(MarkerTrackingRequest(participant_id="alice"))

    assert [(marker.marker_type, marker.value) for marker in result.markers] == [
        (MarkerType.ARUCO, "42")
    ]


def test_marker_tool_preserves_nearby_repeated_aruco_ids() -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, 23, 6)
    canvas = np.full((20, 30), 255, dtype=np.uint8)
    canvas[7:13, 5:11] = marker
    canvas[7:13, 13:19] = marker
    markers = marker_tracking_module._extract_aruco_markers(
        marker_tracking_module._full_frame_scans(canvas),
        cv2.aruco.ArucoDetector(dictionary),
    )

    assert [tracked.value for tracked in markers] == ["23", "23"]


@pytest.mark.asyncio
async def test_marker_tool_recovers_small_aruco_from_downsampled_scene() -> None:
    frame = _downsampled_aruco_scene()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    native_detector = cv2.aruco.ArucoDetector(dictionary)
    _corners, native_ids, _rejected = native_detector.detectMarkers(frame)
    assert native_ids is not None
    assert native_ids.reshape(-1).tolist() == [23]

    endpoint = _Endpoint(frame)
    tool = MarkerTrackingTool(
        endpoint=endpoint,
        marker_types=(MarkerType.ARUCO,),
    )
    await endpoint.seed()

    result = await tool.execute(MarkerTrackingRequest(participant_id="alice"))

    assert [(marker.marker_type, marker.value) for marker in result.markers] == [
        (MarkerType.ARUCO, "23"),
        (MarkerType.ARUCO, "42"),
    ]
    small = next(marker for marker in result.markers if marker.value == "42")
    assert all(210 <= point.x <= 225 for point in small.corners)
    assert all(285 <= point.y <= 300 for point in small.corners)


@pytest.mark.asyncio
async def test_marker_tool_distinguishes_no_marker_from_no_frame() -> None:
    endpoint = _Endpoint(np.full((100, 100), 255, dtype=np.uint8))
    tool = MarkerTrackingTool(endpoint=endpoint, frame_timeout_s=0.01)
    await endpoint.seed()

    no_marker = await tool.execute(MarkerTrackingRequest(participant_id="alice"))
    no_frame = await tool.execute(MarkerTrackingRequest(participant_id="missing"))

    assert no_marker.available is True
    assert no_marker.markers == []
    assert no_marker.message is not None and "No enabled marker" in no_marker.message
    assert no_frame.available is False
    assert no_frame.markers == []
    assert no_frame.message is not None and "No camera frame" in no_frame.message


def test_marker_tool_validates_request_and_initialization() -> None:
    endpoint = _Endpoint(np.full((32, 48), 255, dtype=np.uint8))

    with pytest.raises(ValidationError):
        MarkerTrackingRequest(participant_id="alice", unsupported=True)
    with pytest.raises(ValueError, match="at least one"):
        MarkerTrackingTool(endpoint=endpoint, marker_types=())
    with pytest.raises(ValueError, match="unknown ArUco dictionary"):
        MarkerTrackingTool(
            endpoint=endpoint,
            marker_types=(MarkerType.ARUCO,),
            aruco_dictionary="UNKNOWN",
        )
    with pytest.raises(ValueError, match="not a valid MarkerType"):
        MarkerTrackingTool(endpoint=endpoint, marker_types=("barcode",))
