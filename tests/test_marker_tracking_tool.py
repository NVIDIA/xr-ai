# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for live QR and ArUco marker tracking."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
from pydantic import ValidationError
from xr_ai_hub import FrameData, FrameSignal, PixelFormat
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


def _qr_image(scale: int = 6) -> np.ndarray:
    pixels = np.array(
        [[0 if value == "1" else 255 for value in row] for row in _QR_MODULES],
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
