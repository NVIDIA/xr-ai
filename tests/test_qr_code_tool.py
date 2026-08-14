# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for OpenCV-backed live QR-code extraction."""

from __future__ import annotations

import time

import numpy as np
import pytest
from pydantic import ValidationError
from xr_ai_hub import FrameData, FrameSignal, PixelFormat
from xr_ai_tools.qr_code import QRCodeRequest, QRCodeTool, decode_qr_codes

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
    modules = np.array(
        [[0 if value == "1" else 255 for value in row] for row in _QR_MODULES],
        dtype=np.uint8,
    )
    return modules.repeat(scale, axis=0).repeat(scale, axis=1)


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


def test_decode_qr_codes_extracts_every_payload_and_quadrilateral() -> None:
    qr = _qr_image()
    canvas = np.full((qr.shape[0] + 40, qr.shape[1] * 2 + 80), 255, dtype=np.uint8)
    canvas[20 : 20 + qr.shape[0], 20 : 20 + qr.shape[1]] = qr
    second_x = 60 + qr.shape[1]
    canvas[20 : 20 + qr.shape[0], second_x : second_x + qr.shape[1]] = qr

    codes = decode_qr_codes(canvas)

    assert [code.data for code in codes] == ["XR AI QR tool", "XR AI QR tool"]
    assert all(code.corners is not None and len(code.corners) == 4 for code in codes)
    assert all(
        point.x >= 0 and point.y >= 0
        for code in codes
        for point in code.corners or ()
    )


@pytest.mark.asyncio
async def test_qr_code_tool_reads_the_current_participant_frame() -> None:
    endpoint = _Endpoint(_qr_image())
    tool = QRCodeTool(endpoint=endpoint)
    await endpoint.seed()

    result = await tool.execute(QRCodeRequest(participant_id="alice"))

    assert result.available is True
    assert result.message is None
    assert [code.data for code in result.codes] == ["XR AI QR tool"]
    assert endpoint.statuses == [("processing", "alice"), ("idle", "alice")]
    assert tool.frames.participants() == ["alice"]
    tool.release("alice")
    assert tool.frames.participants() == []


@pytest.mark.asyncio
async def test_qr_code_tool_accepts_sync_and_async_custom_extractors() -> None:
    calls: list[tuple[str, tuple[int, int]]] = []

    def sync_extractor(image):
        calls.append(("sync", image.size))
        return [{"data": "sync backend"}]

    async def async_extractor(image):
        calls.append(("async", image.size))
        return [{"data": "model backend", "corners": None}]

    results = []
    for extractor in (sync_extractor, async_extractor):
        endpoint = _Endpoint(np.full((32, 48), 255, dtype=np.uint8))
        tool = QRCodeTool(endpoint=endpoint, extractor=extractor)
        await endpoint.seed()
        results.append(
            await tool.execute(QRCodeRequest(participant_id="alice"))
        )

    assert calls == [("sync", (48, 32)), ("async", (48, 32))]
    assert [result.codes[0].data for result in results] == [
        "sync backend",
        "model backend",
    ]
    assert all(result.codes[0].corners is None for result in results)


@pytest.mark.asyncio
async def test_qr_code_tool_distinguishes_no_code_from_no_frame() -> None:
    endpoint = _Endpoint(np.full((100, 100), 255, dtype=np.uint8))
    tool = QRCodeTool(endpoint=endpoint, frame_timeout_s=0.01)
    await endpoint.seed()

    no_code = await tool.execute(QRCodeRequest(participant_id="alice"))
    no_frame = await tool.execute(QRCodeRequest(participant_id="missing"))

    assert no_code.available is True
    assert no_code.codes == []
    assert no_code.message is not None and "No readable QR code" in no_code.message
    assert no_frame.available is False
    assert no_frame.codes == []
    assert no_frame.message is not None and "No camera frame" in no_frame.message


def test_qr_code_request_is_strict_and_decoder_validates_pixels() -> None:
    with pytest.raises(ValidationError):
        QRCodeRequest(participant_id="alice", unsupported=True)
    with pytest.raises(TypeError, match="uint8"):
        decode_qr_codes(np.zeros((10, 10), dtype=np.float32))
    with pytest.raises(ValueError, match="grayscale or color"):
        decode_qr_codes(np.zeros((10,), dtype=np.uint8))
