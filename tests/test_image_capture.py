# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-driven client image capture contracts."""

from __future__ import annotations

import asyncio
import io
import json
import time

import pytest
from PIL import Image
from xr_ai_hub import (
    ClientImageCaptureSource,
    FrameData,
    FrameSignal,
    ImageCaptureData,
    ImageCaptureUnavailable,
    PixelFormat,
)
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.image import ImageRegistry


@pytest.fixture(autouse=True)
def _run_cpu_helpers_inline(monkeypatch) -> None:
    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)


def _jpeg() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (2, 3), "blue").save(output, "JPEG")
    return output.getvalue()


class _Endpoint:
    def __init__(self, *, respond: bool = True) -> None:
        self.connected_participants = frozenset({"alice"})
        self.respond = respond
        self.requests = []
        self.cancels = []

    def on_frame(self, callback) -> None:
        self.frame_callback = callback

    def on_image_capture(self, callback):
        self.image_callback = callback
        return lambda: None

    def on_participant(self, callback) -> None:
        self.participant_callback = callback

    async def request_image_capture(self, request) -> None:
        self.requests.append(request)
        if self.respond:
            await self.image_callback(ImageCaptureData(
                participant_id=request.participant_id,
                request_id=request.request_id,
                pts_us=123,
                mime_type="image/jpeg",
                data=_jpeg(),
            ))

    async def cancel_image_capture(self, cancel) -> None:
        self.cancels.append(cancel)

    async def request_frame(self, signal) -> FrameData:
        return FrameData(
            seq=signal.seq,
            pts_us=signal.pts_us,
            width=2,
            height=2,
            fmt=PixelFormat.RGB24,
            data=bytes([0, 0, 255] * 4),
            participant_id=signal.participant_id,
            track_id=signal.track_id,
        )


async def test_current_frame_falls_back_to_client_capture() -> None:
    endpoint = _Endpoint()
    images = ImageRegistry()
    tool = CurrentFrameTool(endpoint=endpoint, images=images)  # type: ignore[arg-type]

    result = await tool._get_current_frame(CurrentFrameRequest(participant_id="alice"))

    assert result.participant_id == "alice"
    assert (result.width, result.height) == (2, 3)
    assert images.resolve(result.image) == _jpeg()
    assert len(endpoint.requests) == 1
    assert endpoint.cancels == []


async def test_current_frame_does_not_request_capture_while_video_is_fresh() -> None:
    endpoint = _Endpoint()
    images = ImageRegistry()
    tool = CurrentFrameTool(endpoint=endpoint, images=images)  # type: ignore[arg-type]
    await endpoint.frame_callback(FrameSignal(
        slot=0,
        seq=7,
        pts_us=time.time_ns() // 1_000,
        width=2,
        height=2,
        fmt=PixelFormat.RGB24,
        data_sz=12,
        participant_id="alice",
        track_id="camera",
    ))

    result = await tool._get_current_frame(CurrentFrameRequest(participant_id="alice"))

    assert result.sequence == 7
    assert endpoint.requests == []


async def test_timed_out_capture_is_cancelled_on_the_client() -> None:
    endpoint = _Endpoint(respond=False)
    source = ClientImageCaptureSource(endpoint, timeout_s=0.01)  # type: ignore[arg-type]

    with pytest.raises(ImageCaptureUnavailable, match="timeout"):
        await source.capture("alice")

    assert len(endpoint.cancels) == 1
    assert endpoint.cancels[0].request_id == endpoint.requests[0].request_id


async def test_cancelled_capture_is_cancelled_on_the_client() -> None:
    endpoint = _Endpoint(respond=False)
    source = ClientImageCaptureSource(endpoint, timeout_s=1)  # type: ignore[arg-type]
    pending = asyncio.create_task(source.capture("alice"))
    await asyncio.sleep(0)

    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert len(endpoint.cancels) == 1
    assert endpoint.cancels[0].request_id == endpoint.requests[0].request_id


async def test_capture_rejects_a_disconnected_participant() -> None:
    endpoint = _Endpoint()
    endpoint.connected_participants = frozenset()
    source = ClientImageCaptureSource(endpoint)  # type: ignore[arg-type]

    with pytest.raises(ImageCaptureUnavailable, match="not connected"):
        await source.capture("alice")


async def test_capture_request_and_image_route_through_real_hub(
    hub,
    make_connector,
    make_processor,
    settle,
) -> None:
    endpoint = make_processor()
    connector = make_connector()
    await connector.register()
    await settle()
    await connector.notify_participant_joined("alice", pts_us=1)
    await settle()

    async def respond(message) -> None:
        if message.topic != "camera.capture.request":
            return
        request = json.loads(message.data)
        await connector.push_image_capture(ImageCaptureData(
            participant_id="alice",
            request_id=request["request_id"],
            pts_us=123,
            mime_type="image/jpeg",
            data=_jpeg(),
        ))

    connector.on_return_data(respond)
    connector_task = asyncio.create_task(connector.run())
    try:
        image = await ClientImageCaptureSource(endpoint, timeout_s=1).capture("alice")
    finally:
        connector_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await connector_task

    assert image.participant_id == "alice"
    assert image.data == _jpeg()
