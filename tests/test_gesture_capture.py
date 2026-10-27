# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic gesture geometry and modular image workflow contracts."""

from __future__ import annotations

import asyncio
import io
import time
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from gesture_capture_worker.agent import GestureCaptureAgent
from gesture_capture_worker.gesture import CameraBoxTrigger, camera_box
from PIL import Image
from pydantic import ValidationError
from xr_ai_hub import FrameSignal, FrameUnavailable, ImageCaptureUnavailable, PixelFormat
from xr_ai_tools import Tool
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool, ImageFrame
from xr_ai_tools.hand_tracking import (
    HandLandmark,
    HandTrackingRequest,
    HandTrackingResult,
    HandTrackingTool,
    TrackedHand,
)
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.image_crop import ImageCropRequest, ImageCropTool, ImageRegion
from xr_ai_tools.still_capture import StillCaptureRequest, StillCaptureTool
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryResult


def _png(size=(400, 200)):
    output = io.BytesIO()
    Image.new("RGB", size, "blue").save(output, "PNG")
    return output.getvalue()


def _hand(corner, thumb, index):
    points = [HandLandmark(x=corner[0], y=corner[1], z=0) for _ in range(21)]
    points[2] = HandLandmark(x=corner[0], y=corner[1], z=0)
    points[5] = HandLandmark(x=corner[0], y=corner[1], z=0)
    points[4] = HandLandmark(x=thumb[0], y=thumb[1], z=0)
    points[8] = HandLandmark(x=index[0], y=index[1], z=0)
    return TrackedHand(landmarks=points, handedness="Left", handedness_score=0.9)


def _box():
    return HandTrackingResult(width=400, height=200, hands=[
        _hand((0.2, 0.2), (0.8, 0.2), (0.2, 0.8)),
        _hand((0.8, 0.8), (0.2, 0.8), (0.8, 0.2)),
    ])


def test_geometry_needs_two_opposing_l_shapes():
    observation = _box()
    region = camera_box(observation, corner_tolerance=0.2, angle_cosine=0.4)
    assert region == ImageRegion(left=0.2, top=0.2, right=0.8, bottom=0.8)
    observation.hands.reverse()
    assert camera_box(observation, corner_tolerance=0.2, angle_cosine=0.4) == region
    observation.hands.pop()
    assert camera_box(observation, corner_tolerance=0.2, angle_cosine=0.4) is None
    observation.hands.append(observation.hands[0])
    assert camera_box(observation, corner_tolerance=0.2, angle_cosine=0.4) is None


def test_hold_release_duplicate_unknown_and_gap():
    region = ImageRegion(left=0.2, top=0.2, right=0.8, bottom=0.8)
    trigger = CameraBoxTrigger(hold_s=0.2, release_s=0.2)
    assert trigger.observe(region, 100_000) is None
    assert trigger.observe(region, 100_000) is None
    assert trigger.observe(region, 300_000) == region
    trigger.unavailable()
    assert trigger.observe(region, 400_000) is None
    assert trigger.observe(None, 500_000) is None
    assert trigger.observe(None, 700_000) is None
    assert trigger.observe(region, 800_000) is None
    # A long absence of observations must not count toward a stable hold.
    assert trigger.observe(region, 2_000_000) is None
    assert trigger.observe(region, 2_200_000) == region


@pytest.mark.parametrize("bounds", [(0.5,0,0.2,1), (0,0,0,1), (-1,0,1,1), (0,0,float('nan'),1)])
def test_invalid_crop_bounds(bounds):
    with pytest.raises(ValidationError):
        ImageRegion(**dict(zip(("left","top","right","bottom"), bounds, strict=True)))


async def test_crop_maps_into_photo_pixels_and_preserves_owner():
    images = ImageRegistry()
    source = images.put(_png((4000, 2000)), owner="alice")
    result = await ImageCropTool(images=images).execute(ImageCropRequest(
        image=source, region=ImageRegion(left=0.25, top=0.25, right=0.75, bottom=0.75),
    ))
    assert (result.width, result.height) == (2000, 1000)
    with Image.open(io.BytesIO(images.resolve(result.image))) as cropped:
        assert cropped.getpixel((100, 100)) == (0, 0, 255)
    images.release_owner("alice")
    with pytest.raises(LookupError):
        images.resolve(result.image)


async def test_video_only_selection_never_requests_a_still():
    from test_image_capture import _Endpoint
    endpoint = _Endpoint()
    tool = CurrentFrameTool(endpoint=endpoint, images=ImageRegistry())
    with pytest.raises(FrameUnavailable):
        await tool.execute(CurrentFrameRequest(participant_id="alice", allow_capture=False))
    assert endpoint.requests == []


async def test_explicit_still_ignores_fresh_video():
    from test_image_capture import _Endpoint
    endpoint = _Endpoint()
    images = ImageRegistry()
    live = CurrentFrameTool(endpoint=endpoint, images=images)
    await endpoint.frame_callback(FrameSignal(
        slot=0, seq=7, pts_us=time.time_ns() // 1_000, width=2, height=2,
        fmt=PixelFormat.RGB24, data_sz=12, participant_id="alice", track_id="camera",
    ))
    assert live.frames.participants() == ["alice"]
    tool = StillCaptureTool(endpoint=endpoint, images=images)
    photo = await tool.execute(StillCaptureRequest(participant_id="alice"))
    assert len(endpoint.requests) == 1
    assert (photo.width, photo.height) == (2, 3)
    tool.close()
    with pytest.raises(ImageCaptureUnavailable, match="closed"):
        await tool.execute(StillCaptureRequest(participant_id="alice"))


async def test_sample_composes_tools_and_captures_once_until_release():
    from test_image_capture import _Endpoint
    images = ImageRegistry()
    endpoint = _Endpoint()
    # Return a large photo with the same framing as the low-resolution observations.
    async def photo(request):
        from xr_ai_hub import ImageCaptureData
        endpoint.requests.append(request)
        await endpoint.image_callback(ImageCaptureData(
            participant_id="alice",request_id=request.request_id,pts_us=1,
            mime_type="image/png",data=_png((4000,2000)),
        ))
    endpoint._request_image_capture = photo
    timestamp = 100_000
    async def frame(request):
        return ImageFrame(image=images.put(_png(),owner="alice"),width=400,height=200,
                          sequence=timestamp,timestamp_us=timestamp,participant_id="alice")
    async def hands(request):
        return _box()
    observed = []
    async def query(request):
        with Image.open(io.BytesIO(images.resolve(request.image))) as image:
            observed.append(image.size)
        return ImageQueryResult(text="TEST OCR")
    agent = GestureCaptureAgent(
        frames=Tool("frame","Frame",CurrentFrameRequest,ImageFrame,frame),
        hands=Tool("hands","Hands",HandTrackingRequest,HandTrackingResult,hands),
        capture=StillCaptureTool(endpoint=endpoint,images=images),crop=ImageCropTool(images=images),
        query=Tool("query","Query",ImageQueryRequest,ImageQueryResult,query),
        trigger=CameraBoxTrigger(hold_s=0.2,release_s=0.2),
        settings=SimpleNamespace(corner_tolerance=0.2,angle_cosine=0.4,query="Read text"),
    )
    assert await agent.step("alice") is None
    timestamp = 300_000
    assert await agent.step("alice") == "TEST OCR"
    timestamp = 500_000
    assert await agent.step("alice") is None
    assert len(endpoint.requests) == 1
    assert observed == [(2400,1200)]


async def test_hand_backend_maps_landmarks_and_close_drains_cancelled_inference(monkeypatch):
    from mediapipe.tasks.python.vision import HandLandmarker
    images = ImageRegistry()
    entered, release, close_entered, closed = Event(), Event(), Event(), Event()
    class Detector:
        def detect(self, image):
            entered.set()
            assert release.wait(5), "test did not release native inference"
            assert not closed.is_set()
            return SimpleNamespace(
                hand_landmarks=[_box().hands[0].landmarks],
                handedness=[[SimpleNamespace(category_name="Left", score=0.9)]],
            )
        def close(self):
            closed.set()
    monkeypatch.setattr(HandLandmarker, "create_from_options", lambda options: Detector())
    tool = HandTrackingTool(images=images, model_path=Path("provided.task"))
    ref = images.put(_png())
    pending = asyncio.create_task(tool.execute(HandTrackingRequest(image=ref)))
    assert await asyncio.to_thread(entered.wait, 5)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    def close():
        close_entered.set()
        tool.close()
    closing = asyncio.create_task(asyncio.to_thread(close))
    try:
        assert await asyncio.to_thread(close_entered.wait, 5)
        assert not closed.is_set()
    finally:
        release.set()
        await closing
    assert closed.is_set()
    with pytest.raises(RuntimeError, match="closed"):
        await tool.execute(HandTrackingRequest(image=ref))


async def test_hand_backend_preserves_source_coordinates(monkeypatch):
    from mediapipe.tasks.python.vision import HandLandmarker
    observation = _box()
    class Detector:
        def detect(self, image):
            assert image.width == 400 and image.height == 200
            return SimpleNamespace(
                hand_landmarks=[hand.landmarks for hand in observation.hands],
                handedness=[[SimpleNamespace(category_name=hand.handedness, score=hand.handedness_score)]
                            for hand in observation.hands],
            )
        def close(self):
            pass
    monkeypatch.setattr(HandLandmarker, "create_from_options", lambda options: Detector())
    images = ImageRegistry()
    tool = HandTrackingTool(images=images, model_path=Path("provided.task"))
    try:
        result = await tool.execute(HandTrackingRequest(image=images.put(_png())))
        assert result == observation
    finally:
        await asyncio.to_thread(tool.close)


async def test_exif_oriented_still_uses_upright_coordinates():
    from test_image_capture import _Endpoint
    from xr_ai_hub import ImageCaptureData
    endpoint = _Endpoint()
    encoded = io.BytesIO()
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (20, 10), "blue").save(encoded, "JPEG", exif=exif)
    async def respond(request):
        await endpoint.image_callback(ImageCaptureData(
            participant_id="alice", request_id=request.request_id, pts_us=1,
            mime_type="image/jpeg", data=encoded.getvalue(),
        ))
    endpoint._request_image_capture = respond
    images = ImageRegistry()
    tool = StillCaptureTool(endpoint=endpoint, images=images)
    try:
        result = await tool.execute(StillCaptureRequest(participant_id="alice"))
        assert (result.width, result.height) == (10, 20)
        with Image.open(io.BytesIO(images.resolve(result.image))) as image:
            assert image.size == (10, 20)
            assert image.getexif().get(274, 1) == 1
    finally:
        tool.close()


async def test_worker_roster_is_idempotent_and_departure_drains_polling(monkeypatch, tmp_path):
    from gesture_capture_worker import __main__ as worker
    from gesture_capture_worker.config import Settings
    from test_image_capture import _Endpoint
    from xr_ai_hub import ParticipantEvent

    running, polling, departed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    closed = []
    constructed = []

    class Endpoint(_Endpoint):
        async def run(self):
            running.set()
            await asyncio.Future()
        async def wait_until_running(self):
            await running.wait()
        def stop(self):
            closed.append("endpoint stopped")
        def close(self):
            closed.append("endpoint closed")

    class Hands:
        def close(self):
            closed.append("hands")

    class Vlm:
        async def close(self):
            closed.append("vlm")

    class Agent:
        def __init__(self, **kwargs):
            constructed.append(self)
        async def step(self, participant_id):
            polling.set()
            try:
                await asyncio.Future()
            finally:
                departed.set()

    endpoint = Endpoint()
    monkeypatch.setattr(worker, "ProcessorEndpoint", lambda **kwargs: endpoint)
    monkeypatch.setattr(worker, "HandTrackingTool", lambda **kwargs: Hands())
    monkeypatch.setattr(worker, "GestureCaptureAgent", Agent)
    monkeypatch.setattr(worker, "load_models_config", lambda path: None)
    monkeypatch.setattr(worker, "make_vlm", lambda *args: Vlm())
    settings = Settings(model_path=tmp_path / "model.task", models=tmp_path / "models.json", query="Read")
    app = asyncio.create_task(worker._run(settings, None))
    try:
        await asyncio.wait_for(running.wait(), 2)
        await endpoint.participant_callback(ParticipantEvent(participant_id="alice", joined=True, pts_us=1))
        await asyncio.wait_for(polling.wait(), 2)
        await endpoint.participant_callback(ParticipantEvent(participant_id="alice", joined=True, pts_us=2))
        assert len(constructed) == 1
        await endpoint.participant_callback(ParticipantEvent(participant_id="alice", joined=False, pts_us=3))
        assert departed.is_set()
    finally:
        app.cancel()
        with pytest.raises(asyncio.CancelledError):
            await app
    assert set(closed) == {"hands", "vlm", "endpoint stopped", "endpoint closed"}
