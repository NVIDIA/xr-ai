# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic gesture geometry and modular image workflow contracts."""

from __future__ import annotations

import asyncio
import io
import math
import time
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from gesture_capture_worker.agent import GestureCaptureAgent
from gesture_capture_worker.circle import PinchCircleTrigger
from gesture_capture_worker.config import CircleSettings, Settings, load_settings
from gesture_capture_worker.gesture import CameraBoxTrigger, RegionSelector, camera_box
from PIL import Image, ImageDraw
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
    for joint, tip in ((3, thumb), (7, index)):
        dx, dy = (tip[0] - corner[0]) * 400, (tip[1] - corner[1]) * 200
        length = math.hypot(dx, dy)
        points[joint] = HandLandmark(x=tip[0] - 12 * dx / length / 400,
                                    y=tip[1] - 12 * dy / length / 200, z=0)
    return TrackedHand(landmarks=points, handedness="Left", handedness_score=0.9)


def _box():
    return HandTrackingResult(width=400, height=200, hands=[
        _hand((0.2, 0.2), (0.8, 0.2), (0.2, 0.8)),
        _hand((0.8, 0.8), (0.2, 0.8), (0.8, 0.2)),
    ])


def test_geometry_needs_two_opposing_l_shapes():
    observation = _box()
    region = camera_box(observation, corner_tolerance=0.2, angle_cosine=0.4)
    assert region == ImageRegion(left=0.215, top=0.23, right=0.785, bottom=0.77)
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


@pytest.mark.parametrize("gesture", ["box", "circle"])
@pytest.mark.parametrize("aligned_photo,photo_size", [
    (False, (4000, 3000)), (True, (4000, 2000)), (True, (4000, 3000)),
])
async def test_sample_composes_tools_with_selected_video_or_aligned_photo(gesture, aligned_photo, photo_size):
    from test_image_capture import _Endpoint
    images = ImageRegistry()
    endpoint = _Endpoint()
    # A deliberately incompatible photo must not affect default video-mode queries.
    async def photo(request):
        from xr_ai_hub import ImageCaptureData
        endpoint.requests.append(request)
        await endpoint.image_callback(ImageCaptureData(
            participant_id="alice",request_id=request.request_id,pts_us=1,
            mime_type="image/png",data=_png(photo_size),
        ))
    endpoint._request_image_capture = photo
    timestamp = 100_000
    async def frame(request):
        return ImageFrame(image=images.put(_png(),owner="alice"),width=400,height=200,
                          sequence=timestamp,timestamp_us=timestamp,participant_id="alice")
    observation = _box()
    async def hands(request):
        return observation
    observed = []
    async def query(request):
        with Image.open(io.BytesIO(images.resolve(request.image))) as image:
            observed.append(image.size)
        assert request.query == "Describe the selected region"
        return ImageQueryResult(text="A blue region")
    agent = GestureCaptureAgent(
        frames=Tool("frame","Frame",CurrentFrameRequest,ImageFrame,frame),
        hands=Tool("hands","Hands",HandTrackingRequest,HandTrackingResult,hands),
        capture=StillCaptureTool(endpoint=endpoint,images=images),crop=ImageCropTool(images=images),
        query=Tool("query","Query",ImageQueryRequest,ImageQueryResult,query),
        selector=RegionSelector(_settings(hold_s=0.2, aligned_photo=aligned_photo)),
        settings=_settings(hold_s=0.2, aligned_photo=aligned_photo),
    )
    if gesture == "box":
        steps = [(100_000, _box()), (300_000, _box()), (500_000, _box())]
        expected_crop = (2280, 1080) if aligned_photo else (228, 108)
    else:
        observations = [_pinch(x, y) for x, y in _circle_points()]
        observations += [_pinch(300, 100, released=True)] * 7
        steps = [(i * 100_000, value) for i, value in enumerate(observations, start=1)]
        expected_crop = (1200, 1200) if aligned_photo else (120, 120)

    async def run_steps():
        nonlocal timestamp, observation
        answers = []
        for timestamp, observation in steps:
            answer = await agent.step("alice")
            if answer is not None:
                answers.append(answer)
                await agent.finish_turn()
        return answers

    if aligned_photo and photo_size == (4000, 3000):
        with pytest.raises(ValueError, match="Photo and video aspect ratios differ"):
            await run_steps()
        assert observed == []
    else:
        assert await run_steps() == ["A blue region"]
        assert observed == [expected_crop]
    assert len(endpoint.requests) == int(aligned_photo)



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
        async def finish_turn(self):
            pass
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
    path = Path(__file__).resolve().parents[1] / "agent-samples/gesture-capture/yaml/gesture_capture_worker.yaml"
    settings = load_settings(path)
    assert settings.aligned_photo is False
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


def _settings(**kwargs):
    return Settings(model_path=Path("model.task"), models=Path("models.json"),
                    query="Describe the selected region", **kwargs)


def _pinch(x, y, *, released=False, handedness="Left", width=400, height=200):
    points = [HandLandmark(x=x / width, y=y / height, z=0) for _ in range(21)]
    # A fixed 40px palm and 4px pinch stay invariant under image aspect ratio.
    points[5] = HandLandmark(x=(x - 20) / width, y=y / height, z=0)
    points[17] = HandLandmark(x=(x + 20) / width, y=y / height, z=0)
    distance = 24 if released else 4
    points[4] = HandLandmark(x=(x - distance / 2) / width, y=y / height, z=0)
    points[8] = HandLandmark(x=(x + distance / 2) / width, y=y / height, z=0)
    return HandTrackingResult(width=width, height=height, hands=[TrackedHand(
        landmarks=points, handedness=handedness, handedness_score=0.9,
    )])


def _circle_points(*, reverse=False, turns=1):
    direction = -1 if reverse else 1
    return [(200 + 60 * math.cos(direction * i * math.tau / 24),
             100 + 60 * math.sin(direction * i * math.tau / 24)) for i in range(24 * turns + 1)]


def _trace(trigger, path, timestamp=0, *, completed=None):
    for x, y in path:
        timestamp += 100_000
        region = trigger.observe(_pinch(x, y), timestamp)
        if completed is None:
            assert region is None
        elif region is not None:
            completed.append(region)
    return timestamp


def _release(trigger, x, y, timestamp):
    for _ in range(5):
        timestamp += 100_000
        assert trigger.observe(_pinch(x, y, released=True), timestamp) is None
    return timestamp


@pytest.mark.parametrize("reverse", [False, True])
def test_circle_completes_while_pinched_once_and_stable_release_rearms(reverse):
    trigger = PinchCircleTrigger(CircleSettings())
    timestamp = 0
    for _ in range(2):
        results = []
        timestamp = _trace(trigger, _circle_points(reverse=reverse, turns=2), timestamp, completed=results)
        assert results == [ImageRegion(left=0.35, top=0.2, right=0.65, bottom=0.8)]
        timestamp = _release(trigger, 260, 100, timestamp)


@pytest.mark.parametrize("path", [
    [(200, 100)] * 30,
    _circle_points()[:18],
    [(180 + i * 3, 100) for i in range(25)],
    [(200 + 60 * math.cos(i * math.tau / 24), 100 + 15 * math.sin(i * math.tau / 24)) for i in range(25)],
    [point for _ in range(4) for point in [(140, 40), (260, 40), (260, 160), (140, 160)]],
    [*_circle_points()[:18], *_circle_points()[17:8:-1], *_circle_points()[9:]],
])
def test_invalid_traces_never_submit(path):
    trigger = PinchCircleTrigger(CircleSettings())
    timestamp = _trace(trigger, path)
    _release(trigger, *path[-1], timestamp)


@pytest.mark.parametrize("interruption", ["gap", "dimensions"])
def test_circle_source_interruptions_discard_failed_path_and_allow_fresh_attempt(interruption):
    selector = RegionSelector(_settings())
    timestamp = _trace(selector, _circle_points()[:12])
    timestamp += 100_000
    x, y = _circle_points()[11]
    observation = _pinch(x, y)
    observation.hands.clear()
    if interruption == "gap":
        timestamp += 1_000_000
    else:
        observation.width = 800
    assert selector.observe(observation, timestamp) is None
    assert not selector._circle.active
    results = []
    timestamp = _trace(selector, _circle_points(), timestamp, completed=results)
    assert len(results) == 1
    timestamp = _release(selector, 260, 100, timestamp)
    results = []
    _trace(selector, _circle_points(), timestamp, completed=results)
    assert len(results) == 1


@pytest.mark.parametrize("hand_count", [0, 2])
def test_circle_brief_misses_keep_path_but_sustained_loss_cancels(hand_count):
    selector = RegionSelector(_settings())
    path = _circle_points()
    timestamp = _trace(selector, path[:12])
    missing = _pinch(*path[11])
    missing.hands = [] if hand_count == 0 else missing.hands * 2
    assert selector.observe(missing, timestamp + 100_000) is None
    results = []
    timestamp = _trace(selector, path[12:], timestamp + 100_000, completed=results)
    assert len(results) == 1
    timestamp = _release(selector, 260, 100, timestamp)
    timestamp = _trace(selector, path[:12], timestamp)
    for _ in range(6):
        timestamp += 100_000
        assert selector.observe(missing, timestamp) is None
    timestamp = _trace(selector, path[12:], timestamp)
    _release(selector, 260, 100, timestamp)


def test_circle_rearm_hysteresis_ignores_single_frame_open_and_threshold_noise():
    trigger = PinchCircleTrigger(CircleSettings())
    results = []
    timestamp = _trace(trigger, _circle_points(), completed=results)
    assert len(results) == 1
    assert trigger.observe(_pinch(260, 100, released=True), timestamp + 100_000) is None
    assert trigger.observe(_pinch(260, 100), timestamp + 200_000) is None
    intermediate = _pinch(260, 100)
    intermediate.hands[0].landmarks[4].x -= 0.01
    intermediate.hands[0].landmarks[8].x += 0.01
    assert trigger.observe(intermediate, timestamp + 300_000) is None
    timestamp = _trace(trigger, _circle_points(), timestamp + 300_000)
    timestamp = _release(trigger, 260, 100, timestamp)
    results = []
    _trace(trigger, _circle_points(), timestamp, completed=results)
    assert len(results) == 1


def test_circle_sample_limit_cancels_without_truncating_start():
    trigger = PinchCircleTrigger(CircleSettings(max_points=16))
    timestamp = _trace(trigger, _circle_points())
    _release(trigger, 260, 100, timestamp)


def test_circle_duplicate_frames_do_not_complete_a_trace():
    selector = RegionSelector(_settings())
    for x, y in _circle_points():
        assert selector.observe(_pinch(x, y), 100_000) is None
    _release(selector, x, y, 100_000)


def test_stationary_pinch_stays_quiet_until_drawing_progresses():
    selector = RegionSelector(_settings())
    timestamp = _trace(selector, [(200, 100)] * 20)
    assert not selector.in_progress
    assert selector.observe(_pinch(201, 100), timestamp + 100_000) is None
    assert not selector.in_progress
    assert selector.observe(_pinch(230, 100), timestamp + 200_000) is None
    assert selector.in_progress


def test_box_candidate_and_latch_survive_brief_missing_or_noisy_geometry():
    selector = RegionSelector(_settings(hold_s=0.6))
    empty = HandTrackingResult(width=400, height=200, hands=[])
    assert selector.observe(_box(), 100_000) is None
    assert selector.observe(_box(), 300_000) is None
    assert selector.observe(empty, 400_000) is None
    noisy = _box()
    noisy.hands[0].landmarks[4].x = 0.5
    assert selector.observe(noisy, 500_000) is None
    assert selector.observe(_box(), 700_000) is not None
    assert selector.observe(empty, 800_000) is None
    assert selector.observe(_box(), 900_000) is None
    assert selector.observe(_box(), 1_100_000) is None
    # Missing sensing, even sustained, cannot count as an intentional release.
    for timestamp in range(1_200_000, 2_000_000, 100_000):
        assert selector.observe(empty, timestamp) is None
    assert selector.observe(_box(), 2_000_000) is None


def test_box_reacquisition_requires_consistent_region_and_bounded_loss():
    trigger = CameraBoxTrigger(hold_s=0.6, release_s=0.4)
    region = ImageRegion(left=0.2, top=0.2, right=0.8, bottom=0.8)
    moved = ImageRegion(left=0.3, top=0.3, right=0.9, bottom=0.9)
    assert trigger.observe(region, 100_000) is None
    trigger.missing(300_000)
    assert trigger.observe(moved, 500_000) is None
    assert trigger.observe(moved, 800_000) is None
    trigger.missing(1_400_000)
    assert trigger.observe(moved, 1_500_000) is None


def test_unknown_frame_interrupts_circle_rearm_confirmation():
    selector = RegionSelector(_settings())
    timestamp = _trace(selector, _circle_points(), completed=[])
    assert selector.observe(_pinch(260, 100, released=True), timestamp + 100_000) is None
    missing = HandTrackingResult(width=400, height=200, hands=[])
    assert selector.observe(missing, timestamp + 200_000) is None
    assert selector.observe(_pinch(260, 100, released=True), timestamp + 300_000) is None
    assert selector.observe(_pinch(260, 100, released=True), timestamp + 500_000) is None
    timestamp = _trace(selector, _circle_points(), timestamp + 500_000)
    timestamp = _release(selector, 260, 100, timestamp)
    results = []
    _trace(selector, _circle_points(), timestamp, completed=results)
    assert len(results) == 1


def test_active_circle_does_not_acquire_box_from_brief_hand_transition():
    selector = RegionSelector(_settings(hold_s=0.2))
    path = _circle_points()
    timestamp = _trace(selector, path[:12])
    for _ in range(3):
        timestamp += 100_000
        assert selector.observe(_box(), timestamp) is None
    results = []
    _trace(selector, path[12:], timestamp, completed=results)
    assert len(results) == 1


@pytest.mark.parametrize("angle", [0, 20])
async def test_box_vlm_input_is_open_interior_with_target_retained(angle):
    from gesture_capture_worker import __main__ as worker
    from test_image_tools import _Vlm
    from xr_ai_tools.vision import ImageQueryTool

    observation = _box()
    radians = math.radians(angle)
    for hand in observation.hands:
        for point in hand.landmarks:
            x, y = point.x * 400 - 200, point.y * 200 - 100
            point.x = (200 + math.cos(radians) * x - math.sin(radians) * y) / 400
            point.y = (100 + math.sin(radians) * x + math.cos(radians) * y) / 200
    image = Image.new("RGB", (400, 200), "white")
    draw = ImageDraw.Draw(image)
    finger_color, target_color = (210, 160, 130), (0, 100, 220)
    for hand in observation.hands:
        base = hand.landmarks[2]
        for tip_index in (4, 8):
            tip = hand.landmarks[tip_index]
            draw.line([(base.x * 400, base.y * 200), (tip.x * 400, tip.y * 200)],
                      fill=finger_color, width=10)
    draw.rectangle((190, 90, 209, 109), fill=target_color)
    output = io.BytesIO()
    image.save(output, "PNG")
    images = ImageRegistry()
    timestamp = 100_000
    async def frame(request):
        return ImageFrame(image=images.put(output.getvalue()), width=400, height=200,
                          sequence=timestamp, timestamp_us=timestamp, participant_id="alice")
    async def hands(request):
        return observation
    vlm = _Vlm("A blue square")
    agent = GestureCaptureAgent(
        frames=Tool("frame", "Frame", CurrentFrameRequest, ImageFrame, frame),
        hands=Tool("hands", "Hands", HandTrackingRequest, HandTrackingResult, hands),
        capture=None, crop=ImageCropTool(images=images),
        query=ImageQueryTool(images=images, vlm=vlm, system_prompt=worker._REGION_SYSTEM_PROMPT),
        selector=RegionSelector(_settings(hold_s=0.2)), settings=_settings(),
    )
    assert await agent.step("alice") is None
    timestamp = 300_000
    assert await agent.step("alice") == "A blue square"
    sent_images, question, system_prompt, _headers = vlm.calls[0]
    with Image.open(io.BytesIO(sent_images[0])) as cropped:
        pixels = list(cropped.get_flattened_data())
        assert pixels.count(target_color) == 400
        assert finger_color not in pixels
    assert question == "Describe the selected region"
    assert system_prompt == worker._REGION_SYSTEM_PROMPT
