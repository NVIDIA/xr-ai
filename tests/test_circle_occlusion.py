# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deliberate grip, bounded hand occlusion, and observed-only circle travel."""

import asyncio
import math
from types import SimpleNamespace

from gesture_capture_worker.circle import PinchCircleTrigger
from gesture_capture_worker.config import CircleSettings
from gesture_capture_worker.feedback import GestureFeedback
from gesture_capture_worker.gesture import RegionSelector
from gesture_capture_worker.grip import Grip
from test_gesture_capture import _circle_points, _pinch, _settings


def _missing():
    observation = _pinch(200, 100)
    observation.hands.clear()
    return observation


def test_one_second_occlusion_retains_grip_path_and_resumes_without_new_engagement():
    selector = RegionSelector(_settings())
    path = _circle_points()
    timestamp = 0
    for point in path[:12]:
        timestamp += 100_000
        assert selector.observe(_pinch(*point), timestamp) is None
    trigger = selector._circle
    before = list(trigger._points)
    for _ in range(12):
        timestamp += 100_000
        assert selector.observe(_missing(), timestamp) is None
        assert trigger._points == before and trigger._grip.state is Grip.PINCHED
        assert selector.in_progress
    timestamp += 100_000
    assert selector.observe(_pinch(*path[11]), timestamp) is None
    assert trigger._points == before  # repeated same position adds no new distinct sample
    results = []
    for point in path[12:]:
        timestamp += 100_000
        result = selector.observe(_pinch(*point), timestamp)
        if result is not None:
            results.append(result)
    assert len(results) == 1


def test_unseen_arc_does_not_supply_winding_or_complete_on_return():
    trigger = PinchCircleTrigger(CircleSettings())
    def point(degrees):
        return 200 + 40 * math.cos(math.radians(degrees)), 100 + 40 * math.sin(math.radians(degrees))
    timestamp = 0
    for degrees in range(0, 256, 15):
        timestamp += 100_000
        assert trigger.observe(_pinch(*point(degrees)), timestamp) is None
    before = list(trigger._points)
    for _ in range(10):
        timestamp += 100_000
        assert trigger.observe(_missing(), timestamp) is None
    timestamp += 100_000
    assert trigger.observe(_pinch(*point(0)), timestamp) is None  # 105° unseen gap supplies NO winding
    assert trigger._points == [*before, point(0)]  # reliable return is retained, but no gap segment
    assert len(trigger._segments) == len(before) - 1
    for degrees in (15, 30):
        timestamp += 100_000
        assert trigger.observe(_pinch(*point(degrees)), timestamp) is None  # observed total still <300°
    timestamp += 100_000
    assert trigger.observe(_pinch(*point(45)), timestamp) is not None


def test_real_return_sample_is_retained_in_fit_bounds_and_point_count():
    trigger = PinchCircleTrigger(CircleSettings())
    for i, point in enumerate(_circle_points()[:10]):
        assert trigger.observe(_pinch(*point), (i + 1) * 100_000) is None
    before = list(trigger._points)
    assert trigger.observe(_missing(), 1_100_000) is None
    assert trigger.observe(_pinch(175, 180), 1_200_000) is None
    assert trigger._points == [*before, (175, 180)]
    assert max(y for _, y in trigger._points) == 180
    assert len(trigger._segments) == len(before) - 1


def test_timeout_cancels_attempt_without_releasing_grip_or_completing():
    trigger = PinchCircleTrigger(CircleSettings())
    for i, point in enumerate(_circle_points()[:10]):
        assert trigger.observe(_pinch(*point), (i + 1) * 100_000) is None
    for timestamp in range(1_100_000, 3_200_000, 100_000):
        assert trigger.observe(_missing(), timestamp) is None
    assert not trigger.active and trigger._grip.state is Grip.PINCHED
    assert trigger.observe(_pinch(200, 100), 3_200_000) is None
    assert trigger.active and trigger._grip.state is Grip.PINCHED
    for timestamp in range(3_400_000, 4_000_000, 100_000):
        assert trigger.observe(_pinch(200, 100, released=True), timestamp) is None
    assert trigger._grip.state is Grip.OPEN and not trigger.active


def test_far_hand_cannot_resume_the_old_trace_and_camera_gap_still_discards_it():
    selector = RegionSelector(_settings())
    for i, point in enumerate(_circle_points()[:10]):
        assert selector.observe(_pinch(*point), (i + 1) * 100_000) is None
    before = list(selector._circle._points)
    for timestamp in range(1_100_000, 2_000_000, 100_000):
        assert selector.observe(_pinch(10, 10), timestamp) is None
        if timestamp <= 1_500_000:
            assert selector._circle._points == before
    assert selector._circle._points == [(10.0, 10.0)]  # fresh stroke only; no handoff of old points
    # A true source-frame gap >max_gap_s cancels even though hand grace is2s.
    assert selector.observe(_missing(), 2_600_000) is None
    assert not selector._circle.active
    assert selector._circle._grip.state is Grip.PINCHED


async def test_progress_hum_renews_on_fresh_frames_during_occlusion_then_stops_at_cancel():
    sent = []
    async def send(chunk):
        sent.append(chunk.samples)
    feedback = GestureFeedback(SimpleNamespace(send_return_audio=send), 'test', enabled=True, max_gap_s=.05)
    selector = RegionSelector(_settings())
    timestamp = 0
    try:
        for point in _circle_points()[:10]:
            timestamp += 100_000
            selector.observe(_pinch(*point), timestamp)
        await feedback.update(selector.in_progress)
        task = feedback._hum_task
        for _ in range(12):
            timestamp += 100_000
            assert selector.observe(_missing(), timestamp) is None
            await feedback.update(selector.in_progress)
            await asyncio.sleep(.02)
            assert feedback._hum_task is task and not task.done()
        assert sent
        for _ in range(10):
            timestamp += 100_000
            selector.observe(_missing(), timestamp)
            await feedback.update(selector.in_progress)
        assert feedback._hum_task is None
    finally:
        await feedback.close()


def test_fast_eight_point_circle_keeps_real_return_sample_and_observed_travel():
    trigger = PinchCircleTrigger(CircleSettings())
    results = []
    timestamp = 0
    for i in range(8):
        degrees = 360 * i / 7
        point = (200 + 40 * math.cos(math.radians(degrees)), 100 + 40 * math.sin(math.radians(degrees)))
        if i == 3:
            for _ in range(8):
                timestamp += 100_000
                assert trigger.observe(_missing(), timestamp) is None
        timestamp += 80_000
        result = trigger.observe(_pinch(*point), timestamp)
        if i == 3:
            assert result is None and len(trigger._points) == 4
            assert len(trigger._segments) == 2  # A→B is not an observed segment
        if result is not None:
            results.append(result)
    assert len(results) == 1  # seven segments minus one gap still observe~309°


def test_hand_grace_camera_freshness_and_postrecognition_wait_are_separate_defaults():
    settings = _settings()
    assert settings.circle.occlusion_grace_s == 2.0
    assert settings.max_gap_s == .5
    assert settings.circle.clear_timeout_s == 2.0
