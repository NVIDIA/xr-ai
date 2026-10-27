# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Positive grip evidence, depth-aware projection and bounded uncertainty."""

import math

import pytest
from gesture_capture_worker.circle import PinchCircleTrigger
from gesture_capture_worker.config import CircleSettings
from gesture_capture_worker.grip import Grip, GripHysteresis
from gesture_capture_worker.hand_motion import match_hand
from test_circle_selection_wait import _path
from test_gesture_capture import _pinch


def _grip():
    return GripHysteresis(pinch_ratio=0.25, release_ratio=0.45, dwell_s=0.2, max_gap_s=0.5)


def test_entry_and_release_need_continuous_positive_evidence():
    grip = _grip()
    assert not grip.observe(0.1, 0)
    assert grip.state is Grip.OPEN
    assert not grip.observe(None, 100_000)
    assert not grip.observe(0.1, 200_000)
    assert not grip.observe(0.1, 300_000)
    assert grip.observe(0.1, 400_000)
    assert grip.state is Grip.PINCHED
    # One open outlier cannot release; intermediate/unknown retains grip but
    # supplies neither trajectory evidence nor elapsed confirmation time.
    assert not grip.observe(0.8, 500_000)
    assert not grip.observe(0.35, 600_000)
    assert grip.state is Grip.PINCHED
    assert grip.observe(0.1, 700_000)
    assert not grip.observe(0.8, 800_000)
    assert not grip.observe(None, 900_000)
    assert not grip.observe(0.8, 1_000_000)
    assert grip.state is Grip.PINCHED
    assert not grip.observe(0.8, 1_100_000)
    assert not grip.observe(0.8, 1_200_000)
    assert grip.state is Grip.OPEN


def test_unknown_timeout_cancels_without_interpreting_loss_as_release():
    grip = _grip()
    for timestamp in (0, 100_000, 200_000):
        grip.observe(0.1, timestamp)
    assert grip.state is Grip.PINCHED
    assert not grip.observe(None, 400_000)
    assert grip.state is Grip.PINCHED
    assert not grip.observe(None, 800_000)
    assert grip.state is Grip.UNKNOWN
    for timestamp in (900_000, 1_000_000, 1_100_000):
        assert not grip.observe(0.1, timestamp)
    assert grip.state is Grip.UNKNOWN
    for timestamp in (1_200_000, 1_300_000, 1_400_000):
        grip.observe(0.8, timestamp)
    assert grip.state is Grip.OPEN


@pytest.mark.parametrize("angle", [0, 60, 85, 180])
@pytest.mark.parametrize("released", [False, True])
def test_depth_distance_preserves_grip_under_hand_rotation(angle, released):
    observation = _pinch(200, 100, released=released)
    # Rotate a known 3D hand around y. Tip and palm distances are preserved
    # while projected distances foreshorten; at nearly edge-on treat as unknown.
    for point in observation.hands[0].landmarks:
        x = (point.x - 0.5) * 400
        point.x = (200 + x * math.cos(math.radians(angle))) / 400
        point.z = -x * math.sin(math.radians(angle)) / 400
    pose = match_hand(observation, None, 2)
    if angle == 85:
        assert pose.ratio is None  # geometry degeneracy, not visibility confidence
    else:
        assert pose.ratio == pytest.approx(0.6 if released else 0.1)


def test_projected_tip_overlap_is_not_positive_pinch_when_depth_separates_tips():
    observation = _pinch(200, 100)
    observation.hands[0].landmarks[4].z = -0.04
    observation.hands[0].landmarks[8].z = 0.04
    pose = match_hand(observation, None, 2)
    assert pose.ratio > 0.45
    grip = _grip()
    for timestamp in (0, 100_000, 200_000):
        assert not grip.observe(pose.ratio, timestamp)
    assert grip.state is Grip.OPEN


def test_immediate_moving_pinch_preserves_first_arc_during_entry_confirmation():
    trigger = PinchCircleTrigger(CircleSettings())
    path = _path(310, 8)
    results = [trigger.observe(_pinch(*p), (i + 1) * 80_000) for i, p in enumerate(path)]
    assert all(result is None for result in results[:-1])
    assert results[-1] is not None  # no stationary hold before the 560ms trace


def test_uncertain_motion_never_enters_confirmed_circle_path():
    trigger = PinchCircleTrigger(CircleSettings(), release_s=0.2)
    for timestamp in (100_000, 200_000, 300_000):
        trigger.observe(_pinch(200, 100), timestamp)
    assert trigger._grip.state is Grip.PINCHED
    before = list(trigger._points)
    middle = _pinch(230, 100)
    middle.hands[0].landmarks[4].x = (230 - 7) / 400
    middle.hands[0].landmarks[8].x = (230 + 7) / 400
    assert trigger.observe(middle, 400_000) is None
    assert trigger._points == before and trigger._grip.state is Grip.PINCHED
    # Sustained middle-band evidence cannot keep an attempt alive indefinitely.
    assert trigger.observe(middle, 900_000) is None
    assert not trigger.active
    assert trigger._grip.state is Grip.UNKNOWN
    for i, p in enumerate(_path(360, 14)):
        assert trigger.observe(_pinch(*p), 1_000_000 + i * 80_000) is None


def test_duplicate_samples_cannot_confirm_grip():
    grip = _grip()
    for _ in range(10):
        assert not grip.observe(0.1, 100_000)
    assert grip.state is Grip.OPEN
    assert not grip.observe(0.1, 200_000)
    assert grip.observe(0.1, 300_000)


def test_pending_selection_does_not_depend_on_grip_evidence():
    from test_circle_selection_wait import _frame, _wait

    wait = _wait()
    observation = _pinch(220, 100)
    observation.hands[0].landmarks[5].x = 219 / 400
    observation.hands[0].landmarks[17].x = 221 / 400
    observation.hands[0].landmarks[5].z = -20 / 400
    observation.hands[0].landmarks[17].z = 20 / 400
    assert match_hand(observation, wait.hand, 2).ratio is None
    for timestamp in range(200_000, 2_100_000, 100_000):
        assert not wait.observe(observation, _frame(timestamp))
    assert wait.observe(observation, _frame(2_100_000))
