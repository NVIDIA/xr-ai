# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression playback separates grip uncertainty from observed pointer motion."""

import pytest
from gesture_capture_worker.circle import PinchCircleTrigger
from gesture_capture_worker.config import CircleSettings
from gesture_capture_worker.grip import Grip, GripHysteresis
from test_circle_selection_wait import _path
from test_gesture_capture import _pinch


def _grip():
    return GripHysteresis(pinch_ratio=0.25, release_ratio=0.45, dwell_s=0.2, max_gap_s=0.5)


def test_two_closed_observations_enter_and_band_retains_observed_motion():
    grip = _grip()
    assert not grip.observe(0.1, 0)
    assert grip.state is Grip.OPEN
    assert not grip.observe(0.35, 80_000)
    assert grip.observe(0.2, 160_000)
    assert grip.state is Grip.PINCHED
    assert grip.observe(0.35, 240_000)
    # Unconfirmed release evidence retains grip and usable observed motion.
    assert grip.observe(0.8, 320_000)
    assert grip.state is Grip.PINCHED
    assert grip.observe(0.3, 400_000)
    assert grip.observe(0.8, 480_000)
    assert not grip.observe(None, 560_000)
    assert grip.observe(0.8, 640_000)
    assert grip.state is Grip.PINCHED
    assert not grip.observe(0.8, 840_000)
    assert grip.state is Grip.OPEN


def test_missing_observations_retain_grip_until_positive_open_evidence():
    grip = _grip()
    grip.observe(0.1, 0)
    assert grip.observe(0.1, 80_000)
    assert not grip.observe(None, 240_000)
    assert grip.state is Grip.PINCHED
    assert not grip.observe(None, 640_000)
    assert grip.state is Grip.PINCHED
    for timestamp in (720_000, 800_000, 880_000):
        assert grip.observe(0.1, timestamp)
    assert grip.state is Grip.PINCHED
    grip.observe(0.8, 960_000)
    grip.observe(None, 1_040_000)
    grip.observe(0.8, 1_120_000)
    assert grip.state is Grip.PINCHED
    grip.observe(0.8, 1_320_000)
    assert grip.state is Grip.OPEN


def test_duplicate_and_missing_samples_do_not_supply_second_closed_observation():
    grip = _grip()
    for _ in range(10):
        assert not grip.observe(0.1, 80_000)
    grip.observe(None, 160_000)
    assert not grip.observe(0.1, 240_000)
    assert grip.observe(0.1, 320_000)


@pytest.mark.parametrize("mode", ["steady", "distance_band_jitter", "alternating_entry_band", "relative_z_jitter"])
def test_default_cadence_playback_recovers_prior_working_strokes(mode):
    # Same 1.12s/14-point trace was compared against actual 360e03d87 and
    # 418f026c sources. Baseline accepted all four; 418 accepted only steady.
    # This is perturbed synthetic playback, not a recorded physical hand.
    trigger = PinchCircleTrigger(CircleSettings())
    results = []
    for i, point in enumerate(_path(310, 14)):
        observation = _pinch(*point)
        if mode in ("distance_band_jitter", "alternating_entry_band"):
            ratio = ([.1, .15, .32, .18, .35, .2, .3][i % 7] if mode == "distance_band_jitter"
                     else [.15, .35][i % 2])
            distance = 40 * ratio
            observation.hands[0].landmarks[4].x = (point[0] - distance / 2) / 400
            observation.hands[0].landmarks[8].x = (point[0] + distance / 2) / 400
        elif mode == "relative_z_jitter":
            observation.hands[0].landmarks[4].z = [0, -.015, -.02][i % 3]
            observation.hands[0].landmarks[8].z = [0, .015, .02][i % 3]
        if trigger.observe(observation, (i + 1) * 80_000):
            results.append((i + 1) * 80_000)
    assert results == [1_120_000]


def test_immediate_moving_pinch_preserves_entry_points():
    trigger = PinchCircleTrigger(CircleSettings())
    results = [trigger.observe(_pinch(*p), (i + 1) * 80_000) for i, p in enumerate(_path(310, 8))]
    assert all(result is None for result in results[:-1])
    assert results[-1] is not None


def test_band_motion_is_observed_but_missing_pointer_is_never_invented():
    trigger = PinchCircleTrigger(CircleSettings())
    trigger.observe(_pinch(200, 100), 80_000)
    trigger.observe(_pinch(205, 100), 160_000)
    middle = _pinch(230, 100)
    middle.hands[0].landmarks[4].x = 223 / 400
    middle.hands[0].landmarks[8].x = 237 / 400
    assert trigger.observe(middle, 240_000) is None
    assert trigger._points[-1] == (230, 100)
    before = list(trigger._points)
    missing = _pinch(240, 100)
    missing.hands.clear()
    assert trigger.observe(missing, 320_000) is None
    assert trigger._points == before
    assert trigger.observe(missing, 2_880_000) is None
    assert not trigger.active
    results = [trigger.observe(_pinch(*p), 2_960_000 + i * 80_000)
               for i, p in enumerate(_path(360, 14))]
    assert sum(result is not None for result in results) == 1


def test_entry_positive_evidence_expires_despite_usable_band_frames():
    grip = _grip()
    assert not grip.observe(.1, 80_000)
    for timestamp in range(160_000, 800_000, 80_000):
        assert not grip.observe(.35, timestamp)
    assert not grip.observe(.1, 800_000)
    assert grip.observe(.1, 880_000)


def test_failed_trace_recovers_from_six_closed_samples_without_reusing_old_stroke():
    trigger = PinchCircleTrigger(CircleSettings())
    path = _path(310, 14)
    for i, point in enumerate(path[:7]):
        assert trigger.observe(_pinch(*point), (i + 1) * 80_000) is None
    missing = _pinch(200, 100)
    missing.hands.clear()
    assert trigger.observe(missing, 2_800_000) is None
    assert not trigger.active
    # The old implementation blocked these indefinitely after FAILED loss.
    for i, ratio in enumerate((.038, .16, .22, .301, .18, .27, .11, .2)):
        observation = _pinch(*path[0])
        observation.hands[0].landmarks[4].x = (path[0][0] - ratio * 20) / 400
        observation.hands[0].landmarks[8].x = (path[0][0] + ratio * 20) / 400
        assert trigger.observe(observation, 2_880_000 + i * 80_000) is None
    assert trigger._grip.state is Grip.PINCHED
    assert trigger._points == [path[0]]
    # Only the new complete trajectory can recognize; none of the old arc survives.
    results = [trigger.observe(_pinch(*point), 3_520_000 + i * 80_000)
               for i, point in enumerate(path)]
    assert sum(region is not None for region in results) == 1


def test_completed_circle_stays_latched_across_loss_and_held_pinch():
    trigger = PinchCircleTrigger(CircleSettings())
    path = _path(310, 14)
    results = [trigger.observe(_pinch(*point), (i + 1) * 80_000) for i, point in enumerate(path)]
    assert sum(region is not None for region in results) == 1
    trigger.reset()
    for i, point in enumerate(path):
        assert trigger.observe(_pinch(*point), 1_800_000 + i * 80_000) is None
    assert not trigger._ready
    for i in range(6):
        assert trigger.observe(_pinch(*path[0], released=True), 3_000_000 + i * 100_000) is None
    results = [trigger.observe(_pinch(*point), 3_600_000 + i * 80_000) for i, point in enumerate(path)]
    assert sum(region is not None for region in results) == 1


@pytest.mark.parametrize("legacy", [False, True])
def test_unconfirmed_open_motion_does_not_strand_circle_pointer(monkeypatch, legacy):
    import math
    if legacy:
        previous = GripHysteresis.observe
        def suppress_open_motion(self, ratio, timestamp):
            permitted = previous(self, ratio, timestamp)
            return permitted and (ratio is None or ratio < self._release)
        monkeypatch.setattr(GripHysteresis, "observe", suppress_open_motion)
    trigger = PinchCircleTrigger(CircleSettings())
    recognized = []
    for i in range(21):
        angle = math.radians(18 * i)
        point = (200 + 90 * math.cos(angle), 100 + 90 * math.sin(angle))
        region = trigger.observe(_pinch(*point, released=5 <= i <= 7), (i + 1) * 100_000)
        if i == 7:
            assert trigger._grip.state is Grip.PINCHED
            assert len(trigger._points) == (5 if legacy else 8)
        if region is not None:
            recognized.append(i)
    # Same samples: previous policy freezes the pointer and resets the stroke;
    # retained-grip policy recognizes the actual observed near-closed loop.
    assert recognized == ([] if legacy else [17])


def test_confirmed_open_sample_cannot_complete_circle():
    trigger = PinchCircleTrigger(CircleSettings())
    path = _path(310, 14)
    for i, point in enumerate(path[:13]):
        assert trigger.observe(_pinch(*point), (i + 1) * 80_000) is None
    # Start release confirmation at a stationary actual observation, then place
    # the last point only on the confirming OPEN frame: it must not recognize.
    assert trigger.observe(_pinch(*path[12], released=True), 1_120_000) is None
    assert trigger._grip.state is Grip.PINCHED
    assert trigger.observe(_pinch(*path[-1], released=True), 1_520_000) is None
    assert trigger._grip.state is Grip.OPEN and not trigger.active
