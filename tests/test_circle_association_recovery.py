# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Baseline-comparative stale-anchor recovery and explicit hand-role ambiguity."""

import math

from gesture_capture_worker.circle import PinchCircleTrigger
from gesture_capture_worker.config import CircleSettings
from gesture_capture_worker.grip import Grip
from test_gesture_capture import _pinch


def _path(center_x=200):
    return [(center_x + 40 * math.cos(i * math.tau / 24), 100 + 40 * math.sin(i * math.tau / 24))
            for i in range(25)]


def test_moved_return_after_loss_matches_pre_occlusion_baseline_recognition():
    # Exact-source comparison:34b72f9c recognizes3.1s; de7d3e077 never
    # recognizes because stale association rejects until2.5s. Synthetic positions
    # reproduce the logged rejection pattern; they are not captured user joints.
    trigger = PinchCircleTrigger(CircleSettings())
    for timestamp in (100_000, 200_000, 300_000):
        assert trigger.observe(_pinch(100, 100), timestamp) is None
    before = list(trigger._points)
    for timestamp in range(400_000, 1_100_000, 100_000):
        missing = _pinch(100, 100)
        missing.hands.clear()
        assert trigger.observe(missing, timestamp) is None
    assert trigger._points == before  # genuine absence still has2s grace
    hits = []
    for i, point in enumerate(_path(300)):
        timestamp = 1_100_000 + i * 100_000
        region = trigger.observe(_pinch(*point), timestamp)
        if i == 0:
            assert trigger._points == [point]  # stale anchor clears old stroke
            assert len(trigger._segments) == 0 and region is None
        if region is not None:
            hits.append(timestamp)
    assert hits == [3_100_000]


def test_only_gesturing_hand_is_selected_among_clearly_open_bystanders():
    trigger = PinchCircleTrigger(CircleSettings())
    hits = []
    for i, point in enumerate(_path()):
        observation = _pinch(*point)
        # Both are near the previous position: proximity alone stays ambiguous.
        observation.hands += _pinch(220, 110, released=True).hands
        region = trigger.observe(observation, (i + 1) * 100_000)
        if region is not None:
            hits.append(region)
    assert len(hits) == 1


def test_two_pinched_or_uncertain_actors_do_not_handoff_old_stroke():
    for second_ratio in (.1, .35):
        trigger = PinchCircleTrigger(CircleSettings())
        for i, point in enumerate(_path()[:8]):
            assert trigger.observe(_pinch(*point), (i + 1) * 100_000) is None
        before = list(trigger._points)
        for i in range(8, 18):
            observation = _pinch(*_path()[i])
            other = _pinch(220, 110)
            other.hands[0].landmarks[4].x = (220 - second_ratio * 20) / 400
            other.hands[0].landmarks[8].x = (220 + second_ratio * 20) / 400
            observation.hands += other.hands
            assert trigger.observe(observation, (i + 1) * 100_000) is None
        assert trigger._points == before
        assert trigger._grip.state is Grip.PINCHED


def test_far_open_hand_does_not_reanchor_or_release_within_occlusion_grace():
    trigger = PinchCircleTrigger(CircleSettings())
    for timestamp in (100_000, 200_000):
        assert trigger.observe(_pinch(100, 100), timestamp) is None
    before = list(trigger._points)
    for timestamp in range(300_000, 1_500_000, 100_000):
        assert trigger.observe(_pinch(340, 100, released=True), timestamp) is None
    assert trigger._points == before and trigger._grip.state is Grip.PINCHED


def test_nearby_band_motion_is_retained_with_passive_open_hand():
    trigger = PinchCircleTrigger(CircleSettings())
    trigger.observe(_pinch(200, 100), 100_000)
    trigger.observe(_pinch(201, 100), 200_000)
    band = _pinch(220, 100)
    band.hands[0].landmarks[4].x = 213 / 400
    band.hands[0].landmarks[8].x = 227 / 400
    band.hands += _pinch(220, 110, released=True).hands
    assert trigger.observe(band, 300_000) is None
    assert trigger._points[-1] == (220.0, 100.0)


def test_completed_turn_can_rearm_from_unique_open_hand_after_moving():
    trigger = PinchCircleTrigger(CircleSettings())
    hits = [trigger.observe(_pinch(*point), (i + 1) * 100_000) for i, point in enumerate(_path())]
    assert sum(region is not None for region in hits) == 1 and not trigger._ready
    for timestamp in range(3_100_000, 3_700_000, 100_000):
        assert trigger.observe(_pinch(10, 10, released=True), timestamp) is None
    assert trigger._ready and trigger._grip.state is Grip.OPEN


def test_completed_release_observation_is_not_hidden_by_far_pinched_bystander():
    trigger = PinchCircleTrigger(CircleSettings())
    for i, point in enumerate(_path()):
        trigger.observe(_pinch(*point), (i + 1) * 100_000)
    position = trigger.hand.pointer
    for timestamp in range(2_700_000, 3_300_000, 100_000):
        observation = _pinch(*position, released=True)
        observation.hands += _pinch(20, 20).hands
        assert trigger.observe(observation, timestamp) is None
    assert trigger._ready and trigger._grip.state is Grip.OPEN


def test_reliable_nearby_hand_keeps_drawing_with_distant_pinched_bystander():
    trigger = PinchCircleTrigger(CircleSettings())
    trigger.observe(_pinch(*_path()[0]), 100_000)
    trigger.observe(_pinch(*_path()[0]), 200_000)
    hits = []
    for i, point in enumerate(_path()):
        observation = _pinch(*point)
        observation.hands += _pinch(20, 20).hands
        region = trigger.observe(observation, 300_000 + i * 100_000)
        if region is not None:
            hits.append(region)
    assert len(hits) == 1


def test_stale_completed_anchor_does_not_use_open_bystander_to_release_held_pinch():
    trigger = PinchCircleTrigger(CircleSettings())
    for i, point in enumerate(_path()):
        trigger.observe(_pinch(*point), (i + 1) * 100_000)
    previous = trigger.hand.pointer
    for timestamp in range(3_100_000, 5_000_000, 100_000):
        observation = _pinch(340, 100)  # held active hand moved beyond stale anchor
        observation.hands += _pinch(*previous, released=True).hands  # open bystander occupies old location
        assert trigger.observe(observation, timestamp) is None
    assert not trigger._ready and trigger._completed_latch
    assert trigger._grip.state is Grip.PINCHED
