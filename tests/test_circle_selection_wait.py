# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Noisy single-hand traces and frozen-circle fresh-frame capture boundaries."""

import math
from types import SimpleNamespace

import pytest
from gesture_capture_worker.agent import GestureCaptureAgent
from gesture_capture_worker.circle import PinchCircleTrigger
from gesture_capture_worker.config import CircleSettings
from gesture_capture_worker.hand_motion import match_hand
from gesture_capture_worker.selection_wait import CircleCaptureWait
from test_gesture_capture import _pinch, _settings
from xr_ai_tools.hand_tracking import HandTrackingResult
from xr_ai_tools.image import ImageReference
from xr_ai_tools.image_crop import ImageRegion


def _frame(timestamp, **kwargs):
    return SimpleNamespace(timestamp_us=timestamp, width=400, height=200, track_id="camera",
                           sequence=timestamp, image=ImageReference(uri=str(timestamp)), **kwargs)


def _path(degrees=310, count=12):
    noise = 0 if degrees == 300 else 1.5
    return [(200 + (60 + math.sin(i) * noise) * math.cos(math.radians(degrees * i / (count - 1))),
             100 + (60 + math.sin(i) * noise) * math.sin(math.radians(degrees * i / (count - 1))))
            for i in range(count)]


@pytest.mark.parametrize("degrees", [300, 310, 360])
def test_fast_near_circle_survives_label_flicker_ghost_and_dropouts(degrees):
    trigger = PinchCircleTrigger(CircleSettings())
    regions = []
    timestamp = 0
    for i, point in enumerate(_path(degrees, 14)):
        timestamp += 80_000
        observation = _pinch(*point, handedness="Right" if i % 2 else "Left")
        if i == 4:
            # Rejected observations don't invent motion or replace the active hand.
            assert trigger.observe(_pinch(20, 20), timestamp) is None
            timestamp += 80_000
            assert trigger.observe(HandTrackingResult(width=400, height=200, hands=[]), timestamp) is None
            timestamp += 80_000
        if i == 7:
            observation.hands += _pinch(20, 20).hands
        result = trigger.observe(observation, timestamp)
        if result is not None:
            regions.append(result)
    assert len(regions) == 1
    assert trigger.observe(_pinch(*_path(degrees, 14)[-1]), timestamp + 80_000) is None


@pytest.mark.parametrize("degrees", [240, 270])
def test_open_arcs_still_rejected(degrees):
    trigger = PinchCircleTrigger(CircleSettings())
    assert all(trigger.observe(_pinch(*p), (i + 1) * 80_000) is None
               for i, p in enumerate(_path(degrees)))


def test_sustained_outliers_do_not_extend_trace_grace():
    trigger = PinchCircleTrigger(CircleSettings())
    path = _path(360, 16)
    for i, p in enumerate(path[:8]):
        assert trigger.observe(_pinch(*p), (i + 1) * 80_000) is None
    for i in range(8, 17):
        assert trigger.observe(_pinch(0, 0), (i + 1) * 80_000) is None
    for i, p in enumerate(path[8:], 18):
        assert trigger.observe(_pinch(*p), (i + 1) * 80_000) is None


def _wait(**kwargs):
    settings = _settings(release_s=0.2, **kwargs)
    hand = match_hand(_pinch(220, 100), None, 2)
    region = ImageRegion(left=0.35, top=0.2, right=0.65, bottom=0.8)
    return CircleCaptureWait(region, hand, _frame(100_000), settings)


def test_drawing_hand_clearance_ignores_distant_bystander_and_requires_stability():
    wait = _wait()
    for timestamp, x in [(200_000, 250), (300_000, 280), (400_000, 300)]:
        observation = _pinch(x, 100)
        observation.hands += _pinch(30, 30).hands
        assert not wait.observe(observation, _frame(timestamp))
    assert wait.observe(_pinch(300, 100), _frame(500_000))


def test_missing_ambiguous_hands_use_bounded_best_effort_timeout():
    wait = _wait()
    for timestamp in range(200_000, 2_100_000, 100_000):
        observation = _pinch(220, 100)
        observation.hands *= 2
        assert not wait.observe(observation, _frame(timestamp))
    assert wait.observe(HandTrackingResult(width=400, height=200, hands=[]), _frame(2_100_000))


@pytest.mark.parametrize("change", ["source", "gap", "dimensions"])
def test_pending_selection_rejects_video_discontinuity(change):
    wait = _wait()
    frame = _frame(200_000)
    if change == "source":
        frame.track_id = "other"
    elif change == "gap":
        frame.timestamp_us = 800_000
    else:
        frame.width = 800
    with pytest.raises(ValueError, match="expired"):
        wait.observe(_pinch(300, 100), frame)


@pytest.mark.parametrize("failure", [False, True])
async def test_agent_pending_circle_observes_during_cooldown_and_queries_new_frame(failure):
    frames = iter([_frame(i * 100_000) for i in range(1, 9)])
    observations = iter([_pinch(220, 100), _pinch(250, 100), _pinch(280, 100), _pinch(300, 100)])
    selector_calls, queried, cues = [], [], []
    async def frame(_):
        return next(frames)
    async def hands(_):
        return next(observations)
    async def crop(request):
        queried.append(request.image)
        return SimpleNamespace(image=request.image)
    async def query(_):
        if failure:
            raise ValueError("query failed")
        return SimpleNamespace(text="target")
    async def recognized():
        cues.append("chime")
    region = ImageRegion(left=0.35, top=0.2, right=0.65, bottom=0.8)
    def select(*_):
        selector_calls.append(1)
        return region
    selector = SimpleNamespace(circle_selected=True, drawing_hand=match_hand(_pinch(220, 100), None, 2),
                               observe=select)
    agent = GestureCaptureAgent(frames=SimpleNamespace(execute=frame), hands=SimpleNamespace(execute=hands),
        crop=SimpleNamespace(execute=crop), query=SimpleNamespace(execute=query), capture=None,
        selector=selector, settings=_settings(release_s=0.1), feedback=SimpleNamespace(recognized=recognized))
    for _ in range(3):
        assert await agent.step("alice") is None
        await agent.finish_turn()  # Worker cleanup must not unlock pending selection.
    if failure:
        with pytest.raises(ValueError, match="query failed"):
            await agent.step("alice")
    else:
        assert await agent.step("alice") == "target"
        assert await agent.step("alice") is None  # Remains gated until delivery.
    assert queried == [ImageReference(uri="400000")] and selector_calls == [1] and cues == ["chime"]
    await agent.finish_turn()
    assert not agent._turn_active


def test_eight_point_fast_circle_completes_without_perfect_closure():
    trigger = PinchCircleTrigger(CircleSettings())
    results = [trigger.observe(_pinch(*p), (i + 1) * 80_000) for i, p in enumerate(_path(310, 8))]
    assert all(result is None for result in results[:-1])
    assert results[-1] is not None


def test_dropout_breaks_clear_confirmation_and_duplicates_do_not_advance():
    wait = _wait()
    assert not wait.observe(_pinch(280, 100), _frame(200_000))
    assert not wait.observe(_pinch(280, 100), _frame(200_000))
    assert not wait.observe(HandTrackingResult(width=400, height=200, hands=[]), _frame(300_000))
    assert not wait.observe(_pinch(280, 100), _frame(400_000))
    assert not wait.observe(_pinch(280, 100), _frame(500_000))
    assert wait.observe(_pinch(280, 100), _frame(600_000))


def test_pending_wait_expires_when_video_stops(monkeypatch):
    wait = _wait()
    monkeypatch.setattr('gesture_capture_worker.selection_wait.time.monotonic',
                        lambda: wait.last_observed + 0.6)
    assert wait.expired


async def test_brief_unavailable_frame_retains_pending_but_breaks_clear_dwell():
    from xr_ai_hub import FrameUnavailable

    async def unavailable(_):
        raise FrameUnavailable("temporarily missing")
    agent = GestureCaptureAgent(frames=SimpleNamespace(execute=unavailable), hands=None,
        capture=None, crop=None, query=None, selector=SimpleNamespace(unavailable=lambda: None), settings=_settings())
    agent._pending = _wait()
    agent._pending.clear_since = 200_000
    agent._turn_active = True
    with pytest.raises(FrameUnavailable):
        await agent.step("alice")
    await agent.finish_turn()
    assert agent._pending is not None and agent._pending.clear_since is None and agent._turn_active
