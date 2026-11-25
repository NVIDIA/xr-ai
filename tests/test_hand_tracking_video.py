# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporal detector ownership and source-frame admission boundaries."""

from types import SimpleNamespace

import pytest
from gesture_capture_worker.agent import GestureCaptureAgent
from test_circle_selection_wait import _frame, _wait
from test_gesture_capture import _settings
from xr_ai_tools.hand_tracking import HandTrackingResult


def _agent(frames):
    events = []
    async def frame(_):
        return next(frames)
    async def detect(request):
        events.append(('detect', request.timestamp_us))
        return HandTrackingResult(width=400, height=200, hands=[])
    def reset():
        events.append(('reset',))
    selector = SimpleNamespace(reset=lambda: events.append(('selector_reset',)),
        unavailable=lambda: events.append(('unavailable',)),
        observe=lambda *args: None)
    agent = GestureCaptureAgent(frames=SimpleNamespace(execute=frame),
        hands=SimpleNamespace(execute=detect, reset=reset), capture=None, crop=None, query=None,
        selector=selector, settings=_settings())
    return agent, events


@pytest.mark.parametrize('changed', [
    {'track_id': 'new-camera'}, {'width': 800}, {'height': 400},
    {'sequence': 0}, {'timestamp_us': 50_000}, {'timestamp_us': 800_000},
])
async def test_source_epoch_resets_before_inference(changed):
    first, second = _frame(100_000), _frame(200_000)
    for key, value in changed.items():
        setattr(second, key, value)
    agent, events = _agent(iter([first, second]))
    await agent.step('participant')
    await agent.step('participant')
    assert events == [('detect', 100_000), ('reset',), ('selector_reset',),
                      ('detect', second.timestamp_us)]


async def test_same_ms_and_identical_frames_skip_without_absence_or_reset():
    agent, events = _agent(iter([_frame(100_100), _frame(100_100), _frame(100_900), _frame(101_000)]))
    for _ in range(4):
        await agent.step('participant')
    assert events == [('detect', 100_100), ('detect', 101_000)]


async def test_unavailable_resets_once_on_next_fresh_frame_not_every_poll():
    agent, events = _agent(iter([_frame(100_000), _frame(100_000), _frame(200_000), _frame(300_000)]))
    await agent.step('participant')
    agent.unavailable()
    agent.unavailable()
    await agent.step('participant')
    assert events.count(('reset',)) == 0
    await agent.step('participant')
    await agent.step('participant')
    assert events.count(('reset',)) == 1
    assert events[-2:] == [('detect', 200_000), ('detect', 300_000)]


async def test_pending_selection_source_change_never_infers_or_queries():
    agent, events = _agent(iter([_frame(100_000), _frame(200_000)]))
    await agent.step('participant')
    agent._pending = _wait()
    agent._turn_active = True
    agent._last_frame = ('old-camera', 100_000, 100_000, 400, 200)
    with pytest.raises(ValueError, match='source interruption'):
        await agent.step('participant')
    assert events[-2:] == [('reset',), ('selector_reset',)]
    assert agent._pending is None
