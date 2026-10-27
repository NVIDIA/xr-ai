# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gesture cue routing, cancellation, and per-participant turn ownership."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from gesture_capture_worker.agent import GestureCaptureAgent
from gesture_capture_worker.feedback import GestureFeedback
from test_gesture_capture import _png, _settings
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.image_crop import ImageCropTool, ImageRegion
from xr_ai_tools.vision import ImageQueryResult


async def test_hum_stop_drains_a_blocked_public_audio_send():
    entered, drained = asyncio.Event(), asyncio.Event()
    active = 0
    async def send(chunk):
        nonlocal active
        active += 1
        entered.set()
        try:
            await asyncio.Future()
        finally:
            active -= 1
            drained.set()
    feedback = GestureFeedback(SimpleNamespace(send_return_audio=send), "alice", enabled=True, max_gap_s=1)
    await feedback.update(True)
    await asyncio.wait_for(entered.wait(), 2)
    await feedback.close()
    assert drained.is_set() and active == 0
    assert feedback._hum_task is None


async def test_cancelling_chime_drains_blocked_audio_and_prevents_later_chunks():
    entered, drained = asyncio.Event(), asyncio.Event()
    chunks = []
    async def send(chunk):
        chunks.append(chunk)
        entered.set()
        try:
            await asyncio.Future()
        finally:
            drained.set()
    feedback = GestureFeedback(SimpleNamespace(send_return_audio=send), "alice", enabled=True, max_gap_s=1)
    task = asyncio.create_task(feedback.recognized())
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await feedback.close()
    assert drained.is_set() and len(chunks) == 1
    await feedback.update(True)
    await feedback.recognized()
    assert len(chunks) == 1


async def test_hum_expires_when_observations_stop_and_disabled_cues_stay_silent():
    chunks = []
    async def send(chunk):
        chunks.append(chunk)
    endpoint = SimpleNamespace(send_return_audio=send)
    feedback = GestureFeedback(endpoint, "alice", enabled=True, max_gap_s=0.03)
    await feedback.update(True)
    await asyncio.wait_for(feedback._hum_task, 2)
    assert chunks and all(chunk.participant_id == "alice" for chunk in chunks)
    await feedback.close()
    count = len(chunks)
    disabled = GestureFeedback(endpoint, "alice", enabled=False, max_gap_s=0.5)
    await disabled.update(True)
    await disabled.recognized()
    await disabled.close()
    assert len(chunks) == count


async def test_hum_and_chime_use_existing_hub_audio_routing(hub, make_connector, make_processor, settle):
    alice, bob = make_connector(), make_connector()
    await alice.register()
    await bob.register()
    await settle()
    await alice.notify_participant_joined("alice", pts_us=1)
    await bob.notify_participant_joined("bob", pts_us=1)
    received, other = [], []
    changed = asyncio.Condition()
    async def receive(chunk):
        async with changed:
            received.append(chunk)
            changed.notify_all()
    async def other_receive(chunk):
        other.append(chunk)
    alice.on_return_audio(receive)
    bob.on_return_audio(other_receive)
    tasks = [asyncio.create_task(alice.run()), asyncio.create_task(bob.run())]
    processor = make_processor()
    await settle()
    feedback = GestureFeedback(processor, "alice", enabled=True, max_gap_s=0.5)
    try:
        await feedback.update(True)
        async with changed:
            await asyncio.wait_for(changed.wait_for(lambda: bool(received)), 2)
        await feedback.update(False)
        hum = np.frombuffer(received[0].data, dtype="<f4")
        assert 0 < float(np.max(np.abs(hum))) <= 0.026
        before = len(received)
        await feedback.recognized()
        async with changed:
            await asyncio.wait_for(changed.wait_for(lambda: len(received) >= before + 13), 2)
        chime = np.concatenate([np.frombuffer(chunk.data, dtype="<f4") for chunk in received[before:]])
        spectrum = np.abs(np.fft.rfft(chime))
        frequencies = np.fft.rfftfreq(len(chime), 1 / 24000)
        peaks = frequencies[np.argpartition(spectrum, -2)[-2:]]
        assert set(peaks) == {880, 1320}
        assert all(chunk.participant_id == "alice" and chunk.track_id == "gesture-feedback" for chunk in received)
        assert other == []
    finally:
        await feedback.close()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("failure", [False, True])
async def test_turn_cooldown_lasts_until_delivery_or_failure_cleanup(failure):
    from xr_ai_tools.current_frame import ImageFrame

    images = ImageRegistry()
    frame_calls = 0
    query_started, release_query = asyncio.Event(), asyncio.Event()
    async def frame(request):
        nonlocal frame_calls
        frame_calls += 1
        return ImageFrame(image=images.put(_png()), width=400, height=200, sequence=frame_calls,
                          timestamp_us=frame_calls * 100_000, participant_id="alice")
    async def hands(request):
        return None
    async def query(request):
        query_started.set()
        await release_query.wait()
        if failure:
            raise ValueError("inference failed")
        return ImageQueryResult(text="target")
    selector = SimpleNamespace(observe=lambda *_args: ImageRegion(left=0.2, top=0.2, right=0.8, bottom=0.8))
    agent = GestureCaptureAgent(
        frames=SimpleNamespace(execute=frame), hands=SimpleNamespace(execute=hands),
        capture=None, crop=ImageCropTool(images=images), query=SimpleNamespace(execute=query),
        selector=selector, settings=_settings(),
    )
    turn = asyncio.create_task(agent.step("alice"))
    await asyncio.wait_for(query_started.wait(), 2)
    assert await agent.step("alice") is None
    assert frame_calls == 1
    release_query.set()
    if failure:
        with pytest.raises(ValueError, match="inference failed"):
            await turn
    else:
        assert await turn == "target"
    # Tool completion alone does not reopen admission before the worker delivers
    # the response (or handles the error) and completes the participant turn.
    assert await agent.step("alice") is None
    assert frame_calls == 1
    await agent.finish_turn()
    failure = False
    assert await agent.step("alice") == "target"
    await agent.finish_turn()
    assert frame_calls == 2


@pytest.mark.parametrize("gesture", ["box", "circle"])
async def test_each_gesture_cues_once_per_turn_and_recovers_after_failure(gesture):
    from gesture_capture_worker.gesture import RegionSelector
    from test_gesture_capture import _box, _circle_points, _pinch
    from xr_ai_tools.current_frame import ImageFrame

    images = ImageRegistry()
    events, answers = [], []
    timestamp = 0
    observation = None
    query_calls = 0
    async def frame(request):
        return ImageFrame(image=images.put(_png()), width=400, height=200, sequence=timestamp,
                          timestamp_us=timestamp, participant_id="alice")
    async def hands(request):
        return observation
    async def query(request):
        nonlocal query_calls
        query_calls += 1
        if query_calls == 1:
            raise ValueError("first turn failed")
        return ImageQueryResult(text="target")
    class Feedback:
        async def update(self, active):
            events.append("hum" if active else "quiet")
        async def recognized(self):
            events.append("chime")
        async def stop(self):
            events.append("stop")
    feedback = Feedback()
    settings = _settings(hold_s=0.2, release_s=0.2)
    agent = GestureCaptureAgent(
        frames=SimpleNamespace(execute=frame), hands=SimpleNamespace(execute=hands), capture=None,
        crop=ImageCropTool(images=images), query=SimpleNamespace(execute=query),
        selector=RegionSelector(settings), settings=settings, feedback=feedback,
    )
    if gesture == "box":
        relaxed = _box()
        relaxed.hands[0].landmarks[4].x = 0.5
        observations = [_box() for _ in range(6)] + [relaxed] * 4 + [_box() for _ in range(6)]
    else:
        path = [_pinch(x, y) for x, y in _circle_points()]
        observations = [_pinch(260, 100)] * 3 + path + [_pinch(260, 100)] * 4
        observations += [_pinch(260, 100, released=True)] * 4 + path
    for observation in observations:
        timestamp += 100_000
        try:
            answer = await agent.step("alice")
            if answer is not None:
                answers.append(answer)
        except ValueError as error:
            assert str(error) == "first turn failed"
            agent.unavailable()
            await feedback.stop()
        finally:
            await agent.finish_turn()
    assert events.count("chime") == 2
    assert "hum" in events and "stop" in events
    assert query_calls == 2 and answers == ["target"]
    if gesture == "circle":
        assert events[:3] == ["quiet"] * 3
