# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the sample's common flow contract and step execution ownership."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       "agent-samples/procedure-capture-guide/worker"))

from procedure_capture_guide_worker.guide.confirmation import ConfirmationPolicy  # noqa: E402
from procedure_capture_guide_worker.guide.flow import FlowUpdate  # noqa: E402
from procedure_capture_guide_worker.guide.step_runner import StepRunner  # noqa: E402
from procedure_capture_guide_worker.guide.timer_flow import TimerFlow  # noqa: E402
from procedure_capture_guide_worker.guide.visual_verification_flow import (  # noqa: E402
    FrameIdentity,
    Observation,
    VisualVerificationFlow,
)


def frame(sequence, *, track="camera"):
    return Observation(True, b"pinned image", FrameIdentity(
        participant_id="alice", track_id=track, sequence=sequence, timestamp_us=sequence * 1000,
    ))


async def wait_for(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


async def test_timer_deadline_bounds_long_poll_interval_and_uses_same_read_clock():
    now = 10.0
    flow = TimerFlow(2, 60, clock=lambda: now)
    updates = []

    async def apply(update):
        updates.append(update)
        return update.status == "ready"

    runner = StepRunner(flow, apply=apply, clock=lambda: now)
    await runner.poll()
    assert updates[-1].status == "pending"
    assert updates[-1].value == flow.read()
    assert runner.next_poll_at == 12
    now = 11.9
    await runner.poll()
    assert len(updates) == 1
    now = 12
    await runner.poll()
    assert runner.status == "completed"
    assert updates[-1].value.expired and updates[-1].value.remaining_s == 0
    await runner.poll()
    assert len(updates) == 2


async def test_visual_flow_counts_distinct_sources_and_resets_unavailable():
    acquire = AsyncMock(side_effect=[frame(1), frame(1), frame(0),
                                     Observation(False, "camera off"), frame(2), frame(2, track="new")])
    verify = AsyncMock(return_value="accepted")
    flow = VisualVerificationFlow(acquire=acquire, verify=verify, interval_s=1, required=2)
    results = [await flow.poll() for _ in range(6)]
    assert [result.value.hits for result in results] == [1, 1, 1, 0, 1, 2]
    assert [result.status for result in results] == ["pending", "pending", "pending", "unavailable", "pending", "ready"]
    assert verify.await_count == 3


async def test_visual_window_expires_old_positive_before_new_evidence():
    now = 0.0
    flow = VisualVerificationFlow(
        acquire=AsyncMock(side_effect=[frame(1), frame(2), frame(3)]),
        verify=AsyncMock(return_value="accepted"), interval_s=1, required=2,
        confirmation=ConfirmationPolicy(required=2, window=3, max_age_s=2), clock=lambda: now,
    )
    assert (await flow.poll()).value.hits == 1
    now = 3
    assert (await flow.poll()).value.hits == 1
    now = 4
    assert (await flow.poll()).status == "ready"


async def test_unknown_and_failed_inference_do_not_confirm_or_leave_old_hits():
    verify = AsyncMock(side_effect=["accepted", "unknown", "accepted", RuntimeError("model down"), "accepted"])
    flow = VisualVerificationFlow(acquire=AsyncMock(side_effect=[frame(i) for i in range(5)]),
                                  verify=verify, interval_s=1, required=2)
    for hits in [1, 0, 1]:
        assert (await flow.poll()).value.hits == hits
    with pytest.raises(RuntimeError, match="model down"):
        await flow.poll()
    assert flow.hits == 0
    assert (await flow.poll()).value.hits == 1


async def test_runner_serializes_observations_and_delivers_completion_once():
    entered, release = asyncio.Event(), asyncio.Event()
    active = 0

    class Flow:
        async def poll(self):
            nonlocal active
            active += 1
            assert active == 1
            entered.set()
            await release.wait()
            active -= 1
            return FlowUpdate("ready", "verified", 1)

    apply = AsyncMock(return_value=True)
    runner = StepRunner(Flow(), apply=apply)
    first = asyncio.create_task(runner.poll())
    await entered.wait()
    second = asyncio.create_task(runner.poll())
    release.set()
    await asyncio.gather(first, second)
    apply.assert_awaited_once()
    assert runner.status == "completed"


async def test_stop_drains_inference_and_blocks_delivery_even_if_operation_swallows_cancel():
    entered, cleanup, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class Flow:
        async def poll(self):
            entered.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cleanup.set()
                await release.wait()
                return FlowUpdate("ready", "late evidence", 1)

    apply = AsyncMock(return_value=True)
    runner = StepRunner(Flow(), apply=apply)
    task = runner.start(name="test-step")
    await entered.wait()
    stop = asyncio.create_task(runner.stop())
    await cleanup.wait()
    assert not stop.done()
    stop.cancel()
    await asyncio.sleep(0)
    assert not stop.done()  # Caller cancellation cannot abandon flow cleanup.
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await stop
    assert task.done() and runner.status == "cancelled"
    apply.assert_not_awaited()


async def test_runner_exposes_execution_failure_instead_of_retrying_programming_errors():
    failure = RuntimeError("bad flow configuration")

    class Flow:
        async def poll(self):
            raise failure

    apply = AsyncMock()
    runner = StepRunner(Flow(), apply=apply)
    with pytest.raises(RuntimeError, match="bad flow configuration"):
        await runner.start(name="failed-flow")
    assert runner.status == "failed" and runner.error is failure
    apply.assert_not_awaited()
    await runner.stop()


@pytest.mark.parametrize("flow_type", ["timer", "visual"])
async def test_both_flows_execute_through_same_runner(flow_type):
    now = 0.0
    if flow_type == "timer":
        flow = TimerFlow(1, .001, clock=lambda: now)
        now = 1.0
    else:
        flow = VisualVerificationFlow(acquire=AsyncMock(return_value=frame(1)),
                                      verify=AsyncMock(return_value="accepted"), interval_s=.001, required=1)
    apply = AsyncMock(side_effect=lambda update: update.status == "ready")
    runner = StepRunner(flow, apply=apply)
    await asyncio.wait_for(runner.start(name="same-contract"), 1)
    assert runner.status == "completed"
    assert apply.await_args.args[0].status == "ready"


async def test_unavailable_visual_inference_retries_without_reusing_old_confirmation():
    verify = AsyncMock(side_effect=["accepted", None, "accepted", "accepted"])
    flow = VisualVerificationFlow(acquire=AsyncMock(side_effect=[frame(i) for i in range(4)]),
                                  verify=verify, interval_s=.001, required=2)
    seen = []

    async def apply(update):
        seen.append((update.status, update.value.hits))
        return update.status == "ready"

    runner = StepRunner(flow, apply=apply)
    await asyncio.wait_for(runner.start(name="recoverable-unavailable"), 1)
    assert seen == [("pending", 1), ("unavailable", 0), ("pending", 1), ("ready", 2)]
