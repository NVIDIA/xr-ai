# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a native timer and report complete, timestamped state without model calls."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class TimerState:
    """A sampled native timer; deadline and sample time label the freshness of remaining time."""

    status: Literal["running", "completed"]
    duration_seconds: float
    started_at_us: int
    deadline_us: int
    updated_at_us: int
    elapsed_seconds: float
    remaining_seconds: float


async def run_timer_flow(
    *,
    duration_seconds: float,
    update_interval_seconds: float,
    on_state: Callable[[TimerState], Awaitable[None]] | None = None,
) -> TimerState:
    """Count from execution start, publish progress and return the terminal state.

    Args:
        duration_seconds: Positive native duration supplied by the runner.
        update_interval_seconds: Positive maximum delay between progress updates.
        on_state: Async callback awaited in this timer's task. Its errors propagate.

    Returns:
        Completed timer state. Cancellation propagates without a success result.

    Monotonic elapsed time keeps clock corrections from changing the duration.
    Wall-clock start/deadline/sample labels are retained for conversation and
    recording. Sleeping is bounded by the remaining duration, so a long update
    interval does not delay completion.
    """
    if (
        not math.isfinite(duration_seconds)
        or not math.isfinite(update_interval_seconds)
        or duration_seconds <= 0
        or update_interval_seconds <= 0
    ):
        raise ValueError("timer duration and update interval must be positive")
    started_at_us = time.time_ns() // 1000
    started_monotonic = time.monotonic()
    deadline_us = started_at_us + round(duration_seconds * 1_000_000)
    while True:
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError
        elapsed_seconds = max(0.0, time.monotonic() - started_monotonic)
        remaining_seconds = max(0.0, duration_seconds - elapsed_seconds)
        state = TimerState(
            "completed" if remaining_seconds == 0 else "running",
            duration_seconds,
            started_at_us,
            deadline_us,
            time.time_ns() // 1000,
            elapsed_seconds,
            remaining_seconds,
        )
        if on_state is not None:
            await on_state(state)
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError
        if remaining_seconds == 0:
            return state
        await asyncio.sleep(min(update_interval_seconds, remaining_seconds))
