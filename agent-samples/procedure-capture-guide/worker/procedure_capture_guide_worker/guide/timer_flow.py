# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify elapsed time with a runtime-owned timer shared by tools and execution.

The flow samples a monotonic clock and wakes by the deadline even when the
configured update interval is longer. Models cannot assign the start time or
change the duration. The runner owns polling and cancellation.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ._workflow_spec import _TimerResult
from .flow import FlowUpdate


@dataclass(frozen=True, slots=True)
class TimerFlow:
    duration_s: int
    interval_s: float
    clock: Callable[[], float] = field(default=time.monotonic, repr=False, compare=False)
    started: float = field(init=False)

    def __post_init__(self):
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("timer duration must be finite and positive")
        if not math.isfinite(self.interval_s) or self.interval_s <= 0:
            raise ValueError("timer interval must be finite and positive")
        object.__setattr__(self, "started", self.clock())

    def read(self) -> _TimerResult:
        elapsed = max(0.0, self.clock() - self.started)
        return _TimerResult(elapsed_s=math.floor(elapsed),
                            remaining_s=max(0, math.ceil(self.duration_s - elapsed)),
                            expired=elapsed >= self.duration_s)

    async def poll(self) -> FlowUpdate[_TimerResult]:
        result = self.read()
        remaining = self.duration_s - max(0.0, self.clock() - self.started)
        delay = min(self.interval_s, remaining) if remaining > 0 else self.interval_s
        return FlowUpdate("ready" if result.expired else "pending", result, delay)
