# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure guide waits using a runtime-owned monotonic clock.

Only the execution backend creates a timer. Read-only tools and background
checks read the same clock, so model replies never establish a start time.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ._workflow_spec import _TimerResult


@dataclass(frozen=True, slots=True)
class GuideTimer:
    started: float
    duration_s: int
    clock: Callable[[], float] = field(default=time.monotonic, repr=False, compare=False)

    def read(self) -> _TimerResult:
        elapsed = max(0.0, self.clock() - self.started)
        return _TimerResult(
            elapsed_s=math.floor(elapsed),
            remaining_s=max(0, math.ceil(self.duration_s - elapsed)),
            expired=elapsed >= self.duration_s,
        )
