# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grip hysteresis: retained grip is independent of usable pointer observations."""

from enum import Enum


class Grip(Enum):
    UNKNOWN = "unknown"
    OPEN = "open"
    PINCHED = "pinched"


class GripHysteresis:
    def __init__(self, *, pinch_ratio: float, release_ratio: float, dwell_s: float, max_gap_s: float) -> None:
        self._pinch = pinch_ratio
        self._release = release_ratio
        self._release_us = dwell_s * 1_000_000
        self._gap_us = max_gap_s * 1_000_000
        self.state = Grip.OPEN
        self._release_start: int | None = None
        self._closed_samples = 0
        self._last_closed: int | None = None
        self._last_observation: int | None = None
        self._timestamp = -1

    def reset(self) -> None:
        """After tracking loss, sustained visible open evidence must rearm."""
        self.state = Grip.UNKNOWN
        self._last_observation = None
        self.unavailable()

    def entry_pending(self, timestamp_us: int) -> bool:
        return self._last_closed is not None and timestamp_us - self._last_closed <= self._gap_us

    def unavailable(self) -> None:
        self._release_start = None
        self._closed_samples = 0
        self._last_closed = None

    def observe(self, ratio: float | None, timestamp_us: int) -> bool:
        """Return whether retained grip permits this observed pointer sample."""
        if timestamp_us <= self._timestamp:
            return False
        self._timestamp = timestamp_us
        if self._last_observation is not None and timestamp_us - self._last_observation > self._gap_us:
            self.reset()
        if ratio is None:
            self.unavailable()
            return False
        self._last_observation = timestamp_us
        if not self.entry_pending(timestamp_us):
            self._closed_samples = 0
        if ratio >= self._release:
            self._closed_samples = 0
            self._last_closed = None
            if self._release_start is None:
                self._release_start = timestamp_us
            if timestamp_us - self._release_start >= self._release_us:
                self.state = Grip.OPEN
            return False
        self._release_start = None
        if ratio <= self._pinch and self.state is Grip.OPEN:
            self._closed_samples += 1
            self._last_closed = timestamp_us
            if self._closed_samples >= 2:
                self.state = Grip.PINCHED
        # The hysteresis band preserves a confirmed grip AND valid pointer
        # motion. It is not evidence that the observed pointer has disappeared.
        return self.state is Grip.PINCHED
