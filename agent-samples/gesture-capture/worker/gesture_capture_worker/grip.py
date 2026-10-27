# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence-driven grip state, independent of pointer motion and selected regions."""

from enum import Enum


class Grip(Enum):
    UNKNOWN = "unknown"
    OPEN = "open"
    PINCHED = "pinched"


class GripHysteresis:
    def __init__(self, *, pinch_ratio: float, release_ratio: float, dwell_s: float, max_gap_s: float) -> None:
        self._pinch = pinch_ratio
        self._release = release_ratio
        self._dwell_us = dwell_s * 1_000_000
        self._gap_us = max_gap_s * 1_000_000
        self.state = Grip.OPEN
        self._candidate: tuple[Grip, int] | None = None
        self._last_evidence: int | None = None
        self._timestamp = -1

    def reset(self) -> None:
        """After loss, visible sustained open evidence is required before closing."""
        self.state = Grip.UNKNOWN
        self._candidate = None
        self._last_evidence = None

    def unavailable(self) -> None:
        self._candidate = None

    def observe(self, ratio: float | None, timestamp_us: int) -> bool:
        """Return whether this sample supplies confirmed closed evidence."""
        if timestamp_us <= self._timestamp:
            return False
        self._timestamp = timestamp_us
        if self._last_evidence is not None and timestamp_us - self._last_evidence > self._gap_us:
            self.reset()
        evidence = (Grip.UNKNOWN if ratio is None else Grip.PINCHED if ratio <= self._pinch
                    else Grip.OPEN if ratio >= self._release else Grip.UNKNOWN)
        if evidence is Grip.UNKNOWN:
            self.unavailable()
            return False
        self._last_evidence = timestamp_us
        # A held pinch returning after timeout must not become a fresh attempt.
        if self.state is Grip.UNKNOWN and evidence is Grip.PINCHED:
            self.unavailable()
            return False
        if evidence is self.state:
            self._candidate = None
        else:
            if self._candidate is None or self._candidate[0] is not evidence:
                self._candidate = (evidence, timestamp_us)
            if timestamp_us - self._candidate[1] >= self._dwell_us:
                self.state = evidence
                self._candidate = None
        return self.state is Grip.PINCHED and evidence is Grip.PINCHED
