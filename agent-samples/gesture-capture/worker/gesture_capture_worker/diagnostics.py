# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in bounded numeric summaries; no media, landmark coordinates or identifiers."""

import json
import logging
import math
import time
from collections import Counter

_LOG = logging.getLogger(__name__)


class GestureDiagnostics:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._started = time.monotonic()
        self._counts: Counter[str] = Counter()
        self._ranges: dict[str, list[float]] = {}
        self._last_timestamp: int | None = None
        self._grip = "open"
        self._pinch_peak: tuple[float, float, float] | None = None

    def count(self, reason: str) -> None:
        if self.enabled:
            self._counts[reason] += 1

    def number(self, name: str, value: float) -> None:
        if self.enabled and math.isfinite(value):
            limits = self._ranges.setdefault(name, [value, value])
            limits[0], limits[1] = min(limits[0], value), max(limits[1], value)

    def pinch(self, ratio: float, tip_distance_px: float, palm_width_px: float) -> None:
        if not self.enabled or not all(math.isfinite(v) for v in (ratio, tip_distance_px, palm_width_px)):
            return
        self.number("pinch_ratio", ratio)
        self.number("tip_distance_px", tip_distance_px)
        self.number("palm_width_px", palm_width_px)
        if self._pinch_peak is None or ratio > self._pinch_peak[0]:
            self._pinch_peak = (ratio, tip_distance_px, palm_width_px)

    def grip(self, state: str) -> None:
        if self.enabled and state in {"open", "pinched", "unknown"}:
            if state != self._grip:
                self.count("grip_to_" + state)
            self._grip = state

    def frame(self, timestamp_us: int, hands: int, backend_ms: float) -> None:
        if not self.enabled:
            return
        self.count("frames")
        self.count("hands_" + str(min(hands, 3)))
        self.number("backend_ms", backend_ms)
        if self._last_timestamp is not None and timestamp_us > self._last_timestamp:
            self.number("frame_interval_ms", (timestamp_us - self._last_timestamp) / 1000)
        self._last_timestamp = timestamp_us

    def emit(self) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self._started < 1 or not self._counts:
            return
        payload = {"event": "gesture_diagnostics", "window_s": round(now - self._started, 3),
                   "grip": self._grip, "counts": dict(self._counts),
                   "ranges": {key: [round(v, 3) for v in limits] for key, limits in self._ranges.items()}}
        if self._pinch_peak is not None:
            payload["pinch_peak"] = dict(zip(("ratio", "tip_distance_px", "palm_width_px"),
                                             (round(v, 3) for v in self._pinch_peak), strict=True))
        _LOG.info("%s", json.dumps(payload, sort_keys=True))
        self._counts.clear()
        self._ranges.clear()
        self._pinch_peak = None
        self._started = now
