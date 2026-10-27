# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample-owned pinch trajectory and circular-region selection."""

from __future__ import annotations

import math

from xr_ai_tools.hand_tracking import HandTrackingResult
from xr_ai_tools.image_crop import ImageRegion

from .config import CircleSettings


def _circle_region(points: list[tuple[float, float]], width: int, height: int,
                   settings: CircleSettings) -> ImageRegion | None:
    if len(points) < settings.min_points:
        return None
    xs, ys = zip(*points, strict=True)
    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
    diameter_x, diameter_y = right - left, bottom - top
    if min(diameter_x, diameter_y) < settings.min_diameter * min(width, height):
        return None
    if min(diameter_x, diameter_y) / max(diameter_x, diameter_y) < settings.min_axis_ratio:
        return None
    radius = (diameter_x + diameter_y) / 4
    center = ((left + right) / 2, (top + bottom) / 2)
    if math.dist(points[0], points[-1]) > settings.closure_tolerance * radius:
        return None
    vectors = [(x - center[0], y - center[1]) for x, y in points]
    if any(abs(math.hypot(*v) / radius - 1) > settings.radial_tolerance for v in vectors):
        return None
    turns = []
    for a, b in zip(vectors, [*vectors[1:], vectors[0]], strict=True):
        if math.dist(a, b) > settings.max_segment_radius * radius:
            return None
        turns.append(math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]))
    winding = sum(turns)
    if not math.isclose(abs(winding), math.tau):
        return None
    backtrack = (sum(abs(turn) for turn in turns) - abs(winding)) / 2
    if backtrack > settings.max_backtrack * math.tau:
        return None
    return ImageRegion(left=left / width, right=right / width, top=top / height, bottom=bottom / height)


class PinchCircleTrigger:
    """Recognize a closed pinched trace once; require stable release to rearm."""

    def __init__(self, settings: CircleSettings, *, release_s: float = 0.4, max_gap_s: float = 0.5) -> None:
        self._settings = settings
        self._release_us = release_s * 1_000_000
        self._max_gap_us = max_gap_s * 1_000_000
        self._release_start: int | None = None
        self._last_seen: int | None = None
        self._timestamp = -1
        self._short_side = 0
        self._points: list[tuple[float, float]] = []
        self._handedness: str | None = None
        self._ready = True

    @property
    def active(self) -> bool:
        return bool(self._points)

    @property
    def drawing(self) -> bool:
        if len(self._points) < 2:
            return False
        xs, ys = zip(*self._points, strict=True)
        return max(max(xs) - min(xs), max(ys) - min(ys)) >= self._settings.min_diameter * self._short_side

    def unavailable(self) -> None:
        self._release_start = None

    def reset(self) -> None:
        self.unavailable()
        self._points.clear()
        self._handedness = None
        # A still-pinched hand returning after loss cannot continue an old trace.
        self._ready = False

    def observe(self, observation: HandTrackingResult, timestamp_us: int) -> ImageRegion | None:
        if timestamp_us <= self._timestamp:
            return None
        self._timestamp = timestamp_us
        if self._last_seen is not None and timestamp_us - self._last_seen > self._max_gap_us:
            self.reset()
        if len(observation.hands) != 1:
            self.unavailable()
            return None
        hand = observation.hands[0]
        width, height = observation.width, observation.height
        self._short_side = min(width, height)
        points = [(p.x * width, p.y * height) for p in hand.landmarks]
        palm = math.dist(points[5], points[17])
        if palm == 0:
            self.unavailable()
            return None
        ratio = math.dist(points[4], points[8]) / palm
        pointer = ((points[4][0] + points[8][0]) / 2, (points[4][1] + points[8][1]) / 2)
        if not (0 <= pointer[0] <= width and 0 <= pointer[1] <= height):
            self.reset()
            return None
        if self._points and (self._handedness != hand.handedness
                             or math.dist(pointer, self._points[-1]) > self._settings.max_jump_palm * palm):
            self.reset()
            return None
        self._last_seen = timestamp_us
        if ratio >= self._settings.release_ratio:
            if self._release_start is None:
                self._release_start = timestamp_us
            if timestamp_us - self._release_start < self._release_us:
                return None
            self._points = []
            self._handedness = None
            self._ready = True
            return None
        self._release_start = None
        if not self._ready:
            return None
        if not self._points and ratio > self._settings.pinch_ratio:
            return None
        if self._points and pointer == self._points[-1]:
            return None
        if len(self._points) >= self._settings.max_points:
            self.reset()
            return None
        self._handedness = hand.handedness
        self._points.append(pointer)
        region = _circle_region(self._points, width, height, self._settings)
        if region is not None:
            self._points = []
            self._ready = False
        return region
