# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample-owned pinch trajectory and circular-region selection."""

from __future__ import annotations

import math

from xr_ai_tools.hand_tracking import HandTrackingResult
from xr_ai_tools.image_crop import ImageRegion

from .config import CircleSettings
from .hand_motion import HandPose, match_hand


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
    # Least-squares circle center avoids bias toward the missing arc of a
    # near-closed loop. Solve the centered two-variable normal equations.
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    centered = [(x - mean_x, y - mean_y) for x, y in points]
    xx = sum(x * x for x, y in centered)
    yy = sum(y * y for x, y in centered)
    xy = sum(x * y for x, y in centered)
    bx = sum(x * (x * x + y * y) / 2 for x, y in centered)
    by = sum(y * (x * x + y * y) / 2 for x, y in centered)
    determinant = xx * yy - xy * xy
    if determinant <= 0:
        return None
    center = (mean_x + (bx * yy - by * xy) / determinant,
              mean_y + (by * xx - bx * xy) / determinant)
    vectors = [(x - center[0], y - center[1]) for x, y in points]
    radius = sum(math.hypot(*v) for v in vectors) / len(vectors)
    if math.dist(points[0], points[-1]) > settings.closure_tolerance * radius * (1 + 1e-9):
        return None
    if any(abs(math.hypot(*v) / radius - 1) > settings.radial_tolerance for v in vectors):
        return None
    turns = []
    for a, b in zip(vectors[:-1], vectors[1:], strict=True):
        if math.dist(a, b) > settings.max_segment_radius * radius * (1 + 1e-9):
            return None
        turns.append(math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]))
    winding = sum(turns)
    if abs(winding) + 1e-9 < math.radians(settings.min_sweep_deg) or abs(winding) > math.tau * 1.1:
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
        self.hand: HandPose | None = None
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
        self.hand = None
        # A still-pinched hand returning after loss cannot continue an old trace.
        self._ready = False

    def observe(self, observation: HandTrackingResult, timestamp_us: int) -> ImageRegion | None:
        if timestamp_us <= self._timestamp:
            return None
        self._timestamp = timestamp_us
        if self._last_seen is not None and timestamp_us - self._last_seen > self._max_gap_us:
            self.reset()
        pose = match_hand(observation, self.hand, self._settings.max_jump_palm)
        if pose is None:
            self.unavailable()
            return None
        width, height = observation.width, observation.height
        self._short_side = min(width, height)
        pointer, ratio = pose.pointer, pose.ratio
        if not (0 <= pointer[0] <= width and 0 <= pointer[1] <= height):
            self.unavailable()
            return None
        if self._points and math.dist(pointer, self._points[-1]) > self._settings.max_jump_palm * pose.scale:
            self.unavailable()
            return None
        self.hand = pose
        self._last_seen = timestamp_us
        if ratio >= self._settings.release_ratio:
            if self._release_start is None:
                self._release_start = timestamp_us
            if timestamp_us - self._release_start < self._release_us:
                return None
            self._points = []
            self.hand = None
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
        self._points.append(pointer)
        region = _circle_region(self._points, width, height, self._settings)
        if region is not None:
            self._points = []
            self._ready = False
        return region
