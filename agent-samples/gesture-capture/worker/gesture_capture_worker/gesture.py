# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample-owned camera-box geometry and hold/release policy."""

from __future__ import annotations

import math

from xr_ai_tools.hand_tracking import HandTrackingResult, TrackedHand
from xr_ai_tools.image_crop import ImageRegion


def _corner(hand: TrackedHand, width: int, height: int, angle_cosine: float):
    points = [(p.x * width, p.y * height) for p in hand.landmarks]
    thumb, index = points[4], points[8]
    a, b = points[2], points[5]
    u = (thumb[0] - a[0], thumb[1] - a[1])
    v = (index[0] - b[0], index[1] - b[1])
    lengths = math.hypot(*u) * math.hypot(*v)
    if not lengths or abs(u[0] * v[0] + u[1] * v[1]) / lengths > angle_cosine:
        return None
    cross = u[0] * v[1] - u[1] * v[0]
    t = ((b[0] - a[0]) * v[1] - (b[1] - a[1]) * v[0]) / cross
    corner = (a[0] + t * u[0], a[1] + t * u[1])
    # The intersection should be near the finger bases, behind both tips.
    if math.dist(corner, a) > math.hypot(*u) or math.dist(corner, b) > math.hypot(*v):
        return None
    if math.dist(corner, thumb) < math.hypot(*u) / 2 or math.dist(corner, index) < math.hypot(*v) / 2:
        return None
    return corner, thumb, index


def camera_box(observation: HandTrackingResult, *, corner_tolerance: float,
               angle_cosine: float) -> ImageRegion | None:
    """Find opposing thumb/index L shapes; return their image-space bounding box."""
    if len(observation.hands) != 2:
        return None
    corners = [_corner(h, observation.width, observation.height, angle_cosine) for h in observation.hands]
    if any(c is None for c in corners):
        return None
    a, at, ai = corners[0]
    b, bt, bi = corners[1]
    span = min(math.dist(a, at), math.dist(a, ai), math.dist(b, bt), math.dist(b, bi))
    if span <= 0:
        return None
    mismatch = min(max(math.dist(at, bi), math.dist(ai, bt)),
                   max(math.dist(at, bt), math.dist(ai, bi)))
    if mismatch > corner_tolerance * span:
        return None
    if math.dist(a, b) < span:
        return None
    points = (a, b, at, ai, bt, bi)
    xs = [p[0] / observation.width for p in points]
    ys = [p[1] / observation.height for p in points]
    if min(xs) < 0 or max(xs) > 1 or min(ys) < 0 or max(ys) > 1:
        return None
    return ImageRegion(left=min(xs), top=min(ys), right=max(xs), bottom=max(ys))


class CameraBoxTrigger:
    """Emit once after a stable hold; rearm only after an observed release."""

    def __init__(self, *, hold_s: float, release_s: float, max_gap_s: float = 0.5,
                 movement_tolerance: float = 0.05) -> None:
        self._max_gap_us = max_gap_s * 1_000_000
        self._hold_us = hold_s * 1_000_000
        self._release_us = release_s * 1_000_000
        self._movement_tolerance = movement_tolerance
        self._candidate: tuple[int, ImageRegion] | None = None
        self._release_start: int | None = None
        self._latched = False
        self._last_timestamp = -1

    def observe(self, region: ImageRegion | None, timestamp_us: int) -> ImageRegion | None:
        if timestamp_us <= self._last_timestamp:
            return None
        if timestamp_us - self._last_timestamp > self._max_gap_us:
            self.unavailable()
        self._last_timestamp = timestamp_us
        if region is None:
            self._candidate = None
            if self._release_start is None:
                self._release_start = timestamp_us
            if timestamp_us - self._release_start >= self._release_us:
                self._latched = False
            return None
        self._release_start = None
        if self._latched:
            return None
        if self._candidate is not None:
            started, previous = self._candidate
            movement = max(abs(getattr(previous, key) - getattr(region, key))
                           for key in ("left", "top", "right", "bottom"))
            if movement <= self._movement_tolerance:
                if timestamp_us - started >= self._hold_us:
                    self._latched = True
                    self._candidate = None
                    return region
                return None
        self._candidate = (timestamp_us, region)
        return None

    def unavailable(self) -> None:
        """Unknown observations break continuity without authorizing another capture."""
        self._candidate = None
        self._release_start = None
