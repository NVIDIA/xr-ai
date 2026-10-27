# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample-owned camera-box geometry and hold/release policy."""

from __future__ import annotations

import math

from xr_ai_tools.hand_tracking import HandTrackingResult, TrackedHand
from xr_ai_tools.image_crop import ImageRegion

from .circle import PinchCircleTrigger
from .config import Settings


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


def _interior_region(corners, hands, width: int, height: int, clearance_ratio: float) -> ImageRegion | None:
    center = tuple((a + b) / 2 for a, b in zip(corners[0][0], corners[1][0], strict=True))
    constraints = []
    for (corner, thumb, index), hand in zip(corners, hands, strict=True):
        for tip, joint in ((thumb, 3), (index, 7)):
            dx, dy = tip[0] - corner[0], tip[1] - corner[1]
            length = math.hypot(dx, dy)
            nx, ny = -dy / length, dx / length
            distance = nx * (center[0] - corner[0]) + ny * (center[1] - corner[1])
            if distance < 0:
                nx, ny, distance = -nx, -ny, -distance
            point = hand.landmarks[joint]
            clearance = math.dist(tip, (point.x * width, point.y * height)) * clearance_ratio
            available = distance - clearance
            if available <= 0:
                return None
            constraints.append((abs(nx), abs(ny), available))

    # A centered axis-aligned rectangle fits every inward finger half-plane
    # iff |nx|*half_width + |ny|*half_height <= available. Its largest area
    # occurs at two active constraints or a stationary point on one constraint.
    candidates = [(c / (2 * a), c / (2 * b)) for a, b, c in constraints if a and b]
    for i, (a, b, c) in enumerate(constraints):
        for d, e, f in constraints[i + 1:]:
            determinant = a * e - b * d
            if determinant:
                candidates.append(((c * e - b * f) / determinant, (a * f - c * d) / determinant))
    valid = [(x, y) for x, y in candidates if x > 0 and y > 0 and all(
        a * x + b * y <= c or math.isclose(a * x + b * y, c) for a, b, c in constraints
    )]
    if not valid:
        return None
    half_width, half_height = max(valid, key=lambda point: point[0] * point[1])
    # Round inward because the crop tool normally rounds out to pixel edges.
    left, right = math.ceil(center[0] - half_width), math.floor(center[0] + half_width)
    top, bottom = math.ceil(center[1] - half_height), math.floor(center[1] + half_height)
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        return None
    return ImageRegion(left=left / width, top=top / height, right=right / width, bottom=bottom / height)


def camera_box(observation: HandTrackingResult, *, corner_tolerance: float,
               angle_cosine: float, finger_clearance_ratio: float = 0.5) -> ImageRegion | None:
    """Find opposing L shapes and crop the estimated open interior between fingers."""
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
    return _interior_region(corners, observation.hands, observation.width, observation.height, finger_clearance_ratio)


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
        self._last_positive = -1

    @property
    def in_progress(self) -> bool:
        return self._candidate is not None and not self._latched

    def observe(self, region: ImageRegion | None, timestamp_us: int) -> ImageRegion | None:
        if timestamp_us <= self._last_timestamp:
            return None
        if timestamp_us - self._last_timestamp > self._max_gap_us:
            self.unavailable()
        self._last_timestamp = timestamp_us
        if region is None:
            if timestamp_us - self._last_positive > self._max_gap_us:
                self._candidate = None
            if self._release_start is None:
                self._release_start = timestamp_us
            if timestamp_us - self._release_start >= self._release_us:
                self._latched = False
                self._candidate = None
            return None
        self._last_positive = timestamp_us
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

    def missing(self, timestamp_us: int) -> None:
        """Keep a brief unknown observation without treating it as release."""
        self._release_start = None
        if timestamp_us - self._last_positive > self._max_gap_us:
            self._candidate = None

    def unavailable(self) -> None:
        """Reset source continuity without authorizing another capture."""
        self._candidate = None
        self._release_start = None


class RegionSelector:
    """Compose box and pinch-circle selection into one sample action boundary."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._box = CameraBoxTrigger(
            hold_s=settings.hold_s, release_s=settings.release_s,
            max_gap_s=settings.max_gap_s, movement_tolerance=settings.movement_tolerance,
        )
        self._circle = PinchCircleTrigger(settings.circle, release_s=settings.release_s, max_gap_s=settings.max_gap_s)
        self.circle_selected = False
        self._timestamp = -1
        self._dimensions: tuple[int, int] | None = None

    @property
    def drawing_hand(self):
        return self._circle.hand

    @property
    def in_progress(self) -> bool:
        return self._circle.drawing or self._box.in_progress

    def observe(self, observation: HandTrackingResult, timestamp_us: int) -> ImageRegion | None:
        if timestamp_us <= self._timestamp:
            return None
        dimensions = (observation.width, observation.height)
        if ((self._timestamp >= 0 and timestamp_us - self._timestamp > self._settings.max_gap_s * 1_000_000)
                or (self._dimensions is not None and dimensions != self._dimensions)):
            self.reset()
        self._timestamp = timestamp_us
        self._dimensions = dimensions
        circle_active = self._circle.active
        region = self._circle.observe(observation, timestamp_us)
        self.circle_selected = region is not None
        box = camera_box(observation, corner_tolerance=self._settings.corner_tolerance,
                         angle_cosine=self._settings.angle_cosine,
                         finger_clearance_ratio=self._settings.finger_clearance_ratio)
        if circle_active or self._circle.active:
            self._box.unavailable()
            return region
        if len(observation.hands) != 2:
            self._box.missing(timestamp_us)
            return region
        return region or self._box.observe(box, timestamp_us)

    def unavailable(self) -> None:
        # The next timestamp expires stale evidence. Unknown sensing must not
        # immediately erase a candidate or complete a deliberate release.
        self._box.missing(self._timestamp)
        self._circle.unavailable()

    def reset(self) -> None:
        self._box.unavailable()
        self._circle.reset()
