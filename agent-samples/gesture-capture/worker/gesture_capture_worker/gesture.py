# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose gesture selection with debounced box hold and release."""

from __future__ import annotations

from xr_ai_tools.hand_tracking import HandTrackingResult
from xr_ai_tools.image_crop import ImageRegion

from .box import camera_box
from .circle import PinchCircleTrigger
from .config import Settings
from .diagnostics import GestureDiagnostics


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

    def __init__(self, settings: Settings, *, diagnostics: GestureDiagnostics | None = None) -> None:
        self._settings = settings
        self._diagnostics = diagnostics or GestureDiagnostics()
        self._box = CameraBoxTrigger(
            hold_s=settings.hold_s, release_s=settings.release_s,
            max_gap_s=settings.max_gap_s, movement_tolerance=settings.movement_tolerance,
        )
        self._circle = PinchCircleTrigger(settings.circle, release_s=settings.release_s,
                                          max_gap_s=settings.max_gap_s, diagnostics=diagnostics)
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
            self._diagnostics.count("selector_dimensions_or_frame_gap")
            self.reset()
        self._timestamp = timestamp_us
        self._dimensions = dimensions
        circle_active = self._circle.active
        region = self._circle.observe(observation, timestamp_us)
        self.circle_selected = region is not None
        box = camera_box(observation, corner_tolerance=self._settings.corner_tolerance,
                         angle_cosine=self._settings.angle_cosine,
                         finger_clearance_ratio=self._settings.finger_clearance_ratio)
        if box is not None:
            self._diagnostics.count("box_candidate")
        if circle_active or self._circle.active:
            self._box.unavailable()
            return region
        if len(observation.hands) != 2:
            self._box.missing(timestamp_us)
            return region
        selected = region or self._box.observe(box, timestamp_us)
        if selected is not None:
            self._diagnostics.count("box_recognized")
        return selected

    def unavailable(self) -> None:
        # The next timestamp expires stale evidence. Unknown sensing must not
        # immediately erase a candidate or complete a deliberate release.
        self._box.missing(self._timestamp)
        self._circle.unavailable()

    def reset(self) -> None:
        self._box.unavailable()
        self._circle.reset()
