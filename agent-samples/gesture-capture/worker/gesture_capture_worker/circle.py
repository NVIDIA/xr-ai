# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample-owned pinch trajectory and circular-region selection."""

from __future__ import annotations

import math

from xr_ai_tools.hand_tracking import HandTrackingResult
from xr_ai_tools.image_crop import ImageRegion

from .config import CircleSettings
from .diagnostics import GestureDiagnostics
from .grip import Grip, GripHysteresis
from .hand_motion import HandPose, hand_poses


def _circle_region(points: list[tuple[float, float]], width: int, height: int,
                   settings: CircleSettings, diagnostics: GestureDiagnostics | None = None,
                   segments: list[tuple[tuple[float, float], tuple[float, float]]] | None = None) -> ImageRegion | None:
    def reject(reason):
        if diagnostics:
            diagnostics.count(reason)
        return None
    if diagnostics and diagnostics.enabled and points:
        diagnostics.number("trace_width_px", max(x for x, _ in points) - min(x for x, _ in points))
        diagnostics.number("trace_height_px", max(y for _, y in points) - min(y for _, y in points))
    if len(points) < settings.min_points:
        return reject("circle_min_points")
    xs, ys = zip(*points, strict=True)
    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
    diameter_x, diameter_y = right - left, bottom - top
    if min(diameter_x, diameter_y) < settings.min_diameter * min(width, height):
        return reject("circle_too_small")
    if min(diameter_x, diameter_y) / max(diameter_x, diameter_y) < settings.min_axis_ratio:
        return reject("circle_elongated")
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
        return reject("circle_degenerate_fit")
    center = (mean_x + (bx * yy - by * xy) / determinant,
              mean_y + (by * xx - bx * xy) / determinant)
    vectors = [(x - center[0], y - center[1]) for x, y in points]
    radius = sum(math.hypot(*v) for v in vectors) / len(vectors)
    if diagnostics:
        diagnostics.number("endpoint_gap_radius", math.dist(points[0], points[-1]) / radius)
    if math.dist(points[0], points[-1]) > settings.closure_tolerance * radius * (1 + 1e-9):
        return reject("circle_endpoint_gap")
    if any(abs(math.hypot(*v) / radius - 1) > settings.radial_tolerance for v in vectors):
        return reject("circle_radial_error")
    turns = []
    observed_segments = segments if segments is not None else list(zip(points[:-1], points[1:], strict=True))
    for start, end in observed_segments:
        a = (start[0] - center[0], start[1] - center[1])
        b = (end[0] - center[0], end[1] - center[1])
        if math.dist(a, b) > settings.max_segment_radius * radius * (1 + 1e-9):
            return reject("circle_sparse_segment")
        turns.append(math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]))
    winding = sum(turns)
    if abs(winding) + 1e-9 < math.radians(settings.min_sweep_deg) or abs(winding) > math.tau * 1.1:
        return reject("circle_sweep")
    backtrack = (sum(abs(turn) for turn in turns) - abs(winding)) / 2
    if backtrack > settings.max_backtrack * math.tau:
        return reject("circle_backtrack")
    return ImageRegion(left=left / width, right=right / width, top=top / height, bottom=bottom / height)


class PinchCircleTrigger:
    """Recognize a closed pinched trace once; require stable release to rearm."""

    def __init__(self, settings: CircleSettings, *, release_s: float = 0.4, max_gap_s: float = 0.5,
                 diagnostics: GestureDiagnostics | None = None) -> None:
        self._settings = settings
        self._diagnostics = diagnostics or GestureDiagnostics()
        hand_gap_s = settings.occlusion_grace_s
        self._anchor_gap_us = max_gap_s * 1_000_000
        self._grip = GripHysteresis(pinch_ratio=settings.pinch_ratio, release_ratio=settings.release_ratio,
                                    dwell_s=release_s, max_gap_s=hand_gap_s)
        self._max_gap_us = hand_gap_s * 1_000_000
        self._last_seen: int | None = None
        self._timestamp = -1
        self._short_side = 0
        self._points: list[tuple[float, float]] = []
        self._candidate_points: list[tuple[float, float]] = []
        self._segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self._candidate_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self._interrupted = False
        self.hand: HandPose | None = None
        self._ready = True
        self._completed_latch = False

    @property
    def active(self) -> bool:
        return bool(self._points or self._candidate_points)

    @property
    def drawing(self) -> bool:
        trajectory = self._points or self._candidate_points
        if len(trajectory) < 2:
            return False
        xs, ys = zip(*trajectory, strict=True)
        return max(max(xs) - min(xs), max(ys) - min(ys)) >= self._settings.min_diameter * self._short_side

    def unavailable(self) -> None:
        self._grip.unavailable()
        self._interrupted = True

    def reset(self) -> None:
        self._diagnostics.count("circle_reset")
        if not self._completed_latch and self.active:
            self._diagnostics.count("fresh_attempt_after_loss")
        self.unavailable()
        self._points.clear()
        self._candidate_points.clear()
        self._segments.clear()
        self._candidate_segments.clear()
        self._interrupted = False
        self.hand = None
        self._last_seen = None
        # Recognition owns the release latch. Failed acquisition discards its
        # path and permits normal pinch entry, without continuing old motion.
        # Tracking loss cancels a stroke, never the remembered physical grip.
        self._diagnostics.grip(self._grip.state.value)
        self._ready = not self._completed_latch

    def _append(self, points, segments, pointer) -> None:
        if points and not self._interrupted and points[-1] != pointer:
            segments.append((points[-1], pointer))
        points.append(pointer)
        self._interrupted = False

    def _select_hand(self, observation: HandTrackingResult, timestamp_us: int) -> HandPose | None:
        poses = hand_poses(observation)
        possible = [pose for pose in poses if pose.ratio < self._settings.release_ratio]
        if (self._completed_latch and len(poses) > 1 and possible and self._last_seen is not None
                and timestamp_us - self._last_seen > self._anchor_gap_us):
            self._diagnostics.count("association_ambiguous")
            return None  # An open bystander cannot release a stale tracked engagement.
        if not self._completed_latch and len(poses) > 1 and possible:
            if len(possible) > 1 and (self.hand is None or self._last_seen is None
                                      or timestamp_us - self._last_seen > self._anchor_gap_us):
                self._diagnostics.count("association_ambiguous")
                return None
            # Exclude clearly open hands, retaining the confirmed grip's band.
            poses = possible
        trajectory = self._points or self._candidate_points
        center_near = [pose for pose in poses if self.hand is None or math.dist(pose.center, self.hand.center)
                       <= self._settings.max_jump_palm * min(pose.scale, self.hand.scale)]
        nearby = [pose for pose in center_near if not trajectory or math.dist(pose.pointer, trajectory[-1])
                  <= self._settings.max_jump_palm * pose.scale]
        if len(nearby) == 1:
            self._diagnostics.count("associated")
            return nearby[0]
        if (self._completed_latch and len(poses) == 1 and poses[0].ratio >= self._settings.release_ratio
                and self._last_seen is not None and timestamp_us - self._last_seen > self._anchor_gap_us):
            # A unique explicit open hand can rearm after moving during the turn;
            # it still needs the normal release dwell and cannot submit a query.
            self._diagnostics.count("association_release_reanchored")
            return poses[0]
        if (len(poses) == 1 and poses[0].ratio <= self._settings.pinch_ratio
                and self._last_seen is not None and timestamp_us - self._last_seen > self._anchor_gap_us):
            # A stale spatial anchor must not strand positive pinch input. Start
            # a NEW stroke: never join this actor/position to the old trajectory.
            self.reset()
            self._diagnostics.count("association_reanchored_new_attempt")
            return poses[0]
        reason = ("no_hands_returned" if not observation.hands else "association_ambiguous" if len(nearby) > 1
                  else "pointer_jump" if center_near else "association_no_usable_nearby_hand")
        self._diagnostics.count(reason)
        return None

    def observe(self, observation: HandTrackingResult, timestamp_us: int) -> ImageRegion | None:
        if timestamp_us <= self._timestamp:
            return None
        self._timestamp = timestamp_us
        if self._last_seen is not None and timestamp_us - self._last_seen > self._max_gap_us:
            if self.active:
                self._diagnostics.count("circle_occlusion_cancelled")
            self.reset()
        pose = self._select_hand(observation, timestamp_us)
        if pose is None:
            self._grip.observe(None, timestamp_us)
            self.unavailable()
            return None
        width, height = observation.width, observation.height
        self._short_side = min(width, height)
        pointer, ratio = pose.pointer, pose.ratio
        self._diagnostics.pinch(ratio, ratio * pose.scale, pose.scale)
        if not (0 <= pointer[0] <= width and 0 <= pointer[1] <= height):
            self._diagnostics.count("pointer_outside")
            self._grip.observe(None, timestamp_us)
            self.unavailable()
            return None
        if self._grip.state is Grip.OPEN and not self._grip.entry_pending(timestamp_us):
            self._candidate_points.clear()
            self._candidate_segments.clear()
        self.hand = pose
        returning = self._interrupted
        was_open = self._grip.state is Grip.OPEN
        strong_pinch = ratio is not None and ratio <= self._settings.pinch_ratio
        if was_open and strong_pinch and self._ready:
            if not self._candidate_points or pointer != self._candidate_points[-1]:
                self._append(self._candidate_points, self._candidate_segments, pointer)
                self._diagnostics.count("entry_pointer_added")
            if len(self._candidate_points) > self._settings.max_points:
                self.reset()
                return None
        elif not was_open or (ratio is not None and ratio >= self._settings.release_ratio):
            self._candidate_points.clear()
            self._candidate_segments.clear()
        closed = self._grip.observe(ratio, timestamp_us)
        self._diagnostics.grip(self._grip.state.value)
        self._diagnostics.count("evidence_closed" if strong_pinch else "evidence_open"
                                if ratio >= self._settings.release_ratio else "evidence_band")
        if closed and was_open:
            self._points = self._candidate_points
            self._segments = self._candidate_segments
            self._candidate_points = []
            self._candidate_segments = []
        if self._grip.state is Grip.UNKNOWN:
            self._points.clear()
            self._segments.clear()
            self._ready = False
        if self._grip.state is Grip.OPEN:
            if not was_open:
                self._ready = True
                self._completed_latch = False
            self._points.clear()
            self._segments.clear()
        self._last_seen = timestamp_us
        if not closed or not self._ready:
            self._diagnostics.count("grip_or_rearm_wait")
            if not self._ready:
                self._diagnostics.count("rearm_wait_after_completion" if self._completed_latch
                                        else "rearm_wait_after_loss")
            return None
        if not self._points or pointer != self._points[-1]:
            if len(self._points) >= self._settings.max_points:
                self.reset()
                return None
            self._append(self._points, self._segments, pointer)
            self._diagnostics.count("pointer_added")
        self._diagnostics.number("circle_points", len(self._points))
        if returning:
            self._interrupted = False
            self._diagnostics.count("circle_reacquired")
            return None
        region = _circle_region(self._points, width, height, self._settings, self._diagnostics, self._segments)
        if region is not None:
            self._diagnostics.count("circle_recognized")
            self._points = []
            self._segments.clear()
            self._ready = False
            self._completed_latch = True
        return region
