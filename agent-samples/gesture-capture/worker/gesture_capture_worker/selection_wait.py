# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Circle-only frozen selection: observe hand clearance, then capture a fresh frame."""

import time

from xr_ai_tools.current_frame import ImageFrame
from xr_ai_tools.hand_tracking import HandTrackingResult
from xr_ai_tools.image_crop import ImageRegion

from .config import Settings
from .diagnostics import GestureDiagnostics
from .hand_motion import HandPose, match_hand


class CircleCaptureWait:
    def __init__(self, region: ImageRegion, hand: HandPose | None, frame: ImageFrame, settings: Settings,
                 diagnostics: GestureDiagnostics | None = None) -> None:
        self._diagnostics = diagnostics or GestureDiagnostics()
        self.region = region
        self.hand = hand
        self.track = frame.track_id
        self.dimensions = (frame.width, frame.height)
        self.timestamp = frame.timestamp_us
        self.deadline = frame.timestamp_us + settings.circle.clear_timeout_s * 1_000_000
        self.settings = settings
        self.clear_since = None
        self.last_observed = time.monotonic()

    @property
    def expired(self) -> bool:
        return time.monotonic() - self.last_observed > self.settings.max_gap_s

    def observe(self, observation: HandTrackingResult, frame: ImageFrame) -> bool:
        """Return ready; reject source discontinuity before considering timeout."""
        timestamp = frame.timestamp_us
        if (frame.track_id != self.track or (frame.width, frame.height) != self.dimensions
                or timestamp - self.timestamp > self.settings.max_gap_s * 1_000_000):
            self._diagnostics.count("pending_source_or_frame_gap")
            raise ValueError("Circle selection expired after video source interruption; draw again")
        if timestamp <= self.timestamp:
            return False
        self.timestamp = timestamp
        self.last_observed = time.monotonic()
        pose = match_hand(observation, self.hand, self.settings.circle.max_jump_palm, self._diagnostics)
        self._diagnostics.count("pending_observation")
        if pose is None:
            self._diagnostics.count("pending_association_unknown")
            self.clear_since = None
        else:
            self.hand = pose
            left, top, right, bottom = pose.bounds
            region = self.region
            overlap = max(0, min(right, region.right * frame.width) - max(left, region.left * frame.width)) * max(
                0, min(bottom, region.bottom * frame.height) - max(top, region.top * frame.height))
            hand_area = (right - left) * (bottom - top)
            region_area = (region.right - region.left) * frame.width * (region.bottom - region.top) * frame.height
            disjoint = (right <= region.left * frame.width or left >= region.right * frame.width
                        or bottom <= region.top * frame.height or top >= region.bottom * frame.height)
            mostly_clear = hand_area > 0 and overlap <= self.settings.circle.clear_overlap * min(hand_area, region_area)
            if disjoint or mostly_clear:
                if self.clear_since is None:
                    self.clear_since = timestamp
            else:
                self.clear_since = None
        timed_out = timestamp >= self.deadline
        cleared = (self.clear_since is not None
                   and timestamp - self.clear_since >= self.settings.release_s * 1_000_000)
        if timed_out or cleared:
            self._diagnostics.count("pending_timeout" if timed_out else "pending_cleared")
        return timed_out or cleared
