# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort sample spatial continuity; handedness labels are not identities."""

import math
from dataclasses import dataclass

from xr_ai_tools.hand_tracking import HandTrackingResult

from .diagnostics import GestureDiagnostics


@dataclass(frozen=True)
class HandPose:
    center: tuple[float, float]
    scale: float
    pointer: tuple[float, float]
    ratio: float
    bounds: tuple[float, float, float, float]


def hand_poses(observation: HandTrackingResult) -> list[HandPose]:
    """Extract image-space geometry without assigning identity or gesture policy."""
    poses = []
    for hand in observation.hands:
        points = [(p.x * observation.width, p.y * observation.height) for p in hand.landmarks]
        scale = math.dist(points[5], points[17])
        if scale <= 0:
            continue
        center = tuple(sum(points[i][axis] for i in (0, 5, 9, 13, 17)) / 5 for axis in (0, 1))
        xs, ys = zip(*points, strict=True)
        poses.append(HandPose(center, scale,
                              tuple((points[4][i] + points[8][i]) / 2 for i in (0, 1)),
                              math.dist(points[4], points[8]) / scale,
                              (min(xs), min(ys), max(xs), max(ys))))
    return poses


def match_hand(observation: HandTrackingResult, previous: HandPose | None, max_jump: float,
               diagnostics: GestureDiagnostics | None = None) -> HandPose | None:
    candidates = [pose for pose in hand_poses(observation) if previous is None
                  or math.dist(pose.center, previous.center) <= max_jump * min(pose.scale, previous.scale)]
    if diagnostics:
        diagnostics.count("associated" if len(candidates) == 1 else "no_hands_returned" if not observation.hands
                          else "association_ambiguous" if len(candidates) > 1 else "association_no_usable_nearby_hand")
    return candidates[0] if len(candidates) == 1 else None
