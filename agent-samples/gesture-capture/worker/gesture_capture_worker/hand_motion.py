# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort sample spatial continuity; handedness labels are not identities."""

import math
from dataclasses import dataclass

from xr_ai_tools.hand_tracking import HandTrackingResult


@dataclass(frozen=True)
class HandPose:
    center: tuple[float, float]
    scale: float
    pointer: tuple[float, float]
    ratio: float | None
    bounds: tuple[float, float, float, float]


def match_hand(observation: HandTrackingResult, previous: HandPose | None, max_jump: float) -> HandPose | None:
    candidates = []
    for hand in observation.hands:
        points = [(p.x * observation.width, p.y * observation.height) for p in hand.landmarks]
        # Relative z has roughly x scale, so use width for z as well. This
        # reduces projection-only tip overlap; it is not metric depth/confidence.
        points_3d = [(p.x * observation.width, p.y * observation.height, p.z * observation.width)
                     for p in hand.landmarks]
        scale = math.dist(points_3d[5], points_3d[17])
        if scale <= 0:
            continue
        center = tuple(sum(points[i][axis] for i in (0, 5, 9, 13, 17)) / 5 for axis in (0, 1))
        if previous and math.dist(center, previous.center) > max_jump * min(scale, previous.scale):
            continue
        xs, ys = zip(*points, strict=True)
        candidates.append(HandPose(center, scale,
                                  tuple((points[4][i] + points[8][i]) / 2 for i in (0, 1)),
                                  (math.dist(points_3d[4], points_3d[8]) / scale
                                   if math.dist(points[5], points[17]) >= 0.2 * scale else None),
                                  (min(xs), min(ys), max(xs), max(ys))))
    return candidates[0] if len(candidates) == 1 else None
