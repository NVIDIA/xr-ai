# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample-owned two-hand framing geometry and inward image-region bounds."""

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


def _finger_lines(corners, hands, width, height, clearance_ratio):
    lines = []
    for (corner, thumb, index), hand in zip(corners, hands, strict=True):
        for tip, joint in ((thumb, 3), (index, 7)):
            point = hand.landmarks[joint]
            clearance = math.dist(tip, (point.x * width, point.y * height)) * clearance_ratio
            lines.append((corner, (tip[0] - corner[0], tip[1] - corner[1]), clearance))
    return lines


def _interior_region(corners, hands, width: int, height: int, clearance_ratio: float) -> ImageRegion | None:
    center = tuple((a + b) / 2 for a, b in zip(corners[0][0], corners[1][0], strict=True))
    return _inset_region(_finger_lines(corners, hands, width, height, clearance_ratio), center, width, height)


def _inset_region(lines, center, width, height):
    constraints = []
    for origin, direction, clearance in lines:
        dx, dy = direction
        length = math.hypot(dx, dy)
        nx, ny = -dy / length, dx / length
        distance = nx * (center[0] - origin[0]) + ny * (center[1] - origin[1])
        if distance < 0:
            nx, ny, distance = -nx, -ny, -distance
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


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _ray_intersection(a, u, b, v):
    determinant = _cross(u, v)
    if math.isclose(determinant, 0, abs_tol=1e-9):
        return None
    delta = (b[0] - a[0], b[1] - a[1])
    t, q = _cross(delta, v) / determinant, _cross(delta, u) / determinant
    if t <= 0 or q <= 0:
        return None
    return a[0] + t * u[0], a[1] + t * u[1]


def _opposed_rays(corners, tolerance):
    a, at, ai = corners[0]
    b, bt, bi = corners[1]
    u, v = (at[0] - a[0], at[1] - a[1]), (ai[0] - a[0], ai[1] - a[1])
    for matched_u, matched_v in ((bt, bi), (bi, bt)):
        p, q = (matched_u[0] - b[0], matched_u[1] - b[1]), (matched_v[0] - b[0], matched_v[1] - b[1])
        if any(x[0] * y[0] + x[1] * y[1] >= 0 or abs(_cross(x, y)) > tolerance * math.hypot(*x) * math.hypot(*y)
               for x, y in ((u, p), (v, q))):
            continue
        c, d = _ray_intersection(a, u, b, q), _ray_intersection(a, v, b, p)
        if c is not None and d is not None:
            return True
    return False


def _tip_region(corners, hands, width, height, tolerance, angle_cosine, clearance_ratio):
    # Each hand supplies adjacent corners. Palms must stay outside that edge;
    # otherwise four tips could surround a crop containing the hands themselves.
    vertices = [(tip, hand_index) for hand_index, corner in enumerate(corners) for tip in corner[1:]]
    center = tuple(sum(point[axis] for point, _ in vertices) / 4 for axis in (0, 1))
    vertices.sort(key=lambda item: math.atan2(item[0][1] - center[1], item[0][0] - center[0]))
    points = [point for point, _ in vertices]
    edges = [(points[(i + 1) % 4][0] - points[i][0], points[(i + 1) % 4][1] - points[i][1]) for i in range(4)]
    lengths = [math.hypot(*edge) for edge in edges]
    if not all(lengths) or any(_cross(edges[i], edges[(i + 1) % 4]) <= 0 for i in range(4)):
        return None
    if any(abs(sum(x * y for x, y in zip(edges[i], edges[(i + 1) % 4], strict=True)))
           > angle_cosine * lengths[i] * lengths[(i + 1) % 4] for i in range(4)):
        return None
    if any(abs(_cross(edges[i], edges[i + 2])) > tolerance * lengths[i] * lengths[i + 2] for i in range(2)):
        return None
    for hand_index, hand in enumerate(hands):
        pair = [i for i, (_, owner) in enumerate(vertices) if owner == hand_index]
        if (pair[1] - pair[0]) % 4 not in (1, 3):
            return None
        start = pair[0] if (pair[0] + 1) % 4 == pair[1] else pair[1]
        origin, edge = points[start], edges[start]
        for joint in (0, 2, 5):
            point = hand.landmarks[joint]
            delta = point.x * width - origin[0], point.y * height - origin[1]
            if _cross(edge, delta) > 0:
                return None
    lines = _finger_lines(corners, hands, width, height, clearance_ratio)
    clearances = []
    for point, owner in vertices:
        joint_id = 3 if point == corners[owner][1] else 7
        joint = hands[owner].landmarks[joint_id]
        clearances.append(math.dist(point, (joint.x * width, joint.y * height)) * clearance_ratio)
    lines += [(points[i], edges[i], max(clearances[i], clearances[(i + 1) % 4])) for i in range(4)]
    return _inset_region(lines, center, width, height)


def camera_box(observation: HandTrackingResult, *, corner_tolerance: float,
               angle_cosine: float, finger_clearance_ratio: float = 0.5) -> ImageRegion | None:
    """Recognize separated tip corners or opposed L rays; no fingertip contact needed."""
    if len(observation.hands) != 2:
        return None
    width, height = observation.width, observation.height
    corners = [_corner(hand, width, height, angle_cosine) for hand in observation.hands]
    if any(corner is None for corner in corners):
        return None
    a, at, ai = corners[0]
    b, bt, bi = corners[1]
    if any(not (0 <= x <= width and 0 <= y <= height) for corner in corners for x, y in corner):
        return None
    span = min(math.dist(a, at), math.dist(a, ai), math.dist(b, bt), math.dist(b, bi))
    mismatch = min(max(math.dist(at, bi), math.dist(ai, bt)), max(math.dist(at, bt), math.dist(ai, bi)))
    # Near tip clusters disambiguate the legacy frame; they are not a condition
    # for the separated layouts. Preserve its crop even with noisy distinct tips.
    if mismatch <= corner_tolerance * span and math.dist(a, b) >= span:
        return _interior_region(corners, observation.hands, width, height, finger_clearance_ratio)
    for (corner, thumb, index), hand in zip(corners, observation.hands, strict=True):
        for tip, joint_id in ((thumb, 3), (index, 7)):
            joint = hand.landmarks[joint_id]
            # New separated layouts need extended fingers, not a folded tip
            # behind its distal joint. The legacy branch above stays unchanged.
            if ((tip[0] - corner[0]) * (tip[0] - joint.x * width)
                    + (tip[1] - corner[1]) * (tip[1] - joint.y * height)) <= 0:
                return None
    if _opposed_rays(corners, corner_tolerance):
        region = _interior_region(corners, observation.hands, width, height, finger_clearance_ratio)
        if region is not None:
            return region
    return _tip_region(corners, observation.hands, width, height, corner_tolerance,
                       angle_cosine, finger_clearance_ratio)
