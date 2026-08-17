# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure spatial calculations for native XR tools."""

from __future__ import annotations

import math
from typing import Literal

from .types import SpatialFrame, Vector3


def _position(x: float, y: float, z: float) -> Vector3:
    return Vector3(x=round(x, 3), y=round(y, 3), z=round(z, 3))


def _horizontal_axes(frame: SpatialFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    forward_x, forward_z = frame.forward.x, frame.forward.z
    magnitude = math.sqrt(forward_x * forward_x + forward_z * forward_z)
    if magnitude < 1e-6:
        right_x, right_z = frame.right.x, frame.right.z
        right_magnitude = math.sqrt(right_x * right_x + right_z * right_z)
        if right_magnitude < 1e-6:
            forward_x, forward_z = 0.0, -1.0
        else:
            forward_x, forward_z = right_z / right_magnitude, -right_x / right_magnitude
    else:
        forward_x, forward_z = forward_x / magnitude, forward_z / magnitude
    return (forward_x, forward_z), (-forward_z, forward_x)


def offset_user_frame(
    frame: SpatialFrame,
    start: Vector3,
    *,
    forward: float = 0.0,
    right: float = 0.0,
    up: float = 0.0,
) -> Vector3:
    """Offset a point along the user's horizontal axes and world-space up.

    The user's forward and right axes are projected onto the horizontal plane.
    Returned coordinates are rounded to millimeter precision.
    """

    (forward_x, forward_z), (right_x, right_z) = _horizontal_axes(frame)
    return _position(
        start.x + forward_x * forward + right_x * right,
        start.y + up,
        start.z + forward_z * forward + right_z * right,
    )


def gaze_target(frame: SpatialFrame, distance: float = 1.5) -> Vector3:
    """Return a point along the user's three-dimensional gaze direction."""

    if distance < 0:
        raise ValueError("distance must be non-negative")
    return _position(
        frame.origin.x + frame.forward.x * distance,
        frame.origin.y + frame.forward.y * distance,
        frame.origin.z + frame.forward.z * distance,
    )


def user_relative(
    frame: SpatialFrame,
    direction: Literal["front", "back", "left", "right", "above", "below"],
    distance: float,
) -> Vector3:
    """Return a point in a named direction from the user's frame origin."""

    if distance < 0:
        raise ValueError("distance must be non-negative; flip the direction instead")
    offsets = {
        "front": (distance, 0.0, 0.0),
        "back": (-distance, 0.0, 0.0),
        "left": (0.0, -distance, 0.0),
        "right": (0.0, distance, 0.0),
        "above": (0.0, 0.0, distance),
        "below": (0.0, 0.0, -distance),
    }
    forward, right, up = offsets[direction]
    return offset_user_frame(frame, frame.origin, forward=forward, right=right, up=up)


def anchor_relative(
    frame: SpatialFrame,
    anchor: Vector3,
    relation: Literal[
        "toward_user",
        "away_from_user",
        "left_of",
        "right_of",
        "above",
        "below",
    ],
    distance: float,
) -> Vector3:
    """Return a point in a named user-relative direction from an anchor."""

    if distance < 0:
        raise ValueError("distance must be non-negative; flip the direction instead")
    offsets = {
        "toward_user": (-distance, 0.0, 0.0),
        "away_from_user": (distance, 0.0, 0.0),
        "left_of": (0.0, -distance, 0.0),
        "right_of": (0.0, distance, 0.0),
        "above": (0.0, 0.0, distance),
        "below": (0.0, 0.0, -distance),
    }
    forward, right, up = offsets[relation]
    return offset_user_frame(frame, anchor, forward=forward, right=right, up=up)


def toward(
    start: Vector3,
    target: Vector3,
    distance: float,
) -> Vector3:
    """Move a point by a signed distance along the line toward a target."""

    delta_x = target.x - start.x
    delta_y = target.y - start.y
    delta_z = target.z - start.z
    magnitude = math.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)
    if magnitude < 1e-9:
        raise ValueError("origin and target coincide")
    scale = distance / magnitude
    return _position(
        start.x + delta_x * scale,
        start.y + delta_y * scale,
        start.z + delta_z * scale,
    )


def midpoint(first: Vector3, second: Vector3) -> Vector3:
    """Return the midpoint between two positions."""

    return _position(
        (first.x + second.x) / 2,
        (first.y + second.y) / 2,
        (first.z + second.z) / 2,
    )


__all__ = [
    "anchor_relative",
    "gaze_target",
    "midpoint",
    "offset_user_frame",
    "toward",
    "user_relative",
]
