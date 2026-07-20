# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure coordinate calculations shared by native functions and compatibility adapters."""

from __future__ import annotations

import math
from typing import Literal

from .schemas import SpatialFrame, Vector3


def _position(x: float, y: float, z: float) -> dict[str, float]:
    return {"x": round(x, 3), "y": round(y, 3), "z": round(z, 3)}


def _require_non_negative_distance(distance_meters: float) -> None:
    if distance_meters < 0:
        raise ValueError("distance_meters must be non-negative; choose the opposite direction instead")


def _horizontal_user_axes(
    user_frame: SpatialFrame,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return normalized forward and right axes projected onto the ground plane."""
    forward_x, forward_z = user_frame.forward.x, user_frame.forward.z
    magnitude = math.sqrt(forward_x * forward_x + forward_z * forward_z)
    if magnitude < 1e-6:
        right_x, right_z = user_frame.right.x, user_frame.right.z
        right_magnitude = math.sqrt(right_x * right_x + right_z * right_z)
        if right_magnitude < 1e-6:
            forward_x, forward_z = 0.0, -1.0
        else:
            forward_x, forward_z = right_z / right_magnitude, -right_x / right_magnitude
    else:
        forward_x, forward_z = forward_x / magnitude, forward_z / magnitude
    return (forward_x, forward_z), (-forward_z, forward_x)


def _offset_in_user_frame(
    user_frame: SpatialFrame,
    start_position: Vector3,
    *,
    forward_meters: float = 0.0,
    right_meters: float = 0.0,
    up_meters: float = 0.0,
) -> dict[str, float]:
    (forward_x, forward_z), (right_x, right_z) = _horizontal_user_axes(user_frame)
    return _position(
        start_position.x + forward_x * forward_meters + right_x * right_meters,
        start_position.y + up_meters,
        start_position.z + forward_z * forward_meters + right_z * right_meters,
    )


def compute_gaze_target(
    user_frame: SpatialFrame,
    distance_meters: float = 1.5,
) -> dict[str, float]:
    """Compute a world position along the user's full three-dimensional gaze ray."""
    _require_non_negative_distance(distance_meters)
    return _position(
        user_frame.origin.x + user_frame.forward.x * distance_meters,
        user_frame.origin.y + user_frame.forward.y * distance_meters,
        user_frame.origin.z + user_frame.forward.z * distance_meters,
    )


def compute_user_relative_position(
    user_frame: SpatialFrame,
    direction_from_user: Literal["front", "back", "left", "right", "above", "below"],
    distance_meters: float = 1.5,
) -> dict[str, float]:
    """Compute a gravity-aligned position in a named direction from the user."""
    _require_non_negative_distance(distance_meters)
    offsets = {
        "front": (distance_meters, 0.0, 0.0),
        "back": (-distance_meters, 0.0, 0.0),
        "left": (0.0, -distance_meters, 0.0),
        "right": (0.0, distance_meters, 0.0),
        "above": (0.0, 0.0, distance_meters),
        "below": (0.0, 0.0, -distance_meters),
    }
    forward_meters, right_meters, up_meters = offsets[direction_from_user]
    return _offset_in_user_frame(
        user_frame,
        user_frame.origin,
        forward_meters=forward_meters,
        right_meters=right_meters,
        up_meters=up_meters,
    )


def compute_position_relative_to_anchor(
    user_frame: SpatialFrame,
    *,
    anchor_position: Vector3,
    relation_to_anchor: Literal[
        "toward_user",
        "away_from_user",
        "left_of",
        "right_of",
        "above",
        "below",
    ],
    distance_meters: float = 0.3,
) -> dict[str, float]:
    """Compute a position relative to an anchor using the user's horizontal axes."""
    _require_non_negative_distance(distance_meters)
    offsets = {
        "toward_user": (-distance_meters, 0.0, 0.0),
        "away_from_user": (distance_meters, 0.0, 0.0),
        "left_of": (0.0, -distance_meters, 0.0),
        "right_of": (0.0, distance_meters, 0.0),
        "above": (0.0, 0.0, distance_meters),
        "below": (0.0, 0.0, -distance_meters),
    }
    forward_meters, right_meters, up_meters = offsets[relation_to_anchor]
    return _offset_in_user_frame(
        user_frame,
        anchor_position,
        forward_meters=forward_meters,
        right_meters=right_meters,
        up_meters=up_meters,
    )


def offset_position_in_user_frame(
    user_frame: SpatialFrame,
    *,
    start_position: Vector3,
    forward_meters: float = 0.0,
    right_meters: float = 0.0,
    up_meters: float = 0.0,
) -> dict[str, float]:
    """Offset a world position along the user's forward, right, and up axes."""
    return _offset_in_user_frame(
        user_frame,
        start_position,
        forward_meters=forward_meters,
        right_meters=right_meters,
        up_meters=up_meters,
    )


def compute_position_toward_or_away_from_reference(
    start_position: Vector3,
    reference_position: Vector3,
    *,
    movement_direction: Literal["toward", "away"],
    distance_meters: float = 0.5,
) -> dict[str, float]:
    """Compute a position moved toward or away from a reference by an exact distance."""
    _require_non_negative_distance(distance_meters)
    delta_x = reference_position.x - start_position.x
    delta_y = reference_position.y - start_position.y
    delta_z = reference_position.z - start_position.z
    magnitude = math.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)
    if magnitude < 1e-9:
        raise ValueError("start_position and reference_position coincide")
    sign = 1.0 if movement_direction == "toward" else -1.0
    scale = sign * distance_meters / magnitude
    return _position(
        start_position.x + delta_x * scale,
        start_position.y + delta_y * scale,
        start_position.z + delta_z * scale,
    )


def compute_midpoint(
    first_position: Vector3,
    second_position: Vector3,
) -> dict[str, float]:
    """Compute the world-space midpoint between two positions."""
    return _position(
        (first_position.x + second_position.x) / 2,
        (first_position.y + second_position.y) / 2,
        (first_position.z + second_position.z) / 2,
    )


def world_offset(
    start_position: Vector3,
    *,
    x_meters: float = 0.0,
    y_meters: float = 0.0,
    z_meters: float = 0.0,
) -> dict[str, float]:
    """Apply world-axis offsets for the legacy Vec MCP compatibility surface."""
    return _position(
        start_position.x + x_meters,
        start_position.y + y_meters,
        start_position.z + z_meters,
    )
