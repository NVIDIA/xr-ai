# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure calculations shared by native functions and compatibility adapters."""

from __future__ import annotations

import math
from typing import Literal

from .schemas import SpatialFrame, Vector3


def _position(x: float, y: float, z: float) -> dict[str, float]:
    return {"x": round(x, 3), "y": round(y, 3), "z": round(z, 3)}


def _require_distance(distance: float) -> None:
    if distance < 0:
        raise ValueError("distance must be non-negative; flip the direction instead")


def _ground_basis(frame: SpatialFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    fx, fz = frame.forward.x, frame.forward.z
    magnitude = math.hypot(fx, fz)
    if magnitude < 1e-6:
        rx, rz = frame.right.x, frame.right.z
        right_magnitude = math.hypot(rx, rz)
        if right_magnitude < 1e-6:
            fx, fz = 0.0, -1.0
        else:
            fx, fz = rz / right_magnitude, -rx / right_magnitude
    else:
        fx, fz = fx / magnitude, fz / magnitude
    return (fx, fz), (-fz, fx)


def offset(
    frame: SpatialFrame,
    origin: Vector3,
    *,
    forward: float = 0.0,
    right: float = 0.0,
    up: float = 0.0,
) -> dict[str, float]:
    """Apply a gravity-aligned frame-relative offset to an origin."""

    (fx, fz), (rx, rz) = _ground_basis(frame)
    return _position(
        origin.x + fx * forward + rx * right,
        origin.y + up,
        origin.z + fz * forward + rz * right,
    )


def world_offset(
    origin: Vector3,
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
) -> dict[str, float]:
    """Apply a world-axis offset used by the legacy Vec MCP surface."""

    return _position(origin.x + dx, origin.y + dy, origin.z + dz)


def position_in_gaze(frame: SpatialFrame, distance: float = 1.5) -> dict[str, float]:
    """Return a point along the frame's full forward vector."""

    return _position(
        frame.origin.x + frame.forward.x * distance,
        frame.origin.y + frame.forward.y * distance,
        frame.origin.z + frame.forward.z * distance,
    )


def place_user_relative(
    frame: SpatialFrame,
    direction: Literal["front", "back", "left", "right", "above", "below"],
    distance: float = 1.5,
) -> dict[str, float]:
    """Place a point in a named gravity-aligned direction from the frame origin."""

    _require_distance(distance)
    offsets = {
        "front": (distance, 0.0, 0.0),
        "back": (-distance, 0.0, 0.0),
        "left": (0.0, -distance, 0.0),
        "right": (0.0, distance, 0.0),
        "above": (0.0, 0.0, distance),
        "below": (0.0, 0.0, -distance),
    }
    forward, right, up = offsets[direction]
    return offset(frame, frame.origin, forward=forward, right=right, up=up)


def place_object_relative(
    frame: SpatialFrame,
    *,
    anchor: Vector3,
    direction: Literal["front", "back", "left", "right", "above", "below"],
    distance: float = 0.3,
) -> dict[str, float]:
    """Place a point in a named gravity-aligned direction from an object anchor."""

    _require_distance(distance)
    offsets = {
        "front": (-distance, 0.0, 0.0),
        "back": (distance, 0.0, 0.0),
        "left": (0.0, -distance, 0.0),
        "right": (0.0, distance, 0.0),
        "above": (0.0, 0.0, distance),
        "below": (0.0, 0.0, -distance),
    }
    forward, right, up = offsets[direction]
    return offset(frame, anchor, forward=forward, right=right, up=up)


def displace_object(
    frame: SpatialFrame,
    *,
    position: Vector3,
    forward: float = 0.0,
    right: float = 0.0,
    up: float = 0.0,
) -> dict[str, float]:
    """Move a position by a signed frame-relative delta."""

    return offset(frame, position, forward=forward, right=right, up=up)


def move_relative_to(
    position: Vector3,
    reference: Vector3,
    *,
    direction: Literal["toward", "away"],
    distance: float = 0.5,
) -> dict[str, float]:
    """Move a position toward or away from a reference by an exact distance."""

    _require_distance(distance)
    dx = reference.x - position.x
    dy = reference.y - position.y
    dz = reference.z - position.z
    magnitude = math.sqrt(dx * dx + dy * dy + dz * dz)
    if magnitude < 1e-9:
        raise ValueError("position and reference coincide")
    scale = (1.0 if direction == "toward" else -1.0) * distance / magnitude
    return _position(
        position.x + dx * scale,
        position.y + dy * scale,
        position.z + dz * scale,
    )


def midpoint(first: Vector3, second: Vector3) -> dict[str, float]:
    """Return the midpoint between two world positions."""

    return _position(
        (first.x + second.x) / 2,
        (first.y + second.y) / 2,
        (first.z + second.z) / 2,
    )


def place_in_container(obj_id: str, container: Vector3) -> dict[str, float | str]:
    """Pair an object id with a container's center position."""

    return {"obj_id": obj_id, **_position(container.x, container.y, container.z)}
