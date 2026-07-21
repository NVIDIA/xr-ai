# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert native OpenXR poses into plain service-wire dictionaries."""

import math
import time
from typing import Any


def rotate_vector(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Rotate a vector by a unit quaternion in x, y, z, w order."""

    qx, qy, qz, qw = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def unavailable_pose(error: str) -> dict[str, Any]:
    return {
        "is_valid": False,
        "position": {"x": 0.0, "y": 1.6, "z": 0.0},
        "forward": {"x": 0.0, "y": 0.0, "z": -1.0},
        "right": {"x": 1.0, "y": 0.0, "z": 0.0},
        "up": {"x": 0.0, "y": 1.0, "z": 0.0},
        "yaw_deg": 0.0,
        "pitch_deg": 0.0,
        "ts": int(time.time() * 1000),
        "error": f"session_not_ready: {error}",
    }


def from_native_pose(data: Any) -> dict[str, Any]:
    pose = data.pose
    quaternion = (
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    )
    qx, qy, qz, qw = quaternion
    forward = rotate_vector(quaternion, (0.0, 0.0, -1.0))
    right = rotate_vector(quaternion, (1.0, 0.0, 0.0))
    up = rotate_vector(quaternion, (0.0, 1.0, 0.0))
    yaw = math.degrees(
        math.atan2(
            2.0 * (qw * qy + qx * qz),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
    )
    pitch = math.degrees(
        math.asin(max(-1.0, min(1.0, 2.0 * (qw * qx - qy * qz))))
    )
    return {
        "is_valid": bool(data.is_valid),
        "position": {
            "x": round(float(pose.position.x), 3),
            "y": round(float(pose.position.y), 3),
            "z": round(float(pose.position.z), 3),
        },
        "forward": dict(zip(("x", "y", "z"), (round(value, 3) for value in forward), strict=True)),
        "right": dict(zip(("x", "y", "z"), (round(value, 3) for value in right), strict=True)),
        "up": dict(zip(("x", "y", "z"), (round(value, 3) for value in up), strict=True)),
        "yaw_deg": round(yaw, 1),
        "pitch_deg": round(pitch, 1),
        "ts": int(time.time() * 1000),
    }
