# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport-neutral values used by spatial-math functions."""

from __future__ import annotations

from pydantic import BaseModel


class _StructuredValue(BaseModel):
    def __str__(self) -> str:
        # Text-only agent bridges call str(); JSON keeps the value reusable as a typed tool input.
        return self.model_dump_json()


class Vector3(_StructuredValue):
    """A position or direction in three-dimensional space."""

    x: float
    y: float
    z: float


class SpatialFrame(_StructuredValue):
    """An origin and orthogonal directions supplied by a tracking capability."""

    origin: Vector3
    forward: Vector3
    right: Vector3
    up: Vector3


class PositionResult(Vector3):
    """A resolved world-space position."""


class ObjectPositionResult(PositionResult):
    """A resolved position paired with the object that should move."""

    obj_id: str


__all__ = ["ObjectPositionResult", "PositionResult", "SpatialFrame", "Vector3"]
