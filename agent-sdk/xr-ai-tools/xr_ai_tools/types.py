# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared request and spatial value types for native XR tools."""

from pydantic import BaseModel, ConfigDict


class StrictRequest(BaseModel):
    """Base model for tool requests that rejects undeclared arguments."""

    model_config = ConfigDict(extra="forbid")


class EmptyRequest(StrictRequest):
    """Request model for a tool that accepts no arguments."""

    pass


class Vector3(BaseModel):
    """A three-dimensional Cartesian vector or position."""

    x: float
    """Component on the x-axis."""

    y: float
    """Component on the y-axis."""

    z: float
    """Component on the z-axis."""


class SpatialFrame(BaseModel):
    """An origin and orthogonal orientation axes in world space."""

    origin: Vector3
    """World-space origin of the frame."""

    forward: Vector3
    """Forward direction of the frame."""

    right: Vector3
    """Right direction of the frame."""

    up: Vector3
    """Up direction of the frame."""


__all__ = ["EmptyRequest", "SpatialFrame", "StrictRequest", "Vector3"]
