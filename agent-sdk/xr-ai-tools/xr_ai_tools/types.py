# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared request and spatial value types for native XR tools."""

from pydantic import BaseModel, ConfigDict


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyRequest(StrictRequest):
    pass


class Vector3(BaseModel):
    x: float
    y: float
    z: float


class SpatialFrame(BaseModel):
    origin: Vector3
    forward: Vector3
    right: Vector3
    up: Vector3


__all__ = ["EmptyRequest", "SpatialFrame", "StrictRequest", "Vector3"]
