# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public value models shared by XR function capabilities."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ServiceResult(BaseModel):
    """A typed capability result that remains structured in text-only tool messages."""

    model_config = ConfigDict(extra="allow")

    def __str__(self) -> str:
        # LangChain coerces non-string tool results with str(); JSON preserves
        # the schema for the next model call while keeping the native value typed.
        return self.model_dump_json()


class Vector3(ServiceResult):
    x: float
    y: float
    z: float


class SpatialFrame(ServiceResult):
    """A transport-neutral coordinate frame for spatial calculations."""

    origin: Vector3
    forward: Vector3
    right: Vector3
    up: Vector3


class Color(ServiceResult):
    r: float
    g: float
    b: float


__all__ = ["Color", "ServiceResult", "SpatialFrame", "Vector3"]
