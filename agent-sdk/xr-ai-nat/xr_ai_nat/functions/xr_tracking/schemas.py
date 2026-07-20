# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed requests and results for XR tracking."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..spatial_math import SpatialFrame, Vector3


class EmptyRequest(BaseModel):
    """A typed no-argument NAT invocation."""

    model_config = ConfigDict(extra="forbid")


class HeadPose(BaseModel):
    """One head-pose sample expressed as world-space basis vectors."""

    is_valid: bool
    position: Vector3
    forward: Vector3
    right: Vector3
    up: Vector3
    yaw_deg: float
    pitch_deg: float
    timestamp_ms: int
    error: str | None = None

    def user_frame(self) -> SpatialFrame:
        return SpatialFrame(
            origin=self.position,
            forward=self.forward,
            right=self.right,
            up=self.up,
        )


class OpenXRHealth(BaseModel):
    """Operational state of the headless OpenXR session."""

    status: Literal["ok"] = "ok"
    session_open: bool
    open_attempts: int
    last_open_error: str | None = None


__all__ = ["EmptyRequest", "HeadPose", "OpenXRHealth"]
