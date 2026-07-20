# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed OpenXR-service contracts and client."""

from typing import Literal

from pydantic import BaseModel

from .._models import _StrictRequest
from .._rpc import RPCClient
from ..spatial_math import Vector3


class HeadPoseRequest(_StrictRequest):
    pass


class OpenXRHealthRequest(_StrictRequest):
    pass


class HeadPose(BaseModel):
    """One head-pose sample expressed as world-space basis vectors."""

    is_valid: bool
    position: Vector3
    forward: Vector3
    right: Vector3
    up: Vector3
    yaw_deg: float
    pitch_deg: float
    ts: int
    error: str | None = None


class OpenXRHealth(BaseModel):
    """Operational state of the headless OpenXR session."""

    status: Literal["ok"] = "ok"
    session_open: bool
    open_attempts: int
    last_open_error: str | None = None


class OpenXRClient:
    """Translate typed tracking operations to the private service protocol."""

    def __init__(self, endpoint: str, *, timeout_s: float = 10.0) -> None:
        self._rpc = RPCClient(endpoint, timeout_s=timeout_s)

    async def get_head_pose(self, request: HeadPoseRequest) -> HeadPose:
        return HeadPose.model_validate(
            await self._rpc.call("get_head_pose", request.model_dump())
        )

    async def get_health(self, request: OpenXRHealthRequest) -> OpenXRHealth:
        return OpenXRHealth.model_validate(
            await self._rpc.call("get_health", request.model_dump(), timeout_s=2.0)
        )

    async def close(self) -> None:
        await self._rpc.close()
