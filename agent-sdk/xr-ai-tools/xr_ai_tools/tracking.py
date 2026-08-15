# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native tools backed by the typed OpenXR tracking service."""

from pydantic import BaseModel

from .rpc import RPCClient
from .tools import Tool
from .types import EmptyRequest, SpatialFrame, Vector3


class HeadPose(BaseModel):
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
    status: str = "ok"
    session_open: bool
    open_attempts: int
    last_open_error: str | None = None


class TrackingTools:
    """Own the OpenXR-service client and current-user-frame tool."""

    def __init__(self, endpoint: str, *, timeout_s: float = 10.0) -> None:
        self._rpc = RPCClient(endpoint, timeout_s=timeout_s)
        self.get_user_frame = Tool(
            "get_user_frame",
            "Get the user's current world-space origin and forward, right, and up axes.",
            EmptyRequest,
            SpatialFrame,
            self._get_user_frame,
        )

    async def _get_user_frame(self, request: EmptyRequest) -> SpatialFrame:
        pose = HeadPose.model_validate(
            await self._rpc.call("get_head_pose", request.model_dump())
        )
        if not pose.is_valid:
            raise RuntimeError(pose.error or "XR tracking is unavailable")
        return SpatialFrame(
            origin=pose.position,
            forward=pose.forward,
            right=pose.right,
            up=pose.up,
        )

    async def get_health(self) -> OpenXRHealth:
        return OpenXRHealth.model_validate(
            await self._rpc.call("get_health", {}, timeout_s=2.0)
        )

    async def health(self) -> bool:
        try:
            return (await self.get_health()).session_open
        except Exception:
            return False

    async def close(self) -> None:
        await self._rpc.close()


__all__ = ["HeadPose", "OpenXRHealth", "TrackingTools"]
