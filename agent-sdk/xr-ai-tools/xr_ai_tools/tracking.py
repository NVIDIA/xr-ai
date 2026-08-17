# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native tools backed by the typed OpenXR tracking service."""

from pydantic import BaseModel

from .rpc import RPCClient
from .tools import Tool
from .types import EmptyRequest, SpatialFrame, Vector3


class HeadPose(BaseModel):
    """Head pose and orientation reported by the OpenXR service."""

    is_valid: bool
    """Whether the remaining pose fields describe a valid tracked pose."""

    position: Vector3
    """World-space head position."""

    forward: Vector3
    """World-space forward direction."""

    right: Vector3
    """World-space right direction."""

    up: Vector3
    """World-space up direction."""

    yaw_deg: float
    """Head yaw in degrees."""

    pitch_deg: float
    """Head pitch in degrees."""

    ts: int
    """Service-provided pose timestamp."""

    error: str | None = None
    """Tracking failure detail when the pose is invalid."""


class OpenXRHealth(BaseModel):
    """OpenXR service health and session state."""

    status: str = "ok"
    """Service health status label."""

    session_open: bool
    """Whether an OpenXR session is currently open."""

    open_attempts: int
    """Number of attempts made to open an OpenXR session."""

    last_open_error: str | None = None
    """Most recent session-open failure, if any."""


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
        """Return detailed OpenXR service and session health."""

        return OpenXRHealth.model_validate(
            await self._rpc.call("get_health", {}, timeout_s=2.0)
        )

    async def health(self) -> bool:
        """Return whether the service is reachable with an open XR session."""

        try:
            return (await self.get_health()).session_open
        except Exception:
            return False

    async def close(self) -> None:
        """Close the underlying service connection."""

        await self._rpc.close()


__all__ = ["HeadPose", "OpenXRHealth", "TrackingTools"]
