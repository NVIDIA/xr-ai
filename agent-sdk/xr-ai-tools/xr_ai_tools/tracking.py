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


class HandPose(BaseModel):
    """Tracked joint positions of one hand."""

    is_tracked: bool
    """Whether the hand is currently tracked."""

    joints: dict[str, Vector3]
    """World-space joint positions keyed by OpenXR hand joint name.

    Names follow ``XR_EXT_hand_tracking`` in lower case, such as ``palm``,
    ``wrist``, ``thumb_tip``, and ``index_proximal``. Only joints whose
    position is currently valid are present.
    """


class HandPoses(BaseModel):
    """Both hands as reported by the OpenXR service."""

    left: HandPose
    """The user's left hand."""

    right: HandPose
    """The user's right hand."""

    ts: int
    """Service-provided timestamp in milliseconds."""

    error: str | None = None
    """Why hands are unavailable, when the service cannot track them."""


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
    """Own the OpenXR-service client and the user-frame and hand tools."""

    def __init__(self, endpoint: str, *, timeout_s: float = 10.0) -> None:
        self._rpc = RPCClient(endpoint, timeout_s=timeout_s)
        self.get_user_frame = Tool(
            "get_user_frame",
            "Get the user's current world-space origin and forward, right, and up axes.",
            EmptyRequest,
            SpatialFrame,
            self._get_user_frame,
        )
        self.get_hand_poses = Tool(
            "get_hand_poses",
            "Get world-space positions of the user's tracked hand joints, such as the palm, wrist, and fingertips.",
            EmptyRequest,
            HandPoses,
            self._get_hand_poses,
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

    async def _get_hand_poses(self, request: EmptyRequest) -> HandPoses:
        hands = HandPoses.model_validate(
            await self._rpc.call("get_hand_poses", request.model_dump())
        )
        if hands.error is not None:
            raise RuntimeError(hands.error)
        return hands

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


__all__ = ["HandPose", "HandPoses", "HeadPose", "OpenXRHealth", "TrackingTools"]
