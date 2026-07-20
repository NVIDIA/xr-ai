# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed OpenXR RPC surface independent of the hardware implementation."""

import asyncio
from typing import Protocol

from xr_ai_nat.functions._rpc import RPCError
from xr_ai_nat.functions.xr_tracking import HeadPose, OpenXRHealth
from xr_ai_nat.functions.xr_tracking.schemas import EmptyRequest


class PoseSource(Protocol):
    def get_pose(self) -> HeadPose: ...

    def health(self) -> OpenXRHealth: ...


class OpenXRService:
    """Validate service calls and delegate hardware access to one pose source."""

    def __init__(self, source: PoseSource) -> None:
        self._source = source

    async def dispatch(self, operation: str, arguments: dict) -> dict:
        EmptyRequest.model_validate(arguments)
        if operation == "get_head_pose":
            pose = await asyncio.to_thread(self._source.get_pose)
            return pose.model_dump(mode="python")
        if operation == "get_health":
            return self._source.health().model_dump(mode="python")
        raise RPCError(f"unknown operation: {operation}", code="unknown_operation")
