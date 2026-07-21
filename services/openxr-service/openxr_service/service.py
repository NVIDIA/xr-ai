# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed OpenXR RPC surface independent of the hardware implementation."""

import asyncio
from typing import Any, Protocol

from pydantic import ValidationError
from xr_ai_nat.functions._rpc import RPCError
from xr_ai_nat.functions.xr_tracking._client import HeadPoseRequest, OpenXRHealthRequest


class PoseSource(Protocol):
    def get_pose(self) -> dict[str, Any]:
        pass

    def health(self) -> dict[str, Any]:
        pass


class OpenXRService:
    """Validate service calls and delegate hardware access to one pose source."""

    def __init__(self, source: PoseSource) -> None:
        self._source = source

    async def dispatch(self, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if operation == "get_head_pose":
            self._validate(HeadPoseRequest, arguments)
            return await asyncio.to_thread(self._source.get_pose)
        if operation == "get_health":
            self._validate(OpenXRHealthRequest, arguments)
            return self._source.health()
        raise RPCError(f"unknown operation: {operation}", code="unknown_operation")

    @staticmethod
    def _validate(request_type: type[HeadPoseRequest] | type[OpenXRHealthRequest], arguments: dict[str, Any]) -> None:
        try:
            request_type.model_validate(arguments)
        except ValidationError as exc:
            raise RPCError("unexpected OpenXR service arguments", code="invalid_request") from exc
