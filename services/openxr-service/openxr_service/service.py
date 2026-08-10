# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed OpenXR RPC surface independent of the hardware implementation."""

import asyncio
from typing import Any, Protocol

from loguru import logger
from pydantic import ValidationError
from xr_ai_tools.rpc import RPCError
from xr_ai_tools.types import EmptyRequest


class PoseSource(Protocol):
    def get_pose(self) -> dict[str, Any]:
        pass

    def health(self) -> dict[str, Any]:
        pass


class OpenXRService:
    """Validate service calls and delegate hardware access to one pose source."""

    def __init__(self, source: PoseSource, *, allow_sim_pose: bool = False) -> None:
        self._source = source
        self._allow_sim_pose = allow_sim_pose
        self._sim_pose: dict[str, Any] | None = None
        if allow_sim_pose:
            logger.warning("sim-pose test hook enabled: any RPC peer can override head tracking")

    async def dispatch(self, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if operation == "get_head_pose":
            self._validate(arguments)
            return await asyncio.to_thread(self._source.get_pose)
        if operation == "set_sim_pose" and self._allow_sim_pose:
            try:
                pose = HeadPose.model_validate({**arguments, "is_valid": True, "error": None})
            except ValidationError as exc:
                raise RPCError("invalid sim pose", code="invalid_request") from exc
            self._sim_pose = pose.model_dump()
            logger.warning("sim pose set; real head tracking is overridden until cleared")
            return {"ok": True}
        if operation == "clear_sim_pose" and self._allow_sim_pose:
            self._sim_pose = None
            return {"ok": True}
        if operation == "get_health":
            self._validate(arguments)
            return self._source.health()
        raise RPCError(f"unknown operation: {operation}", code="unknown_operation")

    @staticmethod
    def _validate(arguments: dict[str, Any]) -> None:
        try:
            EmptyRequest.model_validate(arguments)
        except ValidationError as exc:
            raise RPCError("unexpected OpenXR service arguments", code="invalid_request") from exc
