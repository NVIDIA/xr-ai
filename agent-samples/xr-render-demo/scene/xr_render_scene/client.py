# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed client for the sample-local scene process."""

from xr_ai_nat.functions._rpc import RPCClient

from .schemas import (
    AddPrimitiveRequest,
    AddPrimitiveResult,
    EmptyRequest,
    MutationResult,
    RemovePrimitiveRequest,
    SceneHealth,
    SceneState,
    StartXRResult,
    UpdatePrimitiveRequest,
)


class SceneClient:
    def __init__(self, endpoint: str, *, timeout_s: float = 10.0) -> None:
        self._rpc = RPCClient(endpoint, timeout_s=timeout_s)

    async def start_xr(self, request: EmptyRequest) -> StartXRResult:
        return StartXRResult.model_validate(
            await self._rpc.call("start_xr", request.model_dump())
        )

    async def add_primitive(self, request: AddPrimitiveRequest) -> AddPrimitiveResult:
        return AddPrimitiveResult.model_validate(
            await self._rpc.call("add_primitive", request.model_dump())
        )

    async def update_primitive(self, request: UpdatePrimitiveRequest) -> MutationResult:
        return MutationResult.model_validate(
            await self._rpc.call(
                "update_primitive",
                request.model_dump(exclude_none=True),
            )
        )

    async def remove_primitive(self, request: RemovePrimitiveRequest) -> MutationResult:
        return MutationResult.model_validate(
            await self._rpc.call("remove_primitive", request.model_dump())
        )

    async def get_scene_state(self, request: EmptyRequest) -> SceneState:
        return SceneState.model_validate(
            await self._rpc.call("get_scene_state", request.model_dump())
        )

    async def get_health(self, request: EmptyRequest) -> SceneHealth:
        return SceneHealth.model_validate(
            await self._rpc.call("get_health", request.model_dump(), timeout_s=2.0)
        )

    async def close(self) -> None:
        await self._rpc.close()


__all__ = ["SceneClient"]
