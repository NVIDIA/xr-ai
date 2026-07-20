# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private typed client for the OpenXR service."""

from .._rpc import RPCClient, RPCError
from .schemas import EmptyRequest, HeadPose, OpenXRHealth


class OpenXRClient:
    """Translate typed tracking operations to the private service protocol."""

    def __init__(self, endpoint: str, *, timeout_s: float = 10.0) -> None:
        self._rpc = RPCClient(endpoint, timeout_s=timeout_s)

    async def get_head_pose(self, request: EmptyRequest | None = None) -> HeadPose:
        arguments = (request or EmptyRequest()).model_dump()
        return HeadPose.model_validate(await self._rpc.call("get_head_pose", arguments))

    async def get_health(self, request: EmptyRequest | None = None) -> OpenXRHealth:
        arguments = (request or EmptyRequest()).model_dump()
        return OpenXRHealth.model_validate(
            await self._rpc.call("get_health", arguments, timeout_s=2.0)
        )

    async def is_available(self) -> bool:
        try:
            await self.get_health()
        except RPCError:
            return False
        return True

    async def close(self) -> None:
        await self._rpc.close()
