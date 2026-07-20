# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NAT functions backed by the long-running OpenXR service."""

from nat.plugin_api import Builder, FunctionGroup, FunctionGroupBaseConfig, register_function_group
from pydantic import Field

from ..spatial_math import SpatialFrame
from ._client import OpenXRClient
from .schemas import EmptyRequest


class XRTrackingFunctionsConfig(FunctionGroupBaseConfig, name="xr_tracking"):
    """Configure current-user tracking functions."""

    endpoint: str = Field(description="Private OpenXR service endpoint.")
    timeout_s: float = Field(default=10.0, gt=0.0)


@register_function_group(config_type=XRTrackingFunctionsConfig)
async def xr_tracking_functions(config: XRTrackingFunctionsConfig, _builder: Builder):
    client = OpenXRClient(config.endpoint, timeout_s=config.timeout_s)
    group = FunctionGroup(config=config)

    async def get_user_frame(request: EmptyRequest) -> SpatialFrame:
        del request
        pose = await client.get_head_pose()
        if not pose.is_valid:
            raise RuntimeError(pose.error or "XR tracking is unavailable")
        return pose.user_frame()

    group.add_function(
        "get_user_frame",
        get_user_frame,
        description=(
            "Get the user's current world-space origin and forward, right, and up axes. "
            "Pass this complete frame to spatial-math functions for user-relative coordinates."
        ),
    )

    try:
        yield group
    finally:
        await client.close()


__all__ = ["XRTrackingFunctionsConfig"]
