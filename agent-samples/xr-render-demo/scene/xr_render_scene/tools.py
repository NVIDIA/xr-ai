# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native tools for the sample-local XR scene capability."""

from xr_ai_tools import Tool

from .client import SceneClient
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


class SceneTools:
    """Own one typed scene client and its Relay-managed tools."""

    def __init__(self, endpoint: str, *, timeout_s: float = 10.0) -> None:
        self.client = SceneClient(endpoint, timeout_s=timeout_s)
        self.get_scene_state = Tool(
            "get_scene_state",
            "Return every current XR object with its ID, type, world position, color, and size.",
            EmptyRequest,
            SceneState,
            self.client.get_scene_state,
        )
        self.update_primitive = Tool(
            "update_primitive",
            "Partially update an existing XR object by ID. Omitted fields remain unchanged.",
            UpdatePrimitiveRequest,
            MutationResult,
            self.client.update_primitive,
        )
        self.add_primitive = Tool(
            "add_primitive",
            "Create a sphere or box at a world position and return its new object ID. Position and size use metres.",
            AddPrimitiveRequest,
            AddPrimitiveResult,
            self.client.add_primitive,
        )
        self.remove_primitive = Tool(
            "remove_primitive",
            "Permanently remove one XR scene object by ID.",
            RemovePrimitiveRequest,
            MutationResult,
            self.client.remove_primitive,
        )
        self.start_xr = Tool(
            "start_xr",
            "Start the sample's LOVR OpenXR renderer if needed.",
            EmptyRequest,
            StartXRResult,
            self.client.start_xr,
        )
        self.get_health = Tool(
            "get_health",
            "Return LOVR lifecycle and scene-delivery status.",
            EmptyRequest,
            SceneHealth,
            self.client.get_health,
        )
        self.tools = (
            self.get_scene_state,
            self.update_primitive,
            self.add_primitive,
            self.remove_primitive,
            self.start_xr,
            self.get_health,
        )

    async def close(self) -> None:
        await self.client.close()


__all__ = ["SceneTools"]
