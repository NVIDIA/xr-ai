# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample-local NAT functions for the XR render scene."""

from nat.plugin_api import Builder, FunctionGroup, FunctionGroupBaseConfig, register_function_group
from pydantic import Field

from .client import SceneClient


class _SceneFunctionsConfig(FunctionGroupBaseConfig):
    endpoint: str = Field(description="XR render scene service endpoint.")
    timeout_s: float = Field(default=10.0, gt=0.0)


class SceneStateFunctionsConfig(_SceneFunctionsConfig, name="xr_render_scene_state"):
    """Configure read-only scene inspection."""


class SceneUpdateFunctionsConfig(_SceneFunctionsConfig, name="xr_render_scene_updates"):
    """Configure updates to existing objects."""


class SceneObjectFunctionsConfig(_SceneFunctionsConfig, name="xr_render_scene_objects"):
    """Configure scene object creation and removal."""


class SceneControlFunctionsConfig(_SceneFunctionsConfig, name="xr_render_scene_control"):
    """Configure LOVR lifecycle operations."""


def _client(config: _SceneFunctionsConfig) -> SceneClient:
    return SceneClient(config.endpoint, timeout_s=config.timeout_s)


@register_function_group(config_type=SceneStateFunctionsConfig)
async def scene_state_functions(config: SceneStateFunctionsConfig, _builder: Builder):
    client = _client(config)
    group = FunctionGroup(config=config)
    group.add_function(
        "get_scene_state",
        client.get_scene_state,
        description=(
            "Return every current XR object with its ID, type, world position, "
            "color, and size."
        ),
    )
    try:
        yield group
    finally:
        await client.close()


@register_function_group(config_type=SceneUpdateFunctionsConfig)
async def scene_update_functions(config: SceneUpdateFunctionsConfig, _builder: Builder):
    client = _client(config)
    group = FunctionGroup(config=config)
    group.add_function(
        "update_primitive",
        client.update_primitive,
        description=(
            "Partially update an existing XR object by ID. Omitted fields remain unchanged."
        ),
    )
    try:
        yield group
    finally:
        await client.close()


@register_function_group(config_type=SceneObjectFunctionsConfig)
async def scene_object_functions(config: SceneObjectFunctionsConfig, _builder: Builder):
    client = _client(config)
    group = FunctionGroup(config=config)
    group.add_function(
        "add_primitive",
        client.add_primitive,
        description=(
            "Create a sphere or box at a world position and return its new object ID. "
            "Position and size use metres."
        ),
    )
    group.add_function(
        "remove_primitive",
        client.remove_primitive,
        description="Permanently remove one XR scene object by ID.",
    )
    try:
        yield group
    finally:
        await client.close()


@register_function_group(config_type=SceneControlFunctionsConfig)
async def scene_control_functions(config: SceneControlFunctionsConfig, _builder: Builder):
    client = _client(config)
    group = FunctionGroup(config=config)
    group.add_function(
        "start_xr",
        client.start_xr,
        description="Start the sample's LOVR OpenXR renderer if needed.",
    )
    group.add_function(
        "get_health",
        client.get_health,
        description="Return LOVR lifecycle and scene-delivery status.",
    )
    try:
        yield group
    finally:
        await client.close()


__all__ = [
    "SceneControlFunctionsConfig",
    "SceneObjectFunctionsConfig",
    "SceneStateFunctionsConfig",
    "SceneUpdateFunctionsConfig",
]
