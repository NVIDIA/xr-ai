# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample-facing NAT functions used by the existing XR scene agent loop."""

import math
import time
from typing import Annotated, Any, Literal

from nat.builder.function import Function
from nat.plugin_api import (
    Builder,
    FunctionGroup,
    FunctionGroupBaseConfig,
    FunctionGroupRef,
    register_function_group,
)
from pydantic import BaseModel, Field
from xr_ai_models import ToolDef


class _EmptyRequest(BaseModel):
    pass


class RenderSpatialToolsConfig(FunctionGroupBaseConfig, name="xr_render_spatial_tools"):
    """Compose tracking and spatial math into the demo's established tool vocabulary."""

    tracking: FunctionGroupRef = FunctionGroupRef("tracking")
    spatial_math: FunctionGroupRef = FunctionGroupRef("spatial_math")


@register_function_group(config_type=RenderSpatialToolsConfig)
async def render_spatial_tools(config: RenderSpatialToolsConfig, builder: Builder):
    tracking_group = await builder.get_function_group(config.tracking)
    tracking = await tracking_group.get_all_functions()
    math_group = await builder.get_function_group(config.spatial_math)
    spatial = await math_group.get_all_functions()
    get_frame = tracking[f"{tracking_group.instance_name}__get_user_frame"]

    async def frame_dict() -> dict[str, Any]:
        frame = await get_frame.ainvoke({})
        return frame.model_dump(mode="python")

    async def get_head_pose(request: _EmptyRequest) -> dict[str, Any]:
        del request
        frame = await frame_dict()
        forward = frame["forward"]
        return {
            "is_valid": True,
            "position": frame["origin"],
            "forward": forward,
            "right": frame["right"],
            "up": frame["up"],
            "yaw_deg": math.degrees(math.atan2(-forward["x"], -forward["z"])),
            "pitch_deg": math.degrees(math.asin(max(-1.0, min(1.0, forward["y"])))),
            "ts": time.time_ns() // 1_000_000,
        }

    async def position_ahead(
        distance: Annotated[float, Field(description="Distance along the user's gaze, in metres.")] = 1.5,
    ) -> dict[str, float]:
        result = await spatial[f"{math_group.instance_name}__compute_gaze_target"].ainvoke(
            {"user_frame": await frame_dict(), "distance_meters": distance}
        )
        return result.model_dump(mode="python")

    async def position_relative(
        forward: float = 0.0,
        right: float = 0.0,
        up: float = 0.0,
        origin_x: float | None = None,
        origin_y: float | None = None,
        origin_z: float | None = None,
    ) -> dict[str, float]:
        frame = await frame_dict()
        origin = frame["origin"]
        result = await spatial[f"{math_group.instance_name}__offset_position_in_user_frame"].ainvoke(
            {
                "user_frame": frame,
                "start_position": {
                    "x": origin["x"] if origin_x is None else origin_x,
                    "y": origin["y"] if origin_y is None else origin_y,
                    "z": origin["z"] if origin_z is None else origin_z,
                },
                "forward_meters": forward,
                "right_meters": right,
                "up_meters": up,
            }
        )
        return result.model_dump(mode="python")

    async def place_user_relative(
        direction: Literal["front", "back", "left", "right", "above", "below"],
        distance: float = 1.5,
    ) -> dict[str, Any]:
        if distance < 0:
            return {"error": "distance must be non-negative; flip the direction instead"}
        result = await spatial[f"{math_group.instance_name}__compute_user_relative_position"].ainvoke(
            {
                "user_frame": await frame_dict(),
                "direction_from_user": direction,
                "distance_meters": distance,
            }
        )
        return result.model_dump(mode="python")

    async def place_object_relative(
        origin_x: float,
        origin_y: float,
        origin_z: float,
        direction: Literal["front", "back", "left", "right", "above", "below", "next_to"],
        distance: float = 0.3,
    ) -> dict[str, Any]:
        if distance < 0:
            return {"error": "distance must be non-negative; flip the direction instead"}
        relation = {
            "front": "toward_user",
            "back": "away_from_user",
            "left": "left_of",
            "right": "right_of",
            "next_to": "right_of",
            "above": "above",
            "below": "below",
        }[direction]
        frame = (
            await frame_dict()
            if direction in {"front", "back", "left", "right", "next_to"}
            else {
                "origin": {"x": 0.0, "y": 0.0, "z": 0.0},
                "forward": {"x": 0.0, "y": 0.0, "z": -1.0},
                "right": {"x": 1.0, "y": 0.0, "z": 0.0},
                "up": {"x": 0.0, "y": 1.0, "z": 0.0},
            }
        )
        result = await spatial[f"{math_group.instance_name}__compute_position_relative_to_anchor"].ainvoke(
            {
                "user_frame": frame,
                "anchor_position": {"x": origin_x, "y": origin_y, "z": origin_z},
                "relation_to_anchor": relation,
                "distance_meters": distance,
            }
        )
        return result.model_dump(mode="python")

    async def displace_object(
        current_x: float,
        current_y: float,
        current_z: float,
        right: float = 0.0,
        up: float = 0.0,
        forward: float = 0.0,
    ) -> dict[str, float]:
        result = await spatial[f"{math_group.instance_name}__offset_position_in_user_frame"].ainvoke(
            {
                "user_frame": await frame_dict(),
                "start_position": {"x": current_x, "y": current_y, "z": current_z},
                "forward_meters": forward,
                "right_meters": right,
                "up_meters": up,
            }
        )
        return result.model_dump(mode="python")

    async def displace_objects(
        object_ids: list[str],
        current_xs: list[float],
        current_ys: list[float],
        current_zs: list[float],
        right: float = 0.0,
        up: float = 0.0,
        forward: float = 0.0,
    ) -> dict[str, Any]:
        if not (len(object_ids) == len(current_xs) == len(current_ys) == len(current_zs)):
            return {"error": ("object_ids / current_xs / current_ys / current_zs must all be the same length")}
        frame = await frame_dict()
        function = spatial[f"{math_group.instance_name}__offset_position_in_user_frame"]
        items = []
        for obj_id, x, y, z in zip(object_ids, current_xs, current_ys, current_zs, strict=True):
            result = await function.ainvoke(
                {
                    "user_frame": frame,
                    "start_position": {"x": x, "y": y, "z": z},
                    "forward_meters": forward,
                    "right_meters": right,
                    "up_meters": up,
                }
            )
            items.append({"obj_id": obj_id, **result.model_dump(mode="python")})
        return {"items": items}

    async def place_inside_by_id(
        movee_id: str,
        container_x: float,
        container_y: float,
        container_z: float,
    ) -> dict[str, Any]:
        return {
            "obj_id": movee_id,
            "x": round(container_x, 3),
            "y": round(container_y, 3),
            "z": round(container_z, 3),
        }

    async def between_anchors(
        a_x: float,
        a_y: float,
        a_z: float,
        b_x: float,
        b_y: float,
        b_z: float,
    ) -> dict[str, float]:
        result = await spatial[f"{math_group.instance_name}__compute_midpoint"].ainvoke(
            {
                "first_position": {"x": a_x, "y": a_y, "z": a_z},
                "second_position": {"x": b_x, "y": b_y, "z": b_z},
            }
        )
        return result.model_dump(mode="python")

    async def world_offset(
        origin_x: float,
        origin_y: float,
        origin_z: float,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
    ) -> dict[str, float]:
        return {"x": origin_x + dx, "y": origin_y + dy, "z": origin_z + dz}

    async def along_direction(
        origin_x: float,
        origin_y: float,
        origin_z: float,
        target_x: float,
        target_y: float,
        target_z: float,
        distance: float = 0.5,
    ) -> dict[str, float]:
        result = await spatial[f"{math_group.instance_name}__compute_position_toward_or_away_from_reference"].ainvoke(
            {
                "start_position": {"x": origin_x, "y": origin_y, "z": origin_z},
                "reference_position": {"x": target_x, "y": target_y, "z": target_z},
                "movement_direction": "toward" if distance >= 0 else "away",
                "distance_meters": abs(distance),
            }
        )
        return result.model_dump(mode="python")

    async def scale_value(current: float, factor: float) -> dict[str, float]:
        return {"value": round(current * factor, 3)}

    group = FunctionGroup(config=config)
    group.add_function(
        "get_head_pose",
        get_head_pose,
        description="Return the current world-space head position and forward, right, and up axes.",
    )
    group.add_function(
        "position_ahead",
        position_ahead,
        description="Compute a world position along the user's gaze for 'in front of me' requests.",
    )
    group.add_function(
        "position_relative",
        position_relative,
        description=(
            "Apply signed forward, right, and up user-frame offsets. Omit origin coordinates "
            "to start at the user, or pass an object's current position to move it."
        ),
    )
    group.add_function(
        "place_user_relative",
        place_user_relative,
        description=(
            "Compute a position in one named direction from the user. Distance is non-negative; "
            "choose front, back, left, right, above, or below to set the direction."
        ),
    )
    group.add_function(
        "place_object_relative",
        place_object_relative,
        description="Compute a position in one named direction from an existing object's world position.",
    )
    group.add_function(
        "displace_object",
        displace_object,
        description="Shift one existing object by signed right, up, and forward user-frame offsets.",
    )
    group.add_function(
        "displace_objects",
        displace_objects,
        description="Apply one signed user-frame offset to parallel lists of existing objects.",
    )
    group.add_function(
        "place_inside_by_id",
        place_inside_by_id,
        description="Return the container position with the ID of the object that should move there.",
    )
    group.add_function(
        "between_anchors",
        between_anchors,
        description="Compute the world-space midpoint between exactly two anchor positions.",
    )
    group.add_function(
        "world_offset",
        world_offset,
        description="Apply signed world-axis dx, dy, and dz offsets to an origin position.",
    )
    group.add_function(
        "along_direction",
        along_direction,
        description="Move an origin toward a target by positive distance or away by negative distance.",
    )
    group.add_function(
        "scale_value",
        scale_value,
        description="Multiply a current numeric size by a scale factor.",
    )
    yield group


class NativeToolbox:
    """Present selected NAT functions to the existing model-service tool loop."""

    def __init__(self, functions: dict[str, Function]) -> None:
        self._functions: dict[str, Function] = {}
        for function in functions.values():
            short_name = function.instance_name.rsplit("__", 1)[-1]
            if short_name in self._functions:
                raise ValueError(f"duplicate native tool name: {short_name}")
            self._functions[short_name] = function

    def definitions(self, *, exclude: set[str] | frozenset[str] = frozenset()) -> list[ToolDef]:
        return [
            ToolDef(
                name=name,
                description=(function.description or "").strip(),
                parameters=function.input_schema.model_json_schema(),
            )
            for name, function in self._functions.items()
            if name not in exclude
        ]

    async def invoke(self, name: str, arguments: dict[str, Any]) -> Any:
        result = await self._functions[name].ainvoke(arguments)
        return _plain(result)


def _plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


__all__ = ["NativeToolbox", "RenderSpatialToolsConfig"]
