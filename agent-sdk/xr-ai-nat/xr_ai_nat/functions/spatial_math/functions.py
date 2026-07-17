# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed NAT functions for deterministic XR spatial calculations."""

import json
from typing import Annotated, Literal

from nat.plugin_api import Builder, FunctionGroup, FunctionGroupBaseConfig, register_function_group
from pydantic import BeforeValidator, Field

from . import _math
from .schemas import ObjectPositionResult, PositionResult, SpatialFrame, Vector3


def _decode_json(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


_Frame = Annotated[
    SpatialFrame,
    BeforeValidator(_decode_json),
    Field(description="Coordinate frame returned by an XR tracking function."),
]
_Position = Annotated[
    Vector3,
    BeforeValidator(_decode_json),
    Field(description="A complete world position represented as one {x, y, z} object."),
]
_Distance = Annotated[float, Field(ge=0, description="Non-negative distance in metres.")]


class SpatialMathFunctionsConfig(FunctionGroupBaseConfig, name="xr_spatial_math"):
    """Configure pure spatial functions over caller-supplied coordinates."""


@register_function_group(config_type=SpatialMathFunctionsConfig)
async def spatial_math_functions(config: SpatialMathFunctionsConfig, _builder: Builder):
    group = FunctionGroup(config=config)

    async def position_in_gaze(frame: _Frame, distance: _Distance = 1.5) -> PositionResult:
        return PositionResult.model_validate(_math.position_in_gaze(frame, distance))

    group.add_function(
        "position_in_gaze",
        position_in_gaze,
        description="Return a position along the user's full 3D gaze direction.",
    )

    async def place_user_relative(
        frame: _Frame,
        direction: Literal["front", "back", "left", "right", "above", "below"],
        distance: _Distance = 1.5,
    ) -> PositionResult:
        return PositionResult.model_validate(
            _math.place_user_relative(frame, direction=direction, distance=distance)
        )

    group.add_function(
        "place_user_relative",
        place_user_relative,
        description=(
            "Return a gravity-aligned position in a named direction from the user. "
            "Use only when the user is the spatial anchor."
        ),
    )

    async def place_object_relative(
        frame: _Frame,
        anchor: _Position,
        direction: Annotated[
            Literal["front", "back", "left", "right", "above", "below"],
            Field(description="Direction from the object anchor; front points toward the user."),
        ],
        distance: _Distance = 0.3,
    ) -> PositionResult:
        return PositionResult.model_validate(
            _math.place_object_relative(
                frame,
                anchor=anchor,
                direction=direction,
                distance=distance,
            )
        )

    group.add_function(
        "place_object_relative",
        place_object_relative,
        description="Return a gravity-aligned position relative to a scene-object anchor.",
    )

    async def displace_object(
        frame: _Frame,
        position: _Position,
        forward: Annotated[float, Field(description="Signed user-forward offset in metres.")] = 0.0,
        right: Annotated[float, Field(description="Signed user-right offset in metres.")] = 0.0,
        up: Annotated[float, Field(description="Signed world-up offset in metres.")] = 0.0,
    ) -> PositionResult:
        return PositionResult.model_validate(
            _math.displace_object(
                frame,
                position=position,
                forward=forward,
                right=right,
                up=up,
            )
        )

    group.add_function(
        "displace_object",
        displace_object,
        description="Offset an existing world position in the user's gravity-aligned frame.",
    )

    async def move_relative_to(
        position: _Position,
        reference: _Position,
        direction: Literal["toward", "away"],
        distance: _Distance = 0.5,
    ) -> PositionResult:
        return PositionResult.model_validate(
            _math.move_relative_to(
                position,
                reference,
                direction=direction,
                distance=distance,
            )
        )

    group.add_function(
        "move_relative_to",
        move_relative_to,
        description="Move one position toward or away from a reference by an exact distance.",
    )

    async def midpoint(first: _Position, second: _Position) -> PositionResult:
        return PositionResult.model_validate(_math.midpoint(first, second))

    group.add_function(
        "midpoint",
        midpoint,
        description="Return the world-space midpoint between two positions.",
    )

    async def place_in_container(
        obj_id: Annotated[str, Field(description="ID of the object being placed inside.")],
        container: _Position,
    ) -> ObjectPositionResult:
        return ObjectPositionResult.model_validate(_math.place_in_container(obj_id, container))

    group.add_function(
        "place_in_container",
        place_in_container,
        description="Pair an object ID with the center position of its container.",
    )

    yield group


__all__ = ["SpatialMathFunctionsConfig"]
