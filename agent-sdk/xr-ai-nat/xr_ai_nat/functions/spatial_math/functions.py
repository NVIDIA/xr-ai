# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-facing NAT functions for pure, framework-neutral coordinate calculations."""

import json
from typing import Annotated, Literal

from nat.plugin_api import Builder, FunctionGroup, FunctionGroupBaseConfig, register_function_group
from pydantic import BeforeValidator, Field

from . import _math as spatial_math
from .schemas import SpatialFrame, Vector3


def _decode_json(value: object) -> object:
    # LangChain may present a typed tool result as text before the model reuses it.
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


_UserFrameArgument = Annotated[
    SpatialFrame,
    BeforeValidator(_decode_json),
    Field(
        description=("Current user coordinate frame returned by an XR tracking function, copied as one JSON object.")
    ),
]
_WorldPositionArgument = Annotated[
    Vector3,
    BeforeValidator(_decode_json),
    Field(
        description=(
            "Complete world position copied as one {x, y, z} object from scene state or a previous spatial-math result."
        )
    ),
]


class SpatialMathFunctionsConfig(FunctionGroupBaseConfig, name="xr_spatial_math"):
    """Configure pure coordinate calculations over caller-supplied positions and frames."""


@register_function_group(config_type=SpatialMathFunctionsConfig)
async def spatial_math_functions(config: SpatialMathFunctionsConfig, _builder: Builder):
    """Build calculations that return coordinates but never mutate scene objects."""

    group = FunctionGroup(config=config)

    async def compute_gaze_target(
        user_frame: _UserFrameArgument,
        distance_meters: Annotated[
            float,
            Field(ge=0, description="Non-negative distance along the full 3D gaze ray, in metres."),
        ] = 1.5,
    ) -> Vector3:
        return Vector3.model_validate(spatial_math.compute_gaze_target(user_frame, distance_meters))

    group.add_function(
        "compute_gaze_target",
        compute_gaze_target,
        description=(
            "Compute a world position along the user's full 3D gaze ray. "
            "Use only for explicit gaze language such as 'where I am looking'. "
            "This function calculates coordinates and does not create or move a scene object."
        ),
    )

    async def compute_user_relative_position(
        user_frame: _UserFrameArgument,
        direction_from_user: Annotated[
            Literal["front", "back", "left", "right", "above", "below"],
            Field(description="Named direction whose anchor is the user."),
        ],
        distance_meters: Annotated[
            float,
            Field(ge=0, description="Non-negative distance from the user, in metres."),
        ] = 1.5,
    ) -> Vector3:
        return Vector3.model_validate(
            spatial_math.compute_user_relative_position(
                user_frame,
                direction_from_user,
                distance_meters,
            )
        )

    group.add_function(
        "compute_user_relative_position",
        compute_user_relative_position,
        description=(
            "Compute a gravity-aligned world position relative to the user, for requests such as "
            "'in front of me' or 'to my left'. The user is always the anchor. This function "
            "calculates coordinates and does not create or move a scene object."
        ),
    )

    async def compute_position_relative_to_anchor(
        user_frame: _UserFrameArgument,
        anchor_position: _WorldPositionArgument,
        relation_to_anchor: Annotated[
            Literal[
                "toward_user",
                "away_from_user",
                "left_of",
                "right_of",
                "above",
                "below",
            ],
            Field(
                description=(
                    "Requested relation to the anchor. Horizontal relations use the user's view: "
                    "toward_user, away_from_user, left_of, or right_of."
                )
            ),
        ],
        distance_meters: Annotated[
            float,
            Field(ge=0, description="Non-negative distance from the anchor, in metres."),
        ] = 0.3,
    ) -> Vector3:
        return Vector3.model_validate(
            spatial_math.compute_position_relative_to_anchor(
                user_frame,
                anchor_position=anchor_position,
                relation_to_anchor=relation_to_anchor,
                distance_meters=distance_meters,
            )
        )

    group.add_function(
        "compute_position_relative_to_anchor",
        compute_position_relative_to_anchor,
        description=(
            "Compute a world position relative to a named scene-object anchor. Pass the anchor's "
            "complete position and describe the relation explicitly as toward/away from the user, "
            "left/right of the anchor, above, or below. This function calculates coordinates and "
            "does not mutate either object."
        ),
    )

    async def offset_position_in_user_frame(
        user_frame: _UserFrameArgument,
        start_position: _WorldPositionArgument,
        forward_meters: Annotated[
            float,
            Field(description="Signed offset along user-forward, in metres; negative means backward."),
        ] = 0.0,
        right_meters: Annotated[
            float,
            Field(description="Signed offset along user-right, in metres; negative means left."),
        ] = 0.0,
        up_meters: Annotated[
            float,
            Field(description="Signed world-up offset, in metres; negative means down."),
        ] = 0.0,
    ) -> Vector3:
        return Vector3.model_validate(
            spatial_math.offset_position_in_user_frame(
                user_frame,
                start_position=start_position,
                forward_meters=forward_meters,
                right_meters=right_meters,
                up_meters=up_meters,
            )
        )

    group.add_function(
        "offset_position_in_user_frame",
        offset_position_in_user_frame,
        description=(
            "Compute a new world position by applying signed forward, right, and up offsets to an "
            "existing position in the user's coordinate frame. This function only returns the new "
            "coordinates; update the scene separately."
        ),
    )

    async def compute_position_toward_or_away_from_reference(
        start_position: _WorldPositionArgument,
        reference_position: _WorldPositionArgument,
        movement_direction: Annotated[
            Literal["toward", "away"],
            Field(description="Whether the result moves toward or away from the reference position."),
        ],
        distance_meters: Annotated[
            float,
            Field(ge=0, description="Non-negative travel distance, in metres."),
        ] = 0.5,
    ) -> Vector3:
        return Vector3.model_validate(
            spatial_math.compute_position_toward_or_away_from_reference(
                start_position,
                reference_position,
                movement_direction=movement_direction,
                distance_meters=distance_meters,
            )
        )

    group.add_function(
        "compute_position_toward_or_away_from_reference",
        compute_position_toward_or_away_from_reference,
        description=(
            "Compute where a starting position ends after moving an exact distance toward or away "
            "from a named reference position. No user frame is involved and no scene object is mutated."
        ),
    )

    async def compute_midpoint(
        first_position: _WorldPositionArgument,
        second_position: _WorldPositionArgument,
    ) -> Vector3:
        return Vector3.model_validate(spatial_math.compute_midpoint(first_position, second_position))

    group.add_function(
        "compute_midpoint",
        compute_midpoint,
        description=(
            "Compute the world-space midpoint between two complete positions. "
            "This function only returns coordinates and does not move either object."
        ),
    )

    yield group


__all__ = ["SpatialMathFunctionsConfig"]
