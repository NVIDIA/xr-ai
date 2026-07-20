# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the native spatial-math function group."""

from __future__ import annotations

import json

import pytest
from nat.builder.workflow_builder import WorkflowBuilder
from xr_ai_nat.functions.spatial_math import SpatialFrame, SpatialMathFunctionsConfig, Vector3

_FRAME = SpatialFrame(
    origin=Vector3(x=1.0, y=1.5, z=2.0),
    forward=Vector3(x=0.0, y=0.0, z=-1.0),
    right=Vector3(x=1.0, y=0.0, z=0.0),
    up=Vector3(x=0.0, y=1.0, z=0.0),
)


@pytest.mark.asyncio
async def test_spatial_math_group_exposes_explicit_coordinate_calculations() -> None:
    async with WorkflowBuilder() as builder:
        await builder.add_function_group("spatial_math", SpatialMathFunctionsConfig())
        group = await builder.get_function_group("spatial_math")
        functions = await group.get_all_functions()

    assert set(functions) == {
        "spatial_math__compute_gaze_target",
        "spatial_math__compute_midpoint",
        "spatial_math__compute_position_relative_to_anchor",
        "spatial_math__compute_position_toward_or_away_from_reference",
        "spatial_math__compute_user_relative_position",
        "spatial_math__offset_position_in_user_frame",
    }

    expected_inputs = {
        "spatial_math__compute_gaze_target": {"user_frame", "distance_meters"},
        "spatial_math__compute_user_relative_position": {
            "user_frame",
            "direction_from_user",
            "distance_meters",
        },
        "spatial_math__compute_position_relative_to_anchor": {
            "user_frame",
            "anchor_position",
            "relation_to_anchor",
            "distance_meters",
        },
        "spatial_math__offset_position_in_user_frame": {
            "user_frame",
            "start_position",
            "forward_meters",
            "right_meters",
            "up_meters",
        },
        "spatial_math__compute_position_toward_or_away_from_reference": {
            "start_position",
            "reference_position",
            "movement_direction",
            "distance_meters",
        },
        "spatial_math__compute_midpoint": {"first_position", "second_position"},
    }
    for name, function in functions.items():
        properties = function.input_schema.model_json_schema()["properties"]
        assert set(properties) == expected_inputs[name]
        assert all(value.get("description") for value in properties.values())


@pytest.mark.asyncio
async def test_spatial_math_functions_accept_structured_and_serialized_values() -> None:
    frame = _FRAME.model_dump()
    async with WorkflowBuilder() as builder:
        await builder.add_function_group("spatial_math", SpatialMathFunctionsConfig())
        group = await builder.get_function_group("spatial_math")
        functions = await group.get_all_functions()

        gaze = await functions["spatial_math__compute_gaze_target"].ainvoke(
            {"user_frame": json.dumps(frame), "distance_meters": 2.0}
        )
        user_relative = await functions["spatial_math__compute_user_relative_position"].ainvoke(
            {
                "user_frame": frame,
                "direction_from_user": "front",
                "distance_meters": 1.5,
            }
        )
        object_relative = await functions["spatial_math__compute_position_relative_to_anchor"].ainvoke(
            {
                "user_frame": frame,
                "anchor_position": {"x": 1.0, "y": 1.5, "z": 0.5},
                "relation_to_anchor": "right_of",
                "distance_meters": 0.3,
            }
        )
        displaced = await functions["spatial_math__offset_position_in_user_frame"].ainvoke(
            {
                "user_frame": frame,
                "start_position": {"x": 4.0, "y": 1.0, "z": 5.0},
                "forward_meters": 2.0,
                "up_meters": 0.2,
            }
        )
        moved = await functions["spatial_math__compute_position_toward_or_away_from_reference"].ainvoke(
            {
                "start_position": {"x": -1.0, "y": 1.0, "z": -2.0},
                "reference_position": {"x": 1.0, "y": 1.0, "z": -2.0},
                "movement_direction": "toward",
                "distance_meters": 0.4,
            }
        )
        midpoint = await functions["spatial_math__compute_midpoint"].ainvoke(
            {
                "first_position": {"x": -1.0, "y": 1.0, "z": -2.0},
                "second_position": {"x": 3.0, "y": 2.0, "z": 0.0},
            }
        )

    assert gaze.model_dump() == {"x": 1.0, "y": 1.5, "z": 0.0}
    assert user_relative.model_dump() == {"x": 1.0, "y": 1.5, "z": 0.5}
    assert object_relative.model_dump() == {"x": 1.3, "y": 1.5, "z": 0.5}
    assert displaced.model_dump() == {"x": 4.0, "y": 1.2, "z": 3.0}
    assert moved.model_dump() == {"x": -0.6, "y": 1.0, "z": -2.0}
    assert midpoint.model_dump() == {"x": 1.0, "y": 1.5, "z": -1.0}


@pytest.mark.asyncio
async def test_anchor_relation_schema_names_the_reference_frame() -> None:
    async with WorkflowBuilder() as builder:
        await builder.add_function_group("spatial_math", SpatialMathFunctionsConfig())
        group = await builder.get_function_group("spatial_math")
        functions = await group.get_all_functions()

    relation = functions[
        "spatial_math__compute_position_relative_to_anchor"
    ].input_schema.model_json_schema()["properties"]["relation_to_anchor"]
    assert set(relation["enum"]) == {
        "toward_user",
        "away_from_user",
        "left_of",
        "right_of",
        "above",
        "below",
    }


@pytest.mark.asyncio
async def test_spatial_math_schema_rejects_negative_named_distance() -> None:
    async with WorkflowBuilder() as builder:
        await builder.add_function_group("spatial_math", SpatialMathFunctionsConfig())
        group = await builder.get_function_group("spatial_math")
        functions = await group.get_all_functions()

        with pytest.raises(ValueError):
            await functions["spatial_math__compute_user_relative_position"].ainvoke(
                {
                    "user_frame": _FRAME.model_dump(),
                    "direction_from_user": "front",
                    "distance_meters": -1.0,
                }
            )
