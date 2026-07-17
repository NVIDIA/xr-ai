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
async def test_spatial_math_group_exposes_minimal_task_shaped_functions() -> None:
    async with WorkflowBuilder() as builder:
        await builder.add_function_group("spatial_math", SpatialMathFunctionsConfig())
        group = await builder.get_function_group("spatial_math")
        functions = await group.get_all_functions()

    assert set(functions) == {
        "spatial_math__displace_object",
        "spatial_math__midpoint",
        "spatial_math__move_relative_to",
        "spatial_math__place_in_container",
        "spatial_math__place_object_relative",
        "spatial_math__place_user_relative",
        "spatial_math__position_in_gaze",
    }


@pytest.mark.asyncio
async def test_spatial_math_functions_accept_structured_and_serialized_values() -> None:
    frame = _FRAME.model_dump()
    async with WorkflowBuilder() as builder:
        await builder.add_function_group("spatial_math", SpatialMathFunctionsConfig())
        group = await builder.get_function_group("spatial_math")
        functions = await group.get_all_functions()

        gaze = await functions["spatial_math__position_in_gaze"].ainvoke(
            {"frame": json.dumps(frame), "distance": 2.0}
        )
        user_relative = await functions["spatial_math__place_user_relative"].ainvoke(
            {"frame": frame, "direction": "front", "distance": 1.5}
        )
        object_relative = await functions["spatial_math__place_object_relative"].ainvoke(
            {
                "frame": frame,
                "anchor": {"x": 1.0, "y": 1.5, "z": 0.5},
                "direction": "right",
                "distance": 0.3,
            }
        )
        displaced = await functions["spatial_math__displace_object"].ainvoke(
            {
                "frame": frame,
                "position": {"x": 4.0, "y": 1.0, "z": 5.0},
                "forward": 2.0,
                "up": 0.2,
            }
        )
        moved = await functions["spatial_math__move_relative_to"].ainvoke(
            {
                "position": {"x": -1.0, "y": 1.0, "z": -2.0},
                "reference": {"x": 1.0, "y": 1.0, "z": -2.0},
                "direction": "toward",
                "distance": 0.4,
            }
        )
        midpoint = await functions["spatial_math__midpoint"].ainvoke(
            {
                "first": {"x": -1.0, "y": 1.0, "z": -2.0},
                "second": {"x": 3.0, "y": 2.0, "z": 0.0},
            }
        )
        contained = await functions["spatial_math__place_in_container"].ainvoke(
            {"obj_id": "sphere-0", "container": {"x": 0.4, "y": 1.2, "z": -1.7}}
        )

    assert gaze.model_dump() == {"x": 1.0, "y": 1.5, "z": 0.0}
    assert user_relative.model_dump() == {"x": 1.0, "y": 1.5, "z": 0.5}
    assert object_relative.model_dump() == {"x": 1.3, "y": 1.5, "z": 0.5}
    assert displaced.model_dump() == {"x": 4.0, "y": 1.2, "z": 3.0}
    assert moved.model_dump() == {"x": -0.6, "y": 1.0, "z": -2.0}
    assert midpoint.model_dump() == {"x": 1.0, "y": 1.5, "z": -1.0}
    assert contained.model_dump() == {
        "x": 0.4,
        "y": 1.2,
        "z": -1.7,
        "obj_id": "sphere-0",
    }


@pytest.mark.asyncio
async def test_spatial_math_schema_rejects_negative_named_distance() -> None:
    async with WorkflowBuilder() as builder:
        await builder.add_function_group("spatial_math", SpatialMathFunctionsConfig())
        group = await builder.get_function_group("spatial_math")
        functions = await group.get_all_functions()

        with pytest.raises(ValueError):
            await functions["spatial_math__place_user_relative"].ainvoke(
                {"frame": _FRAME.model_dump(), "direction": "front", "distance": -1.0}
            )
