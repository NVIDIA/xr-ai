# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for native spatial calculations."""

import pytest
from xr_ai_tools.spatial import (
    anchor_relative,
    gaze_target,
    midpoint,
    offset_user_frame,
    toward,
    user_relative,
)
from xr_ai_tools.types import SpatialFrame, Vector3

_FRAME = SpatialFrame(
    origin=Vector3(x=1.0, y=1.5, z=2.0),
    forward=Vector3(x=0.0, y=0.0, z=-1.0),
    right=Vector3(x=1.0, y=0.0, z=0.0),
    up=Vector3(x=0.0, y=1.0, z=0.0),
)


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (lambda: gaze_target(_FRAME, 2.0), Vector3(x=1.0, y=1.5, z=0.0)),
        (
            lambda: user_relative(_FRAME, "front", 1.5),
            Vector3(x=1.0, y=1.5, z=0.5),
        ),
        (
            lambda: anchor_relative(
                _FRAME,
                Vector3(x=1.0, y=1.5, z=0.5),
                "right_of",
                0.3,
            ),
            Vector3(x=1.3, y=1.5, z=0.5),
        ),
        (
            lambda: offset_user_frame(
                _FRAME,
                Vector3(x=4.0, y=1.0, z=5.0),
                forward=2.0,
                up=0.2,
            ),
            Vector3(x=4.0, y=1.2, z=3.0),
        ),
        (
            lambda: toward(
                Vector3(x=-1.0, y=1.0, z=-2.0),
                Vector3(x=1.0, y=1.0, z=-2.0),
                0.4,
            ),
            Vector3(x=-0.6, y=1.0, z=-2.0),
        ),
        (
            lambda: midpoint(
                Vector3(x=-1.0, y=1.0, z=-2.0),
                Vector3(x=3.0, y=2.0, z=0.0),
            ),
            Vector3(x=1.0, y=1.5, z=-1.0),
        ),
    ],
)
def test_spatial_calculations(operation, expected: Vector3) -> None:
    assert operation() == expected


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        (
            SpatialFrame(
                origin=Vector3(x=0.0, y=0.0, z=0.0),
                forward=Vector3(x=0.0, y=1.0, z=0.0),
                right=Vector3(x=1.0, y=0.0, z=0.0),
                up=Vector3(x=0.0, y=0.0, z=1.0),
            ),
            Vector3(x=0.0, y=0.0, z=-1.0),
        ),
        (
            SpatialFrame(
                origin=Vector3(x=0.0, y=0.0, z=0.0),
                forward=Vector3(x=0.0, y=1.0, z=0.0),
                right=Vector3(x=0.0, y=1.0, z=0.0),
                up=Vector3(x=0.0, y=0.0, z=1.0),
            ),
            Vector3(x=0.0, y=0.0, z=-1.0),
        ),
    ],
)
def test_horizontal_forward_falls_back_for_vertical_or_degenerate_axes(
    frame: SpatialFrame,
    expected: Vector3,
) -> None:
    assert offset_user_frame(frame, frame.origin, forward=1.0) == expected


@pytest.mark.parametrize(
    "operation",
    [
        lambda: gaze_target(_FRAME, -1.0),
        lambda: user_relative(_FRAME, "front", -1.0),
        lambda: anchor_relative(_FRAME, _FRAME.origin, "left_of", -1.0),
    ],
)
def test_named_distances_reject_negative_values(operation) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        operation()
