# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract for the shared value models in ``xr_ai_nat.functions.types``."""
from __future__ import annotations

import importlib
import json

import pytest
import xr_ai_nat.functions.types as types_module
from xr_ai_nat.functions.types import ServiceResult, SpatialFrame, Vector3


def test_value_models_stringify_as_json() -> None:
    """Text-only agent bridges call ``str()``; the models render as JSON so the
    value stays reusable as a typed tool input (unchanged from the old base)."""
    vec = Vector3(x=1.0, y=2.0, z=3.0)
    assert str(vec) == vec.model_dump_json()
    assert json.loads(str(vec)) == {"x": 1.0, "y": 2.0, "z": 3.0}

    frame = SpatialFrame(
        origin=Vector3(x=0.0, y=0.0, z=0.0),
        forward=Vector3(x=0.0, y=0.0, z=-1.0),
        right=Vector3(x=1.0, y=0.0, z=0.0),
        up=Vector3(x=0.0, y=1.0, z=0.0),
    )
    assert json.loads(str(frame))["forward"] == {"x": 0.0, "y": 0.0, "z": -1.0}


def test_service_result_retains_unknown_fields() -> None:
    """``ServiceResult`` sets ``extra="allow"``, so unknown fields are retained
    (not dropped) — a deliberate change from the old ``extra="ignore"`` base."""
    vec = Vector3.model_validate({"x": 1.0, "y": 2.0, "z": 3.0, "w": 9.0})
    assert vec.model_dump()["w"] == 9.0
    assert json.loads(str(vec))["w"] == 9.0
    assert isinstance(vec, ServiceResult)


def test_service_result_is_exported() -> None:
    """``ServiceResult`` is the intended shared base for capability result models,
    so it is part of the public module surface."""
    assert "ServiceResult" in types_module.__all__


def test_spatial_math_schemas_is_a_deprecated_alias() -> None:
    """The moved ``spatial_math.schemas`` submodule stays as a deprecated
    forwarding alias so existing imports keep working (types now live in
    ``functions.types``)."""
    import xr_ai_nat.functions.spatial_math.schemas as legacy

    with pytest.warns(DeprecationWarning):
        legacy = importlib.reload(legacy)
    assert legacy.Vector3 is Vector3
    assert legacy.SpatialFrame is SpatialFrame
