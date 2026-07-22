# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed requests and results for the XR render demo scene."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyRequest(_Request):
    pass


class StartXRResult(BaseModel):
    status: str
    error: str | None = None


class AddPrimitiveRequest(_Request):
    prim_type: str = Field(description="Primitive type: sphere or box.")
    x: float = Field(default=0.0, description="World-space X coordinate in metres.")
    y: float = Field(default=1.6, description="World-space Y coordinate in metres.")
    z: float = Field(default=-1.5, description="World-space Z coordinate in metres.")
    r: float = Field(default=0.2, description="Red color component from 0 to 1.")
    g: float = Field(default=0.9, description="Green color component from 0 to 1.")
    b: float = Field(default=1.0, description="Blue color component from 0 to 1.")
    size: float = Field(default=0.1, description="Sphere radius or box half-edge in metres.")


class AddPrimitiveResult(BaseModel):
    id: str
    ok: bool
    reason: str | None = None


class UpdatePrimitiveRequest(_Request):
    obj_id: str = Field(description="Existing scene object ID.")
    prim_type: str | None = Field(default=None, description="New primitive type.")
    x: float | None = Field(default=None, description="New world-space X coordinate.")
    y: float | None = Field(default=None, description="New world-space Y coordinate.")
    z: float | None = Field(default=None, description="New world-space Z coordinate.")
    r: float | None = Field(default=None, description="New red color component.")
    g: float | None = Field(default=None, description="New green color component.")
    b: float | None = Field(default=None, description="New blue color component.")
    size: float | None = Field(default=None, description="New radius or half-edge in metres.")


class RemovePrimitiveRequest(_Request):
    obj_id: str = Field(description="Existing scene object ID.")


class MutationResult(BaseModel):
    ok: bool
    reason: str | None = None
    new_id: str | None = None


class Vector3(BaseModel):
    x: float
    y: float
    z: float


class Color(BaseModel):
    r: float
    g: float
    b: float


class SceneObject(BaseModel):
    id: str
    type: str
    position: Vector3
    color: Color
    size: float


class SceneState(BaseModel):
    objects: list[SceneObject]


class SceneHealth(BaseModel):
    status: Literal["ok"] = "ok"
    lovr_started: bool
    spawn_error: str | None = None
    render_drops: int


__all__ = [
    "AddPrimitiveRequest",
    "AddPrimitiveResult",
    "EmptyRequest",
    "MutationResult",
    "RemovePrimitiveRequest",
    "SceneHealth",
    "SceneObject",
    "SceneState",
    "StartXRResult",
    "UpdatePrimitiveRequest",
]
