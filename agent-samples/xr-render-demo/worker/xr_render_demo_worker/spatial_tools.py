# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample-specific tools that compose tracking with spatial math."""

from __future__ import annotations

import json
import math
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from xr_ai_tools import Tool
from xr_ai_tools.spatial import (
    anchor_relative,
    gaze_target,
    midpoint,
    offset_user_frame,
    toward,
    user_relative,
)
from xr_ai_tools.tracking import TrackingTools
from xr_ai_tools.types import EmptyRequest, SpatialFrame, Vector3


class _Request(BaseModel):
    """Ignore provider-added fields that are unrelated to the selected tool."""


class PositionAheadRequest(_Request):
    distance: float = Field(default=1.5, ge=0.0, description="Distance along the user's gaze, in metres.")


class PositionRelativeRequest(_Request):
    forward: float = Field(default=0.0, description="Signed metres along the user forward axis.")
    right: float = Field(default=0.0, description="Signed metres along the user right axis.")
    up: float = Field(default=0.0, description="Signed metres along world up.")
    origin_x: float | None = Field(default=None, description="Optional world-space origin X.")
    origin_y: float | None = Field(default=None, description="Optional world-space origin Y.")
    origin_z: float | None = Field(default=None, description="Optional world-space origin Z.")


class PlaceUserRelativeRequest(_Request):
    direction: Literal["front", "back", "left", "right", "above", "below"] = Field(
        description="Named direction relative to the user."
    )
    distance: float = Field(default=1.5, ge=0.0, description="Non-negative distance in metres.")


class PlaceObjectRelativeRequest(_Request):
    origin_x: float = Field(description="Anchor world-space X.")
    origin_y: float = Field(description="Anchor world-space Y.")
    origin_z: float = Field(description="Anchor world-space Z.")
    direction: Literal["front", "back", "left", "right", "above", "below", "next_to"] = Field(
        description="Named relation to the anchor in the user frame."
    )
    distance: float = Field(default=0.3, ge=0.0, description="Non-negative distance in metres.")


class DisplaceObjectRequest(_Request):
    current_x: float = Field(description="Current world-space X.")
    current_y: float = Field(description="Current world-space Y.")
    current_z: float = Field(description="Current world-space Z.")
    right: float = Field(default=0.0, description="Signed metres along the user right axis.")
    up: float = Field(default=0.0, description="Signed metres along world up.")
    forward: float = Field(default=0.0, description="Signed metres along the user forward axis.")


class DisplaceObjectsRequest(_Request):
    object_ids: list[str] = Field(description="Object IDs aligned with the coordinate arrays.")
    current_xs: list[float] = Field(description="Current world-space X values.")
    current_ys: list[float] = Field(description="Current world-space Y values.")
    current_zs: list[float] = Field(description="Current world-space Z values.")
    right: float = Field(default=0.0, description="Signed metres along the user right axis.")
    up: float = Field(default=0.0, description="Signed metres along world up.")
    forward: float = Field(default=0.0, description="Signed metres along the user forward axis.")


class PlaceInsideRequest(_Request):
    movee_id: str = Field(description="ID of the object to move.")
    container_x: float = Field(description="Container world-space X.")
    container_y: float = Field(description="Container world-space Y.")
    container_z: float = Field(description="Container world-space Z.")


class BetweenAnchorsRequest(_Request):
    a_x: float = Field(description="First anchor world-space X.")
    a_y: float = Field(description="First anchor world-space Y.")
    a_z: float = Field(description="First anchor world-space Z.")
    b_x: float = Field(description="Second anchor world-space X.")
    b_y: float = Field(description="Second anchor world-space Y.")
    b_z: float = Field(description="Second anchor world-space Z.")


class WorldOffsetRequest(_Request):
    origin_x: float = Field(description="Origin world-space X.")
    origin_y: float = Field(description="Origin world-space Y.")
    origin_z: float = Field(description="Origin world-space Z.")
    dx: float = Field(default=0.0, description="Signed world-space X offset.")
    dy: float = Field(default=0.0, description="Signed world-space Y offset.")
    dz: float = Field(default=0.0, description="Signed world-space Z offset.")


class AlongDirectionRequest(_Request):
    origin_x: float = Field(description="Origin world-space X.")
    origin_y: float = Field(description="Origin world-space Y.")
    origin_z: float = Field(description="Origin world-space Z.")
    target_x: float = Field(description="Target world-space X.")
    target_y: float = Field(description="Target world-space Y.")
    target_z: float = Field(description="Target world-space Z.")
    distance: float = Field(default=0.5, description="Signed distance in metres; negative moves away.")


class ScaleValueRequest(_Request):
    current: float = Field(description="Current scalar size.")
    factor: float = Field(description="Multiplier applied to the current size.")


class HeadPoseResult(BaseModel):
    is_valid: bool = True
    position: Vector3
    forward: Vector3
    right: Vector3
    up: Vector3
    yaw_deg: float
    pitch_deg: float
    ts: int


class FlexibleResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    error: str | None = None


class BatchResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    items: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class ScalarResult(BaseModel):
    value: float


def _render_flexible(result: FlexibleResult) -> str:
    return result.model_dump_json(exclude_none=True)


def _render_batch(result: BatchResult) -> str:
    data = result.model_dump(exclude_none=True)
    if result.error is not None:
        data.pop("items", None)
    return json.dumps(data)


class RenderSpatialTools:
    """Compose tracking and pure math into the demo's established vocabulary."""

    def __init__(self, tracking: TrackingTools) -> None:
        self.tracking = tracking
        self.get_head_pose = Tool(
            "get_head_pose",
            "Return the current world-space head position and forward, right, and up axes.",
            EmptyRequest,
            HeadPoseResult,
            self._get_head_pose,
        )
        self.position_ahead = Tool(
            "position_ahead",
            "Compute a world position along the user's gaze for 'in front of me' requests.",
            PositionAheadRequest,
            Vector3,
            self._position_ahead,
        )
        self.position_relative = Tool(
            "position_relative",
            "Apply signed forward, right, and up user-frame offsets.",
            PositionRelativeRequest,
            Vector3,
            self._position_relative,
        )
        self.place_user_relative = Tool(
            "place_user_relative",
            "Compute a position in one named direction from the user.",
            PlaceUserRelativeRequest,
            FlexibleResult,
            self._place_user_relative,
            render_result=_render_flexible,
        )
        self.place_object_relative = Tool(
            "place_object_relative",
            "Compute a position in one named direction from an existing object's world position.",
            PlaceObjectRelativeRequest,
            FlexibleResult,
            self._place_object_relative,
            render_result=_render_flexible,
        )
        self.displace_object = Tool(
            "displace_object",
            "Shift one existing object by signed right, up, and forward user-frame offsets.",
            DisplaceObjectRequest,
            Vector3,
            self._displace_object,
        )
        self.displace_objects = Tool(
            "displace_objects",
            "Apply one signed user-frame offset to parallel lists of existing objects.",
            DisplaceObjectsRequest,
            BatchResult,
            self._displace_objects,
            render_result=_render_batch,
        )
        self.place_inside_by_id = Tool(
            "place_inside_by_id",
            "Return the container position with the ID of the object that should move there.",
            PlaceInsideRequest,
            FlexibleResult,
            self._place_inside,
            render_result=_render_flexible,
        )
        self.between_anchors = Tool(
            "between_anchors",
            "Compute the world-space midpoint between exactly two anchor positions.",
            BetweenAnchorsRequest,
            Vector3,
            self._between_anchors,
        )
        self.world_offset = Tool(
            "world_offset",
            "Apply signed world-axis dx, dy, and dz offsets to an origin position.",
            WorldOffsetRequest,
            Vector3,
            self._world_offset,
        )
        self.along_direction = Tool(
            "along_direction",
            "Move an origin toward a target by positive distance or away by negative distance.",
            AlongDirectionRequest,
            Vector3,
            self._along_direction,
        )
        self.scale_value = Tool(
            "scale_value",
            "Multiply a current numeric size by a scale factor.",
            ScaleValueRequest,
            ScalarResult,
            lambda request: ScalarResult(value=round(request.current * request.factor, 3)),
        )
        self.tools = (
            self.get_head_pose,
            self.position_ahead,
            self.position_relative,
            self.place_user_relative,
            self.place_object_relative,
            self.displace_object,
            self.displace_objects,
            self.place_inside_by_id,
            self.between_anchors,
            self.world_offset,
            self.along_direction,
            self.scale_value,
        )

    async def _frame(self) -> SpatialFrame:
        return await self.tracking.get_user_frame.execute(EmptyRequest())

    async def _get_head_pose(self, _request: EmptyRequest) -> HeadPoseResult:
        frame = await self._frame()
        return HeadPoseResult(
            position=frame.origin,
            forward=frame.forward,
            right=frame.right,
            up=frame.up,
            yaw_deg=math.degrees(math.atan2(-frame.forward.x, -frame.forward.z)),
            pitch_deg=math.degrees(math.asin(max(-1.0, min(1.0, frame.forward.y)))),
            ts=time.time_ns() // 1_000_000,
        )

    async def _position_ahead(self, request: PositionAheadRequest) -> Vector3:
        return gaze_target(await self._frame(), request.distance)

    async def _position_relative(self, request: PositionRelativeRequest) -> Vector3:
        frame = await self._frame()
        origin = frame.origin
        return offset_user_frame(
            frame,
            Vector3(
                x=origin.x if request.origin_x is None else request.origin_x,
                y=origin.y if request.origin_y is None else request.origin_y,
                z=origin.z if request.origin_z is None else request.origin_z,
            ),
            forward=request.forward,
            right=request.right,
            up=request.up,
        )

    async def _place_user_relative(self, request: PlaceUserRelativeRequest) -> FlexibleResult:
        if request.distance < 0:
            return FlexibleResult(error="distance must be non-negative; flip the direction instead")
        return FlexibleResult.model_validate(
            user_relative(await self._frame(), request.direction, request.distance).model_dump()
        )

    async def _place_object_relative(self, request: PlaceObjectRelativeRequest) -> FlexibleResult:
        if request.distance < 0:
            return FlexibleResult(error="distance must be non-negative; flip the direction instead")
        relation = {
            "front": "toward_user",
            "back": "away_from_user",
            "left": "left_of",
            "right": "right_of",
            "next_to": "right_of",
            "above": "above",
            "below": "below",
        }[request.direction]
        frame = await self._frame()
        result = anchor_relative(
            frame,
            Vector3(x=request.origin_x, y=request.origin_y, z=request.origin_z),
            relation,
            request.distance,
        )
        return FlexibleResult.model_validate(result.model_dump())

    async def _displace_object(self, request: DisplaceObjectRequest) -> Vector3:
        return offset_user_frame(
            await self._frame(),
            Vector3(x=request.current_x, y=request.current_y, z=request.current_z),
            forward=request.forward,
            right=request.right,
            up=request.up,
        )

    async def _displace_objects(self, request: DisplaceObjectsRequest) -> BatchResult:
        if not (
            len(request.object_ids)
            == len(request.current_xs)
            == len(request.current_ys)
            == len(request.current_zs)
        ):
            return BatchResult(error="object_ids / current_xs / current_ys / current_zs must all be the same length")
        frame = await self._frame()
        items = []
        for obj_id, x, y, z in zip(
            request.object_ids,
            request.current_xs,
            request.current_ys,
            request.current_zs,
            strict=True,
        ):
            result = offset_user_frame(
                frame,
                Vector3(x=x, y=y, z=z),
                forward=request.forward,
                right=request.right,
                up=request.up,
            )
            items.append({"obj_id": obj_id, **result.model_dump()})
        return BatchResult(items=items)

    async def _place_inside(self, request: PlaceInsideRequest) -> FlexibleResult:
        return FlexibleResult.model_validate(
            {
                "obj_id": request.movee_id,
                "x": round(request.container_x, 3),
                "y": round(request.container_y, 3),
                "z": round(request.container_z, 3),
            }
        )

    async def _between_anchors(self, request: BetweenAnchorsRequest) -> Vector3:
        return midpoint(
            Vector3(x=request.a_x, y=request.a_y, z=request.a_z),
            Vector3(x=request.b_x, y=request.b_y, z=request.b_z),
        )

    async def _world_offset(self, request: WorldOffsetRequest) -> Vector3:
        return Vector3(
            x=request.origin_x + request.dx,
            y=request.origin_y + request.dy,
            z=request.origin_z + request.dz,
        )

    async def _along_direction(self, request: AlongDirectionRequest) -> Vector3:
        return toward(
            Vector3(x=request.origin_x, y=request.origin_y, z=request.origin_z),
            Vector3(x=request.target_x, y=request.target_y, z=request.target_z),
            request.distance,
        )


__all__ = ["RenderSpatialTools"]
