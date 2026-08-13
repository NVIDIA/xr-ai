# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native tool composition for the xr-render-demo application."""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from xr_ai_models import VLMService
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.capabilities import (
    EmptyRequest,
    HistoricalVisionTool,
    SpatialFrame,
    TextMemoryTool,
    TrackingTools,
    Vector3,
    VideoMemoryTools,
)
from xr_ai_tools.live_vision import LiveVisionTool
from xr_ai_tools.spatial import (
    anchor_relative,
    gaze_target,
    midpoint,
    offset_user_frame,
    toward,
    user_relative,
)
from xr_render_scene import SceneTools


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PositionAheadRequest(_Request):
    distance: float = Field(default=1.5, description="Distance along the user's gaze, in metres.")


class PositionRelativeRequest(_Request):
    forward: float = 0.0
    right: float = 0.0
    up: float = 0.0
    origin_x: float | None = None
    origin_y: float | None = None
    origin_z: float | None = None


class PlaceUserRelativeRequest(_Request):
    direction: Literal["front", "back", "left", "right", "above", "below"]
    distance: float = 1.5


class PlaceObjectRelativeRequest(_Request):
    origin_x: float
    origin_y: float
    origin_z: float
    direction: Literal["front", "back", "left", "right", "above", "below", "next_to"]
    distance: float = 0.3


class DisplaceObjectRequest(_Request):
    current_x: float
    current_y: float
    current_z: float
    right: float = 0.0
    up: float = 0.0
    forward: float = 0.0


class DisplaceObjectsRequest(_Request):
    object_ids: list[str]
    current_xs: list[float]
    current_ys: list[float]
    current_zs: list[float]
    right: float = 0.0
    up: float = 0.0
    forward: float = 0.0


class PlaceInsideRequest(_Request):
    movee_id: str
    container_x: float
    container_y: float
    container_z: float


class BetweenAnchorsRequest(_Request):
    a_x: float
    a_y: float
    a_z: float
    b_x: float
    b_y: float
    b_z: float


class WorldOffsetRequest(_Request):
    origin_x: float
    origin_y: float
    origin_z: float
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0


class AlongDirectionRequest(_Request):
    origin_x: float
    origin_y: float
    origin_z: float
    target_x: float
    target_y: float
    target_z: float
    distance: float = 0.5


class ScaleValueRequest(_Request):
    current: float
    factor: float


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


class NativeCapabilities:
    """Own every native tool and service client used by the render worker."""

    def __init__(
        self,
        *,
        scene_endpoint: str,
        openxr_endpoint: str,
        video_memory_endpoint: str,
        frame_endpoint: Any,
        vlm: VLMService,
        text_memory_dir: str | Path,
        frame_max_age_s: float = 2.0,
        frame_timeout_s: float = 5.0,
    ) -> None:
        self.scene = SceneTools(scene_endpoint)
        self.tracking = TrackingTools(openxr_endpoint)
        self.spatial = RenderSpatialTools(self.tracking)
        self.video = VideoMemoryTools(video_memory_endpoint)
        self.live_vision = LiveVisionTool(
            endpoint=frame_endpoint,
            vlm=vlm,
            system_prompt="Answer directly from the visible camera image in one short plain-English sentence.",
            frame_max_age_s=frame_max_age_s,
            frame_timeout_s=frame_timeout_s,
            manage_status=False,
        )
        self.past_vision = HistoricalVisionTool(video=self.video, vlm=vlm)
        self.text_memory = TextMemoryTool(text_memory_dir)
        all_tools = (
            *self.scene.tools,
            *self.spatial.tools,
            *self.video.tools,
            self.live_vision,
            self.past_vision,
        )
        self.all = ToolSet(all_tools)
        self.model = ToolSet(
            tool
            for tool in all_tools
            if tool.name not in {"start_xr", "get_health", "get_frame_from_time"}
        )

    def release(self, participant_id: str) -> None:
        self.live_vision.release(participant_id)

    async def close(self) -> None:
        await asyncio.gather(
            self.scene.close(),
            self.tracking.close(),
            self.video.close(),
        )


__all__ = ["NativeCapabilities"]
