# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scene operations shared by native functions and the MCP adapter."""

import asyncio

from loguru import logger
from xr_ai_nat.functions._rpc import RPCError

from .engine import SceneDispatcher
from .schemas import (
    AddPrimitiveRequest,
    EmptyRequest,
    RemovePrimitiveRequest,
    UpdatePrimitiveRequest,
)


class SceneService:
    def __init__(self, dispatcher: SceneDispatcher) -> None:
        self._dispatcher = dispatcher
        self._spawn_task: asyncio.Task[None] | None = None

    async def dispatch(self, operation: str, arguments: dict) -> dict:
        if operation == "start_xr":
            EmptyRequest.model_validate(arguments)
            return self._start_xr()
        if operation == "add_primitive":
            return await self._add(AddPrimitiveRequest.model_validate(arguments))
        if operation == "update_primitive":
            return await self._update(UpdatePrimitiveRequest.model_validate(arguments))
        if operation == "remove_primitive":
            return await self._remove(RemovePrimitiveRequest.model_validate(arguments))
        if operation == "get_scene_state":
            EmptyRequest.model_validate(arguments)
            return self._dispatcher.scene_snapshot()
        if operation == "get_health":
            EmptyRequest.model_validate(arguments)
            return self._dispatcher.health_snapshot()
        raise RPCError(f"unknown operation: {operation}", code="unknown_operation")

    def _start_xr(self) -> dict:
        health = self._dispatcher.health_snapshot()
        if health["lovr_started"]:
            return {"status": "already_started"}
        if health["spawn_error"] is not None:
            return {"status": "error", "error": health["spawn_error"]}

        if self._spawn_task is None or self._spawn_task.done():
            self._spawn_task = asyncio.create_task(
                self._spawn_lovr(),
                name="lovr-spawn",
            )
        return {"status": "starting"}

    async def _spawn_lovr(self) -> None:
        try:
            await self._dispatcher.start_lovr_once()
        except Exception:
            logger.exception("xr-render-scene: start_xr crashed")

    async def close(self) -> None:
        task, self._spawn_task = self._spawn_task, None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _add(self, request: AddPrimitiveRequest) -> dict:
        position = {"x": request.x, "y": request.y, "z": request.z}
        color = {"r": request.r, "g": request.g, "b": request.b}
        obj_id = self._dispatcher.add(
            request.prim_type,
            position,
            color,
            request.size,
        )
        result = await self._dispatcher.forward(
            "scene.add",
            {
                "id": obj_id,
                "type": request.prim_type,
                "position": [request.x, request.y, request.z],
                "color": [request.r, request.g, request.b],
                "size": request.size,
            },
        )
        logger.debug(
            "xr-render-scene: add_primitive id={} type={}",
            obj_id,
            request.prim_type,
        )
        return {"id": obj_id, **result}

    async def _update(self, request: UpdatePrimitiveRequest) -> dict:
        obj = self._dispatcher.get_object(request.obj_id)
        if obj is None:
            return {"ok": False, "reason": "not_found"}

        props: dict = {}
        position = {key: value for key, value in (
            ("x", request.x), ("y", request.y), ("z", request.z)
        ) if value is not None}
        color = {key: value for key, value in (
            ("r", request.r), ("g", request.g), ("b", request.b)
        ) if value is not None}
        if position:
            props["position"] = position
        if color:
            props["color"] = color
        if request.size is not None:
            props["size"] = request.size
        if props:
            self._dispatcher.update(request.obj_id, props)

        if request.prim_type is not None and request.prim_type != obj["type"]:
            merged = self._dispatcher.get_object(request.obj_id)
            position, color = merged["position"], merged["color"]
            self._dispatcher.remove(request.obj_id)
            await self._dispatcher.forward("scene.remove", {"id": request.obj_id})
            new_id = self._dispatcher.add(
                request.prim_type,
                position,
                color,
                merged["size"],
            )
            result = await self._dispatcher.forward(
                "scene.add",
                {
                    "id": new_id,
                    "type": request.prim_type,
                    "position": [position["x"], position["y"], position["z"]],
                    "color": [color["r"], color["g"], color["b"]],
                    "size": merged["size"],
                },
            )
            return {
                "ok": result.get("ok", True),
                "reason": result.get("reason"),
                "new_id": new_id,
            }

        if not props:
            return {"ok": True}

        current = self._dispatcher.get_object(request.obj_id)
        wire: dict = {"id": request.obj_id}
        if "position" in props:
            value = current["position"]
            wire["position"] = [value["x"], value["y"], value["z"]]
        if "color" in props:
            value = current["color"]
            wire["color"] = [value["r"], value["g"], value["b"]]
        if "size" in props:
            wire["size"] = current["size"]
        return await self._dispatcher.forward("scene.update", wire)

    async def _remove(self, request: RemovePrimitiveRequest) -> dict:
        if not self._dispatcher.remove(request.obj_id):
            return {"ok": False, "reason": "not_found"}
        return await self._dispatcher.forward(
            "scene.remove",
            {"id": request.obj_id},
        )


__all__ = ["SceneService"]
