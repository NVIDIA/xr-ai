# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contract tests for the sample-local native scene functions."""

import asyncio
import uuid

import pytest
from nat.builder.workflow_builder import WorkflowBuilder
from xr_ai_nat.functions._rpc import RPCServer
from xr_render_scene import (
    SceneControlFunctionsConfig,
    SceneObjectFunctionsConfig,
    SceneStateFunctionsConfig,
    SceneUpdateFunctionsConfig,
)
from xr_render_scene.service import SceneService


class _MemoryDispatcher:
    def __init__(self) -> None:
        self.objects: dict[str, dict] = {}
        self.started = False

    async def start_lovr_once(self) -> dict:
        self.started = True
        return {"status": "started"}

    def health_snapshot(self) -> dict:
        return {
            "status": "ok",
            "lovr_started": self.started,
            "spawn_error": None,
            "render_drops": 0,
        }

    def scene_snapshot(self) -> dict:
        return {
            "objects": [
                {"id": object_id, **value}
                for object_id, value in self.objects.items()
            ]
        }

    def add(self, prim_type: str, position: dict, color: dict, size: float) -> str:
        object_id = f"{prim_type}-{len(self.objects)}"
        self.objects[object_id] = {
            "type": prim_type,
            "position": dict(position),
            "color": dict(color),
            "size": size,
        }
        return object_id

    def get_object(self, object_id: str) -> dict | None:
        return self.objects.get(object_id)

    def update(self, object_id: str, properties: dict) -> bool:
        current = self.objects.get(object_id)
        if current is None:
            return False
        for key, value in properties.items():
            if isinstance(value, dict):
                current[key].update(value)
            else:
                current[key] = value
        return True

    def remove(self, object_id: str) -> bool:
        return self.objects.pop(object_id, None) is not None

    async def forward(self, _operation: str, _value: dict) -> dict:
        return {"ok": True}


class _BlockingStartDispatcher(_MemoryDispatcher):
    def __init__(self) -> None:
        super().__init__()
        self.started_event = asyncio.Event()
        self.cancelled = False

    async def start_lovr_once(self) -> dict:
        self.started_event.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


@pytest.mark.asyncio
async def test_native_scene_groups_share_the_typed_scene_service() -> None:
    endpoint = f"ipc:///tmp/scene-functions-{uuid.uuid4().hex}"
    server = RPCServer(endpoint, SceneService(_MemoryDispatcher()).dispatch)
    task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.02)
    try:
        async with WorkflowBuilder() as builder:
            await builder.add_function_group(
                "objects",
                SceneObjectFunctionsConfig(endpoint=endpoint),
            )
            await builder.add_function_group(
                "updates",
                SceneUpdateFunctionsConfig(endpoint=endpoint),
            )
            await builder.add_function_group(
                "state",
                SceneStateFunctionsConfig(endpoint=endpoint),
            )
            await builder.add_function_group(
                "control",
                SceneControlFunctionsConfig(endpoint=endpoint),
            )

            objects = await (await builder.get_function_group("objects")).get_all_functions()
            updates = await (await builder.get_function_group("updates")).get_all_functions()
            state = await (await builder.get_function_group("state")).get_all_functions()

            created = await objects["objects__add_primitive"].ainvoke(
                {"prim_type": "sphere", "r": 1.0, "g": 0.0, "b": 0.0}
            )
            await updates["updates__update_primitive"].ainvoke(
                {"obj_id": created.id, "x": 0.5}
            )
            snapshot = await state["state__get_scene_state"].ainvoke({})

        assert snapshot.objects[0].id == created.id
        assert snapshot.objects[0].position.x == 0.5
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_scene_service_tracks_and_cancels_pending_spawn() -> None:
    dispatcher = _BlockingStartDispatcher()
    service = SceneService(dispatcher)

    assert service._start_xr() == {"status": "starting"}
    first_task = service._spawn_task
    assert service._start_xr() == {"status": "starting"}
    assert service._spawn_task is first_task

    await dispatcher.started_event.wait()
    await service.close()

    assert dispatcher.cancelled
    assert service._spawn_task is None
