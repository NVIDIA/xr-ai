# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for service RPC and native XR-tracking functions."""

import asyncio
import contextlib
import math
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from nat.builder.workflow_builder import WorkflowBuilder
from openxr_service.pose import from_native_pose
from openxr_service.service import OpenXRService
from xr_ai_nat.functions._rpc import RPCClient, RPCError, RPCServer
from xr_ai_nat.functions.spatial_math import SpatialFrame, Vector3
from xr_ai_nat.functions.xr_tracking import HeadPose, OpenXRHealth, XRTrackingFunctionsConfig


def _endpoint(_tmp_path: Path) -> str:
    return f"ipc:///tmp/xr-{uuid.uuid4().hex}"


@contextlib.asynccontextmanager
async def _running_server(endpoint: str, dispatch):
    server = RPCServer(endpoint, dispatch)
    task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.02)
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_rpc_correlates_concurrent_responses(tmp_path: Path) -> None:
    async def dispatch(operation: str, arguments: dict) -> dict:
        await asyncio.sleep(float(arguments["delay"]))
        return {"operation": operation, "value": arguments["value"]}

    endpoint = _endpoint(tmp_path)
    async with _running_server(endpoint, dispatch):
        async with RPCClient(endpoint) as client:
            slow, fast = await asyncio.gather(
                client.call("echo", {"value": "slow", "delay": 0.05}),
                client.call("echo", {"value": "fast", "delay": 0.0}),
            )

    assert slow["value"] == "slow"
    assert fast["value"] == "fast"


@pytest.mark.asyncio
async def test_rpc_preserves_remote_error_codes(tmp_path: Path) -> None:
    async def dispatch(_operation: str, _arguments: dict) -> dict:
        raise RPCError("tracking unavailable", code="unavailable")

    endpoint = _endpoint(tmp_path)
    async with _running_server(endpoint, dispatch):
        async with RPCClient(endpoint) as client:
            with pytest.raises(RPCError) as error:
                await client.call("get_head_pose")

    assert error.value.code == "unavailable"


@pytest.mark.asyncio
async def test_tracking_function_returns_typed_user_frame(tmp_path: Path) -> None:
    pose = HeadPose(
        is_valid=True,
        position=Vector3(x=1.0, y=1.7, z=2.0),
        forward=Vector3(x=0.0, y=0.0, z=-1.0),
        right=Vector3(x=1.0, y=0.0, z=0.0),
        up=Vector3(x=0.0, y=1.0, z=0.0),
        yaw_deg=0.0,
        pitch_deg=0.0,
        ts=10,
    )

    async def dispatch(operation: str, _arguments: dict) -> dict:
        if operation == "get_head_pose":
            return pose.model_dump(mode="python")
        if operation == "get_health":
            return OpenXRHealth(session_open=True, open_attempts=1).model_dump(mode="python")
        raise AssertionError(operation)

    endpoint = _endpoint(tmp_path)
    async with _running_server(endpoint, dispatch), WorkflowBuilder() as builder:
        await builder.add_function_group(
            "tracking",
            XRTrackingFunctionsConfig(endpoint=endpoint),
        )
        group = await builder.get_function_group("tracking")
        functions = await group.get_all_functions()
        function = functions["tracking__get_user_frame"]
        result = await function.ainvoke({}, to_type=SpatialFrame)

    assert result.origin == Vector3(x=1.0, y=1.7, z=2.0)
    assert function.single_output_schema is SpatialFrame


@pytest.mark.asyncio
async def test_openxr_service_keeps_the_wire_dict_only() -> None:
    pose = {"is_valid": True, "ts": 10}
    health = {"status": "ok", "session_open": True, "open_attempts": 1}

    class Source:
        def get_pose(self) -> dict:
            return pose

        def health(self) -> dict:
            return health

    service = OpenXRService(Source())

    assert await service.dispatch("get_head_pose", {}) is pose
    assert await service.dispatch("get_health", {}) is health
    with pytest.raises(RPCError) as error:
        await service.dispatch("get_head_pose", {"unexpected": True})
    assert error.value.code == "invalid_request"


def _native_pose(quaternion: tuple[float, float, float, float]) -> SimpleNamespace:
    return SimpleNamespace(
        is_valid=True,
        pose=SimpleNamespace(
            position=SimpleNamespace(x=1.0, y=1.7, z=-2.0),
            orientation=SimpleNamespace(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]),
        ),
    )


@pytest.mark.parametrize(
    ("quaternion", "forward", "right", "up", "yaw_deg", "pitch_deg"),
    [
        (
            (0.0, 0.0, 0.0, 1.0),
            {"x": 0.0, "y": 0.0, "z": -1.0},
            {"x": 1.0, "y": 0.0, "z": 0.0},
            {"x": 0.0, "y": 1.0, "z": 0.0},
            0.0,
            0.0,
        ),
        (
            (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)),
            {"x": -1.0, "y": 0.0, "z": 0.0},
            {"x": 0.0, "y": 0.0, "z": -1.0},
            {"x": 0.0, "y": 1.0, "z": 0.0},
            90.0,
            0.0,
        ),
        (
            (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
            {"x": 0.0, "y": 1.0, "z": 0.0},
            {"x": 1.0, "y": 0.0, "z": 0.0},
            {"x": 0.0, "y": 0.0, "z": 1.0},
            0.0,
            90.0,
        ),
        (
            (-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
            {"x": 0.0, "y": -1.0, "z": 0.0},
            {"x": 1.0, "y": 0.0, "z": 0.0},
            {"x": 0.0, "y": 0.0, "z": -1.0},
            0.0,
            -90.0,
        ),
    ],
)
def test_pose_conversion_preserves_openxr_axes(
    quaternion: tuple[float, float, float, float],
    forward: dict[str, float],
    right: dict[str, float],
    up: dict[str, float],
    yaw_deg: float,
    pitch_deg: float,
) -> None:
    result = from_native_pose(_native_pose(quaternion))

    assert result["position"] == {"x": 1.0, "y": 1.7, "z": -2.0}
    assert result["forward"] == forward
    assert result["right"] == right
    assert result["up"] == up
    assert result["yaw_deg"] == yaw_deg
    assert result["pitch_deg"] == pitch_deg
    assert isinstance(result["ts"], int)
