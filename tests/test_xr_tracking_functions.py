# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for service RPC and native XR-tracking functions."""

import asyncio
import contextlib
import uuid
from pathlib import Path

import pytest
from nat.builder.workflow_builder import WorkflowBuilder
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
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_rpc_correlates_concurrent_responses(tmp_path: Path) -> None:
    async def dispatch(operation: str, arguments: dict) -> dict:
        await asyncio.sleep(float(arguments["delay"]))
        return {"operation": operation, "value": arguments["value"]}

    endpoint = _endpoint(tmp_path)
    async with _running_server(endpoint, dispatch):
        client = RPCClient(endpoint)
        try:
            slow, fast = await asyncio.gather(
                client.call("echo", {"value": "slow", "delay": 0.05}),
                client.call("echo", {"value": "fast", "delay": 0.0}),
            )
        finally:
            await client.close()

    assert slow["value"] == "slow"
    assert fast["value"] == "fast"


@pytest.mark.asyncio
async def test_rpc_preserves_remote_error_codes(tmp_path: Path) -> None:
    async def dispatch(_operation: str, _arguments: dict) -> dict:
        raise RPCError("tracking unavailable", code="unavailable")

    endpoint = _endpoint(tmp_path)
    async with _running_server(endpoint, dispatch):
        client = RPCClient(endpoint)
        try:
            with pytest.raises(RPCError) as error:
                await client.call("get_head_pose")
        finally:
            await client.close()

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
        timestamp_ms=10,
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
