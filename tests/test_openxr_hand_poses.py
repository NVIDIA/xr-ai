# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hand poses from the OpenXR service through the tracking tools."""

import asyncio
import contextlib
import uuid
from types import SimpleNamespace

import pytest
from openxr_service.pose import HAND_JOINTS, from_native_hands, unavailable_hands
from openxr_service.service import OpenXRService
from xr_ai_tools.rpc import RPCServer
from xr_ai_tools.tracking import HandPoses, TrackingTools
from xr_ai_tools.types import EmptyRequest


def _joints(valid: set[str]) -> list:
    return [
        (SimpleNamespace(x=index * 0.1, y=1.0, z=-0.5), name in valid)
        for index, name in enumerate(HAND_JOINTS)
    ]


def test_hand_joint_names_cover_the_default_joint_set_in_order():
    assert len(HAND_JOINTS) == 26
    assert (HAND_JOINTS[0], HAND_JOINTS[1], HAND_JOINTS[5], HAND_JOINTS[25]) == (
        "palm", "wrist", "thumb_tip", "little_tip",
    )


def test_only_valid_joints_of_tracked_hands_are_reported():
    hands = from_native_hands({"left": _joints({"palm", "index_tip"}), "right": None})

    assert hands["left"] == {
        "is_tracked": True,
        "joints": {
            "palm": {"x": 0.0, "y": 1.0, "z": -0.5},
            "index_tip": {"x": 1.0, "y": 1.0, "z": -0.5},
        },
    }
    assert hands["right"] == {"is_tracked": False, "joints": {}}
    assert hands["error"] is None


def test_runtime_without_hand_tracking_reports_why():
    hands = from_native_hands({})

    assert not hands["left"]["is_tracked"] and not hands["right"]["is_tracked"]
    assert "not supported" in hands["error"]


class _Source:
    def __init__(self, hands):
        self._hands = hands

    def get_pose(self):
        raise AssertionError("not used")

    def get_hands(self):
        return self._hands

    def health(self):
        return {}


@contextlib.asynccontextmanager
async def _serving(hands):
    endpoint = f"ipc:///tmp/xr-{uuid.uuid4().hex}"
    server = RPCServer(endpoint, OpenXRService(_Source(hands)).dispatch)
    task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.02)
    tools = TrackingTools(endpoint)
    try:
        yield tools
    finally:
        await tools.close()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_hand_tool_returns_typed_hands():
    async with _serving(from_native_hands({"left": _joints({"wrist"}), "right": None})) as tools:
        hands = await tools.get_hand_poses.execute(EmptyRequest())

    assert isinstance(hands, HandPoses)
    assert hands.left.is_tracked and set(hands.left.joints) == {"wrist"}
    assert not hands.right.is_tracked


@pytest.mark.asyncio
async def test_hand_tool_raises_while_the_session_is_not_ready():
    async with _serving(unavailable_hands("no headset")) as tools:
        with pytest.raises(RuntimeError, match="no headset"):
            await tools.get_hand_poses.execute(EmptyRequest())
