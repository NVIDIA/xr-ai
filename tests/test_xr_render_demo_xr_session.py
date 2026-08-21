# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""XR session startup failure handling: expected scene RPC and launch
failures must notify the participant, never escape the hub callback."""
from __future__ import annotations

import time
from types import SimpleNamespace

from xr_ai_hub import DataMessage
from xr_render_demo_worker.xr_session import XRSessionController
from xr_render_scene.schemas import SceneHealth, StartXRResult


class _Transport:
    def __init__(self) -> None:
        self.sent: list[DataMessage] = []
        self.endpoint = SimpleNamespace(on_data=lambda callback: None)

    def set_target_participant(self, participant_id: str) -> None:
        pass

    async def send_return_data(self, message: DataMessage) -> None:
        self.sent.append(message)


def _tool(handler):
    return SimpleNamespace(execute=handler)


def _start_event() -> DataMessage:
    return DataMessage(
        participant_id="alice", topic="xr.session.started",
        pts_us=time.time_ns() // 1_000, data=b"")


def _controller(start_xr, get_health) -> tuple[XRSessionController, _Transport]:
    transport = _Transport()
    controller = XRSessionController(
        transport=transport, start_xr=_tool(start_xr), get_render_health=_tool(get_health))
    return controller, transport


async def test_start_rpc_exception_notifies_failure() -> None:
    async def start(_request):
        raise RuntimeError("rpc timeout")

    async def health(_request):
        raise AssertionError("health must not be polled after a failed start")

    controller, transport = _controller(start, health)
    await controller._on_data(_start_event())

    assert controller.started is False
    assert [message.topic for message in transport.sent] == ["render.failed"]


async def test_start_error_result_notifies_failure() -> None:
    async def start(_request):
        return StartXRResult(status="error", error="lovr missing")

    async def health(_request):
        raise AssertionError("health must not be polled after an error result")

    controller, transport = _controller(start, health)
    await controller._on_data(_start_event())

    assert controller.started is False
    assert [message.topic for message in transport.sent] == ["render.failed"]


async def test_spawn_error_notifies_failure() -> None:
    async def start(_request):
        return StartXRResult(status="started")

    async def health(_request):
        return SceneHealth(lovr_started=False, spawn_error="exec failed", render_drops=0)

    controller, transport = _controller(start, health)
    await controller._on_data(_start_event())

    assert controller.started is False
    assert [message.topic for message in transport.sent] == ["render.failed"]


async def test_successful_start_notifies_ready() -> None:
    async def start(_request):
        return StartXRResult(status="started")

    async def health(_request):
        return SceneHealth(lovr_started=True, render_drops=0)

    controller, transport = _controller(start, health)
    await controller._on_data(_start_event())

    assert controller.started is True
    assert [message.topic for message in transport.sent] == ["render.ready"]
