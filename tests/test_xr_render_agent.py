# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-runtime behavior for the xr-render-demo worker."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from pydantic import BaseModel
from xr_ai_hub import DataMessage
from xr_ai_runtime import (
    Agent,
    AgentRuntime,
    MessageMetadata,
    RuntimeContext,
    subscribe,
)
from xr_ai_tools import Tool
from xr_ai_voice import (
    VOICE_OUTPUT_TOPIC,
    UserQuery,
    VoiceInterrupted,
    VoiceOutput,
    VoiceParticipantLeft,
)
from xr_render_scene import EmptyRequest

_WORKER_DIR = (
    Path(__file__).resolve().parent.parent
    / "agent-samples"
    / "xr-render-demo"
    / "worker"
)
sys.path.insert(0, str(_WORKER_DIR))

from xr_render_demo_worker.agent import (  # noqa: E402
    INTERRUPTED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    USER_QUERY_TOPIC,
    RenderAgent,
)
from xr_render_demo_worker.models import SceneReply, SceneRequest  # noqa: E402
from xr_render_demo_worker.xr_session import XRSessionController  # noqa: E402


class _StubSupervisor:
    """Supervisor stub that blocks until cancelled, then raises."""

    def __init__(self, reply: str = "ok") -> None:
        self.reply = reply
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def handle(self, request: SceneRequest) -> SceneReply:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return SceneReply(response=self.reply)


class _QuickSupervisor:
    """Supervisor stub that returns immediately."""

    def __init__(self, reply: str = "hello") -> None:
        self.reply = reply

    async def handle(self, request: SceneRequest) -> SceneReply:
        return SceneReply(response=self.reply)


class _MultiStubSupervisor:
    """Supervisor stub that blocks independently per participant."""

    def __init__(self, participants: set[str]) -> None:
        self._participants = participants
        self.started: set[str] = set()
        self.cancelled: set[str] = set()
        self.all_started = asyncio.Event()
        self.all_cancelled = asyncio.Event()

    async def handle(self, request: SceneRequest) -> SceneReply:
        pid = request.participant_id
        self.started.add(pid)
        if self.started == self._participants:
            self.all_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.add(pid)
            if self.cancelled == self._participants:
                self.all_cancelled.set()
            raise
        return SceneReply(response=f"ack:{pid}")


class _Endpoint:
    def __init__(self) -> None:
        self.callback = None

    def on_data(self, callback) -> None:
        self.callback = callback


class _Transport:
    def __init__(self) -> None:
        self.endpoint = _Endpoint()
        self.target: str | None = None
        self.sent = []

    def set_target_participant(self, participant_id: str) -> None:
        self.target = participant_id

    async def send_return_data(self, message) -> None:
        self.sent.append(message)


class _ToolResult(BaseModel):
    status: str = "ok"
    lovr_started: bool = False
    spawn_error: str | None = None


def _tool(name: str, handler=None) -> Tool:
    return Tool(
        name,
        f"{name} test tool.",
        EmptyRequest,
        _ToolResult,
        handler or (lambda _request: _ToolResult()),
    )


class _VoiceRecorder(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[VoiceOutput, MessageMetadata]] = []
        self.open_responses: set[str] = set()
        self.final = asyncio.Event()
        self.changed = asyncio.Event()

    @subscribe(VOICE_OUTPUT_TOPIC)
    async def record(self, output: VoiceOutput, ctx: RuntimeContext) -> None:
        if output.response_id:
            if output.final:
                if output.response_id not in self.open_responses:
                    raise ValueError(f"orphan response terminator: {output.response_id}")
                self.open_responses.discard(output.response_id)
            else:
                self.open_responses.add(output.response_id)
        self.events.append((output, ctx.metadata))
        self.changed.set()
        if output.final:
            self.final.set()

    async def wait_for(self, count: int) -> None:
        while len(self.events) < count:
            self.changed.clear()
            await self.changed.wait()

    def abort_all(self) -> None:
        self.open_responses.clear()


async def test_render_agent_publishes_voice_output() -> None:
    supervisor = _QuickSupervisor(reply="hello")
    output = _VoiceRecorder()
    runtime = AgentRuntime()
    runtime.register("xr-render", RenderAgent(supervisor))
    runtime.register("test-output", output)

    async with runtime:
        await runtime.publish(
            USER_QUERY_TOPIC,
            UserQuery(text="hello", timestamp_us=1),
            participant_id="pid-1",
            source="test",
        )
        await asyncio.wait_for(output.final.wait(), 1.0)

    chunks = [chunk for chunk, _metadata in output.events]
    assert [chunk.text for chunk in chunks] == ["hello", ""]
    assert [chunk.final for chunk in chunks] == [False, True]
    assert [chunk.timestamp_us for chunk in chunks] == [1, 1]
    assert len({chunk.response_id for chunk in chunks}) == 1


async def test_render_agent_supersedes_a_participant_turn() -> None:
    supervisor = _StubSupervisor()
    output = _VoiceRecorder()
    runtime = AgentRuntime()
    runtime.register("xr-render", RenderAgent(supervisor))
    runtime.register("test-output", output)

    async with runtime:
        await runtime.publish(
            USER_QUERY_TOPIC,
            UserQuery(text="first", timestamp_us=1),
            participant_id="pid-1",
            source="test",
        )
        await asyncio.wait_for(supervisor.started.wait(), 1.0)
        await runtime.publish(
            USER_QUERY_TOPIC,
            UserQuery(text="second", timestamp_us=2),
            participant_id="pid-1",
            source="test",
        )
        await asyncio.wait_for(supervisor.cancelled.wait(), 1.0)


async def test_interruption_cancels_participant_turn() -> None:
    supervisor = _StubSupervisor()
    output = _VoiceRecorder()
    runtime = AgentRuntime()
    runtime.register("xr-render", RenderAgent(supervisor))
    runtime.register("test-output", output)

    async with runtime:
        await runtime.publish(
            USER_QUERY_TOPIC,
            UserQuery(text="first", timestamp_us=1),
            participant_id="pid-1",
            source="test",
        )
        await asyncio.wait_for(supervisor.started.wait(), 1.0)
        await runtime.publish(
            INTERRUPTED_TOPIC,
            VoiceInterrupted(),
            participant_id="pid-1",
            source="voice",
        )
        await asyncio.wait_for(supervisor.cancelled.wait(), 1.0)

    assert output.events == []


async def test_participant_departure_cancels_turn_and_fires_callback() -> None:
    supervisor = _StubSupervisor()
    departed: list[str] = []
    output = _VoiceRecorder()
    runtime = AgentRuntime()
    runtime.register("xr-render", RenderAgent(supervisor, on_participant_left=departed.append))
    runtime.register("test-output", output)

    async with runtime:
        await runtime.publish(
            USER_QUERY_TOPIC,
            UserQuery(text="first", timestamp_us=1),
            participant_id="pid-1",
            source="test",
        )
        await asyncio.wait_for(supervisor.started.wait(), 1.0)
        await runtime.publish(
            PARTICIPANT_LEFT_TOPIC,
            VoiceParticipantLeft(),
            participant_id="pid-1",
            source="voice",
        )
        await asyncio.wait_for(supervisor.cancelled.wait(), 1.0)

    assert departed == ["pid-1"]
    assert output.events == []


async def test_global_interruption_cancels_every_participant() -> None:
    participants = {"alice", "bob"}
    supervisor = _MultiStubSupervisor(participants)
    output = _VoiceRecorder()
    runtime = AgentRuntime()
    runtime.register("xr-render", RenderAgent(supervisor))
    runtime.register("test-output", output)

    async with runtime:
        for pid in participants:
            await runtime.publish(
                USER_QUERY_TOPIC,
                UserQuery(text="wait", timestamp_us=1),
                participant_id=pid,
                source="test",
            )
        await asyncio.wait_for(supervisor.all_started.wait(), 1.0)
        await runtime.publish(
            INTERRUPTED_TOPIC,
            VoiceInterrupted(),
            participant_id=None,
            source="voice",
        )
        await asyncio.wait_for(supervisor.all_cancelled.wait(), 1.0)

    assert output.events == []


async def test_lifecycle_starts_xr_waits_for_lovr_and_acknowledges_participant() -> None:
    transport = _Transport()
    controller = XRSessionController(
        transport=transport,  # type: ignore[arg-type]
        start_xr=_tool("start_xr"),
        get_render_health=_tool(
            "get_render_health",
            lambda _request: _ToolResult(lovr_started=True),
        ),
    )
    controller.attach()

    await transport.endpoint.callback(
        DataMessage(
            participant_id="alice",
            topic="xr.session.started",
            pts_us=1,
            data=b"",
        )
    )

    assert transport.target == "alice"
    assert [(m.participant_id, m.topic) for m in transport.sent] == [
        ("alice", "render.ready")
    ]


async def test_lifecycle_waits_until_lovr_reports_started(monkeypatch) -> None:
    calls = 0

    async def health(_request):
        nonlocal calls
        calls += 1
        return _ToolResult(lovr_started=calls == 2)

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    controller = XRSessionController(
        transport=_Transport(),  # type: ignore[arg-type]
        start_xr=_tool("start_xr"),
        get_render_health=_tool("get_render_health", health),
    )

    assert await controller._wait_until_ready(timeout_s=1.0) is True  # noqa: SLF001
    assert calls == 2
