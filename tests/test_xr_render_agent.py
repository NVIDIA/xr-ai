# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-runtime behavior for the xr-render-demo worker."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from pydantic import BaseModel
from xr_ai_runtime import (
    Agent,
    AgentRuntime,
    MessageMetadata,
    RuntimeContext,
    subscribe,
)
from xr_ai_tools import Tool, ToolSet
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
from xr_render_demo_worker.lifecycle import XRSessionLifecycle  # noqa: E402


class _Scene:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.closed = asyncio.Event()
        self.departed: list[str] = []

    async def handle_query(self, _pid: str, text: str):
        async def response():
            try:
                yield f"ack:{text}"
                if text == "first":
                    self.started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        self.cancelled.set()
                        raise
                yield f"done:{text}"
            finally:
                self.closed.set()

        return response()

    async def handle_notice(self, _pid: str, text: str):
        yield text

    def reset_history(self) -> None:
        pass

    async def on_participant_left(self, participant_id: str) -> None:
        self.departed.append(participant_id)


class _MultiScene:
    def __init__(self, participants: set[str]) -> None:
        self._participants = participants
        self.started: set[str] = set()
        self.closed: set[str] = set()
        self.all_started = asyncio.Event()
        self.all_closed = asyncio.Event()

    async def handle_query(self, participant_id: str, _text: str):
        async def response():
            try:
                yield f"ack:{participant_id}"
                self.started.add(participant_id)
                if self.started == self._participants:
                    self.all_started.set()
                await asyncio.Event().wait()
            finally:
                self.closed.add(participant_id)
                if self.closed == self._participants:
                    self.all_closed.set()

        return response()

    async def handle_notice(self, _participant_id: str, text: str):
        yield text

    async def on_participant_left(self, _participant_id: str) -> None:
        pass


class _Endpoint:
    def on_data(self, _callback) -> None:
        pass


class _Transport:
    def __init__(self) -> None:
        self.endpoint = _Endpoint()


class _ToolResult(BaseModel):
    status: str = "ok"


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


async def test_render_agent_publishes_incremental_voice_output() -> None:
    scene = _Scene()
    output = _VoiceRecorder()
    runtime = AgentRuntime()
    runtime.register("xr-render", RenderAgent(scene, ()))  # type: ignore[arg-type]
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
    metadata = [event_metadata for _chunk, event_metadata in output.events]
    assert [chunk.text for chunk in chunks] == ["ack:hello", "done:hello", ""]
    assert [chunk.final for chunk in chunks] == [False, False, True]
    assert [chunk.timestamp_us for chunk in chunks] == [1, 1, 1]
    assert len({chunk.response_id for chunk in chunks}) == 1
    assert chunks[0].response_id == metadata[0].parent_message_id
    assert {event.participant_id for event in metadata} == {"pid-1"}
    assert {event.source for event in metadata} == {"xr-render"}
    assert len({event.correlation_id for event in metadata}) == 1


async def test_render_agent_supersedes_a_participant_turn() -> None:
    scene = _Scene()
    output = _VoiceRecorder()
    runtime = AgentRuntime()
    runtime.register("xr-render", RenderAgent(scene, ()))  # type: ignore[arg-type]
    runtime.register("test-output", output)

    async with runtime:
        await runtime.publish(
            USER_QUERY_TOPIC,
            UserQuery(text="first", timestamp_us=1),
            participant_id="pid-1",
            source="test",
        )
        await asyncio.wait_for(scene.started.wait(), 1.0)
        await runtime.publish(
            USER_QUERY_TOPIC,
            UserQuery(text="second", timestamp_us=2),
            participant_id="pid-1",
            source="test",
        )
        await asyncio.wait_for(scene.cancelled.wait(), 1.0)
        await asyncio.wait_for(output.wait_for(5), 1.0)

    chunks = [chunk for chunk, _metadata in output.events]
    assert [chunk.text for chunk in chunks] == [
        "ack:first",
        "",
        "ack:second",
        "done:second",
        "",
    ]
    assert [chunk.final for chunk in chunks] == [False, True, False, False, True]
    assert chunks[0].response_id == chunks[1].response_id
    assert chunks[2].response_id == chunks[3].response_id == chunks[4].response_id
    assert chunks[0].response_id != chunks[2].response_id


async def test_launch_failure_notice_reaches_voice_with_interrupt_and_terminator() -> None:
    scene = _Scene()
    output = _VoiceRecorder()
    runtime = AgentRuntime()
    runtime.register("xr-render", RenderAgent(scene, ()))  # type: ignore[arg-type]
    runtime.register("test-output", output)
    lifecycle = XRSessionLifecycle(
        transport=_Transport(),  # type: ignore[arg-type]
        scene_loop=scene,  # type: ignore[arg-type]
        tools=ToolSet(
            (
                Tool(
                    "start_xr",
                    "Start XR.",
                    EmptyRequest,
                    _ToolResult,
                    lambda _request: _ToolResult(),
                ),
            )
        ),
        runtime=runtime,
    )

    async with runtime:
        await lifecycle._notify_launch_failed("pid-1")  # noqa: SLF001
        await asyncio.wait_for(output.final.wait(), 1.0)

    chunks = [chunk for chunk, _metadata in output.events]
    metadata = [event_metadata for _chunk, event_metadata in output.events]
    assert len(chunks) == 2
    assert "Launch XR again" in chunks[0].text
    assert chunks[0].interrupt is True
    assert chunks[0].final is False
    assert chunks[1].text == ""
    assert chunks[1].final is True
    assert chunks[0].response_id == chunks[1].response_id
    assert {event.participant_id for event in metadata} == {"pid-1"}


async def test_interruption_closes_scene_without_orphan_terminator() -> None:
    scene = _Scene()
    output = _VoiceRecorder()
    runtime = AgentRuntime()
    runtime.register("xr-render", RenderAgent(scene, ()))  # type: ignore[arg-type]
    runtime.register("test-output", output)

    async with runtime:
        await runtime.publish(
            USER_QUERY_TOPIC,
            UserQuery(text="first", timestamp_us=1),
            participant_id="pid-1",
            source="test",
        )
        await asyncio.wait_for(scene.started.wait(), 1.0)
        output.abort_all()
        await runtime.publish(
            INTERRUPTED_TOPIC,
            VoiceInterrupted(),
            participant_id="pid-1",
            source="voice",
        )
        await asyncio.wait_for(scene.closed.wait(), 1.0)

    assert [chunk.text for chunk, _metadata in output.events] == ["ack:first"]
    assert output.events[0][0].final is False


async def test_participant_departure_cancels_without_terminator_and_releases_state() -> None:
    scene = _Scene()
    output = _VoiceRecorder()
    runtime = AgentRuntime()
    runtime.register("xr-render", RenderAgent(scene, ()))  # type: ignore[arg-type]
    runtime.register("test-output", output)

    async with runtime:
        await runtime.publish(
            USER_QUERY_TOPIC,
            UserQuery(text="first", timestamp_us=1),
            participant_id="pid-1",
            source="test",
        )
        await asyncio.wait_for(scene.started.wait(), 1.0)
        output.abort_all()
        await runtime.publish(
            PARTICIPANT_LEFT_TOPIC,
            VoiceParticipantLeft(),
            participant_id="pid-1",
            source="voice",
        )

    assert scene.departed == ["pid-1"]
    assert [chunk.text for chunk, _metadata in output.events] == ["ack:first"]


async def test_global_interruption_cancels_every_participant_without_terminators() -> None:
    scene = _MultiScene({"alice", "bob"})
    output = _VoiceRecorder()
    runtime = AgentRuntime()
    runtime.register("xr-render", RenderAgent(scene, ()))  # type: ignore[arg-type]
    runtime.register("test-output", output)

    async with runtime:
        for participant_id in ("alice", "bob"):
            await runtime.publish(
                USER_QUERY_TOPIC,
                UserQuery(text="wait", timestamp_us=1),
                participant_id=participant_id,
                source="test",
            )
        await asyncio.wait_for(scene.all_started.wait(), 1.0)
        output.abort_all()
        await runtime.publish(
            INTERRUPTED_TOPIC,
            VoiceInterrupted(),
            participant_id=None,
            source="voice",
        )
        await asyncio.wait_for(scene.all_closed.wait(), 1.0)

    assert {chunk.text for chunk, _metadata in output.events} == {
        "ack:alice",
        "ack:bob",
    }
    assert all(not chunk.final for chunk, _metadata in output.events)


async def test_launch_failure_after_runtime_shutdown_is_ignored() -> None:
    runtime = AgentRuntime()
    lifecycle = XRSessionLifecycle(
        transport=_Transport(),  # type: ignore[arg-type]
        scene_loop=_Scene(),  # type: ignore[arg-type]
        tools=ToolSet(()),
        runtime=runtime,
    )

    await lifecycle._notify_launch_failed("pid-1")  # noqa: SLF001
