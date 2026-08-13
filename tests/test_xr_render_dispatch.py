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
    VoiceOutput,
)
from xr_render_scene import EmptyRequest

_WORKER_DIR = (
    Path(__file__).resolve().parent.parent
    / "agent-samples"
    / "xr-render-demo"
    / "worker"
)
sys.path.insert(0, str(_WORKER_DIR))

from agent import RenderDemoAgent  # noqa: E402
from dispatch import (  # noqa: E402
    RENDER_NOTICE_TOPIC,
    USER_QUERY_TOPIC,
    RenderAgent,
    RenderNotice,
)


class _Scene:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def handle_query(self, _pid: str, text: str):
        async def response():
            if text == "first":
                self.started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled.set()
                    raise
            yield f"ack:{text}"
            yield f"done:{text}"

        return response()

    async def handle_notice(self, _pid: str, text: str):
        yield text

    def reset_history(self) -> None:
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
        self.final = asyncio.Event()

    @subscribe(VOICE_OUTPUT_TOPIC)
    async def record(self, output: VoiceOutput, ctx: RuntimeContext) -> None:
        self.events.append((output, ctx.metadata))
        if output.final:
            self.final.set()


class _QueryRecorder(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[RenderNotice, MessageMetadata]] = []
        self.received = asyncio.Event()

    @subscribe(RENDER_NOTICE_TOPIC)
    async def record(self, notice: RenderNotice, ctx: RuntimeContext) -> None:
        self.events.append((notice, ctx.metadata))
        self.received.set()


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
    runtime = AgentRuntime()
    runtime.register("xr-render", RenderAgent(scene, ()))  # type: ignore[arg-type]

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


async def test_launch_failure_notice_enters_the_render_notice_topic() -> None:
    runtime = AgentRuntime()
    queries = _QueryRecorder()
    runtime.register("test-query", queries)
    lifecycle = RenderDemoAgent(
        transport=_Transport(),  # type: ignore[arg-type]
        scene_agent=_Scene(),  # type: ignore[arg-type]
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
        await asyncio.wait_for(queries.received.wait(), 1.0)

    notice, metadata = queries.events[0]
    assert metadata.participant_id == "pid-1"
    assert metadata.source == "xr-session"
    assert notice.interrupt_output is True
    assert "Launch XR again" in notice.text
