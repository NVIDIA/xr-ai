# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supervisor turn-lifecycle tests: scene-lock serialization, failure
publishing, and two-turn memory persistence — all over fakes, no LLM."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from xr_ai_tools import Tool
from xr_ai_tools.text_memory import (
    AddTranscriptRequest,
    ConversationEntry,
    RecallConversationRequest,
    RecallConversationResult,
)
from xr_ai_voice import UserQuery
from xr_render_demo_eval import harness
from xr_render_demo_worker.agent import RenderAgent
from xr_render_demo_worker.models import SceneRequest
from xr_render_demo_worker.supervisor import SceneSupervisor


class _RecordingMemory:
    """In-memory text memory: add_transcript really stores, recall really reads."""

    def __init__(self) -> None:
        self.records: list[AddTranscriptRequest] = []
        self.add_transcript = Tool(
            "add_transcript", "Store.", AddTranscriptRequest, None, self._add)
        self.recall_conversation = Tool(
            "recall_conversation", "Recall.", RecallConversationRequest,
            RecallConversationResult, self._recall)

    async def _add(self, req: AddTranscriptRequest) -> None:
        self.records.append(req)

    async def _recall(self, req: RecallConversationRequest) -> RecallConversationResult:
        entries = [
            ConversationEntry(
                timestamp_us=index,
                role=record.source_id.rsplit(":", 1)[1],
                text=record.text,
            )
            for index, record in enumerate(self.records)
            if record.source_id.startswith(f"{req.participant_id}:")
        ]
        return RecallConversationResult(entries=entries)


def _make_supervisor(memory: _RecordingMemory | None = None) -> tuple[SceneSupervisor, harness.FakeScene]:
    fake = harness.FakeScene({}, harness._build_pose(), "", "", "")
    scene, tracking, fake_memory, _frame, _query = fake.make_tools()
    supervisor = SceneSupervisor(
        llm=None,
        scene=scene,
        tracking=tracking,
        text_memory=memory or fake_memory,
        subagent_tools=[],
    )
    return supervisor, fake


async def test_scene_mutations_serialize_across_participants(monkeypatch) -> None:
    """Two participants' turns must not interleave inside the snapshot/mutate/
    verify window: the scene lock admits one turn at a time."""
    supervisor, _fake = _make_supervisor()
    active = 0
    max_active = 0

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.05)
        active -= 1
        return SimpleNamespace(content="All set?", messages=list(messages), tool_calls=())

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    replies = await asyncio.gather(
        supervisor.handle(SceneRequest(transcript="Make it bigger.", participant_id="alice")),
        supervisor.handle(SceneRequest(transcript="Paint it teal.", participant_id="bob")),
    )
    assert [reply.response for reply in replies] == ["All set?", "All set?"]
    assert max_active == 1


async def test_two_turn_memory_without_preseeding(monkeypatch) -> None:
    """A truncated turn is persisted by the supervisor itself, so the next
    turn's completion splices against real recalled memory."""
    memory = _RecordingMemory()
    supervisor, _fake = _make_supervisor(memory)
    seen_user_messages: list[str] = []

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        seen_user_messages.extend(m.content for m in messages if m.role == "user")
        return SimpleNamespace(content="Placed it?", messages=list(messages), tool_calls=())

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    first = await supervisor.handle(
        SceneRequest(transcript="Put the sphere on the", participant_id="alice"))
    assert "what?" in first.response.lower()
    assert [record.source_id for record in memory.records] == ["alice:user", "alice:agent"]

    second = await supervisor.handle(
        SceneRequest(transcript="On the box.", participant_id="alice"))
    assert second.response == "Placed it?"
    assert any(
        "User request: Put the sphere on the box." in content
        for content in seen_user_messages
    )
    assert [record.source_id for record in memory.records] == [
        "alice:user", "alice:agent", "alice:user", "alice:agent",
    ]


async def test_mixed_vision_and_mutation_request_still_verifies(monkeypatch) -> None:
    """A change-requesting utterance that only delegated a non-mutating
    subagent must still get the verification pass."""
    supervisor, _fake = _make_supervisor()
    loop_calls = 0

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        nonlocal loop_calls
        loop_calls += 1
        return SimpleNamespace(
            content="The room looks tidy.",
            messages=list(messages),
            tool_calls=(SimpleNamespace(call=SimpleNamespace(name="vision_agent")),),
        )

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    reply = await supervisor.handle(SceneRequest(
        transcript="Look at the room and create a sphere.", participant_id="alice"))
    assert reply.response == "The room looks tidy."
    assert loop_calls == 2


async def test_failing_supervisor_publishes_failure_notice() -> None:
    """A supervisor crash still produces one complete spoken failure notice,
    never silence."""
    published = []

    class _Ctx:
        metadata = SimpleNamespace(participant_id="alice", message_id="trace-1")

        async def publish(self, topic, value):
            published.append(value)

    class _ExplodingSupervisor:
        async def handle(self, request):
            raise RuntimeError("boom")

    agent = RenderAgent(_ExplodingSupervisor())
    await agent._run_turn(UserQuery(text="Make a sphere.", timestamp_us=1), _Ctx())

    assert len(published) == 1
    assert published[0].text == "Something went wrong. Please try again."
    assert published[0].final is True
    assert published[0].response_id == "trace-1"
