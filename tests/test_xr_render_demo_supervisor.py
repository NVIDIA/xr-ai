# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supervisor turn-lifecycle tests: scene-lock serialization, failure
publishing, memory persistence, and vision-reply redaction, all over fakes
(plus the real transcript store for restart survival), no LLM."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from xr_ai_tools import Tool
from xr_ai_tools.text_memory import (
    AddTranscriptRequest,
    ConversationEntry,
    QueryTranscriptsRequest,
    QueryTranscriptsResult,
    RecallConversationRequest,
    RecallConversationResult,
    TranscriptSegment,
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
        self.query_transcripts = Tool(
            "query_transcripts", "Query.", QueryTranscriptsRequest,
            QueryTranscriptsResult, self._query)

    async def _add(self, req: AddTranscriptRequest) -> None:
        self.records.append(req)

    async def _recall(self, req: RecallConversationRequest) -> RecallConversationResult:
        # Mirrors the real text_memory._recall contract: exactly the :user
        # and :agent sources with stored timestamps; append order is
        # chronological here.
        entries = [
            ConversationEntry(
                timestamp_us=record.timestamp_us,
                role=record.source_id.rsplit(":", 1)[1],
                text=record.text,
            )
            for record in self.records
            if record.source_id in (f"{req.participant_id}:user", f"{req.participant_id}:agent")
        ]
        return RecallConversationResult(entries=entries)

    async def _query(self, req: QueryTranscriptsRequest) -> QueryTranscriptsResult:
        segments = [
            TranscriptSegment(timestamp_us=record.timestamp_us, text=record.text)
            for record in self.records
            if record.source_id == req.source_id
            and req.start_us <= record.timestamp_us <= req.end_us
        ]
        return QueryTranscriptsResult(segments=segments)


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
    # The creation half never happened: the claim-free vision answer keeps
    # its content and gains the no-change fact.
    assert reply.response.startswith("The room looks tidy.")
    assert reply.response.endswith("Nothing in the scene was changed.")
    assert loop_calls == 2


async def test_verification_never_offers_repeat_when_mutation_undelegated(monkeypatch) -> None:
    """A change-requesting utterance whose first pass delegated no mutating
    subagent must get a verification nudge with no repeat-your-answer out."""
    supervisor, _fake = _make_supervisor()
    nudges: list[str] = []

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        nudges.extend(m.content for m in messages if m.role == "user" and "Verified scene" in m.content)
        return SimpleNamespace(
            content="Recolored it to match the wall.",
            messages=list(messages),
            tool_calls=(),
        )

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    reply = await supervisor.handle(SceneRequest(
        transcript="Make the sphere the same color as the wall.", participant_id="alice"))
    assert len(nudges) == 1
    assert "repeat your final answer" not in nudges[0]
    assert "never" in nudges[0]
    # The model repeated its false success anyway; the deterministic guard
    # must replace it, and the scene must be untouched.
    assert "Recolored" not in reply.response
    assert "nothing in the scene was changed" in reply.response
    assert _fake.objects == {}


async def test_gate_keeps_claim_free_explanations(monkeypatch) -> None:
    """A claim-free failure explanation keeps its why; the no-change fact is
    appended, never substituted."""
    supervisor, _fake = _make_supervisor()

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        return SimpleNamespace(
            content="The camera is unavailable, so I could not read your shirt color.",
            messages=list(messages),
            tool_calls=(),
        )

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    reply = await supervisor.handle(SceneRequest(
        transcript="Make the box the color of my shirt.", participant_id="alice"))
    assert reply.response.startswith("The camera is unavailable")
    assert reply.response.endswith("Nothing in the scene was changed.")


def test_status_questions_are_not_mutation_intent() -> None:
    from xr_render_demo_worker.supervisor import _wants_mutation

    assert not _wants_mutation("Did you move the cube?")
    assert not _wants_mutation("Have you added the sphere yet?")
    assert not _wants_mutation("Was the cube removed?")
    assert _wants_mutation("Can you move the cube?")
    assert _wants_mutation("Move the cube.")


async def test_already_satisfied_reply_stands_on_evidence(monkeypatch) -> None:
    """A recolor that found the requested state already holding records
    satisfied evidence; the model's reply stands despite no scene diff."""
    from xr_render_demo_worker._trace import current_mutation_evidence

    supervisor, _fake = _make_supervisor()

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        current_mutation_evidence.get().satisfied += 1
        return SimpleNamespace(
            content="The sphere is already green.",
            messages=list(messages),
            tool_calls=(SimpleNamespace(call=SimpleNamespace(name="appearance_agent")),),
        )

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    reply = await supervisor.handle(SceneRequest(
        transcript="Make the sphere green.", participant_id="alice"))
    assert reply.response == "The sphere is already green."


async def test_rejected_mutation_without_evidence_gets_honest_reply(monkeypatch) -> None:
    """A delegated mutating agent whose tool call was rejected leaves no
    evidence; a non-question reply is replaced with the honest text."""
    supervisor, _fake = _make_supervisor()

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        return SimpleNamespace(
            content="Recolored the box to match your shirt.",
            messages=list(messages),
            tool_calls=(SimpleNamespace(call=SimpleNamespace(name="appearance_agent")),),
        )

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    reply = await supervisor.handle(SceneRequest(
        transcript="Make the box the color of my shirt.", participant_id="alice"))
    assert "nothing in the scene was changed" in reply.response


async def test_verification_preserves_clarifying_questions(monkeypatch) -> None:
    """A question reply after a no-change verification pass is a legitimate
    ask-back, not an unsupported claim."""
    supervisor, _fake = _make_supervisor()

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        return SimpleNamespace(
            content="What color is the wall?",
            messages=list(messages),
            tool_calls=(),
        )

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    reply = await supervisor.handle(SceneRequest(
        transcript="Make the sphere the same color as the wall.", participant_id="alice"))
    assert reply.response == "What color is the wall?"


async def test_verification_offers_repeat_after_mutating_delegation(monkeypatch) -> None:
    """When a mutating subagent did run and reported no change, the nudge
    keeps the repeat-answer path for honest failure replies."""
    supervisor, _fake = _make_supervisor()
    nudges: list[str] = []

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        nudges.extend(m.content for m in messages if m.role == "user" and "Verified scene" in m.content)
        return SimpleNamespace(
            content="I couldn't find that object.",
            messages=list(messages),
            tool_calls=(SimpleNamespace(call=SimpleNamespace(name="appearance_agent")),),
        )

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    await supervisor.handle(SceneRequest(
        transcript="Paint the sphere crimson.", participant_id="alice"))
    assert len(nudges) == 1
    assert "repeat your final answer" in nudges[0]


async def test_turn_tasks_run_in_forked_relay_context(monkeypatch) -> None:
    """Detached turn tasks must not inherit the subscriber's live Relay scope
    stack; each turn gets a forked context."""
    import nemo_relay
    from xr_render_demo_worker import agent as agent_module

    forks = 0
    real_fork = nemo_relay.fork_asyncio_context

    def counting_fork():
        nonlocal forks
        forks += 1
        return real_fork()

    monkeypatch.setattr(agent_module.nemo_relay, "fork_asyncio_context", counting_fork)

    class _QuickSupervisor:
        async def handle(self, request):
            return SimpleNamespace(response="ok")

    published = []

    class _Ctx:
        def __init__(self, pid):
            self.metadata = SimpleNamespace(participant_id=pid, message_id=f"t-{pid}")

        async def publish(self, topic, value):
            published.append(value)

    agent = RenderAgent(_QuickSupervisor())
    await agent.answer_user(UserQuery(text="hi", timestamp_us=1), _Ctx("alice"))
    await agent.answer_user(UserQuery(text="hi", timestamp_us=1), _Ctx("bob"))
    for _ in range(200):
        if len(published) == 2:
            break
        await asyncio.sleep(0.01)
    assert forks == 2
    assert len(published) == 2


async def test_supervisor_eval_fails_on_exception_after_delegation(monkeypatch) -> None:
    """The routing tier must FAIL a case whose workflow raises even when the
    expected agent was already delegated before the exception."""
    from xr_render_demo_eval import supervisor as eval_supervisor
    from xr_render_demo_worker.models import SubagentTask

    case = next(c for c in eval_supervisor.CASES if c.expect_agent)

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        tool = toolset.get(case.expect_agent)
        await tool.execute(SubagentTask(instruction="do the thing"))
        raise RuntimeError("boom after delegation")

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)
    assert await eval_supervisor.run_case(case) is False


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


async def test_vision_reply_redacted_from_inline_context(monkeypatch) -> None:
    """A vision turn's reply is redacted from the next turn's inline history
    while text memory keeps the original for explicit recall."""
    memory = _RecordingMemory()
    supervisor, _fake = _make_supervisor(memory)
    contexts: list[str] = []
    turn = 0

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        nonlocal turn
        turn += 1
        contexts.append(messages[1].content)
        if turn == 1:
            return SimpleNamespace(
                content="You are holding a red notebook.",
                messages=list(messages),
                tool_calls=(SimpleNamespace(call=SimpleNamespace(name="vision_agent")),),
            )
        return SimpleNamespace(
            content="You are holding a blue mug.",
            messages=list(messages),
            tool_calls=(SimpleNamespace(call=SimpleNamespace(name="vision_agent")),),
        )

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    await supervisor.handle(SceneRequest(
        transcript="What am I holding?", participant_id="alice"))
    reply = await supervisor.handle(SceneRequest(
        transcript="What am I holding now?", participant_id="alice"))

    assert "notebook" not in contexts[1]
    assert "[reported what the camera showed at that moment]" in contexts[1]
    assert reply.response == "You are holding a blue mug."
    recalled = await memory.recall_conversation.execute(
        RecallConversationRequest(participant_id="alice"))
    assert any(
        entry.role == "agent" and "notebook" in entry.text for entry in recalled.entries
    )


async def test_vision_redaction_survives_supervisor_restart(monkeypatch, tmp_path) -> None:
    """Provenance lives in the transcript store, so a recycled worker still
    redacts sightings recorded by its predecessor."""
    from xr_ai_tools.text_memory import TextMemoryTools

    memory = TextMemoryTools(tmp_path)
    first, _fake1 = _make_supervisor(memory)
    contexts: list[str] = []
    turn = 0

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        nonlocal turn
        turn += 1
        contexts.append(messages[1].content)
        if turn == 1:
            return SimpleNamespace(
                content="You are holding a red notebook.",
                messages=list(messages),
                tool_calls=(SimpleNamespace(call=SimpleNamespace(name="vision_agent")),),
            )
        return SimpleNamespace(
            content="You are holding a blue mug.",
            messages=list(messages),
            tool_calls=(SimpleNamespace(call=SimpleNamespace(name="vision_agent")),),
        )

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    await first.handle(SceneRequest(
        transcript="What am I holding?", participant_id="alice", timestamp_us=1_000_000))

    second, _fake2 = _make_supervisor(memory)
    reply = await second.handle(SceneRequest(
        transcript="What am I holding now?", participant_id="alice", timestamp_us=2_000_000))

    assert "notebook" not in contexts[1]
    assert "[reported what the camera showed at that moment]" in contexts[1]
    assert reply.response == "You are holding a blue mug."


async def test_non_vision_replies_survive_redaction(monkeypatch) -> None:
    """Only vision turns are redacted: a non-vision reply shows verbatim in
    later inline history, and one participant's sightings never touch
    another's entries."""
    memory = _RecordingMemory()
    supervisor, _fake = _make_supervisor(memory)
    contexts: list[str] = []
    scripted = [
        ("You are holding a red notebook.", True),
        ("The scene has one sphere.", False),
        ("You are holding a red notebook.", False),
        ("Okay.", False),
        ("Okay.", False),
    ]
    turn = 0

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        nonlocal turn
        contexts.append(messages[1].content)
        content, vision = scripted[turn]
        turn += 1
        calls = (SimpleNamespace(call=SimpleNamespace(name="vision_agent")),) if vision else ()
        return SimpleNamespace(content=content, messages=list(messages), tool_calls=calls)

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    await supervisor.handle(SceneRequest(
        transcript="What am I holding?", participant_id="alice"))
    await supervisor.handle(SceneRequest(
        transcript="Describe the scene.", participant_id="alice"))
    await supervisor.handle(SceneRequest(
        transcript="Guess what I am holding.", participant_id="bob"))
    await supervisor.handle(SceneRequest(
        transcript="Thanks.", participant_id="alice"))
    await supervisor.handle(SceneRequest(
        transcript="Thanks.", participant_id="bob"))

    alice_context = contexts[3]
    assert "[reported what the camera showed at that moment]" in alice_context
    assert "The scene has one sphere." in alice_context
    assert "notebook" not in alice_context
    # bob's identical reply was produced without a look: alice's tag must not
    # redact it in bob's history.
    bob_context = contexts[4]
    assert "You are holding a red notebook." in bob_context


async def test_untagged_legacy_history_does_not_break_turns(monkeypatch, tmp_path) -> None:
    """An agent-vision tag whose timestamp matches no agent entry leaves
    that history inline unredacted; the turn itself proceeds normally."""
    from xr_ai_tools.text_memory import TextMemoryTools

    memory = TextMemoryTools(tmp_path)
    await memory.add_transcript.execute(AddTranscriptRequest(
        source_id="alice:user", timestamp_us=1_000_000, text="What am I holding?"))
    await memory.add_transcript.execute(AddTranscriptRequest(
        source_id="alice:agent", timestamp_us=2_000_000,
        text="You are holding a red notebook."))
    await memory.add_transcript.execute(AddTranscriptRequest(
        source_id="alice:agent-vision", timestamp_us=1_000_000,
        text="You are holding a red notebook."))

    supervisor, _fake = _make_supervisor(memory)
    contexts: list[str] = []

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        contexts.append(messages[1].content)
        return SimpleNamespace(
            content="You are holding a blue mug.",
            messages=list(messages),
            tool_calls=(SimpleNamespace(call=SimpleNamespace(name="vision_agent")),),
        )

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    reply = await supervisor.handle(SceneRequest(
        transcript="What am I holding now?", participant_id="alice", timestamp_us=3_000_000))

    assert reply.response == "You are holding a blue mug."
    assert "You are holding a red notebook." in contexts[0]
    assert "What am I holding?" in contexts[0]


async def test_retry_loop_vision_reply_is_tagged(monkeypatch) -> None:
    """A sighting produced by the verification retry is tagged like one from
    the first loop."""
    from xr_render_demo_worker._trace import current_mutation_evidence

    memory = _RecordingMemory()
    supervisor, _fake = _make_supervisor(memory)
    calls = 0

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(content="Checking.", messages=list(messages), tool_calls=())
        current_mutation_evidence.get().satisfied += 1
        return SimpleNamespace(
            content="You are holding a mug.",
            messages=list(messages),
            tool_calls=(SimpleNamespace(call=SimpleNamespace(name="vision_agent")),),
        )

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    await supervisor.handle(SceneRequest(
        transcript="Change the wall and tell me what I am holding.", participant_id="alice"))
    assert calls == 2
    assert any(r.source_id == "alice:agent-vision" for r in memory.records)


async def test_replaced_reply_is_not_tagged(monkeypatch) -> None:
    """A canned no-change replacement is not a sighting, even when vision ran;
    it must show verbatim in the next turn's history."""
    memory = _RecordingMemory()
    supervisor, _fake = _make_supervisor(memory)
    contexts: list[str] = []

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        contexts.append(messages[1].content)
        return SimpleNamespace(
            content="Recolored the wall to match what you are holding.",
            messages=list(messages),
            tool_calls=(SimpleNamespace(call=SimpleNamespace(name="vision_agent")),),
        )

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    await supervisor.handle(SceneRequest(
        transcript="Recolor the wall to match my mug.", participant_id="alice"))
    assert not any(r.source_id == "alice:agent-vision" for r in memory.records)

    await supervisor.handle(SceneRequest(
        transcript="Did anything change?", participant_id="alice"))
    assert "I couldn't make that change" in contexts[-1]


async def test_mixed_vision_and_mutation_reply_stays_inline(monkeypatch) -> None:
    """A turn that both looked and mutated keeps its reply verbatim in
    history; redacting it would erase the confirmation later turns resolve
    references against."""
    from xr_render_demo_worker._trace import current_mutation_evidence

    memory = _RecordingMemory()
    supervisor, _fake = _make_supervisor(memory)
    contexts: list[str] = []

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        contexts.append(messages[1].content)
        current_mutation_evidence.get().satisfied += 1
        return SimpleNamespace(
            content="Added a cone; you are holding a mug.",
            messages=list(messages),
            tool_calls=(
                SimpleNamespace(call=SimpleNamespace(name="vision_agent")),
                SimpleNamespace(call=SimpleNamespace(name="object_agent")),
            ),
        )

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    await supervisor.handle(SceneRequest(
        transcript="Add a cone and tell me what I am holding.", participant_id="alice"))
    assert not any(r.source_id == "alice:agent-vision" for r in memory.records)

    await supervisor.handle(SceneRequest(
        transcript="Make it bigger.", participant_id="alice"))
    assert "Added a cone; you are holding a mug." in contexts[-1]


async def test_same_participant_duplicate_text_not_cross_redacted(monkeypatch) -> None:
    """Timestamp keying: a later non-vision reply with text identical to a
    tagged sighting stays inline verbatim."""
    memory = _RecordingMemory()
    supervisor, _fake = _make_supervisor(memory)
    contexts: list[str] = []
    scripted = [
        ("You are holding a red notebook.", True),
        ("You are holding a red notebook.", False),
        ("Okay.", False),
    ]
    turn = 0

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        nonlocal turn
        contexts.append(messages[1].content)
        content, vision = scripted[turn]
        turn += 1
        calls = (SimpleNamespace(call=SimpleNamespace(name="vision_agent")),) if vision else ()
        return SimpleNamespace(content=content, messages=list(messages), tool_calls=calls)

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    await supervisor.handle(SceneRequest(
        transcript="What am I holding?", participant_id="alice"))
    await supervisor.handle(SceneRequest(
        transcript="Say that again.", participant_id="alice"))
    await supervisor.handle(SceneRequest(
        transcript="Thanks.", participant_id="alice"))

    final_context = contexts[-1]
    assert final_context.count("You are holding a red notebook.") == 1
    assert final_context.count("[reported what the camera showed at that moment]") == 1


async def test_vision_plus_memory_reply_stays_inline(monkeypatch) -> None:
    """A turn that also recalled memory keeps its reply verbatim; redaction
    would erase stable recalled facts along with the sighting."""
    memory = _RecordingMemory()
    supervisor, _fake = _make_supervisor(memory)

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        return SimpleNamespace(
            content="You asked for a mug earlier; you are holding one now.",
            messages=list(messages),
            tool_calls=(
                SimpleNamespace(call=SimpleNamespace(name="vision_agent")),
                SimpleNamespace(call=SimpleNamespace(name="memory_agent")),
            ),
        )

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    await supervisor.handle(SceneRequest(
        transcript="Am I holding what I asked for?", participant_id="alice"))
    assert not any(r.source_id == "alice:agent-vision" for r in memory.records)


async def test_tag_lookup_failure_redacts_all_agent_history(monkeypatch) -> None:
    """With provenance unknown, every agent entry is treated as a possible
    stale sighting."""
    memory = _RecordingMemory()
    supervisor, _fake = _make_supervisor(memory)
    contexts: list[str] = []

    async def failing_query(req):
        raise RuntimeError("store offline")

    memory.query_transcripts = Tool(
        "query_transcripts", "Query.", QueryTranscriptsRequest,
        QueryTranscriptsResult, failing_query)

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        contexts.append(messages[1].content)
        return SimpleNamespace(content="Hello there.", messages=list(messages), tool_calls=())

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    await supervisor.handle(SceneRequest(transcript="Hi.", participant_id="alice"))
    await supervisor.handle(SceneRequest(transcript="Hi again.", participant_id="alice"))

    assert "Hello there." not in contexts[1]
    assert "[reported what the camera showed at that moment]" in contexts[1]
    assert "Hi." in contexts[1]


async def test_failed_tag_write_skips_reply_persist(monkeypatch) -> None:
    """A sighting whose tag write fails is dropped from recall entirely; an
    untagged sighting must never land in the :agent source."""
    memory = _RecordingMemory()
    original_add = memory._add

    async def add(req: AddTranscriptRequest) -> None:
        if req.source_id.endswith(":agent-vision"):
            raise RuntimeError("store offline")
        await original_add(req)

    memory.add_transcript = Tool("add_transcript", "Store.", AddTranscriptRequest, None, add)
    supervisor, _fake = _make_supervisor(memory)

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        return SimpleNamespace(
            content="You are holding a mug.",
            messages=list(messages),
            tool_calls=(SimpleNamespace(call=SimpleNamespace(name="vision_agent")),),
        )

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    reply = await supervisor.handle(SceneRequest(
        transcript="What am I holding?", participant_id="alice"))
    assert reply.response == "You are holding a mug."
    assert [r.source_id for r in memory.records] == ["alice:user"]


def test_clear_transcript_artifacts_leaves_unrelated_files(tmp_path) -> None:
    """Startup cleanup removes only the store's own files."""
    from xr_render_demo_worker.app import _clear_transcript_artifacts

    (tmp_path / "alice_agent.jsonl").write_text("{}")
    (tmp_path / "alice_agent.identity").write_text("alice:agent")
    (tmp_path / "notes.txt").write_text("keep me")
    _clear_transcript_artifacts(tmp_path)
    assert not (tmp_path / "alice_agent.jsonl").exists()
    assert not (tmp_path / "alice_agent.identity").exists()
    assert (tmp_path / "notes.txt").read_text() == "keep me"
