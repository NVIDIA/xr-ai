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
    from xr_render_demo_worker._trace import current_turn_evidence

    supervisor, _fake = _make_supervisor()

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        current_turn_evidence.get().satisfied += 1
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


def test_perception_claim_patterns() -> None:
    """The gate fires on the model's dominant hands-and-body register and
    stays off questions, hedges, memory recall, and look-at answers."""
    from xr_render_demo_worker.supervisor import _claims_perception

    assert _claims_perception("You are holding a camera.")
    assert _claims_perception("You're wearing a red scarf.")
    assert _claims_perception("You’re holding a camera.")
    assert _claims_perception("You are currently holding a camera.")
    assert _claims_perception("You are still holding a camera.")
    assert _claims_perception("You appear to be holding a camera.")
    assert _claims_perception("You seem to be carrying a canvas bag.")
    assert _claims_perception("The bottle in your hand is green.")
    assert _claims_perception("It looks like a camera is in your right hand.")
    assert _claims_perception("You have a stapler in your grasp.")
    assert _claims_perception("You are holding a camera. Want me to recolor it?")

    assert not _claims_perception("You are holding a camera?")
    assert not _claims_perception("What are you holding?")
    assert not _claims_perception("I can't tell what you're holding.")
    assert not _claims_perception("I cannot see what you are holding right now.")
    assert not _claims_perception("I don't know what you are wearing.")
    assert not _claims_perception("The camera is unavailable, so you are holding something I can't name.")
    assert not _claims_perception("You were holding a green mug.")
    assert not _claims_perception("You are looking at the red cube.")
    assert not _claims_perception("You are pointing at the sphere I created.")
    assert not _claims_perception("The sphere is in front of you.")
    assert not _claims_perception("I looked at the camera feed.")
    assert not _claims_perception("")

    assert _claims_perception("I can't see clearly, but you are holding a camera.")
    assert _claims_perception("I can't see the color, you are holding a camera.")
    assert _claims_perception("I can't see the feed and you are holding a camera.")
    assert _claims_perception("I can't see clearly — you are holding a camera.")
    assert _claims_perception("Here's what I found: you are holding a camera.")
    assert _claims_perception("I can't be certain; however you are holding a camera.")
    assert _claims_perception("I can't be sure, yet you are holding a camera.")
    assert _claims_perception("Although I can't see well, you are holding a camera.")
    assert _claims_perception("You are holding a camera, not sure of the color.")

    # Historic-present recorded phrasing still counts as a claim; the gate
    # steers it toward past tense instead of exempting it.
    assert _claims_perception("In the recorded frame, you are holding a purple mug.")
    assert _claims_perception("A moment ago you were near the desk; you are holding a mug.")


async def test_recorded_only_claim_is_steered_then_replaced(monkeypatch) -> None:
    """With only a recorded-frame observation, a present-tense claim gets the
    recorded-moment nudge, and a persisting claim is replaced with the honest
    recorded-only sentence."""
    from xr_render_demo_worker._trace import current_turn_evidence
    from xr_render_demo_worker.supervisor import _RECORDED_ONLY_REPLY

    supervisor, _fake = _make_supervisor()
    nudges: list[str] = []

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        current_turn_evidence.get().observed_recorded = 1
        nudges.extend(m.content for m in messages
                      if m.role == "user" and "past tense" in m.content)
        return SimpleNamespace(
            content="In the recorded frame, you are holding a purple mug.",
            messages=list(messages),
            tool_calls=(),
        )

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    reply = await supervisor.handle(SceneRequest(
        transcript="What was I holding ten seconds ago?", participant_id="alice"))
    assert len(nudges) == 1
    assert reply.response == _RECORDED_ONLY_REPLY


async def test_recorded_claim_rephrased_stands(monkeypatch) -> None:
    """The recorded-moment nudge that yields a past-tense answer ships."""
    from xr_render_demo_worker._trace import current_turn_evidence

    supervisor, _fake = _make_supervisor()
    calls = 0

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        nonlocal calls
        calls += 1
        current_turn_evidence.get().observed_recorded = 1
        if calls == 1:
            return SimpleNamespace(
                content="In the recorded frame, you are holding a purple mug.",
                messages=list(messages),
                tool_calls=(),
            )
        return SimpleNamespace(
            content="Ten seconds ago you were holding a purple mug.",
            messages=list(messages),
            tool_calls=(),
        )

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    reply = await supervisor.handle(SceneRequest(
        transcript="What was I holding ten seconds ago?", participant_id="alice"))
    assert reply.response == "Ten seconds ago you were holding a purple mug."
    assert calls == 2


async def test_observed_empty_retry_keeps_done(monkeypatch) -> None:
    """A retry that observes but returns no text must not be replaced with
    the can't-say sentence; the observation happened."""
    from xr_render_demo_worker._trace import current_turn_evidence

    supervisor, _fake = _make_supervisor()
    calls = 0

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(
                content="You are holding a camera.",
                messages=list(messages),
                tool_calls=(),
            )
        current_turn_evidence.get().observed += 1
        return SimpleNamespace(content="", messages=list(messages), tool_calls=())

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    reply = await supervisor.handle(SceneRequest(
        transcript="What am I holding?", participant_id="alice"))
    assert calls == 2
    assert reply.response == "Done."


async def test_unobserved_claim_is_retried_then_replaced(monkeypatch) -> None:
    """A hands claim with no camera observation is sent back to look once,
    then replaced with the honest can't-say sentence."""
    from xr_render_demo_worker.supervisor import _UNOBSERVED_REPLY

    supervisor, _fake = _make_supervisor()
    nudges: list[str] = []
    iterations: list[int] = []

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        iterations.append(max_iterations)
        nudges.extend(m.content for m in messages
                      if m.role == "user" and "no camera observation" in m.content)
        return SimpleNamespace(
            content="You are holding a camera.",
            messages=list(messages),
            tool_calls=(),
        )

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    reply = await supervisor.handle(SceneRequest(
        transcript="What am I holding?", participant_id="alice"))
    assert len(nudges) == 1
    assert iterations == [12, 6]
    assert reply.response == _UNOBSERVED_REPLY


async def test_observed_claim_stands(monkeypatch) -> None:
    """A camera observation this turn backs the claim, so the reply stands
    and no retry runs."""
    from xr_render_demo_worker._trace import current_turn_evidence

    supervisor, _fake = _make_supervisor()
    calls = 0

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        nonlocal calls
        calls += 1
        current_turn_evidence.get().observed += 1
        return SimpleNamespace(
            content="You are holding a green mug.",
            messages=list(messages),
            tool_calls=(),
        )

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    reply = await supervisor.handle(SceneRequest(
        transcript="What am I holding?", participant_id="alice"))
    assert reply.response == "You are holding a green mug."
    assert calls == 1


async def test_retry_with_observation_stands(monkeypatch) -> None:
    """The forced look supplies the missing evidence, so the retry's answer
    replaces the guess and survives the gate."""
    from xr_render_demo_worker._trace import current_turn_evidence

    supervisor, _fake = _make_supervisor()
    calls = 0

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(
                content="You are holding a camera.",
                messages=list(messages),
                tool_calls=(),
            )
        current_turn_evidence.get().observed += 1
        return SimpleNamespace(
            content="You are holding a boba milk.",
            messages=list(messages),
            tool_calls=(),
        )

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    reply = await supervisor.handle(SceneRequest(
        transcript="What am I holding?", participant_id="alice"))
    assert reply.response == "You are holding a boba milk."
    assert calls == 2


async def test_both_gates_apply_to_final_text(monkeypatch) -> None:
    """A turn that trips the mutation gate and the perception gate keeps
    both verdicts, and the perception retry threads from the latest loop."""
    from xr_render_demo_worker.supervisor import _UNOBSERVED_REPLY

    supervisor, _fake = _make_supervisor()
    calls = 0
    perception_context: list[list[str]] = []

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        nonlocal calls
        calls += 1
        asks = [m.content for m in messages if m.role == "user"]
        if any("no camera observation" in ask for ask in asks):
            perception_context.append(asks)
        return SimpleNamespace(
            content="Recolored the box to match what you are holding.",
            messages=list(messages),
            tool_calls=(SimpleNamespace(call=SimpleNamespace(name="appearance_agent")),),
        )

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    reply = await supervisor.handle(SceneRequest(
        transcript="Make the box the color of the thing I'm holding.", participant_id="alice"))
    assert calls == 3
    assert reply.response == f"{_UNOBSERVED_REPLY} Nothing in the scene was changed."
    assert any("Verified scene changes this turn" in ask for ask in perception_context[0])


async def test_perception_retry_failure_falls_back(monkeypatch) -> None:
    """A retry that never completes still fails honestly rather than
    shipping the unbacked claim."""
    from xr_ai_tools.tool_calling import ToolLoopError
    from xr_render_demo_worker.supervisor import _UNOBSERVED_REPLY

    supervisor, _fake = _make_supervisor()
    calls = 0

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(
                content="You are holding a camera.",
                messages=list(messages),
                tool_calls=(),
            )
        raise ToolLoopError(
            "vision unreachable", messages=(), tool_calls=(), iterations=6
        )

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    reply = await supervisor.handle(SceneRequest(
        transcript="What am I holding?", participant_id="alice"))
    assert reply.response == _UNOBSERVED_REPLY


async def test_empty_perception_retry_falls_back(monkeypatch) -> None:
    """An empty retry reply must not degrade to the generic "Done."."""
    from xr_render_demo_worker.supervisor import _UNOBSERVED_REPLY

    supervisor, _fake = _make_supervisor()
    calls = 0

    async def fake_loop(messages, toolset, call_model, max_iterations=12):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            content="You are holding a camera." if calls == 1 else "",
            messages=list(messages),
            tool_calls=(),
        )

    monkeypatch.setattr("xr_render_demo_worker.supervisor.run_tool_loop", fake_loop)

    reply = await supervisor.handle(SceneRequest(
        transcript="What am I holding?", participant_id="alice"))
    assert reply.response == _UNOBSERVED_REPLY


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
