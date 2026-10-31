# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boundary tests for sample-local SOP composition and reusable checks."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from xr_ai_interaction import ParticipantJoined, ParticipantLeft
from xr_ai_models import DecisionAnswer, DecisionResponse
from xr_ai_runtime import AgentRuntime
from xr_ai_tools import ToolSet
from xr_ai_tools.current_frame import ImageFrame
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import ImageQueryResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-samples/tea-making-sample/worker"))

from tea_making_worker._conversation_context import _context_owner  # noqa: E402
from tea_making_worker._observations import _ObservationPolicy  # noqa: E402
from tea_making_worker._sop_agent import _SOPAgent  # noqa: E402
from tea_making_worker._visual_step import _VisualValidationStep  # noqa: E402
from tea_making_worker.events import PARTICIPANT_JOINED_TOPIC, PARTICIPANT_LEFT_TOPIC  # noqa: E402
from test_tea_making_sample import _activate_test_step, _routed_foreground, _ScriptedDecisions  # noqa: E402


def _sop(decisions=None, **budgets):
    foreground = _routed_foreground(decisions or SimpleNamespace())
    return _SOPAgent(
        workflow=foreground._guidance.workflow,
        llm=foreground._llm, decisions=foreground._decisions,
        images=foreground._images, vlm=foreground._vlm, rag=foreground._rag,
        image_query=SimpleNamespace(),
        application_tools=foreground._background_tools,
        prompt=foreground._prompt, **budgets,
    )


@pytest.mark.asyncio
async def test_one_registered_sop_agent_owns_engine_history_notices_and_cleanup():
    sop = _sop()
    runtime = AgentRuntime()
    runtime.register("sop", sop)
    sop.bind_runtime(runtime)
    async with runtime:
        await runtime.publish(PARTICIPANT_JOINED_TOPIC, ParticipantJoined(), participant_id="alice")
        session = sop._guidance.store.get("alice")
        sop._guidance.store.start(session)
        await sop._guidance._flush(session)
        text = "\n".join(block.text for block in sop._context_blocks("alice"))
        assert "step.enter" in text and "identify" in text
        assert sop._context.blocks("bob") == ()
        sop._context.record_exchange("alice", "What next?", "Show the package.", timestamp_us=1, exchange_id="one")
        assert "Show the package" in "\n".join(block.text for block in sop._context.blocks("alice"))
        await runtime.publish(PARTICIPANT_LEFT_TOPIC, ParticipantLeft(), participant_id="alice")
        assert sop._guidance.store.find("alice") is None
        assert sop._context.blocks("alice") == ()
        # An already-produced old event must not repopulate a departed session.
        from tea_making_worker.workflow_state import WorkflowEvent
        await sop._remember_event(session, WorkflowEvent("step.ready", "identify", "old", {}))
        assert sop._context.blocks("alice") == ()
        assert not sop._guidance._tasks
        await sop.stop()


@pytest.mark.asyncio
async def test_completed_dialogue_reaches_generation_and_choice_without_owning_state():
    decisions = _ScriptedDecisions(("action", "answer"), ("action", "answer"))
    sop = _sop(decisions)
    _activate_test_step(sop._guidance, "fill_water")
    # Establish focus via normal router admission, then ask a referential follow-up.
    decisions.choices.appendleft(("destination", "workflow"))
    before = dict(sop._guidance.store.get("active").state)
    try:
        await sop._answer("What should I do?", "active")
        await sop._answer("Explain that again", "active")
        messages = sop._llm.chat.await_args.args[0]
        assert "What should I do?" in messages[1].content
        assert json.loads(messages[-1].content)["state"]["water_filled"] is False
        assert sop._guidance.store.get("active").state == before
        assert any(state.get("history") for state, _questions in decisions.calls)
    finally:
        await sop.stop()


@pytest.mark.asyncio
async def test_context_compaction_does_not_block_engine_and_is_drained_at_stop():
    sop = _sop(context_budget_bytes=1024, compaction_input_budget_bytes=1024)
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def compact(*_args, **_kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    sop._llm.chat.side_effect = compact
    with _context_owner("active"):
        for index in range(16):
            sop._context.record_exchange(
                "active", "question " * 20, "answer " * 20, timestamp_us=index, exchange_id=index
            )
    await asyncio.wait_for(entered.wait(), 2)
    session = sop._guidance.store.get("active")
    sop._guidance.store.start(session)
    assert session.active and session.step_id == "identify"
    await sop.stop()
    assert cancelled.is_set()
    assert sop._context.blocks("active") == ()


def _visual_fixture(*, observable=True):
    registry = ImageRegistry()
    frame = ImageFrame(
        image=registry.put(b"pixels"), participant_id="alice", track_id="view", sequence=1,
        timestamp_us=10, width=1, height=1,
    )
    policy = _ObservationPolicy.model_validate({
        "kind": "choice", "structured": True,
        "instructions": "Is the latch visibly closed?",
        "criteria": {"accepted": "Closed latch", "rejected": "Open latch", "unknown": "Unclear latch"},
        "updates": {},
    })
    decisions = SimpleNamespace(decide=AsyncMock(return_value=DecisionResponse({
        "evidence": DecisionAnswer("accepted", 1, {"accepted": 1, "rejected": 0, "unknown": 0})
    }, "fake", {}, {})))
    observer = SimpleNamespace(execute=AsyncMock(return_value=ImageQueryResult(text=json.dumps({
        "observable": observable, "description": "The latch is closed." if observable else "Not answerable."
    }))))
    step = _VisualValidationStep(
        question="Describe the latch position.", policy=policy, observer=observer,
        decisions=decisions, llm=SimpleNamespace(), tools=ToolSet(()), timeout_s=1,
    )
    return step, frame, decisions


@pytest.mark.asyncio
async def test_generic_visual_step_has_no_workflow_mutation_or_irrelevant_decision_context():
    step, frame, decisions = _visual_fixture()
    expected = {"unrelated_progress": True, "target": 100}
    result = await step.check(frame, expected=expected)
    assert result.outcome.evidence == "accepted"
    assert result.frame_key == ("view", 1, 10)
    assert expected == {"unrelated_progress": True, "target": 100}
    assert decisions.decide.await_args.args[0] == {"observation": "The latch is closed."}
    assert "100" not in step.observer.execute.await_args.args[0].query


@pytest.mark.asyncio
async def test_unobservable_visual_step_is_unknown_without_a_clef_call():
    step, frame, decisions = _visual_fixture(observable=False)
    result = await step.check(frame, expected={})
    assert result.outcome.evidence == "unknown"
    decisions.decide.assert_not_awaited()


@pytest.mark.asyncio
async def test_replaced_session_cannot_receive_an_inflight_observation():
    sop = _sop()
    _activate_test_step(sop._guidance, "fill_water")
    old = sop._guidance.store.get("active")
    entered, release = asyncio.Event(), asyncio.Event()

    async def decide(*_args, **_kwargs):
        entered.set()
        await release.wait()
        return DecisionResponse({"evidence": DecisionAnswer("accepted", 1, {"accepted": 1})}, "fake", {}, {})

    sop._guidance._decisions = SimpleNamespace(decide=decide)
    task = asyncio.create_task(sop._guidance._observe(
        old, sop._guidance.workflow.step("fill_water"), "Water visible", dict(old.state), old.revision,
    ))
    await entered.wait()
    sop._guidance.store.release("active")
    _activate_test_step(sop._guidance, "fill_water")
    current = sop._guidance.store.get("active")
    release.set()
    assert await task is False
    assert current.evidence_hits == old.evidence_hits == 0
    await sop.stop()


@pytest.mark.asyncio
async def test_departed_participant_query_cannot_recreate_context_or_execution():
    from tea_making_worker.events import USER_QUERY_TOPIC
    from xr_ai_interaction import UserQuery

    sop = _sop()
    runtime = AgentRuntime()
    runtime.register("sop", sop)
    sop.bind_runtime(runtime)
    async with runtime:
        await runtime.publish(PARTICIPANT_JOINED_TOPIC, ParticipantJoined(), participant_id="alice")
        await runtime.publish(PARTICIPANT_LEFT_TOPIC, ParticipantLeft(), participant_id="alice")
        await runtime.publish(USER_QUERY_TOPIC, UserQuery(text="Start again", timestamp_us=1), participant_id="alice")
        assert sop._guidance.store.find("alice") is None
        assert sop._context.blocks("alice") == ()
        assert not sop._tasks
        sop._llm.chat.assert_not_awaited()
        await sop.stop()


@pytest.mark.asyncio
async def test_stop_waits_for_join_before_draining_its_new_monitor():
    sop = _sop()
    entered, release, stopping = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_flush = sop._guidance._flush

    async def flush(session):
        entered.set()
        await release.wait()
        await original_flush(session)

    sop._guidance._flush = flush
    runtime = AgentRuntime()
    runtime.register("sop", sop)
    sop.bind_runtime(runtime)
    async with runtime:
        join = asyncio.create_task(runtime.publish(
            PARTICIPANT_JOINED_TOPIC, ParticipantJoined(), participant_id="alice",
        ))
        await entered.wait()

        async def stop():
            stopping.set()
            await sop.stop()

        shutdown = asyncio.create_task(stop())
        await stopping.wait()
        assert not shutdown.done()
        release.set()
        await join
        await shutdown
        assert not sop._guidance._tasks
        assert not sop._guidance._connected
        assert sop._context.blocks("alice") == ()


@pytest.mark.asyncio
async def test_reusable_visual_check_honors_its_model_deadline_and_drains_cancellation():
    step, frame, decisions = _visual_fixture()
    cancelled = asyncio.Event()

    async def blocked(_request):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    step.observer.execute.side_effect = blocked
    step.timeout_s = 0.01
    with pytest.raises(TimeoutError):
        await step.check(frame, expected={})
    assert cancelled.is_set()
    decisions.decide.assert_not_awaited()
