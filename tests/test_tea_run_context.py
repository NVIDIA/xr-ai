# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run rollover, native context freshness, and conversation integration."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_tea_making_sample import _activate_test_step
from test_tea_sop_agent import _sop
from xr_ai_runtime import AgentRuntime


def _channel(sop, participant, source):
    block = next(block for block in sop._run_contexts.blocks(participant) if block.text.startswith(f"Source {source} "))
    return json.loads(block.text.split("\n", 1)[1])


def _text(blocks):
    return "\n".join(block.text for block in blocks)


@pytest.mark.asyncio
async def test_each_question_gets_fresh_operation_context_without_timer_question_dispatch(monkeypatch):
    import tea_making_worker.sop._tools as tools

    sop = _sop()
    _activate_test_step(sop._guidance, "steep_timer")
    session = sop._guidance.store.get("active")
    session.state["steeping_started_at_us"] = 20_000_000
    try:
        cases = [(32, "How long remains?", 168), (39, "Explain that again", 161), (54, "What is going on?", 146)]
        for now_s, question, remaining in cases:
            monkeypatch.setattr(tools.time, "time_ns", lambda: now_s * 1_000_000_000)
            await sop._generate_answer(question, "active", workflow_context=True)
            state = _channel(sop, "active", "sop.execution")
            assert state["operations"]["active_operation"] == {
                "available": True, "elapsed_s": now_s - 20, "remaining_s": remaining, "expired": False,
            }
            assert "remaining_s" in sop._llm.chat.await_args.args[0][1].content
            assert all(tool.name != "clock__timer" for tool in sop._llm.chat.await_args.kwargs["tools"])
    finally:
        await sop.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["restart", "reset", "complete"])
async def test_run_handoff_preserves_dialogue_and_closes_old_state(operation):
    sop = _sop()
    runtime = AgentRuntime()
    sop.bind_runtime(runtime)
    sop._context.update = AsyncMock(wraps=sop._context.update)
    async with runtime:
        _activate_test_step(sop._guidance, "steep_timer")
        session = sop._guidance.store.get("active")
        old_id = session.run_id
        await sop._guidance._flush(session)
        await sop._refresh_conversation_context("active")
        old_manager = sop._run_contexts._runs["active"].manager
        old_live = sop._run_contexts._runs["active"].live
        sop._remember_exchange("active", "Remember my preference", "I will.", "preference")
        if operation == "restart":
            sop._guidance.store.restart(session)
        elif operation == "reset":
            sop._guidance.store.reset(session)
        else:
            sop._guidance.store.commit(session, {"steeping_complete": True}, "")
            sop._guidance.store.advance(session, skip=False)
        await sop._guidance._flush(session)
        assert old_manager.blocks("active") == ()
        assert old_live.blocks("active") == ()
        history = _text(sop._context.blocks("active"))
        assert "sop.handoff" in history
        handoff = next(
            call.args[2].text for call in sop._context.update.await_args_list if call.args[1] == "sop.handoff"
        )
        assert "HISTORICAL WORKFLOW HANDOFF" in handoff and old_id in handoff
        assert "Remember my preference" in history
        if operation == "restart":
            await sop._refresh_conversation_context("active")
            assert session.run_id != old_id
            state = _channel(sop, "active", "sop.execution")
            assert state["run_id"] == session.run_id
            assert state["operations"] == {}
            assert "steeping_started_at_us" not in state["state"]
        else:
            assert session.run_id is None and not session.active
            assert session.state == sop._guidance.workflow.initial_state()
            assert not sop._run_contexts.blocks("active")
        assert not sop._context.blocks("other")
        await sop.stop()


@pytest.mark.asyncio
async def test_reset_discards_run_observations_and_partial_results_before_next_run():
    sop = _sop()
    _activate_test_step(sop._guidance, "fill_water")
    session = sop._guidance.store.get("active")
    session.evidence["fill_water"] = {"evidence": "unknown", "observed_at_us": 123, "observation": "Obscured"}
    await sop._refresh_conversation_context("active")
    assert _channel(sop, "active", "sop.evidence")["observations"]["fill_water"]["evidence"] == "unknown"
    sop._guidance.store.reset(session)
    sop._guidance.store.start(session)
    await sop._refresh_conversation_context("active")
    assert _channel(sop, "active", "sop.evidence")["observations"] == {}
    await sop.stop()


@pytest.mark.asyncio
async def test_reset_during_context_sampling_retries_without_installing_old_timer_result():
    sop = _sop()
    _activate_test_step(sop._guidance, "steep_timer")
    session = sop._guidance.store.get("active")
    entered, release = asyncio.Event(), asyncio.Event()
    original = sop._guidance._named_tools

    def delayed_tools(session, names):
        from tea_making_worker.sop._tools import TimerRequest, TimerResult
        from xr_ai_tools import Tool, ToolSet

        async def timer(_request):
            entered.set()
            await release.wait()
            return TimerResult(elapsed_s=5, remaining_s=175, expired=False)

        if names == ("clock__timer",):
            return ToolSet((Tool("clock__timer", "Test native projection", TimerRequest, TimerResult, timer),))
        return original(session, names)

    sop._guidance._named_tools = delayed_tools
    refresh = asyncio.create_task(sop._refresh_conversation_context("active"))
    await entered.wait()
    sop._guidance.store.restart(session)
    release.set()
    await refresh
    current = _channel(sop, "active", "sop.execution")
    assert current["step"]["id"] == "identify"
    assert current["operations"] == {}
    await sop.stop()


@pytest.mark.asyncio
async def test_generic_context_projection_uses_yaml_bindings_without_tea_state_names(tmp_path, monkeypatch):
    import tea_making_worker.sop._tools as tools
    import yaml
    from tea_making_worker.sop import load_workflow
    from tea_making_worker.sop._engine import GuidanceAgent

    raw = {
        "task": {"name": "Generic operation", "start_step": "wait"},
        "state": {"begin": {"type": "integer", "initial": 1_000_000},
                  "length": {"type": "integer", "initial": 30}, "done": {"type": "boolean", "initial": False}},
        "steps": [{
            "id": "wait", "reads": ["begin", "length"], "writes": ["done"],
            "trigger": {"function": "clock__timer", "interval_s": 1, "arguments": {}},
            "context": {"process": {"function": "clock__timer", "arguments": {
                "started_at_us": "$state.begin", "duration_s": "$state.length"}}},
            "voice": {"prompt": "Explain progress", "tools": []},
            "observation": {"kind": "result", "match": {"expired": True}, "updates": {"done": True}},
            "complete_when": {"done": True},
        }],
    }
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(raw))
    engine = GuidanceAgent(workflow=load_workflow(path), llm=SimpleNamespace(), decisions=SimpleNamespace(),
                           current_frame=SimpleNamespace(), image_query=SimpleNamespace(), rag=SimpleNamespace())
    engine.store.start(engine.store.get("p"))
    monkeypatch.setattr(tools.time, "time_ns", lambda: 11_000_000_000)
    result = await engine.context_snapshot("p")
    assert result["operations"]["process"]["remaining_s"] == 20
    assert result["state"] == {"begin": 1_000_000, "length": 30, "done": False}
    await engine.stop()


@pytest.mark.asyncio
async def test_question_uses_new_state_when_background_confirmation_changes_revision_during_admission():
    from xr_ai_models import DecisionAnswer, DecisionResponse

    entered, release = asyncio.Event(), asyncio.Event()

    class Decisions:
        async def decide(self, _state, questions, **_kwargs):
            key = next(iter(questions))
            choice = "workflow" if key == "destination" else "answer"
            if key == "action":
                entered.set()
                await release.wait()
            return DecisionResponse({key: DecisionAnswer(choice, 1, {choice: 1})}, "fixture", {}, {})

    sop = _sop(Decisions())
    _activate_test_step(sop._guidance, "heat_water")
    session = sop._guidance.store.get("active")
    question = asyncio.create_task(sop._answer("What is happening?", "active"))
    await entered.wait()
    sop._guidance.store.observe(session, "accepted")
    sop._guidance.store.observe(session, "accepted")
    assert sop._guidance.store.commit(session, {"temperature_ready": True}, "").accepted
    release.set()
    answer, _, _ = await question
    assert answer != sop._policy.stale_message
    payload = json.loads(sop._llm.chat.await_args.args[0][-1].content)
    assert payload["state"]["temperature_ready"] is True
    await sop.stop()


@pytest.mark.asyncio
async def test_after_reset_history_is_available_but_current_execution_is_explicitly_idle():
    sop = _sop()
    _activate_test_step(sop._guidance, "steep_timer")
    await sop._refresh_conversation_context("active")
    session = sop._guidance.store.get("active")
    sop._guidance.store.reset(session)
    await sop._generate_answer("How did it go?", "active")
    messages = sop._llm.chat.await_args.args[0]
    assert json.loads(messages[-1].content)["execution"] == {"active": False, "run_id": None}
    assert not sop._run_contexts.blocks("active")
    assert "sop.handoff" in _text(sop._context.blocks("active"))
    await sop.stop()


@pytest.mark.asyncio
async def test_rollover_drains_old_run_compaction_and_keeps_persistent_dialogue():
    sop = _sop()
    runtime = AgentRuntime()
    sop.bind_runtime(runtime)
    entered, cancelled = asyncio.Event(), asyncio.Event()
    calls = 0

    async def compact(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        first = calls == 1
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            if first:
                cancelled.set()

    sop._llm.chat.side_effect = compact
    async with runtime:
        session = sop._guidance.store.get("p")
        sop._guidance.store.start(session)
        for _ in range(12):
            sop._guidance.store.record(session, "step.ready", "A configured workflow update. " * 12)
        await sop._guidance._flush(session)
        old = sop._run_contexts._runs["p"].manager
        await asyncio.wait_for(entered.wait(), 2)
        sop._remember_exchange("p", "Remember this", "Understood", "note")
        sop._guidance.store.restart(session)
        await sop._guidance._flush(session)
        assert cancelled.is_set()
        assert old.blocks("p") == ()
        assert sop._run_contexts._runs["p"].identity == session.run_id
        assert "Remember this" in _text(sop._context.blocks("p"))
        await sop.stop()


@pytest.mark.asyncio
async def test_finishing_old_frame_cannot_repopulate_new_runs_duplicate_frame_marker():
    from test_tea_making_sample import _decisions
    from xr_ai_tools.current_frame import ImageFrame
    from xr_ai_tools.vision import ImageQueryResult

    sop = _sop()
    engine = sop._guidance
    engine._decisions = _decisions("accepted")
    _activate_test_step(engine, "fill_water")
    session = engine.store.get("active")
    session.evidence_hits = 1
    frame = ImageFrame(
        image=sop._images.images.put(b"frame"), participant_id="active", track_id="view",
        sequence=1, timestamp_us=1, width=1, height=1,
    )
    engine._current_frame = SimpleNamespace(execute=AsyncMock(return_value=frame))
    engine._image_query = SimpleNamespace(execute=AsyncMock(return_value=ImageQueryResult(
        text='{"observable":true,"description":"Visible water inside the vessel."}',
    )))
    entered, release = asyncio.Event(), asyncio.Event()
    observe = engine._observe

    async def finishing(*args, **kwargs):
        result = await observe(*args, **kwargs)
        entered.set()
        await release.wait()
        return result

    engine._observe = finishing
    tick = asyncio.create_task(engine._tick(session))
    await entered.wait()
    assert session.state["water_filled"]
    engine.store.restart(session)
    release.set()
    await tick
    assert session.observed_frame is None
    assert not session.evidence
    await sop.stop()


@pytest.mark.asyncio
async def test_live_channels_are_supplied_separately_from_past_dialogue(monkeypatch):
    import tea_making_worker.sop._tools as tools

    sop = _sop()
    _activate_test_step(sop._guidance, "steep_timer")
    sop._guidance.store.get("active").state["steeping_started_at_us"] = 20_000_000
    sop._remember_exchange("active", "Earlier question", "An old answer: 12 seconds.", "earlier")
    monkeypatch.setattr(tools.time, "time_ns", lambda: 32_000_000_000)
    await sop._generate_answer("What is happening now?", "active", workflow_context=True)
    messages = sop._llm.chat.await_args.args[0]
    history = next(m.content for m in messages if m.content.startswith("HISTORICAL CONTEXT"))
    current = next(m.content for m in messages if m.content.startswith("CURRENT EXECUTION CONTEXT"))
    assert "An old answer: 12 seconds" in history
    assert "An old answer" not in current
    assert "Source sop.execution current state" in current
    assert '"remaining_s": 168' in current
    assert messages.index(next(m for m in messages if m.content == current)) > 1
    await sop.stop()
