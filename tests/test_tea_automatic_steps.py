# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target verification, native progression policy, and visual context retention."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-samples/tea-making-sample/worker"))

from tea_making_worker.agents.sop._state import WorkflowStore  # noqa: E402
from tea_making_worker.sop import load_workflow  # noqa: E402
from test_tea_making_sample import _SAMPLE, _decisions, _identity_guidance  # noqa: E402


def _engine(mode="automatic"):
    quote = "White tea: brew at 82 C for 3 minutes."
    engine = _identity_guidance([{
        "values": {"target_temperature_c": {"value": 82, "unit": "celsius"},
                   "steep_duration_s": {"value": 3, "unit": "minutes"}},
        "sources": {name: {"passage": 0, "quote": quote} for name in ("target_temperature_c", "steep_duration_s")},
    }])
    workflow = load_workflow(_SAMPLE / "yaml/workflow.yaml")
    if mode != "automatic":
        workflow = replace(workflow, advancement=mode)
    engine.workflow = workflow
    engine.store = WorkflowStore(workflow)
    engine._decisions = _decisions("accepted", "accepted", "accepted", "accepted")
    return engine


async def _see(engine, session, observation, sequence):
    step = engine.workflow.step(session.step_id)
    return await engine._observe(
        session, step, observation, dict(session.state), session.revision,
        frame_key=(step.id, "camera", sequence, sequence * 1000),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["automatic", "manual"])
async def test_complete_procedure_supports_auto_default_and_explicit_manual_mode(mode, monkeypatch):
    import tea_making_worker.agents.sop._tools as tools

    engine = _engine(mode)
    session = engine.store.get("p")
    engine.store.start(session)
    run_id = session.run_id
    await _see(engine, session, '{"tea_name":"Northwind White Tea"}', 1)
    assert session.state["target_temperature_c"] == 82
    assert session.step_id == ("fill_water" if mode == "automatic" else "identify")
    if mode == "manual":
        assert "Say next when ready" in engine.store.drain_notices(session)[0]
        engine.store.advance(session, skip=False)
    else:
        assert "Say next" not in engine.store.drain_notices(session)[0]
    await _see(engine, session, "Water surface visibly inside vessel.", 2)
    assert session.step_id == "fill_water"
    await _see(engine, session, "Water surface visibly inside vessel.", 3)
    if mode == "manual":
        engine.store.advance(session, skip=False)
    assert session.step_id == "heat_water"
    snapshot = await engine.context_snapshot("p")
    assert snapshot["evidence"] == {}
    assert session.evidence == {}
    # A rising reading is insufficient. Only target-reaching observations confirm.
    await _see(engine, session, '{"reading":60,"unit":"celsius"}', 4)
    assert session.evidence_hits == 0 and not session.state["temperature_ready"]
    await _see(engine, session, '{"reading":82,"unit":"celsius"}', 5)
    assert session.evidence_hits == 1 and session.step_id == "heat_water"
    await _see(engine, session, '{"reading":180,"unit":"fahrenheit"}', 6)
    if mode == "manual":
        engine.store.advance(session, skip=False)
    assert session.step_id == "start_steeping" and session.state["temperature_ready"]
    snapshot = await engine.context_snapshot("p")
    assert snapshot["evidence"] == {}
    await _see(engine, session, "Tea bag submerged in water in the same cup.", 7)
    await _see(engine, session, "Tea bag submerged in water in the same cup.", 8)
    started = session.state["steeping_started_at_us"]
    if mode == "manual":
        engine.store.advance(session, skip=False)
    assert session.step_id == "steep_timer"
    assert session.state["steeping_started_at_us"] == started
    snapshot = await engine.context_snapshot("p")
    assert snapshot["evidence"] == {}
    assert snapshot["operations"]["active_operation"]["available"]
    monkeypatch.setattr(tools.time, "time_ns", lambda: (started + 181_000_000) * 1000)
    await engine._tick(session)
    if mode == "manual":
        assert session.active and session.state["steeping_complete"]
        engine.store.advance(session, skip=False)
    assert not session.active and session.run_id is None
    assert session.evidence_hits == 0 and session.observed_frame is None
    assert session.state == engine.workflow.initial_state()
    events = engine.store.drain_events(session)
    assert [e.step_id for e in events if e.event == "step.ready"] == [
        "identify", "fill_water", "heat_water", "start_steeping", "steep_timer",
    ]
    assert sum(e.event == "workflow.complete" for e in events) == 1
    completed = next(e for e in events if e.event == "workflow.complete")
    assert completed.run_id == run_id and completed.state["steeping_complete"]
    observations = [event for event in events if event.event == "agent.observation_complete"]
    assert observations[-1].step_id == "steep_timer"
    assert all(event.run_id == run_id for event in observations)
    steeped = next(e for e in events if e.event == "step.ready" and e.step_id == "start_steeping")
    assert steeped.evidence["start_steeping"]["observation"].startswith("Tea bag submerged")
    assert "start_steeping" not in completed.evidence
    assert engine._llm.chat.await_count == 1
    await engine.stop()


def test_checked_in_policy_and_parser_default_are_automatic(tmp_path):
    import yaml

    raw = yaml.safe_load((_SAMPLE / "yaml/workflow.yaml").read_text())
    assert load_workflow(_SAMPLE / "yaml/workflow.yaml").advancement == "automatic"
    del raw["task"]["advancement"]
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert load_workflow(path).advancement == "automatic"
    raw["task"]["advancement"] = "sometimes"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="task.advancement"):
        load_workflow(path)


@pytest.mark.asyncio
async def test_next_cannot_bypass_temperature_confirmation_in_automatic_mode():
    engine = _engine()
    session = engine.store.get("p")
    engine.store.start(session)
    engine.store.advance(session, skip=True)
    engine.store.advance(session, skip=True)
    assert session.step_id == "heat_water"
    for sequence, reading in enumerate([50, 60, 80, 92], 1):
        await _see(engine, session, json.dumps({"reading": reading, "unit": "celsius"}), sequence)
        # The setup uses the existing explicit-skip recommendation of 93 C.
        assert session.step_id == "heat_water" and not session.state["temperature_ready"]
    assert "not complete" in engine.store.advance(session, skip=False)
    await _see(engine, session, '{"reading":93,"unit":"celsius"}', 5)
    await _see(engine, session, '{"reading":null,"unit":null}', 6)
    assert session.evidence_hits == 1
    await _see(engine, session, '{"reading":92,"unit":"celsius"}', 7)
    assert session.evidence_hits == 0
    await _see(engine, session, '{"reading":93,"unit":"celsius"}', 8)
    await _see(engine, session, '{"reading":94,"unit":"celsius"}', 9)
    assert session.step_id == "start_steeping"
    await engine.stop()


@pytest.mark.asyncio
async def test_auto_transition_clears_live_evidence_and_preserves_history_without_a_user_question():
    from unittest.mock import AsyncMock

    from test_tea_run_context import _channel
    from test_tea_sop_agent import _sop
    from xr_ai_runtime import AgentRuntime

    sop = _sop()
    engine = sop._workflow_engine
    engine.workflow = load_workflow(_SAMPLE / "yaml/workflow.yaml")
    engine.store = WorkflowStore(engine.workflow)
    engine._decisions = _decisions("accepted", "accepted")
    runtime = AgentRuntime()
    sop.bind_runtime(runtime)
    async with runtime:
        session = engine.store.get("p")
        engine.store.start(session)
        engine.store.advance(session, skip=True)
        await engine._flush(session)
        manager = sop._run_contexts._runs["p"].manager
        manager.update = AsyncMock(wraps=manager.update)
        description = "A continuous liquid surface is visible inside the vessel. " * 8
        await _see(engine, session, description, 1)
        await _see(engine, session, description, 2)
        assert session.step_id == "heat_water"
        await engine._flush(session)
        evidence = _channel(sop, "p", "sop.evidence")
        assert evidence["observations"] == {}
        assert _channel(sop, "p", "sop.execution")["step"]["id"] == "heat_water"
        history = [json.loads(call.args[2].text) for call in manager.update.await_args_list]
        ready = next(event for event in history if event["event"] == "step.ready")
        assert ready["observation"]["observation"] == description
        assert ready["observation"]["frame_key"] == ["fill_water", "camera", 2, 2000]
        # Refreshing for a question must not restore the completed step's evidence.
        await sop._refresh_conversation_context("p")
        evidence = _channel(sop, "p", "sop.evidence")
        assert evidence["observations"] == {}
        assert _channel(sop, "p", "sop.execution")["step"]["id"] == "heat_water"
        await sop.stop()


@pytest.mark.asyncio
async def test_delayed_skip_cannot_skip_the_step_entered_by_automatic_completion():
    from unittest.mock import AsyncMock

    from tea_making_worker.agents.sop._tools import AdvanceRequest, workflow_management_tools

    engine = _engine()
    session = engine.store.get("p")
    engine.store.start(session)
    engine.store.advance(session, skip=True)
    tool = next(
        t for t in workflow_management_tools(engine.store, session, AsyncMock()) if t.name == "workflow__advance"
    )
    engine.store.observe(session, "accepted")
    engine.store.observe(session, "accepted")
    engine.store.commit(session, {"water_filled": True}, "")
    assert session.step_id == "heat_water"
    result = await tool.execute(AdvanceRequest(skip=True))
    assert "changed" in result.message
    assert session.step_id == "heat_water" and not session.state["temperature_ready"]
    await engine.stop()


def test_automatic_transition_bad_message_binding_cannot_partially_commit_completion():
    workflow = load_workflow(_SAMPLE / "yaml/workflow.yaml")
    steps = {**workflow.steps, "fill_water": replace(workflow.step("fill_water"), enter_message="{{ unavailable }}")}
    store = WorkflowStore(replace(workflow, steps=steps))
    session = store.get("p")
    store.start(session)
    store.observe(session, "accepted")
    before, revision = dict(session.state), session.revision
    with pytest.raises(ValueError, match="missing state"):
        store.commit(session, {
            "tea_name": "Fixture tea", "target_temperature_c": 80, "steep_duration_s": 180,
            "guidance_source": "fixture", "temperature_source": "fixture", "duration_source": "fixture",
            "tea_ready": True,
        }, "")
    assert session.state == before and session.revision == revision
    assert session.step_id == "identify" and not store.step_complete(session)
