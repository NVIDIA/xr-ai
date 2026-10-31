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
from xr_ai_interaction import ParticipantJoined, ParticipantLeft, ResponseOutput, UserQuery
from xr_ai_models import DecisionAnswer, DecisionResponse
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, subscribe
from xr_ai_tools import ToolSet
from xr_ai_tools.current_frame import ImageFrame
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import ImageQueryResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-samples/tea-making-sample/worker"))

from tea_making_worker import SOPAgent  # noqa: E402
from tea_making_worker.conversation._capture import _context_owner  # noqa: E402
from tea_making_worker.events import PARTICIPANT_JOINED_TOPIC, PARTICIPANT_LEFT_TOPIC, ParticipantCleanupComplete  # noqa: E402
from tea_making_worker.flows import (
    ObservationPolicy,  # noqa: E402
    VisualValidationStep,  # noqa: E402
)
from test_tea_making_sample import (
    _activate_test_step,
    _attach_router,
    _routed_foreground,
    _ScriptedDecisions,
    _test_destinations,
)  # noqa: E402


def _sop(decisions=None, *, workflow=None, prompt=None, **budgets):
    foreground = _routed_foreground(decisions or SimpleNamespace())
    return _attach_router(SOPAgent(
        workflow=workflow or foreground._workflow_engine.workflow,
        llm=foreground._llm, decisions=foreground._decisions,
        images=foreground._images, vlm=foreground._vlm, rag=foreground._rag,
        image_query=SimpleNamespace(),
        application_tools=foreground._background_tools,
        prompt=foreground._prompt if prompt is None else prompt,
        current_view_prompt=foreground._view_prompt, **budgets,
    ))



def _register_sop(runtime, sop):
    from tea_making_worker._capture import _record_routing
    from tea_making_worker._cleanup import _CleanupAcks
    from tea_making_worker.events import INTERRUPTED_TOPIC, USER_QUERY_TOPIC
    from xr_ai_routing import AgentRouter
    from xr_ai_voice import VOICE_OUTPUT_TOPIC

    runtime.register("sop", sop)
    sop.bind_runtime(runtime)
    sop._test_idle = asyncio.Event()

    async def capture_status(status, participant):
        await _record_routing(status, participant, snapshot=sop._snapshot)
        if status == "idle":
            sop._test_idle.set()

    router = AgentRouter(
        _test_destinations(sop), sop._decisions,
        query_topic=USER_QUERY_TOPIC, output_topic=VOICE_OUTPUT_TOPIC,
        participant_joined_topic=PARTICIPANT_JOINED_TOPIC,
        interrupted_topic=INTERRUPTED_TOPIC,
        initialize_participant=sop.participant_joined, cleanup_participant=sop.participant_left,
        set_status=capture_status,
    )
    runtime.register("routing", router.runtime_subscriber())
    runtime.register("cleanup", _CleanupAcks(router))
    return router


@pytest.mark.asyncio
async def test_one_registered_sop_agent_owns_engine_history_notices_and_cleanup():
    sop = _sop()
    runtime = AgentRuntime()
    router = _register_sop(runtime, sop)
    async with runtime:
        await runtime.publish(PARTICIPANT_JOINED_TOPIC, ParticipantJoined(), participant_id="alice")
        session = sop._workflow_engine.store.get("alice")
        sop._workflow_engine.store.start(session)
        await sop._workflow_engine._flush(session)
        text = "\n".join(block.text for block in sop._context_blocks("alice"))
        assert "step.enter" in text and "identify" in text
        assert sop._context.blocks("bob") == ()
        sop._context.record_exchange("alice", "What next?", "Show the package.", timestamp_us=1, exchange_id="one")
        assert "Show the package" in "\n".join(block.text for block in sop._context.blocks("alice"))
        await runtime.publish(PARTICIPANT_LEFT_TOPIC, ParticipantLeft(), participant_id="alice")
        assert sop._workflow_engine.store.find("alice") is None
        assert sop._context.blocks("alice") == ()
        # An already-produced old event must not repopulate a departed session.
        from tea_making_worker.agents.sop._state import WorkflowEvent
        await sop._remember_event(session, WorkflowEvent("step.ready", "identify", "old", {}))
        assert sop._context.blocks("alice") == ()
        assert not sop._workflow_engine._tasks
        await router.stop()
        await sop.stop()


@pytest.mark.asyncio
async def test_completed_dialogue_reaches_generation_and_choice_without_owning_state():
    decisions = _ScriptedDecisions(("action", "answer"), ("action", "answer"))
    sop = _sop(decisions)
    _activate_test_step(sop._workflow_engine, "fill_water")
    # Establish focus via normal router admission, then ask a referential follow-up.
    decisions.choices.appendleft(("destination", "workflow"))
    before = dict(sop._workflow_engine.store.get("active").state)
    try:
        await sop._answer("What should I do?", "active")
        await sop._answer("Explain that again", "active")
        messages = sop._llm.chat.await_args.args[0]
        assert "What should I do?" in messages[1].content
        assert json.loads(messages[-1].content)["state"]["water_filled"] is False
        assert sop._workflow_engine.store.get("active").state == before
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
    session = sop._workflow_engine.store.get("active")
    sop._workflow_engine.store.start(session)
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
    policy = ObservationPolicy.model_validate({
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
    step = VisualValidationStep(
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
    _activate_test_step(sop._workflow_engine, "fill_water")
    old = sop._workflow_engine.store.get("active")
    entered, release = asyncio.Event(), asyncio.Event()

    async def decide(*_args, **_kwargs):
        entered.set()
        await release.wait()
        return DecisionResponse({"evidence": DecisionAnswer("accepted", 1, {"accepted": 1})}, "fake", {}, {})

    sop._workflow_engine._decisions = SimpleNamespace(decide=decide)
    task = asyncio.create_task(sop._workflow_engine._observe(
        old, sop._workflow_engine.workflow.step("fill_water"), "Water visible", dict(old.state), old.revision,
    ))
    await entered.wait()
    sop._workflow_engine.store.release("active")
    _activate_test_step(sop._workflow_engine, "fill_water")
    current = sop._workflow_engine.store.get("active")
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
    router = _register_sop(runtime, sop)
    async with runtime:
        await runtime.publish(PARTICIPANT_JOINED_TOPIC, ParticipantJoined(), participant_id="alice")
        await runtime.publish(PARTICIPANT_LEFT_TOPIC, ParticipantLeft(), participant_id="alice")
        await runtime.publish(USER_QUERY_TOPIC, UserQuery(text="Start again", timestamp_us=1), participant_id="alice")
        assert sop._workflow_engine.store.find("alice") is None
        assert sop._context.blocks("alice") == ()
        sop._llm.chat.assert_not_awaited()
        await router.stop()
        await sop.stop()


@pytest.mark.asyncio
async def test_stop_waits_for_join_before_draining_its_new_monitor():
    sop = _sop()
    entered, release, stopping = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_flush = sop._workflow_engine._flush

    async def flush(session):
        entered.set()
        await release.wait()
        await original_flush(session)

    sop._workflow_engine._flush = flush
    runtime = AgentRuntime()
    router = _register_sop(runtime, sop)
    async with runtime:
        join = asyncio.create_task(runtime.publish(
            PARTICIPANT_JOINED_TOPIC, ParticipantJoined(), participant_id="alice",
        ))
        await entered.wait()

        async def stop():
            stopping.set()
            await router.stop()
            await sop.stop()

        shutdown = asyncio.create_task(stop())
        await stopping.wait()
        assert not shutdown.done()
        release.set()
        await join
        await shutdown
        assert not sop._workflow_engine._tasks
        assert not sop._workflow_engine._connected
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


@pytest.mark.asyncio
async def test_main_yaml_camera_policy_reaches_observer_and_streaming_model_requests(tmp_path):
    from tea_making_worker.config import load_config
    from xr_ai_tools.vision import ImageQueryRequest, ImageQueryTool

    configured = "Describe the overhead workbench view; retain readable identifiers."
    path = tmp_path / "worker.yaml"
    import yaml
    from test_tea_making_sample import _SAMPLE
    raw = yaml.safe_load((_SAMPLE / "yaml/tea_making_worker.yaml").read_text())
    raw["current_view_prompt"] = configured
    path.write_text(yaml.safe_dump(raw))
    config = load_config(path)
    foreground = _routed_foreground(SimpleNamespace())
    registry = foreground._images.images
    frame = ImageFrame(
        image=registry.put(b"pixels"), participant_id="p", track_id="camera", sequence=1,
        timestamp_us=10, width=1, height=1,
    )
    received = []

    async def stream_images(_images, _question, **kwargs):
        received.append(kwargs["system_prompt"])
        yield "Visible workbench."

    vlm = SimpleNamespace(
        ask_images=AsyncMock(return_value=SimpleNamespace(content=json.dumps({
            "observable": True, "description": "The water surface is visible inside the vessel.",
        }))),
        stream_images=stream_images,
    )
    foreground._images.get_current_frame = SimpleNamespace(execute=AsyncMock(return_value=frame))
    sop = SOPAgent(
        workflow=foreground._workflow_engine.workflow,
        llm=foreground._llm, decisions=foreground._decisions,
        images=foreground._images, vlm=vlm, rag=foreground._rag,
        image_query=ImageQueryTool(images=registry, vlm=vlm, system_prompt=config.current_view_prompt),
        application_tools=foreground._background_tools,
        prompt=config.foreground_prompt, current_view_prompt=config.current_view_prompt,
    )
    try:
        _activate_test_step(sop._workflow_engine, "fill_water")
        session = sop._workflow_engine.store.get("active")
        await sop._workflow_engine._trigger("active", sop._workflow_engine.workflow.step("fill_water"), session.state)
        assert vlm.ask_images.await_args.kwargs["system_prompt"] == configured
        chunks = [part.text async for part in sop._vision.stream(ImageQueryRequest(image=frame.image, query="Read it"))]
        assert chunks == ["Visible workbench."] and received == [configured]
    finally:
        await sop.stop()
        await foreground.stop()


@pytest.mark.asyncio
async def test_yaml_workflow_voice_and_shared_prompt_reach_live_answer_request(tmp_path):
    import yaml
    from tea_making_worker.sop import load_workflow
    from test_tea_making_sample import _SAMPLE
    from xr_ai_models import ChatResponse

    raw = yaml.safe_load((_SAMPLE / "yaml/workflow.yaml").read_text())
    custom = "Explain the active procedure using concise complete sentences and retain measured units."
    raw["task"]["foreground"]["active_prompt"] = custom
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(raw))
    sop = _sop(workflow=load_workflow(path), prompt="Use the configured application response style.")
    sop._llm = SimpleNamespace(chat=AsyncMock(return_value=ChatResponse(
        content="Ready.", reasoning="", tool_calls=[], finish_reason="stop", raw={},
    )))
    try:
        _activate_test_step(sop._workflow_engine, "fill_water")
        await sop._generate_answer("What should I do?", "active", workflow_context=True)
        system = sop._llm.chat.await_args.args[0][0].content
        assert system.startswith(sop._prompt)
        assert custom in system
        assert sop._workflow_engine.workflow.step("fill_water").voice.prompt in system
        assert "at most two short sentences" not in system
    finally:
        await sop.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["interrupt", "depart"])
async def test_sdk_runtime_drains_cancelled_query_without_late_output_history_or_capture(tmp_path, operation):
    from tea_making_worker._capture import _RecordedLanguage
    from tea_making_worker.events import INTERRUPTED_TOPIC, USER_QUERY_TOPIC
    from test_tea_sop_capture import _capture, _records
    from xr_ai_interaction import Interrupted, UserQuery
    from xr_ai_models import ChatResponse
    from xr_ai_voice import VOICE_OUTPUT_TOPIC

    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    decisions = _ScriptedDecisions(("destination", "workflow"), ("action", "answer"))
    sop = _sop(decisions)

    async def blocked(*_args, **_kwargs):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            # Even a backend which suppresses cancellation must not publish or
            # commit its eventual response. SDK ResponseOutputs enforces this.
            cancelled.set()
            await release.wait()
        return ChatResponse("Late result", None, None, "stop", {})

    sop._llm = _RecordedLanguage(SimpleNamespace(chat=blocked))

    class Sink(Agent):
        def __init__(self):
            super().__init__()
            self.outputs = []

        @subscribe(VOICE_OUTPUT_TOPIC)
        async def receive(self, value: ResponseOutput, ctx: RuntimeContext):
            self.outputs.append(value)

    capture, _ = await _capture(tmp_path)
    runtime = AgentRuntime()
    router = _register_sop(runtime, sop)
    sink = runtime.register("sink", Sink())
    async with capture.activate():
        async with runtime:
            await runtime.publish(PARTICIPANT_JOINED_TOPIC, ParticipantJoined(), participant_id="active")
            _activate_test_step(sop._workflow_engine, "fill_water")
            sop._workflow_engine.store.get("active").next_tick = float("inf")
            await runtime.publish(
                USER_QUERY_TOPIC, UserQuery(text="What next?", timestamp_us=1), participant_id="active",
            )
            await asyncio.wait_for(entered.wait(), 2)
            topic, event = ((INTERRUPTED_TOPIC, Interrupted()) if operation == "interrupt"
                            else (PARTICIPANT_LEFT_TOPIC, ParticipantLeft()))
            cleanup = asyncio.create_task(runtime.publish(topic, event, participant_id="active"))
            await asyncio.wait_for(cancelled.wait(), 2)
            assert not cleanup.done()
            assert sink.outputs == []
            release.set()
            await asyncio.wait_for(cleanup, 2)
            assert sink.outputs == []
            assert "Late result" not in "\n".join(block.text for block in sop._context.blocks("active"))
            if operation == "depart":
                assert sop._workflow_engine.store.find("active") is None
                await runtime.publish(
                    USER_QUERY_TOPIC, UserQuery(text="Again", timestamp_us=2), participant_id="active",
                )
                assert sop._workflow_engine.store.find("active") is None
            await router.stop()
            await sop.stop()
    records = [row for _entry, rows in _records(capture) for row in rows]
    assert any(row["kind"] == "language.finished" for row in records)
    assert not any(row["kind"] == "answer.produced" for row in records)
    routing = [rows for entry, rows in _records(capture) if entry["channel"] == "routing"]
    assert len(routing) == 1
    assert routing[0][-1]["kind"] == "attempt.finished" and routing[0][-1]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_sdk_departure_acknowledges_original_generation_once_before_rejoin():
    from tea_making_worker.events import PARTICIPANT_CLEANUP_COMPLETE_TOPIC

    class Sink(Agent):
        def __init__(self):
            super().__init__()
            self.generation = None
            self.acks = []

        @subscribe(PARTICIPANT_LEFT_TOPIC)
        async def left(self, event: ParticipantLeft, ctx: RuntimeContext):
            self.generation = ctx.metadata.message_id

        @subscribe(PARTICIPANT_CLEANUP_COMPLETE_TOPIC)
        async def ack(self, value: ParticipantCleanupComplete, ctx: RuntimeContext):
            self.acks.append(value)

    sop = _sop()
    runtime = AgentRuntime()
    router = _register_sop(runtime, sop)
    sink = runtime.register("sink", Sink())
    entered, release = asyncio.Event(), asyncio.Event()
    clear = sop._run_contexts.clear

    async def blocked_clear(participant):
        entered.set()
        await release.wait()
        await clear(participant)

    sop._run_contexts.clear = blocked_clear
    async with runtime:
        await runtime.publish(PARTICIPANT_JOINED_TOPIC, ParticipantJoined(), participant_id="p")
        old = sop._workflow_engine.store.get("p")
        departure = asyncio.create_task(runtime.publish(PARTICIPANT_LEFT_TOPIC, ParticipantLeft(), participant_id="p"))
        await asyncio.wait_for(entered.wait(), 2)
        rejoin = asyncio.create_task(runtime.publish(PARTICIPANT_JOINED_TOPIC, ParticipantJoined(), participant_id="p"))
        assert not departure.done() and not rejoin.done()
        release.set()
        await asyncio.wait_for(asyncio.gather(departure, rejoin), 2)
        current = sop._workflow_engine.store.find("p")
        assert current is not None and current is not old
        assert "p" in sop._workflow_engine._connected
        assert [(ack.producer, ack.generation) for ack in sink.acks] == [
            ("foreground", sink.generation), ("guidance", sink.generation),
        ]
        await router.stop()
        await sop.stop()


@pytest.mark.asyncio
async def test_optional_application_adapters_decline_without_their_capability_tools():
    from xr_ai_routing import AgentTurn

    sop = _sop()
    sop._background_tools = lambda _participant: ToolSet(())
    try:
        routes = _test_destinations(sop)
        for name in ("watch", "transcript", "journal", "history"):
            result = await routes[name].admit(AgentTurn("p", name, "Please check"), focused=False)
            assert result.disposition == "decline"
    finally:
        await sop.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_sdk_capture_balances_request_scope_and_retains_routing_admission_and_authorization(tmp_path, failure):
    from tea_making_worker._capture import _RecordedDecisions, _RecordedLanguage
    from tea_making_worker.events import USER_QUERY_TOPIC
    from test_tea_sop_capture import _capture, _records

    choices = [("destination", "workflow"), ("action", "answer" if failure else "start")]
    if not failure:
        choices.append(("authorization", "command"))
    sop = _sop(_RecordedDecisions(_ScriptedDecisions(*choices)))
    if failure:
        sop._llm = _RecordedLanguage(SimpleNamespace(chat=AsyncMock(side_effect=RuntimeError("model failed"))))
    runtime = AgentRuntime()
    router = _register_sop(runtime, sop)
    expected = []

    class Requests(Agent):
        @subscribe(USER_QUERY_TOPIC)
        async def receive(self, message: UserQuery, ctx: RuntimeContext):
            expected.append(ctx.metadata.message_id)

    runtime.register("request-identity", Requests())
    capture, _ = await _capture(tmp_path)
    async with capture.activate():
        async with runtime:
            await runtime.publish(PARTICIPANT_JOINED_TOPIC, ParticipantJoined(), participant_id="active")
            if failure:
                _activate_test_step(sop._workflow_engine, "fill_water")
                sop._workflow_engine.store.get("active").next_tick = float("inf")
            await runtime.publish(USER_QUERY_TOPIC, UserQuery(text="Explain" if failure else "Start", timestamp_us=1),
                                  participant_id="active")
            await asyncio.wait_for(sop._test_idle.wait(), 2)
            await router.stop()
            await sop.stop()
    attempts = _records(capture)
    route = next((entry, rows) for entry, rows in attempts if entry["channel"] == "routing")
    assert route[0]["request_id"] == expected[0]
    assert route[1][-1]["kind"] == "attempt.finished"
    assert route[1][-1]["status"] == ("failed" if failure else "completed")
    calls = [row for row in route[1] if row["kind"] == "decision.started"]
    assert [set(row["questions"]) for row in calls] == (
        [{"destination"}, {"action"}] if failure else [{"destination"}, {"action"}, {"authorization"}]
    )
    assert all(row["request_id"] == expected[0] for _entry, rows in attempts for row in rows)


@pytest.mark.asyncio
async def test_sop_notice_delivery_preserves_update_metadata_and_ignores_departed_participants():
    """One SOP subscriber delivers notices without a second registered voice agent."""
    from tea_making_worker.events import GUIDANCE_NOTICE_TOPIC, GuidanceNotice
    from xr_ai_voice import VOICE_OUTPUT_TOPIC

    delivered_updates = []

    class NoticeSink(Agent):
        @subscribe(VOICE_OUTPUT_TOPIC)
        async def receive(self, message: ResponseOutput, ctx: RuntimeContext) -> None:
            delivered_updates.append((ctx.metadata.participant_id, message))

    sop = _sop()
    runtime = AgentRuntime()
    router = _register_sop(runtime, sop)
    runtime.register("notice-sink", NoticeSink())
    async with runtime:
        try:
            await runtime.publish(PARTICIPANT_JOINED_TOPIC, ParticipantJoined(), participant_id="alice")
            notice = GuidanceNotice(timestamp_us=1234, text="A configured workflow notice.")
            await runtime.publish(GUIDANCE_NOTICE_TOPIC, notice, participant_id="bob")
            assert delivered_updates == []
            await runtime.publish(GUIDANCE_NOTICE_TOPIC, notice, participant_id="alice")
            assert len(delivered_updates) == 1
            participant_id, output = delivered_updates[0]
            assert participant_id == "alice"
            assert output.kind == "update"
            assert output.text == notice.text
            assert output.timestamp_us == notice.timestamp_us
            await runtime.publish(PARTICIPANT_LEFT_TOPIC, ParticipantLeft(), participant_id="alice")
            await runtime.publish(GUIDANCE_NOTICE_TOPIC, notice, participant_id="alice")
            assert len(delivered_updates) == 1
        finally:
            await router.stop()
            await sop.stop()
