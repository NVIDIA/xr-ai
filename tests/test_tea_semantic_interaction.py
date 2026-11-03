# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single semantic tea entry delegates real turns to SDK Conversation."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from xr_ai_models import DecisionAnswer, DecisionResponse
from xr_ai_routing import AgentRouter, AgentTurn
from xr_ai_interaction import ResponseOutput
from xr_ai_runtime import Agent, RuntimeContext, subscribe

from test_tea_sop_agent import _ConversationLanguage, _runner, _until
from tea_making_worker.application.tea_interaction import _TeaInteraction
from tea_making_worker.application.tea_sop_agent import _InteractionProposal
from tea_making_worker.application.tea_events import USER_QUERY_TOPIC
from xr_ai_conversation import Conversation
from xr_ai_voice import VOICE_OUTPUT_TOPIC


class _SemanticDecisions:
    def __init__(self):
        self.operation = "start"
        self.target = "workflow"
        self.calls = []

    async def decide(self, state, questions, **_kwargs):
        self.calls.append((state, questions))
        answers = {}
        for name, question in questions.items():
            choice = self.operation if name == "operation" else self.target
            answers[name] = DecisionAnswer(choice, 1.0, {label: float(label == choice) for label in question.criteria})
        return DecisionResponse(answers, "fixture", {}, {})


@pytest.mark.asyncio
async def test_single_entry_applies_effect_then_uses_the_same_conversation_for_qa():
    observed = asyncio.Event()

    async def observe(_request):
        observed.set()
        await asyncio.Event().wait()

    sop, runtime, context = _runner(observer=SimpleNamespace(execute=observe))
    decisions = _SemanticDecisions()
    language = _ConversationLanguage()
    conversation = Conversation(language, context=context, runtime=runtime, output_topic=VOICE_OUTPUT_TOPIC)
    entry = _TeaInteraction(sop, conversation, decisions)
    router = AgentRouter(
        {"tea": entry}, decisions, default_destination="tea", output=AsyncMock(),
        query_topic=USER_QUERY_TOPIC, output_topic=VOICE_OUTPUT_TOPIC,
        initialize_participant=sop.participant_joined, cleanup_participant=sop.participant_left,
    )
    async with runtime:
        try:
            response = await router.handle(AgentTurn("p", "start", "Let's make tea."))
            await asyncio.wait_for(observed.wait(), 1)
            assert sop.context_snapshot("p")["status"] == "running"
            assert len(decisions.calls) == 2
            assert set(decisions.calls[0][1]) == {"operation"}
            assert set(decisions.calls[1][1]) == {"referent"}
            assert decisions.calls[1][0]["candidate_operation"]["label"] == "start"
            assert language.messages == []
            assert response.text == sop.config.messages["identify"]
            assert sop.context_snapshot("p")["last_interaction"]["outcome"] == "workflow_start_new_run"
            decisions.operation, decisions.target = "none", "not_requested"
            await router.handle(AgentTurn("p", "question", "Why are we waiting?"))
            assert len(decisions.calls) == 3
            assert len(language.messages) == 1
            assert sop.context_snapshot("p")["status"] == "running"
            assert sop.context_snapshot("p")["last_interaction"]["outcome"] == "no_workflow_change"
            assert decisions.calls[2][0]["previous_exchange"]["user"] == "Let's make tea."
        finally:
            await router.stop()
            await sop.stop()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
async def test_completed_semantic_target_is_not_rebound_to_current_step():
    sop, runtime, context = _runner(observer=SimpleNamespace(execute=AsyncMock(side_effect=asyncio.CancelledError)))
    await sop.participant_joined("p")
    session = sop._sessions["p"]
    session.status, session.run_id, session.step_index = "running", "run", 3
    session.flow_statuses.update(heat_water="completed", start_steeping="running")
    snapshot = sop._interaction_snapshot("p")
    async with runtime:
        try:
            await sop._apply_interaction(
                snapshot, AgentTurn("p", "water-ready", "The water's ready, skip."),
                _InteractionProposal(DecisionAnswer("skip", 1.0, {"skip": 1.0}), DecisionAnswer("heat_water", 1.0, {"heat_water": 1.0})),
                started=AsyncMock(),
            )
            assert session.step_index == 3 and not session.skipping
            assert session.flow_statuses["start_steeping"] == "running"
            assert session.last_interaction["outcome"] == "referenced_step_already_finished"
        finally:
            await sop.stop()
            await context.close()


def _proposal(operation, target, *, unit=None, operation_mass=1.0, target_mass=1.0):
    """Make native proposals while retaining uncertainty, rather than just labels."""
    def answer(choice, mass):
        return DecisionAnswer(choice, mass, {choice: mass, "unclear": 1 - mass})
    return _InteractionProposal(
        answer(operation, operation_mass), answer(target, target_mass),
        answer(unit, 1.0) if unit else None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation_mass,target_mass", [(0.32, 1.0), (1.0, 0.4), (0.5, 1.0)])
async def test_ambiguous_operation_or_referent_cannot_stop_the_native_workflow(operation_mass, target_mass):
    sop, runtime, context = _runner()
    await sop.participant_joined("p")
    session = sop._sessions["p"]
    session.status, session.run_id, session.step_index = "running", "run", 2
    session.flow_statuses["heat_water"] = "running"
    started = AsyncMock()
    async with runtime:
        try:
            await sop._apply_interaction(
                sop._interaction_snapshot("p"), AgentTurn("p", "uncertain", "Super clear."),
                _proposal("stop", "workflow", operation_mass=operation_mass, target_mass=target_mass),
                started=started,
            )
            assert session.status == "running"
            assert session.last_interaction["outcome"] == "no_workflow_change"
            started.assert_not_awaited()
        finally:
            await sop.stop()
            await context.close()


@pytest.mark.asyncio
async def test_cancellation_resistant_semantic_backend_cannot_commit_or_record_a_turn():
    entered = asyncio.Event()
    resume = asyncio.Event()
    decisions = _SemanticDecisions()
    original_decide = decisions.decide

    async def swallowing_decide(*args, **kwargs):
        entered.set()
        try:
            await resume.wait()
        except asyncio.CancelledError:
            await resume.wait()
        return await original_decide(*args, **kwargs)

    decisions.decide = swallowing_decide
    sop, runtime, context = _runner()
    await sop.participant_joined("p")
    conversation = SimpleNamespace(answer=AsyncMock())
    entry = _TeaInteraction(sop, conversation, decisions)
    started = AsyncMock()
    async with runtime:
        try:
            pending = asyncio.create_task(entry.answer(AgentTurn("p", "canceled", "Start tea making."), started=started))
            await entered.wait()
            pending.cancel()
            resume.set()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert sop._sessions["p"].status == "idle"
            assert sop._sessions["p"].last_interaction is None
            started.assert_not_awaited()
            conversation.answer.assert_not_awaited()
        finally:
            await sop.stop()
            await context.close()


@pytest.mark.asyncio
async def test_step_and_unit_proposals_revalidate_after_started_yields():
    sop, runtime, context = _runner()
    await sop.participant_joined("p")
    session = sop._sessions["p"]
    session.status, session.run_id, session.step_index = "running", "run", 2
    session.flow_statuses["heat_water"] = "running"
    snapshot = sop._interaction_snapshot("p")

    async def advance_during_response_claim():
        session.revision += 1
        session.step_index = 3
        session.flow_statuses.update(heat_water="completed", start_steeping="running")

    async with runtime:
        try:
            await sop._apply_interaction(
                snapshot, AgentTurn("p", "late-water", "The water is ready, skip; it is Celsius."),
                _proposal("skip", "heat_water", unit="celsius"), started=advance_during_response_claim,
            )
            assert session.step_index == 3 and not session.skipping
            assert session.display_unit is None
            assert session.last_interaction["outcome"] == "referenced_step_already_finished"
        finally:
            await sop.stop()
            await context.close()


@pytest.mark.asyncio
async def test_whole_workflow_stop_survives_same_run_automatic_step_change():
    sop, runtime, context = _runner()
    await sop.participant_joined("p")
    session = sop._sessions["p"]
    session.status, session.run_id, session.step_index = "running", "run", 2
    session.flow_statuses["heat_water"] = "running"
    snapshot = sop._interaction_snapshot("p")
    session.revision += 1
    session.step_index = 3
    session.flow_statuses.update(heat_water="completed", start_steeping="running")
    started = AsyncMock()
    async with runtime:
        try:
            await sop._apply_interaction(
                snapshot, AgentTurn("p", "stop", "Stop this tea-making guide."),
                _proposal("stop", "workflow"), started=started,
            )
            assert session.status == "idle"
            assert session.last_interaction["outcome"] == "workflow_stop_applied"
            started.assert_awaited_once()
        finally:
            await sop.stop()
            await context.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["stopped", "completed"])
async def test_skip_does_not_claim_a_native_effect_after_the_run_has_ended(status):
    sop, runtime, context = _runner()
    await sop.participant_joined("p")
    session = sop._sessions["p"]
    session.status, session.run_id, session.step_index = status, "run", 2
    session.flow_statuses["heat_water"] = status
    started = AsyncMock()
    async with runtime:
        try:
            await sop._apply_interaction(
                sop._interaction_snapshot("p"), AgentTurn("p", "skip", "Skip the water check."),
                _proposal("skip", "heat_water"), started=started,
            )
            assert session.status == status and not session.skipping
            assert session.last_interaction["outcome"] == "workflow_not_running"
            started.assert_not_awaited()
        finally:
            await sop.stop()
            await context.close()


@pytest.mark.asyncio
async def test_runtime_start_has_one_reply_and_autonomous_identity_has_one_summary_and_next_step():
    from xr_ai_interaction import UserQuery
    from test_tea_sop_agent import _router

    identity_release = asyncio.Event()
    outputs = []

    class Capture(Agent):
        @subscribe(VOICE_OUTPUT_TOPIC)
        async def output(self, event: ResponseOutput, _context: RuntimeContext):
            outputs.append(event)

    async def observe(request):
        if "Return JSON matching" in request.query:
            await identity_release.wait()
            from xr_ai_tools.vision import ImageQueryResult
            return ImageQueryResult(text='{"tea_name":"Northwind White Tea","tea_type":"white tea"}')
        await asyncio.Event().wait()

    sop, runtime, context = _runner(observer=SimpleNamespace(execute=observe))
    runtime.register("capture-outputs", Capture())
    router, conversation, language = _router(sop, runtime, context)
    async with runtime:
        try:
            await runtime.publish(USER_QUERY_TOPIC, UserQuery(text="start tea making", timestamp_us=100), participant_id="p")
            await _until(lambda: any(event.kind == "reply" and event.final for event in outputs))
            assert len({event.response_id for event in outputs if event.kind == "reply"}) == 1
            assert not [event for event in outputs if event.kind == "update"]
            assert language.messages == []
            prompt = "\n".join(block.text for block in context.blocks("p"))
            assert '"current_step": "identify"' in prompt
            assert sop.config.messages["identify"] in prompt
            identity_release.set()
            await _until(lambda: any(event.kind == "update" for event in outputs))
            updates = [event.text for event in outputs if event.kind == "update"]
            assert len(updates) == 1
            assert "Northwind White Tea" in updates[0] and "80" in updates[0]
            assert sop.config.messages["fill_water"] in updates[0]
            assert sop.context_snapshot("p")["resolved_settings"]["sources"]["temperature"]["kind"] == "retrieved"
            assert language.messages == []  # Native guidance does not require generated user turns.
        finally:
            await router.stop()
            await sop.stop()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
async def test_unavailable_language_does_not_block_native_start_or_replay_its_action():
    from test_tea_sop_agent import _router

    class FailedLanguage:
        calls = 0
        async def stream_events(self, _messages):
            self.calls += 1
            raise RuntimeError("language endpoint unavailable")
            yield  # Keep the real streaming protocol, including an empty failed stream.

    async def observe(_request):
        await asyncio.Event().wait()

    sop, runtime, context = _runner(observer=SimpleNamespace(execute=observe))
    language = FailedLanguage()
    router, conversation, _ = _router(sop, runtime, context, conversation_language=language)
    async with runtime:
        try:
            response = await router.handle(AgentTurn("p", "start-no-language", "start tea making"))
            assert response.text == sop.config.messages["identify"]
            assert language.calls == 0
            session = sop._sessions["p"]
            assert session.status == "running" and session.run_id
            assert session.last_interaction["outcome"] == "workflow_start_new_run"
            original_run = session.run_id
            with pytest.raises(RuntimeError, match="language endpoint unavailable"):
                await router.handle(AgentTurn("p", "failed-question", "What is the current step?"))
            assert language.calls == 1 and session.run_id == original_run and session.status == "running"
            router._output.assert_awaited_once()  # Existing SDK failure is only for the failed QA reply.
        finally:
            await router.stop()
            await sop.stop()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
async def test_informational_turn_uses_fresh_context_after_autonomous_step_change():
    entered = asyncio.Event()
    release = asyncio.Event()
    decisions = _SemanticDecisions()
    decisions.operation, decisions.target = "none", "not_requested"
    decide = decisions.decide

    async def wait_for_progress(*args, **kwargs):
        entered.set()
        await release.wait()
        return await decide(*args, **kwargs)

    decisions.decide = wait_for_progress
    sop, runtime, context = _runner()
    await sop.participant_joined("p")
    session = sop._sessions["p"]
    session.status, session.run_id, session.step_index = "running", "run", 2
    session.flow_statuses["heat_water"] = "running"
    language = _ConversationLanguage()
    conversation = Conversation(language, context=context, runtime=runtime, output_topic=VOICE_OUTPUT_TOPIC)
    router = AgentRouter(
        {"tea": _TeaInteraction(sop, conversation, decisions)}, decisions,
        default_destination="tea", output=AsyncMock(),
        query_topic=USER_QUERY_TOPIC, output_topic=VOICE_OUTPUT_TOPIC,
        initialize_participant=sop.participant_joined, cleanup_participant=sop.participant_left,
    )
    async with runtime:
        try:
            task = asyncio.create_task(router.handle(AgentTurn("p", "status", "What is happening now?")))
            await entered.wait()
            session.step_index = 3
            session.revision += 1
            session.flow_statuses.update(heat_water="completed", start_steeping="running")
            await sop._publish_context(session)
            release.set()
            await task
            assert session.last_interaction["outcome"] == "no_workflow_change"
            messages = "\n".join(message.content for message in language.messages[-1])
            assert '"current_step": "start_steeping"' in messages
            assert "request_became_stale" not in messages
        finally:
            release.set()
            await router.stop()
            await sop.stop()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
async def test_temperature_readiness_keeps_raw_unit_and_user_declaration_separate(monkeypatch):
    from tea_making_worker.application import tea_sop_agent as module
    from tea_making_worker.application.tea_data_models import _TemperatureObservation

    sop, _runtime, context = _runner()
    await sop.participant_joined("p")
    session = sop._sessions["p"]
    session.display_unit = "fahrenheit"
    arguments = {}

    async def inspect_flow(**kwargs):
        arguments.update(kwargs)

    monkeypatch.setattr(module, "run_visual_verification_flow", inspect_flow)
    try:
        await sop._temperature_check(session, 80)
        can_assess = arguments["can_assess"]
        partial = _TemperatureObservation(value=96, role="current", role_evidence="Current temperature label")
        original = partial.model_dump()
        assert can_assess(partial)
        assert partial.model_dump() == original and partial.unit is None
        assert "fahrenheit" in arguments["completion"].instructions
        assert "176" in arguments["completion"].instructions
        for reading in (
            _TemperatureObservation(value=210, unit_text="°F", role="setpoint", role_evidence="Target label"),
            _TemperatureObservation(value=210, unit_text="°F", role="unknown"),
            _TemperatureObservation(value=96, unit_text="°C", role="current", role_evidence="Current label"),
            _TemperatureObservation(value=96, role="current"),
        ):
            assert not can_assess(reading)
        numeric_only = _TemperatureObservation(value=96, unit_text="°", role="current", role_evidence="96°")
        assert numeric_only.value == 96 and numeric_only.role_evidence is None
        assert not can_assess(numeric_only)
        assert _TemperatureObservation(value=96, unit_text="°").unit is None
        assert _TemperatureObservation(value=210, unit_text="MAX").unit is None
    finally:
        await sop.stop()
        await context.close()


@pytest.mark.asyncio
async def test_stale_instrument_declaration_cannot_overwrite_current_same_step_fact():
    sop, runtime, context = _runner()
    await sop.participant_joined("p")
    session = sop._sessions["p"]
    session.status, session.run_id, session.step_index = "running", "run", 2
    session.flow_statuses["heat_water"] = "running"
    snapshot = sop._interaction_snapshot("p")
    session.display_unit = "celsius"
    session.display_unit_source = {"kind":"user_declared", "turn_id":"newer", "text":"It is Celsius."}
    session.revision += 1
    started = AsyncMock()
    async with runtime:
        try:
            await sop._apply_interaction(
                snapshot, AgentTurn("p", "older", "The instrument uses Fahrenheit."),
                _proposal("none", "not_requested", unit="fahrenheit"), started=started,
            )
            assert session.display_unit == "celsius" and session.display_unit_source["turn_id"] == "newer"
            assert session.last_interaction["outcome"] == "request_became_stale"
            started.assert_not_awaited()
        finally:
            await sop.stop()
            await context.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("ending_request", ["stop tea making", "reset"])
async def test_stop_and_reset_end_into_idle_after_drain_without_erasing_conversation(ending_request):
    from test_tea_sop_agent import _router

    observing = asyncio.Event()
    cleaning = asyncio.Event()
    finish_cleanup = asyncio.Event()

    async def observe(_request):
        observing.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await finish_cleanup.wait()

    sop, runtime, context = _runner(observer=SimpleNamespace(execute=observe))
    router, conversation, language = _router(sop, runtime, context)
    async with runtime:
        try:
            await router.handle(AgentTurn("p", "start", "start tea making"))
            await observing.wait()
            session = sop._sessions["p"]
            original_run = session.run_id
            ending = asyncio.create_task(router.handle(AgentTurn("p", "end", ending_request)))
            await cleaning.wait()
            assert not ending.done() and session.run_id == original_run
            finish_cleanup.set()
            await ending
            assert session.status == "idle" and session.run_id is None and session.step_index == -1
            assert not session.active_tasks and not session.flow_states and session.procedure is None
            assert session.last_run["run_id"] == original_run
            assert session.last_interaction["operation"] == "stop"
            messages = "\n".join(block.text for block in context.blocks("p"))
            assert language.messages == []
            assert "start tea making" in messages  # User history survives workflow ending.
            previous_summary = dict(session.last_run)
            await router.handle(AgentTurn("p", "end-again", ending_request))
            assert session.status == "idle" and session.last_run == previous_summary
            assert "reset" not in sop.config.control_question.criteria
        finally:
            finish_cleanup.set()
            await router.stop()
            await sop.stop()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["Why are we waiting?", "stop tea making"])
async def test_authoritative_semantic_input_limit_is_no_effect_with_full_conversation(query):
    from xr_ai_models import DecisionInputTooLargeError
    from xr_ai_routing import AgentExchange

    sop, runtime, context = _runner()
    await sop.participant_joined("p")
    session = sop._sessions["p"]
    session.status, session.run_id, session.step_index = "running", "run", 2
    session.flow_statuses["heat_water"] = "running"
    decisions = SimpleNamespace(decide=AsyncMock(side_effect=DecisionInputTooLargeError("native token cap")))
    language = _ConversationLanguage()
    conversation = Conversation(language, context=context, runtime=runtime, output_topic=VOICE_OUTPUT_TOPIC)
    router = AgentRouter(
        {"tea": _TeaInteraction(sop, conversation, decisions)}, decisions, default_destination="tea",
        output=AsyncMock(), query_topic=USER_QUERY_TOPIC, output_topic=VOICE_OUTPUT_TOPIC,
        initialize_participant=sop.participant_joined, cleanup_participant=sop.participant_left,
    )
    history = (AgentExchange("Remember my preferred mug.", "Your preferred mug is the blue one."),
               AgentExchange("How long does steeping take?", "It depends on the identified tea."))
    async with runtime:
        try:
            await router.handle(AgentTurn("p", "limit", query, history))
            assert session.status == "running" and session.run_id == "run" and session.step_index == 2
            assert session.last_interaction["outcome"] == "not_interpreted_input_limit"
            assert session.last_interaction["native_effect"] is False
            messages = "\n".join(message.content for message in language.messages[-1])
            assert "preferred mug" in messages and "How long does steeping take?" in messages
            assert query in messages and "not_interpreted_input_limit" in messages
            decision_state = decisions.decide.await_args.args[0]
            assert decision_state["previous_exchange"]["user"] == history[-1].user
            assert "preferred mug" not in str(decision_state)
        finally:
            await router.stop()
            await sop.stop()
            await conversation.close()
            await context.close()
