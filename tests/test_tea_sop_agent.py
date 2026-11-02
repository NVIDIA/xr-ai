# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Production runner, SDK Conversation and router lifecycle boundaries."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import nemo_relay
import pytest
from xr_ai_conversation import ContextManager, Conversation
from xr_ai_models import ChatResponse, ChatStreamEvent, DecisionAnswer, DecisionResponse
from xr_ai_routing import AgentRouter, AgentTurn, RoutingAid
from xr_ai_runtime import AgentRuntime, RuntimeContext
from xr_ai_tools.current_frame import ImageFrame
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.rag import RetrievedChunk, RetrieveResult
from xr_ai_tools.vision import ImageQueryResult
from xr_ai_voice import VOICE_OUTPUT_TOPIC

_SAMPLE = Path(__file__).resolve().parents[1] / "agent-samples/tea-making-sample"
sys.path.insert(0, str(_SAMPLE / "worker"))

from tea_making_worker import SOPAgent  # noqa: E402
from tea_making_worker.agents.sop.agent import SOP_CONTEXT_SOURCE  # noqa: E402
from tea_making_worker.config import load_config  # noqa: E402
from tea_making_worker.events import ParticipantCleanupComplete  # noqa: E402
from tea_making_worker.flows import run_timer_flow  # noqa: E402


class _Decisions:
    """Deterministic inference substitute; routing/execution remains actual SDK code."""

    controls = {
        "start tea making": "start",
        "stop tea making": "stop",
        "next step": "next",
        "skip": "skip",
        "reset": "reset",
        "restart": "restart",
    }

    async def decide(self, state, questions, **_kwargs):
        answers = {}
        request = state.get("request", "") if isinstance(state, dict) else ""
        for key, question in questions.items():
            if key == "destination":
                choice = "workflow" if request in self.controls else "conversation"
            elif key == "control":
                choice = self.controls.get(request, "none")
            elif key == "authorization":
                choice = "command" if request in self.controls else "not_command"
            else:
                choice = "positive"
            answers[key] = DecisionAnswer(
                choice,
                1.0,
                {label: float(label == choice) for label in question.criteria},
            )
        return DecisionResponse(answers, "fixture", {}, {})


def _runner(*, config=None, observer=None, llm=None):
    selected = load_config(_SAMPLE / "yaml/tea_making_worker.yaml")
    config = config or replace(selected.sop, poll_interval_seconds=0.001, timer_update_interval_seconds=0.005)
    runtime = AgentRuntime()
    context = ContextManager(
        SimpleNamespace(),
        summary_context_topics=(SOP_CONTEXT_SOURCE,),
        dialogue_budget_bytes=12000,
        topic_budget_bytes=64000,
        compaction_input_budget_bytes=40000,
    )
    registry = ImageRegistry()
    image = registry.put(b"fixture")
    sequence = 0

    async def acquire(_request):
        nonlocal sequence
        sequence += 1
        return ImageFrame(
            image=image,
            participant_id="p",
            track_id="camera",
            sequence=sequence,
            timestamp_us=sequence * 1000,
            width=2,
            height=2,
        )

    async def observe(request):
        text = (
            '{"tea_name":"Northwind White Tea"}'
            if request.query.startswith(config.identification_question)
            else (
                "A clear water surface is visible; current display reads 80 degrees Celsius; "
                "tea is visibly immersed in the same cup."
            )
        )
        return ImageQueryResult(text=text)

    source_text = "Northwind White Tea: 80 degrees Celsius for 0.02 seconds."
    llm = llm or SimpleNamespace(
        chat=AsyncMock(
            return_value=ChatResponse(
                json.dumps(
                    {
                        "values": {
                            "temperature": {"value": 80, "unit": "celsius"},
                            "steep_duration": {"value": 0.02, "unit": "seconds"},
                        },
                        "sources": {
                            name: {"passage": 0, "quote": source_text} for name in ("temperature", "steep_duration")
                        },
                    }
                ),
                None,
                None,
                "stop",
                {},
            )
        )
    )
    sop = SOPAgent(
        config=config,
        llm=llm,
        decisions=_Decisions(),
        images=SimpleNamespace(get_current_frame=SimpleNamespace(execute=acquire)),
        image_query=observer or SimpleNamespace(execute=observe),
        rag=SimpleNamespace(
            retrieve=SimpleNamespace(
                execute=AsyncMock(
                    return_value=RetrieveResult(
                        results=[
                            RetrievedChunk(text=source_text, source="tea.md", score=1),
                        ]
                    ),
                )
            )
        ),
        context=context,
        runtime=runtime,
    )
    return sop, runtime, context


class _ConversationLanguage:
    def __init__(self):
        self.messages = []

    async def stream_events(self, messages):
        self.messages.append(messages)
        yield ChatStreamEvent(kind="text", text="The latest SOP context is available.")
        yield ChatStreamEvent(kind="finish", finish_reason="stop")


def _router(sop, runtime, context, conversation_language=None):
    language = conversation_language or _ConversationLanguage()
    conversation = Conversation(
        language,
        context=context,
        instructions=sop.config.conversation_instructions,
        system_prompt="Use supplied facts; do not execute workflow controls.",
        routing=RoutingAid("conversation", sop.config.conversation_description),
        runtime=runtime,
        output_topic=VOICE_OUTPUT_TOPIC,
    )
    router = AgentRouter(
        {"workflow": sop, "conversation": conversation},
        sop._decisions,
        query_topic=__import__("tea_making_worker.events", fromlist=["USER_QUERY_TOPIC"]).USER_QUERY_TOPIC,
        output_topic=VOICE_OUTPUT_TOPIC,
        output=AsyncMock(),
        initialize_participant=sop.participant_joined,
        cleanup_participant=sop.participant_left,
    )
    runtime.register("routing", router.runtime_subscriber())
    return router, conversation, language


async def _until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_actual_python_procedure_completes_all_five_steps():
    sop, runtime, context = _runner()
    router, conversation, _language = _router(sop, runtime, context)
    async with runtime:
        try:
            result = await router.handle(AgentTurn("p", "start", "start tea making"))
            assert result.text == sop.config.messages["start"]
            await _until(lambda: sop._sessions["p"].status == "completed")
            snapshot = sop.context_snapshot("p")
            assert snapshot["resolved_settings"]["temperature_c"] == 80
            assert snapshot["resolved_settings"]["sources"]["temperature"]["kind"] == "retrieved"
            assert list(snapshot["completed_or_skipped_flows"]) == [
                "identify",
                "fill_water",
                "heat_water",
                "start_steeping",
                "steep_timer",
            ]
            assert snapshot["active_flows"] == {}
            timer = snapshot["completed_or_skipped_flows"]["steep_timer"]["state"]
            assert timer["remaining_seconds"] == 0
            assert timer["updated_at_us"] >= timer["started_at_us"]
        finally:
            await router.stop()
            await sop.stop()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
async def test_questions_reach_sdk_conversation_while_sop_runs_and_stop_is_a_control():
    water_started = asyncio.Event()

    async def observe(request):
        if "Return JSON matching" in request.query:
            return ImageQueryResult(text='{"tea_name":"Northwind White Tea"}')
        # Hold an actual model/tool trace scope open while QA and stop proceed.
        # A flow must inherit trace lineage without sharing its mutable stack.
        with nemo_relay.scope.scope("blocked-visual-tool", nemo_relay.ScopeType.Tool):
            water_started.set()
            await asyncio.Event().wait()

    sop, runtime, context = _runner(observer=SimpleNamespace(execute=observe))
    router, conversation, language = _router(sop, runtime, context)
    async with runtime:
        try:
            await router.handle(AgentTurn("p", "start", "start tea making"))
            await asyncio.wait_for(water_started.wait(), 1)
            procedure = sop._sessions["p"].procedure
            response = await router.handle(AgentTurn("p", "qa", "What tea did we identify?"))
            assert "latest SOP context" in response.text
            assert sop._sessions["p"].procedure is procedure and not procedure.done()
            messages = "\n".join(message.content for message in language.messages[-1])
            assert "Northwind White Tea" in messages
            assert '"temperature_c": 80.0' in messages
            assert '"current_step": "fill_water"' in messages
            assert '"kind": "retrieved"' in messages
            await router.handle(AgentTurn("p", "stop", "stop tea making"))
            assert procedure.done()
            assert sop.context_snapshot("p")["status"] == "stopped"
            assert sop.context_snapshot("p")["active_flows"] == {}
        finally:
            await router.stop()
            await sop.stop()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
async def test_skip_retains_partial_settings_and_defaults_only_missing_fields():
    fallback_started = asyncio.Event()
    water_started = asyncio.Event()

    async def chat(messages, **_kwargs):
        payload = json.loads(messages[1].content)
        if payload["passages"]:
            return ChatResponse(
                json.dumps(
                    {
                        "values": {"temperature": {"value": 80, "unit": "celsius"}},
                        "sources": {"temperature": {"passage": 0, "quote": "80 degrees Celsius"}},
                    }
                ),
                None,
                None,
                "stop",
                {},
            )
        fallback_started.set()
        await asyncio.Event().wait()

    async def observe(request):
        if "Return JSON matching" in request.query:
            return ImageQueryResult(text='{"tea_name":"Northwind White Tea"}')
        water_started.set()
        await asyncio.Event().wait()

    sop, runtime, context = _runner(llm=SimpleNamespace(chat=chat), observer=SimpleNamespace(execute=observe))
    router, conversation, _language = _router(sop, runtime, context)
    async with runtime:
        try:
            await router.handle(AgentTurn("p", "start", "start tea making"))
            await asyncio.wait_for(fallback_started.wait(), 1)
            await router.handle(AgentTurn("p", "skip", "skip"))
            await asyncio.wait_for(water_started.wait(), 1)
            snapshot = sop.context_snapshot("p")
            assert snapshot["current_step"] == "fill_water"
            assert snapshot["resolved_settings"]["temperature_c"] == 80
            assert snapshot["resolved_settings"]["duration_seconds"] == 180
            assert snapshot["resolved_settings"]["sources"]["temperature"]["kind"] == "retrieved"
            assert snapshot["resolved_settings"]["sources"]["steep_duration"]["kind"] == "default"
            assert snapshot["completed_or_skipped_flows"]["identify"]["outcome"] == "skipped"
        finally:
            await router.stop()
            await sop.stop()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
async def test_manual_next_cannot_bypass_incomplete_step():
    selected = load_config(_SAMPLE / "yaml/tea_making_worker.yaml")
    sop, runtime, context = _runner(config=replace(selected.sop, advancement="manual", poll_interval_seconds=0.001))
    router, conversation, _language = _router(sop, runtime, context)
    async with runtime:
        try:
            await router.handle(AgentTurn("p", "start", "start tea making"))
            await _until(lambda: sop._sessions["p"].ready)
            assert sop.context_snapshot("p")["current_step"] == "identify"
            await router.handle(AgentTurn("p", "next", "next step"))
            await _until(lambda: sop.context_snapshot("p")["current_step"] == "fill_water")
            # Block this step before it can become ready; a query never skips it.
            session = sop._sessions["p"]
            if not session.ready:
                response = await router.handle(AgentTurn("p", "early-next", "next step"))
                assert response.text == sop.config.messages["not_ready"]
            assert session.status == "running"
        finally:
            await router.stop()
            await sop.stop()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query_text",
    [
        "What does reset do?",
        "She told me to stop tea making.",
        "Do not start tea making.",
        "How do I go to the next step?",
        "Would you be able to reset it?",
    ],
)
async def test_informational_negated_and_reported_controls_are_questions(query_text):
    sop, runtime, context = _runner()
    router, conversation, language = _router(sop, runtime, context)
    async with runtime:
        try:
            await router.handle(AgentTurn("p", "question", query_text))
            assert language.messages
            assert sop.context_snapshot("p")["status"] == "idle"
        finally:
            await router.stop()
            await sop.stop()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
async def test_scoped_parallel_flow_keeps_its_own_state_across_foreground_steps():
    selected = load_config(_SAMPLE / "yaml/tea_making_worker.yaml")
    sop, runtime, context = _runner(config=replace(selected.sop, advancement="manual", poll_interval_seconds=0.001))
    router, conversation, _language = _router(sop, runtime, context)
    async with runtime:
        try:
            await router.handle(AgentTurn("p", "start", "start tea making"))
            await _until(lambda: sop._sessions["p"].ready)
            session = sop._sessions["p"]
            async with asyncio.TaskGroup() as tasks:
                spanning_flow = tasks.create_task(
                    run_timer_flow(
                        duration_seconds=100,
                        update_interval_seconds=100,
                        on_state=partial(sop._flow_state, session, session.run_id, "spanning_timer"),
                    )
                )
                await _until(lambda: "spanning_timer" in sop.context_snapshot("p")["active_flows"])
                await router.handle(AgentTurn("p", "next", "next step"))
                await _until(
                    lambda: sop._sessions["p"].ready and sop.context_snapshot("p")["current_step"] == "fill_water"
                )
                await router.handle(AgentTurn("p", "next2", "next step"))
                await _until(lambda: sop.context_snapshot("p")["current_step"] == "heat_water")
                assert "spanning_timer" in sop.context_snapshot("p")["active_flows"]
                await router.handle(AgentTurn("p", "stop", "stop tea making"))
                assert spanning_flow.done()
            assert sop.context_snapshot("p")["active_flows"] == {}
        finally:
            await router.stop()
            await sop.stop()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
async def test_runtime_departure_drains_work_before_ack_and_readmits_fresh_session():
    from tea_making_worker._cleanup import _CleanupAcks
    from tea_making_worker.events import (
        PARTICIPANT_CLEANUP_COMPLETE_TOPIC,
        PARTICIPANT_JOINED_TOPIC,
        PARTICIPANT_LEFT_TOPIC,
        USER_QUERY_TOPIC,
    )
    from xr_ai_interaction import ParticipantJoined, ParticipantLeft, UserQuery
    from xr_ai_runtime import Agent, subscribe

    inference_started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    backend_released = asyncio.Event()

    async def observe(request):
        if "Return JSON matching" in request.query:
            return ImageQueryResult(text='{"tea_name":"Northwind White Tea"}')
        inference_started.set()
        try:
            await backend_released.wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await backend_released.wait()
        return ImageQueryResult(text="late visual result")

    sop, runtime, context = _runner(observer=SimpleNamespace(execute=observe))
    language = _ConversationLanguage()
    conversation = Conversation(
        language,
        context=context,
        runtime=runtime,
        output_topic=VOICE_OUTPUT_TOPIC,
        instructions=sop.config.conversation_instructions,
    )
    router = AgentRouter(
        {"workflow": sop, "conversation": conversation},
        sop._decisions,
        query_topic=USER_QUERY_TOPIC,
        output_topic=VOICE_OUTPUT_TOPIC,
        participant_joined_topic=PARTICIPANT_JOINED_TOPIC,
        initialize_participant=sop.participant_joined,
        cleanup_participant=sop.participant_left,
    )
    runtime.register("routing", router.runtime_subscriber())
    runtime.register("cleanup", _CleanupAcks(router))
    acknowledgments = []

    class Recorder(Agent):
        @subscribe(PARTICIPANT_CLEANUP_COMPLETE_TOPIC)
        async def observe(self, message: ParticipantCleanupComplete, ctx: RuntimeContext):
            acknowledgments.append((message.generation, message.producer, ctx.metadata.participant_id))

    runtime.register("recorder", Recorder())
    async with runtime:
        try:
            await runtime.publish(PARTICIPANT_JOINED_TOPIC, ParticipantJoined(), participant_id="p")
            await _until(lambda: "p" in sop._sessions)
            await runtime.publish(
                USER_QUERY_TOPIC, UserQuery(text="start tea making", timestamp_us=100), participant_id="p"
            )
            await asyncio.wait_for(inference_started.wait(), 1)
            old_session = sop._sessions["p"]
            leave_task = asyncio.create_task(
                runtime.publish(
                    PARTICIPANT_LEFT_TOPIC,
                    ParticipantLeft(),
                    participant_id="p",
                ),
                context=nemo_relay.fork_asyncio_context(),
            )
            await asyncio.wait_for(cancellation_seen.wait(), 1)
            assert not leave_task.done()
            assert acknowledgments == []
            rejoin_task = asyncio.create_task(
                runtime.publish(
                    PARTICIPANT_JOINED_TOPIC,
                    ParticipantJoined(),
                    participant_id="p",
                ),
                context=nemo_relay.fork_asyncio_context(),
            )
            backend_released.set()
            await asyncio.wait_for(asyncio.gather(leave_task, rejoin_task), 1)
            await _until(lambda: "p" in sop._sessions and sop._sessions["p"] is not old_session)
            assert sop.context_snapshot("p")["status"] == "idle"
            assert sop.context_snapshot("p")["active_flows"] == {}
            assert len(acknowledgments) == 1
            assert len({item[0] for item in acknowledgments}) == 1
            assert {item[1] for item in acknowledgments} == {"guidance"}
            assert not old_session.active_tasks
        finally:
            backend_released.set()
            await router.stop()
            await sop.stop()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
async def test_superseding_stop_acknowledgment_does_not_abandon_committed_stop():
    inference_started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    backend_released = asyncio.Event()

    async def observe(request):
        if "Return JSON matching" in request.query:
            return ImageQueryResult(text='{"tea_name":"Northwind White Tea"}')
        inference_started.set()
        try:
            await backend_released.wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await backend_released.wait()
        return ImageQueryResult(text="late visual result")

    sop, runtime, context = _runner(observer=SimpleNamespace(execute=observe))
    router, conversation, language = _router(sop, runtime, context)
    async with runtime:
        try:
            await router.handle(AgentTurn("p", "start", "start tea making"))
            await asyncio.wait_for(inference_started.wait(), 1)
            stop_task = asyncio.create_task(router.handle(AgentTurn("p", "stop", "stop tea making")))
            await asyncio.wait_for(cancellation_seen.wait(), 1)
            question_task = asyncio.create_task(router.handle(AgentTurn("p", "qa", "What happened?")))
            backend_released.set()
            results = await asyncio.wait_for(
                asyncio.gather(stop_task, question_task, return_exceptions=True),
                1,
            )
            assert isinstance(results[0], asyncio.CancelledError)
            assert sop.context_snapshot("p")["status"] == "stopped"
            assert sop.context_snapshot("p")["active_flows"] == {}
            assert language.messages
        finally:
            backend_released.set()
            await router.stop()
            await sop.stop()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["skip", "stop"])
async def test_delayed_controls_bind_step_or_run_as_appropriate(control):
    water_started = asyncio.Event()
    water_released = asyncio.Event()
    heat_started = asyncio.Event()
    heat_released = asyncio.Event()
    classification_started = asyncio.Event()
    classification_released = asyncio.Event()
    selected = load_config(_SAMPLE / "yaml/tea_making_worker.yaml")

    async def observe(request):
        if "Return JSON matching" in request.query:
            return ImageQueryResult(text='{"tea_name":"Northwind White Tea"}')
        if request.query == selected.sop.water.question:
            water_started.set()
            await water_released.wait()
        else:
            heat_started.set()
            await heat_released.wait()
        return ImageQueryResult(text="clear positive evidence")

    class DelayedDecisions(_Decisions):
        async def decide(self, state, questions, **kwargs):
            if "control" in questions and state["request"] in {"skip", "stop tea making"}:
                classification_started.set()
                await classification_released.wait()
            return await super().decide(state, questions, **kwargs)

    sop, runtime, context = _runner(observer=SimpleNamespace(execute=observe))
    sop._decisions = DelayedDecisions()
    router, conversation, _language = _router(sop, runtime, context)
    async with runtime:
        try:
            await router.handle(AgentTurn("p", "start", "start tea making"))
            await asyncio.wait_for(water_started.wait(), 1)
            control_task = asyncio.create_task(
                router.handle(
                    AgentTurn(
                        "p",
                        "delayed",
                        "skip" if control == "skip" else "stop tea making",
                    )
                )
            )
            await asyncio.wait_for(classification_started.wait(), 1)
            water_released.set()
            await asyncio.wait_for(heat_started.wait(), 1)
            classification_released.set()
            result = await asyncio.wait_for(control_task, 1)
            if control == "skip":
                assert result.text == sop.config.messages["changed"]
                assert sop.context_snapshot("p")["current_step"] == "heat_water"
                assert sop.context_snapshot("p")["status"] == "running"
            else:
                assert result.text == sop.config.messages["stop"]
                assert sop.context_snapshot("p")["status"] == "stopped"
        finally:
            water_released.set()
            heat_released.set()
            classification_released.set()
            await router.stop()
            await sop.stop()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
async def test_sdk_conversation_receives_full_recent_flow_context_not_only_a_completion_flag():
    selected = load_config(_SAMPLE / "yaml/tea_making_worker.yaml")
    third_observation_started = asyncio.Event()
    observation_number = 0

    async def observe(request):
        nonlocal observation_number
        if "Return JSON matching" in request.query:
            return ImageQueryResult(text='{"tea_name":"Northwind White Tea"}')
        observation_number += 1
        if observation_number == 3:
            third_observation_started.set()
            await asyncio.Event().wait()
        return ImageQueryResult(
            text=(
                f"Observation {observation_number}: "
                "a continuous water surface and reflection are visible inside the pot."
            )
        )

    sop, runtime, context = _runner(
        config=replace(selected.sop, required_positives=3, history_limit=2, poll_interval_seconds=0.001),
        observer=SimpleNamespace(execute=observe),
    )
    router, conversation, language = _router(sop, runtime, context)
    async with runtime:
        try:
            await router.handle(AgentTurn("p", "start", "start tea making"))
            await asyncio.wait_for(third_observation_started.wait(), 1)
            await router.handle(AgentTurn("p", "qa", "What is visible and why are we waiting?"))
            messages = "\n".join(message.content for message in language.messages[-1])
            assert "Observation 1: a continuous water surface" in messages
            assert "Observation 2: a continuous water surface" in messages
            assert '"positives_in_window": 2' in messages
            assert '"required_positives": 3' in messages
            assert '"window_size": 4' in messages
            assert '"frame_key":' in messages and '"updated_at_us":' in messages
            assert '"temperature_c": 80.0' in messages
            assert selected.sop.water.completion.criteria["positive"] in messages
            assert "truncated" not in messages.lower() or '"truncated": false' in messages
            assert len(messages.encode()) < 64000
        finally:
            await router.stop()
            await sop.stop()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
async def test_skipping_unconfirmed_tea_contact_does_not_start_a_timer():
    selected = load_config(_SAMPLE / "yaml/tea_making_worker.yaml")
    contact_started = asyncio.Event()

    async def observe(request):
        if "Return JSON matching" in request.query:
            return ImageQueryResult(text='{"tea_name":"Northwind White Tea"}')
        if request.query == selected.sop.tea_bag.question:
            contact_started.set()
            await asyncio.Event().wait()
        return ImageQueryResult(text="Clear positive evidence.")

    sop, runtime, context = _runner(observer=SimpleNamespace(execute=observe))
    router, conversation, _language = _router(sop, runtime, context)
    async with runtime:
        try:
            await router.handle(AgentTurn("p", "start", "start tea making"))
            await asyncio.wait_for(contact_started.wait(), 1)
            await router.handle(AgentTurn("p", "skip", "skip"))
            await _until(lambda: sop.context_snapshot("p")["status"] == "completed")
            snapshot = sop.context_snapshot("p")
            assert snapshot["flow_outcomes"]["start_steeping"] == "skipped"
            assert snapshot["flow_outcomes"]["steep_timer"] == "skipped"
            assert "steep_timer" not in sop._sessions["p"].flow_states
        finally:
            await router.stop()
            await sop.stop()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
async def test_stopping_runner_drains_every_parallel_child_before_returning():
    water_started = asyncio.Event()

    async def observe(request):
        if "Return JSON matching" in request.query:
            return ImageQueryResult(text='{"tea_name":"Northwind White Tea"}')
        water_started.set()
        await asyncio.Event().wait()

    sop, runtime, context = _runner(observer=SimpleNamespace(execute=observe))
    router, conversation, _language = _router(sop, runtime, context)
    child_started = [asyncio.Event(), asyncio.Event()]
    child_cancelled = [asyncio.Event(), asyncio.Event()]
    release_children = asyncio.Event()

    async def receive(index, state):
        session = sop._sessions["p"]
        await sop._flow_state(session, session.run_id, f"parallel-{index}", state)
        child_started[index].set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            child_cancelled[index].set()
            await release_children.wait()
            raise

    async with runtime:
        try:
            await router.handle(AgentTurn("p", "start", "start tea making"))
            await asyncio.wait_for(water_started.wait(), 1)
            session = sop._sessions["p"]
            children = [
                asyncio.create_task(
                    run_timer_flow(
                        duration_seconds=100,
                        update_interval_seconds=100,
                        on_state=partial(receive, index),
                    )
                )
                for index in range(2)
            ]
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in child_started)), 1)
            stop_task = asyncio.create_task(router.handle(AgentTurn("p", "stop", "stop tea making")))
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in child_cancelled)), 1)
            assert not stop_task.done()
            assert not all(task.done() for task in children)
            release_children.set()
            await asyncio.wait_for(stop_task, 1)
            assert all(task.done() for task in children)
            assert session.active_tasks == {}
        finally:
            release_children.set()
            await router.stop()
            await sop.stop()
            await conversation.close()
            await context.close()
