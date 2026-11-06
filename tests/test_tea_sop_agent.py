# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Production runner, SDK Conversation and router lifecycle boundaries."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import nemo_relay
import pytest
from xr_ai_conversation import ContextManager, Conversation
from xr_ai_models import ChatResponse, ChatStreamEvent, DecisionAnswer, DecisionResponse
from xr_ai_routing import AgentRouter, AgentTurn, ParticipantAgentFactory, RoutingAid
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, subscribe
from xr_ai_tools.current_frame import CurrentFrameTool, ImageFrame
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.rag import RetrievedChunk, RetrieveResult
from xr_ai_tools.vision import ImageQueryResult
from xr_ai_voice import VOICE_OUTPUT_TOPIC

_SAMPLE = Path(__file__).resolve().parents[1] / "agent-samples/tea-making-sample"
sys.path.insert(0, str(_SAMPLE / "worker"))

from tea_making_worker.application.application_config import load_config  # noqa: E402
from tea_making_worker.application.tea_events import (  # noqa: E402
    GUIDANCE_NOTICE_TOPIC,
    GUIDANCE_RECORD_TOPIC,
    USER_QUERY_TOPIC,
    GuidanceNotice,
    GuidanceRecord,
)
from tea_making_worker.application.tea_sop_agent import (  # noqa: E402
    SOP_CONTEXT_SOURCE,
    SOPAgent,
)
from tea_making_worker.components import run_timer_flow  # noqa: E402


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


def _runner(*, config=None, observer=None, llm=None, frames=None):
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
    registry = frames.images if frames is not None else ImageRegistry()
    image = registry.put(b"fixture", owner="p")
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
                            name: {"applicability": "applicable", "passage_id": "P1", "quote": source_text}
                            for name in ("temperature", "steep_duration")
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
    if frames is not None:
        frames.execute = acquire
    sop = SOPAgent(
        participant_id="p",
        config=config,
        llm=llm,
        decisions=_Decisions(),
        frames=frames or SimpleNamespace(execute=acquire),
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
        self.response = "The latest SOP context is available."

    async def stream_events(self, messages):
        self.messages.append(messages)
        yield ChatStreamEvent(kind="text", text=self.response)
        yield ChatStreamEvent(kind="finish", finish_reason="stop")


def _router(sop, runtime, context, conversation_language=None, *, scoped_instances=None, **router_options):
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
    created = False

    async def create_sop(participant_id):
        nonlocal created
        if not created and participant_id == sop._session.participant_id:
            scoped_sop = sop
            created = True
        else:
            scoped_sop = SOPAgent(
                participant_id=participant_id,
                config=sop.config,
                llm=sop._llm,
                decisions=sop._decisions,
                frames=sop._frames,
                image_query=sop._image_query,
                rag=sop._rag,
                context=context,
                runtime=runtime,
            )
        try:
            await scoped_sop.initialize()
        except BaseException:
            await scoped_sop.close()
            raise
        if scoped_instances is not None:
            scoped_instances.append(scoped_sop)
        return scoped_sop

    async def close_sop(scoped_sop):
        await scoped_sop.close()

    async def cleanup_participant(participant_id):
        await context.clear(participant_id)

    router = AgentRouter(
        {"workflow": ParticipantAgentFactory(create=create_sop, close=close_sop), "conversation": conversation},
        sop._decisions,
        query_topic=USER_QUERY_TOPIC,
        output_topic=VOICE_OUTPUT_TOPIC,
        output=AsyncMock(),
        cleanup_participant=cleanup_participant,
        **router_options,
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
            await _until(lambda: sop._session.status == "completed")
            snapshot = sop.context_snapshot()
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
            await sop.close()
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
            procedure = sop._session.procedure
            response = await router.handle(AgentTurn("p", "qa", "What tea did we identify?"))
            assert "latest SOP context" in response.text
            assert sop._session.procedure is procedure and not procedure.done()
            messages = "\n".join(message.content for message in language.messages[-1])
            assert "Northwind White Tea" in messages
            assert '"temperature_c": 80.0' in messages
            assert '"current_step": "fill_water"' in messages
            assert '"kind": "retrieved"' in messages
            await router.handle(AgentTurn("p", "stop", "stop tea making"))
            assert procedure.done()
            assert sop.context_snapshot()["status"] == "stopped"
            assert sop.context_snapshot()["active_flows"] == {}
        finally:
            await router.stop()
            await sop.close()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
async def test_skip_retains_partial_settings_and_defaults_only_missing_fields():
    fallback_started = asyncio.Event()
    water_started = asyncio.Event()

    async def chat(messages, **_kwargs):
        payload = json.loads(messages[1].content)
        if "sources" in payload["response_schema"]["properties"]:
            return ChatResponse(
                json.dumps(
                    {
                        "values": {"temperature": {"value": 80, "unit": "celsius"}},
                        "sources": {
                            "temperature": {
                                "applicability": "applicable",
                                "passage_id": "P1",
                                "quote": "80 degrees Celsius",
                            }
                        },
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
            snapshot = sop.context_snapshot()
            assert snapshot["current_step"] == "fill_water"
            assert snapshot["resolved_settings"]["temperature_c"] == 80
            assert snapshot["resolved_settings"]["duration_seconds"] == 180
            assert snapshot["resolved_settings"]["sources"]["temperature"]["kind"] == "retrieved"
            assert snapshot["resolved_settings"]["sources"]["steep_duration"]["kind"] == "default"
            assert snapshot["completed_or_skipped_flows"]["identify"]["outcome"] == "skipped"
        finally:
            await router.stop()
            await sop.close()
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
            await _until(lambda: sop._session.ready)
            assert sop.context_snapshot()["current_step"] == "identify"
            await router.handle(AgentTurn("p", "next", "next step"))
            await _until(lambda: sop.context_snapshot()["current_step"] == "fill_water")
            # Block this step before it can become ready; a query never skips it.
            session = sop._session
            if not session.ready:
                response = await router.handle(AgentTurn("p", "early-next", "next step"))
                assert response.text == sop.config.messages["not_ready"]
            assert session.status == "running"
        finally:
            await router.stop()
            await sop.close()
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
            assert sop.context_snapshot()["status"] == "idle"
        finally:
            await router.stop()
            await sop.close()
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
            await _until(lambda: sop._session.ready)
            session = sop._session
            async with asyncio.TaskGroup() as tasks:
                spanning_record = sop._launch_flow(
                    session,
                    "spanning_timer",
                    lambda on_state: run_timer_flow(
                        duration_seconds=100,
                        update_interval_seconds=100,
                        on_state=on_state,
                    ),
                    task_group=tasks,
                )
                spanning_flow = spanning_record.task
                await _until(lambda: "spanning_timer" in sop.context_snapshot()["active_flows"])
                await router.handle(AgentTurn("p", "next", "next step"))
                await _until(lambda: sop._session.ready and sop.context_snapshot()["current_step"] == "fill_water")
                await router.handle(AgentTurn("p", "next2", "next step"))
                await _until(lambda: sop.context_snapshot()["current_step"] == "heat_water")
                assert "spanning_timer" in sop.context_snapshot()["active_flows"]
                await router.handle(AgentTurn("p", "stop", "stop tea making"))
                assert spanning_flow.done()
            assert sop.context_snapshot()["active_flows"] == {}
        finally:
            await router.stop()
            await sop.close()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
async def test_runtime_departure_drains_work_before_releasing_frames_and_readmits_fresh_session():
    from tea_making_worker.application.tea_events import (
        PARTICIPANT_JOINED_TOPIC,
        PARTICIPANT_LEFT_TOPIC,
        USER_QUERY_TOPIC,
    )
    from xr_ai_interaction import ParticipantJoined, ParticipantLeft, UserQuery

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

    registry = ImageRegistry()
    frames = CurrentFrameTool(endpoint=Mock(), images=registry)
    sop, runtime, context = _runner(observer=SimpleNamespace(execute=observe), frames=frames)
    language = _ConversationLanguage()
    conversation = Conversation(
        language,
        context=context,
        runtime=runtime,
        output_topic=VOICE_OUTPUT_TOPIC,
        instructions=sop.config.conversation_instructions,
    )
    released_participants = []
    instances = []

    async def create_sop(participant_id):
        scoped_sop = (
            sop
            if not instances
            else SOPAgent(
                participant_id=participant_id,
                config=sop.config,
                llm=sop._llm,
                decisions=sop._decisions,
                frames=frames,
                image_query=sop._image_query,
                rag=sop._rag,
                context=context,
                runtime=runtime,
            )
        )
        instances.append(scoped_sop)
        await scoped_sop.initialize()
        return scoped_sop

    async def close_sop(scoped_sop):
        await scoped_sop.close()

    async def cleanup_participant(participant_id):
        await context.clear(participant_id)
        frames.release(participant_id)
        released_participants.append(participant_id)

    router = AgentRouter(
        {"workflow": ParticipantAgentFactory(create=create_sop, close=close_sop), "conversation": conversation},
        sop._decisions,
        query_topic=USER_QUERY_TOPIC,
        output_topic=VOICE_OUTPUT_TOPIC,
        participant_joined_topic=PARTICIPANT_JOINED_TOPIC,
        participant_left_topic=PARTICIPANT_LEFT_TOPIC,
        cleanup_participant=cleanup_participant,
    )
    runtime.register("routing", router.runtime_subscriber())
    async with runtime:
        try:
            await runtime.publish(PARTICIPANT_JOINED_TOPIC, ParticipantJoined(), participant_id="p")
            await _until(lambda: len(instances) == 1)
            await runtime.publish(
                USER_QUERY_TOPIC, UserQuery(text="start tea making", timestamp_us=100), participant_id="p"
            )
            await asyncio.wait_for(inference_started.wait(), 1)
            old_session = sop._session
            leave_task = asyncio.create_task(
                runtime.publish(
                    PARTICIPANT_LEFT_TOPIC,
                    ParticipantLeft(),
                    participant_id="p",
                ),
                context=nemo_relay.fork_asyncio_context(),
            )
            await asyncio.wait_for(cancellation_seen.wait(), 1)
            old_session.procedure.cancel()
            await asyncio.sleep(0)
            assert old_session.flows["fill_water"].task.cancelling() == 1
            assert not leave_task.done()
            assert released_participants == []
            assert len(registry) == 1
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
            await _until(lambda: len(instances) == 2)
            assert instances[1] is not instances[0]
            assert instances[1].context_snapshot()["status"] == "idle"
            assert instances[1].context_snapshot()["active_flows"] == {}
            assert not instances[0]._session.connected
            assert released_participants == ["p"]
            assert len(registry) == 0
            new_image = registry.put(b"new session image", owner="p")
            assert registry.resolve(new_image) == b"new session image"
            assert all(record.task is None for record in old_session.flows.values())
        finally:
            backend_released.set()
            await router.stop()
            await sop.close()
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
            assert sop.context_snapshot()["status"] == "stopped"
            assert sop.context_snapshot()["active_flows"] == {}
            assert language.messages
        finally:
            backend_released.set()
            await router.stop()
            await sop.close()
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
                assert sop.context_snapshot()["current_step"] == "heat_water"
                assert sop.context_snapshot()["status"] == "running"
            else:
                assert result.text == sop.config.messages["stop"]
                assert sop.context_snapshot()["status"] == "stopped"
        finally:
            water_released.set()
            heat_released.set()
            classification_released.set()
            await router.stop()
            await sop.close()
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
            await sop.close()
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
            await _until(lambda: sop.context_snapshot()["status"] == "completed")
            snapshot = sop.context_snapshot()
            assert snapshot["flow_outcomes"]["start_steeping"] == "skipped"
            assert snapshot["flow_outcomes"]["steep_timer"] == "skipped"
            assert sop._session.flows["steep_timer"].state is None
        finally:
            await router.stop()
            await sop.close()
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

    async def receive(index, on_state, state):
        await on_state(state)
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
            session = sop._session
            children = [
                sop._launch_flow(
                    session,
                    f"parallel-{index}",
                    lambda on_state, index=index: run_timer_flow(
                        duration_seconds=100,
                        update_interval_seconds=100,
                        on_state=partial(receive, index, on_state),
                    ),
                ).task
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
            assert all(record.task is None for record in session.flows.values())
        finally:
            release_children.set()
            await router.stop()
            await sop.close()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
async def test_type_only_identity_labels_runner_context_without_a_missing_name():
    water_started = asyncio.Event()

    async def observe(request):
        if "Return JSON matching" in request.query:
            return ImageQueryResult(text='{"tea_type":"white tea"}')
        water_started.set()
        await asyncio.Event().wait()

    sop, runtime, context = _runner(observer=SimpleNamespace(execute=observe))
    router, conversation, _language = _router(sop, runtime, context)
    async with runtime:
        try:
            await router.handle(AgentTurn("p", "start", "start tea making"))
            await asyncio.wait_for(water_started.wait(), 1)
            snapshot = sop.context_snapshot()
            assert snapshot["resolved_settings"]["tea_name"] == "white tea"
            assert snapshot["current_step"] == "fill_water"
            query = sop._rag.retrieve.execute.await_args.args[0].query
            assert query == "white tea " + sop.config.retrieval_query_suffix
        finally:
            await router.stop()
            await sop.close()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
async def test_conversation_context_exposes_only_entered_steps_and_established_settings():
    """Startup capabilities must not receive the future plan or unused defaults."""
    identification_started = asyncio.Event()
    release_identification = asyncio.Event()
    water_started = asyncio.Event()

    async def observe(request):
        if "Return JSON matching" in request.query:
            identification_started.set()
            await release_identification.wait()
            return ImageQueryResult(text='{"tea_name":"Northwind White Tea"}')
        water_started.set()
        await asyncio.Event().wait()

    sop, runtime, context = _runner(observer=SimpleNamespace(execute=observe))
    router, conversation, language = _router(sop, runtime, context)

    async def ask(question, request_id):
        await router.handle(AgentTurn("p", request_id, question))
        return "\n".join(message.content for message in language.messages[-1])

    async with runtime:
        try:
            messages = await ask("What can you do?", "capabilities")
            assert sop.context_snapshot() == {
                "status": "idle",
                "active_flows": {},
                "historical_previous_run": None,
            }
            for absent in ("procedure_steps", "resolved_settings", "93", "180", "Fill a kettle", "Heat the water"):
                assert absent not in messages
            assert "tea-making guidance" in messages

            await router.handle(AgentTurn("p", "start", "start tea making"))
            await asyncio.wait_for(identification_started.wait(), 1)
            messages = await ask("What are we doing?", "identifying")
            snapshot = sop.context_snapshot()
            assert list(snapshot["entered_step_instructions"]) == ["identify"]
            assert "resolved_settings" not in snapshot
            assert "Hold the tea package" in messages
            assert "Heat the water" not in messages and '"temperature_c": 93' not in messages

            release_identification.set()
            await asyncio.wait_for(water_started.wait(), 1)
            messages = await ask("What is the current step?", "water")
            snapshot = sop.context_snapshot()
            assert list(snapshot["entered_step_instructions"]) == ["identify", "fill_water"]
            assert snapshot["resolved_settings"]["temperature_c"] == 80
            assert "Northwind White Tea" in messages and "Fill a kettle" in messages
            assert "Heat the water" not in messages and "Steep for" not in messages

            await router.handle(AgentTurn("p", "stop", "stop tea making"))
            snapshot = sop.context_snapshot()
            assert snapshot["status"] == "stopped" and snapshot["active_flows"] == {}
            await router.handle(AgentTurn("p", "reset", "reset"))
            messages = await ask("Is guidance running?", "after-reset")
            snapshot = sop.context_snapshot()
            assert snapshot["status"] == "idle" and snapshot["active_flows"] == {}
            assert "resolved_settings" not in snapshot and "entered_step_instructions" not in snapshot
            assert snapshot["historical_previous_run"]["status"] == "stopped"
            assert '"historical_previous_run"' in messages
        finally:
            release_identification.set()
            await router.stop()
            await sop.close()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
async def test_consecutive_timer_questions_receive_fresh_native_samples_and_lifecycle_context(monkeypatch):
    """Live source values replace previous values while old answers stay historical."""
    import time

    from tea_making_worker.application.tea_events import PARTICIPANT_JOINED_TOPIC
    from tea_making_worker.components import timer_flow
    from xr_ai_interaction import ParticipantJoined

    elapsed = 47.0
    started_at_us = time.time_ns() // 1000 - 47_000_000
    monkeypatch.setattr(
        timer_flow,
        "time",
        SimpleNamespace(
            monotonic=lambda: elapsed,
            time_ns=lambda: (started_at_us + round(elapsed * 1_000_000)) * 1000,
        ),
    )
    sop, runtime, context = _runner()
    router, conversation, language = _router(sop, runtime, context, participant_joined_topic=PARTICIPANT_JOINED_TOPIC)
    async with runtime:
        try:
            await sop.initialize()
            session = sop._session
            session.status = "running"
            session.run_id = "timer-regression"
            session.step_index = 4
            # Native timer begins at this monotonic origin, then advances under
            # the test clock. The actual callback/SDK context path is unchanged.
            timer_record = sop._launch_flow(
                session,
                "steep_timer",
                lambda on_state: run_timer_flow(
                    duration_seconds=150,
                    update_interval_seconds=0.001,
                    on_state=on_state,
                ),
            )
            timer = timer_record.task
            session.procedure = timer
            await _until(lambda: timer_record.state is not None)
            for number, (advance, remaining, question) in enumerate(
                (
                    (47, 103, "How long do I have?"),
                    (81, 69, "Okay what about now?"),
                    (92, 58, "How much longer do I need to wait?"),
                )
            ):
                elapsed = 47 + advance
                await _until(lambda: timer_record.state.remaining_seconds == remaining)
                language.response = f"{remaining} seconds remaining"
                response = await router.handle(AgentTurn("p", f"timer-{number}", question))
                assert response.text == language.response
                messages = "\n".join(message.content for message in language.messages[-1])
                assert f'"remaining_seconds": {remaining}.0' in messages
                assert '"sampled_at_utc":' in messages
                if remaining != 103:
                    assert '"remaining_seconds": 103.0' not in messages
                    assert "103 seconds remaining" in messages
            await router.handle(AgentTurn("p", "stop-timer", "stop tea making"))
            assert timer.done()
            await router.handle(AgentTurn("p", "reset-timer", "reset"))
            assert sop.context_snapshot()["active_flows"] == {}
            assert sop.context_snapshot()["historical_previous_run"]["run_id"] == "timer-regression"
            await router.participant_left("p")
            await runtime.publish(PARTICIPANT_JOINED_TOPIC, ParticipantJoined(), participant_id="p")
            await router.handle(AgentTurn("p", "fresh-session", "What can you do?"))
            messages = "\n".join(message.content for message in language.messages[-1])
            assert "103 seconds remaining" not in messages and "timer-regression" not in messages
        finally:
            await router.stop()
            await sop.close()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
async def test_public_router_can_start_after_reset_and_after_each_completed_procedure():
    """Complete real five-step runs repeatedly without clearing conversation history."""
    sop, runtime, context = _runner()
    router, conversation, _language = _router(sop, runtime, context)
    async with runtime:
        try:
            run_ids = []
            for number in range(3):
                response = await router.handle(AgentTurn("p", f"start-{number}", "start tea making"))
                assert response.text == sop.config.messages["start"]
                await _until(lambda: sop.context_snapshot()["status"] == "completed")
                session = sop._session
                run_ids.append(session.run_id)
                await _until(lambda: session.procedure.done())
                assert all(record.task is None for record in session.flows.values())
                if number == 0:
                    response = await router.handle(AgentTurn("p", "reset-between", "reset"))
                    assert response.text == sop.config.messages["reset"]
                    assert sop.context_snapshot()["status"] == "idle"
            assert len(set(run_ids)) == 3
            assert sop.context_snapshot()["historical_previous_run"]["run_id"] == run_ids[-2]
        finally:
            await router.stop()
            await sop.close()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
async def test_start_queued_during_active_reset_waits_for_old_inference_cleanup():
    first_water = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release_old_inference = asyncio.Event()
    second_water = asyncio.Event()
    water_calls = 0

    async def observe(request):
        nonlocal water_calls
        if "Return JSON matching" in request.query:
            return ImageQueryResult(text='{"tea_type":"white tea"}')
        water_calls += 1
        if water_calls == 1:
            first_water.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_seen.set()
                await release_old_inference.wait()
            return ImageQueryResult(text="A late old-run observation.")
        second_water.set()
        await asyncio.Event().wait()

    sop, runtime, context = _runner(observer=SimpleNamespace(execute=observe))
    router, conversation, _language = _router(sop, runtime, context)
    async with runtime:
        try:
            await router.handle(AgentTurn("p", "start-old", "start tea making"))
            await asyncio.wait_for(first_water.wait(), 1)
            old_run = sop._session.run_id
            old_procedure = sop._session.procedure
            resetting = asyncio.create_task(router.handle(AgentTurn("p", "reset-active", "reset")))
            await asyncio.wait_for(cancellation_seen.wait(), 1)
            restarting = asyncio.create_task(router.handle(AgentTurn("p", "start-new", "start tea making")))
            release_old_inference.set()
            await asyncio.gather(resetting, return_exceptions=True)
            response = await asyncio.wait_for(restarting, 1)
            await asyncio.wait_for(second_water.wait(), 1)
            assert response.text == sop.config.messages["start"]
            assert old_procedure.done() and sop._session.run_id != old_run
            assert "A late old-run observation." not in json.dumps(sop.context_snapshot())
        finally:
            release_old_inference.set()
            await router.stop()
            await sop.close()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
async def test_unresolved_identification_notifies_once_without_defaults_or_wording_spam():
    notices = []
    observation_calls = 0
    all_observations_seen = asyncio.Event()

    async def resolve(messages, **_kwargs):
        payload = json.loads(messages[1].content)
        values = (
            {"temperature": None, "steep_duration": None}
            if "sources" in payload["response_schema"]["properties"]
            else {"temperature": {"value": 85, "unit": "celsius"}, "steep_duration": {"value": 3, "unit": "minutes"}}
        )
        return ChatResponse(json.dumps({"values": values}), None, (), "stop", {})

    language = SimpleNamespace(chat=AsyncMock(side_effect=resolve))

    class Capture(Agent):
        @subscribe(GUIDANCE_NOTICE_TOPIC)
        async def notice(self, event: GuidanceNotice, _context: RuntimeContext):
            notices.append(event.text)

    async def observe(_request):
        nonlocal observation_calls
        observation_calls += 1
        if observation_calls == 1:
            return ImageQueryResult(text="null")
        if observation_calls == 2:
            language.chat.assert_not_awaited()
            return ImageQueryResult(text='{"tea_name":"Moon Orchard"}')
        if observation_calls == 3:
            return ImageQueryResult(text='{"tea_name":" Moon Orchard "}')
        if observation_calls == 4:
            return ImageQueryResult(text="null")
        all_observations_seen.set()
        await asyncio.Event().wait()

    sop, runtime, context = _runner(observer=SimpleNamespace(execute=observe), llm=language)
    runtime.register("notices", Capture())
    router, conversation, _ = _router(sop, runtime, context)
    async with runtime:
        try:
            await router.handle(AgentTurn("p", "start-pending", "start tea making"))
            await asyncio.wait_for(all_observations_seen.wait(), 1)
            pending_notices = [text for text in notices if text.startswith("I can read")]
            assert len(pending_notices) == 1
            assert "water temperature and steeping time" in pending_notices[0]
            assert "skip" in pending_notices[0] and "Show the tea type" in pending_notices[0]
            assert language.chat.await_count == 2
            sop._rag.retrieve.execute.assert_awaited_once()
            state = sop.context_snapshot()
            assert "resolved_settings" not in state
            assert state["active_flows"]["identify"]["visual_data"]["tea_name"] == "Moon Orchard"
            assert state["active_flows"]["identify"]["resolved_data"]["temperature"] is None

            await router.handle(AgentTurn("p", "reset-pending", "reset"))
            observation_calls = 0
            language.chat.reset_mock()
            all_observations_seen.clear()
            await router.handle(AgentTurn("p", "start-pending-again", "start tea making"))
            await asyncio.wait_for(all_observations_seen.wait(), 1)
            assert len([text for text in notices if text.startswith("I can read")]) == 2
        finally:
            await router.stop()
            await sop.close()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
async def test_explicit_restart_launches_new_first_inference_while_start_keeps_active_run():
    first_inferences = []

    async def observe(request):
        first_inferences.append(request.query)
        await asyncio.Event().wait()

    sop, runtime, context = _runner(observer=SimpleNamespace(execute=observe))
    sop._decisions.controls = {
        **sop._decisions.controls,
        "restart tea making": "restart",
        "let's make tea": "start",
    }
    router, conversation, _language = _router(sop, runtime, context)
    async with runtime:
        try:
            await router.handle(AgentTurn("p", "first", "let's make tea"))
            await _until(lambda: len(first_inferences) == 1)
            old_run = sop._session.run_id
            old_procedure = sop._session.procedure
            await router.handle(AgentTurn("p", "restart-first", "restart tea making"))
            await _until(lambda: len(first_inferences) == 2)
            new_run = sop._session.run_id
            assert old_procedure.done() and new_run != old_run
            response = await router.handle(AgentTurn("p", "already-active", "let's make tea"))
            assert response.text == sop.config.messages["already_running"]
            assert sop._session.run_id == new_run and len(first_inferences) == 2
        finally:
            await router.stop()
            await sop.close()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
async def test_reset_then_no_tea_starts_without_previous_identity_settings_or_rag_calls():
    previous_step_started = asyncio.Event()
    no_tea_seen = asyncio.Event()
    show_tea = True

    async def observe(request):
        if "Return JSON matching" in request.query:
            if show_tea:
                return ImageQueryResult(text='{"tea_name":"White tea","tea_type":"white tea"}')
            no_tea_seen.set()
            return ImageQueryResult(text='{"tea_name":null,"tea_type":null}')
        previous_step_started.set()
        await asyncio.Event().wait()

    sop, runtime, context = _runner(observer=SimpleNamespace(execute=observe))
    router, conversation, _language = _router(sop, runtime, context)
    async with runtime:
        try:
            await router.handle(AgentTurn("p", "identified-run", "start tea making"))
            await asyncio.wait_for(previous_step_started.wait(), 1)
            previous_run = sop._session.run_id
            previous_procedure = sop._session.procedure
            sop._rag.retrieve.execute.assert_awaited_once()
            sop._llm.chat.assert_awaited_once()

            await router.handle(AgentTurn("p", "reset-before-empty", "reset"))
            show_tea = False
            await router.handle(AgentTurn("p", "no-tea-run", "start tea making"))
            await asyncio.wait_for(no_tea_seen.wait(), 1)
            await _until(lambda: bool(sop.context_snapshot()["active_flows"]["identify"]["observations"]))
            snapshot = sop.context_snapshot()
            current = snapshot["active_flows"]["identify"]
            assert snapshot["run_id"] != previous_run and previous_procedure.done()
            assert current["visual_data"] is None
            assert current["resolved_data"] == {"temperature": None, "steep_duration": None}
            assert current["sources"] == {} and "resolved_settings" not in snapshot
            assert not current["observations"][-1]["usable"]
            sop._rag.retrieve.execute.assert_awaited_once()
            sop._llm.chat.assert_awaited_once()
            assert snapshot["historical_previous_run"]["run_id"] == previous_run
        finally:
            await router.stop()
            await sop.close()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
async def test_step_entry_failure_does_not_construct_flow(monkeypatch):
    """Failed entry side effects cannot leave an unawaited flow coroutine."""
    sop, runtime, context = _runner()
    await runtime.start()
    await sop.initialize()
    session = sop._session
    session.status = "running"
    session.run_id = "entry-failure"
    factory = Mock(side_effect=AssertionError("flow must remain lazy"))
    monkeypatch.setattr(sop, "_notice", AsyncMock(side_effect=RuntimeError("notice failed")))
    try:
        with pytest.raises(RuntimeError, match="notice failed"):
            await sop._run_step(session, 0, factory)
        factory.assert_not_called()
        assert session.flows["identify"].task is None
    finally:
        await sop.close()
        await runtime.stop()
        await context.close()


@pytest.mark.asyncio
async def test_old_flow_callback_cannot_replace_new_execution_state():
    """A label and run ID do not authorize a callback from a replaced record."""

    @dataclass
    class State:
        status: str
        updated_at_us: int
        evidence: str

    sop, runtime, context = _runner()
    await runtime.start()
    await sop.initialize()
    session = sop._session
    session.status = "running"
    session.run_id = "same-run"
    callbacks = []
    release_replacement = asyncio.Event()

    async def original(on_state):
        callbacks.append(on_state)

    async def replacement(on_state):
        callbacks.append(on_state)
        await on_state(State(status="running", updated_at_us=1, evidence="new"))
        await release_replacement.wait()

    try:
        original_record = sop._launch_flow(session, "monitor", original)
        await original_record.task
        replacement_record = sop._launch_flow(session, "monitor", replacement)
        await _until(lambda: replacement_record.state is not None)
        await callbacks[0](State(status="completed", updated_at_us=2, evidence="old"))
        assert session.flows["monitor"] is replacement_record
        assert replacement_record.state.evidence == "new"
        assert original_record.state is None
    finally:
        release_replacement.set()
        await replacement_record.task
        await sop.close()
        await runtime.stop()
        await context.close()


@pytest.mark.asyncio
async def test_required_parallel_failure_reaches_procedure_supervisor(monkeypatch):
    """TaskGroup failure remains an execution failure, not a cleanup-only result."""
    sop, runtime, context = _runner()
    router, conversation, _language = _router(sop, runtime, context)
    sibling_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()

    async def sibling(_on_state):
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            sibling_cancelled.set()

    async def failing(_on_state):
        await sibling_started.wait()
        raise RuntimeError("required monitor failed")

    async def procedure(session):
        async with asyncio.TaskGroup() as tasks:
            sop._launch_flow(session, "monitor", sibling, task_group=tasks)
            sop._launch_flow(session, "required", failing, task_group=tasks)
            await asyncio.Event().wait()

    monkeypatch.setattr(sop, "_procedure", procedure)
    async with runtime:
        try:
            await router.handle(AgentTurn("p", "start", "start tea making"))
            await _until(lambda: sop._session.status == "failed")
            await asyncio.wait_for(sibling_cancelled.wait(), 1)
            await sop._session.procedure
            assert all(record.task is None for record in sop._session.flows.values())
            assert "TaskGroup" in sop._session.issue
        finally:
            await router.stop()
            await sop.close()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
async def test_repeated_procedure_cancellation_drains_child_cleanup(monkeypatch):
    """A second cancellation cannot tear down an already-cleaning-up flow."""
    sop, runtime, context = _runner()
    router, conversation, _language = _router(sop, runtime, context)
    child_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    finish_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def flow(_on_state):
        child_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await finish_cleanup.wait()
            cleanup_finished.set()

    async def procedure(session):
        await sop._run_step(session, 0, flow)

    monkeypatch.setattr(sop, "_procedure", procedure)
    async with runtime:
        try:
            await router.handle(AgentTurn("p", "start", "start tea making"))
            await asyncio.wait_for(child_started.wait(), 1)
            session = sop._session
            procedure_task = session.procedure
            record = session.flows["identify"]
            procedure_task.cancel()
            await asyncio.wait_for(cleanup_started.wait(), 1)
            procedure_task.cancel()
            await asyncio.sleep(0)
            assert not procedure_task.done() and not record.task.done()
            assert record.task.cancelling() == 1
            finish_cleanup.set()
            with pytest.raises(asyncio.CancelledError):
                await procedure_task
            assert cleanup_finished.is_set()
            assert record.task is None
        finally:
            finish_cleanup.set()
            await router.stop()
            await sop.close()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("advancement", ("automatic", "manual"))
@pytest.mark.parametrize("operation", ("skip", "reset"))
async def test_control_during_step_entry_never_constructs_abandoned_flow(monkeypatch, advancement, operation):
    """Skip and reset apply before launch, under either advancement policy."""
    sop, runtime, context = _runner()
    sop.config = replace(sop.config, advancement=advancement)
    router, conversation, _language = _router(sop, runtime, context)
    entry_started = asyncio.Event()
    release_entry = asyncio.Event()
    next_started = asyncio.Event()
    first_factory = Mock(side_effect=AssertionError("skipped flow must remain lazy"))
    original_notice = sop._notice

    async def notice(session, text):
        if text == sop.config.messages["identify"].format_map(sop._resolved_settings(session)):
            entry_started.set()
            await release_entry.wait()
        await original_notice(session, text)

    async def next_flow(_on_state):
        next_started.set()
        await asyncio.Event().wait()

    async def procedure(session):
        await sop._run_step(session, 0, first_factory)
        await sop._run_step(session, 1, next_flow)

    monkeypatch.setattr(sop, "_notice", notice)
    monkeypatch.setattr(sop, "_procedure", procedure)
    async with runtime:
        try:
            await router.handle(AgentTurn("p", "start", "start tea making"))
            await asyncio.wait_for(entry_started.wait(), 1)
            response = await router.handle(AgentTurn("p", "control-entry", operation))
            assert response.text == sop.config.messages[operation]
            release_entry.set()
            if operation == "skip":
                await asyncio.wait_for(next_started.wait(), 1)
                assert sop._session.flows["identify"].status == "skipped"
                assert sop.context_snapshot()["current_step"] == "fill_water"
            else:
                assert sop.context_snapshot()["status"] == "idle"
                assert not next_started.is_set()
            first_factory.assert_not_called()
        finally:
            release_entry.set()
            await router.stop()
            await sop.close()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
async def test_scoped_sop_instances_isolate_runs_and_keep_shared_clients_open(monkeypatch):
    """Leaving one participant drains only its runner, including never-routed joins."""
    from tea_making_worker.application.tea_events import PARTICIPANT_JOINED_TOPIC
    from xr_ai_interaction import ParticipantJoined

    template, runtime, context = _runner()
    instances = []
    records = []

    class Capture(Agent):
        @subscribe(GUIDANCE_RECORD_TOPIC)
        async def record(self, message: GuidanceRecord, delivery: RuntimeContext):
            records.append((delivery.metadata.participant_id, message.run_id))

    runtime.register("scoped-records", Capture())
    template._llm.close = AsyncMock()
    template._decisions.close = AsyncMock()
    template._rag.close = AsyncMock()
    template._image_query.close = AsyncMock()

    async def procedure(sop, session):
        await sop._run_step(
            session,
            4,
            lambda on_state: run_timer_flow(
                duration_seconds=100,
                update_interval_seconds=100,
                on_state=on_state,
            ),
        )

    monkeypatch.setattr(SOPAgent, "_procedure", procedure)
    router, conversation, language = _router(
        template,
        runtime,
        context,
        scoped_instances=instances,
        participant_joined_topic=PARTICIPANT_JOINED_TOPIC,
    )
    async with runtime:
        try:
            for participant in ("p", "q", "joined-only"):
                await runtime.publish(PARTICIPANT_JOINED_TOPIC, ParticipantJoined(), participant_id=participant)
            await _until(lambda: len(instances) == 3)
            first, second, joined_only = instances
            assert len({id(instance) for instance in instances}) == 3
            for instance in instances:
                assert instance._llm is template._llm
                assert instance._decisions is template._decisions
                assert instance._frames is template._frames
                assert instance._image_query is template._image_query
                assert instance._rag is template._rag
            await router.handle(AgentTurn("p", "start-p", "start tea making"))
            await router.handle(AgentTurn("q", "start-q", "start tea making"))
            await _until(
                lambda: all(
                    "steep_timer" in instance.context_snapshot()["active_flows"] for instance in (first, second)
                )
            )
            assert first._session.run_id != second._session.run_id
            assert ("p", first._session.run_id) in records
            assert ("q", second._session.run_id) in records
            second_task = second._session.flows["steep_timer"].task
            first_record = first._session.flows["steep_timer"]
            first_state = first_record.state
            await router.participant_left("p")
            assert not first._session.connected and first_record.task is None
            assert not second_task.done() and second._session.connected
            assert second.context_snapshot()["status"] == "running"
            await router.handle(AgentTurn("q", "question-q", "What is the current step?"))
            current_messages = "\n".join(message.content for message in language.messages[-1])
            assert second._session.run_id in current_messages
            assert first._session.run_id not in current_messages
            await first._flow_state(
                first._session, "steep_timer", first_record, replace(first_state, remaining_seconds=1)
            )
            assert first_record.state is first_state
            await router.participant_left("joined-only")
            assert not joined_only._session.connected and joined_only._session.procedure is None
            await runtime.publish(PARTICIPANT_JOINED_TOPIC, ParticipantJoined(), participant_id="p")
            await _until(lambda: len(instances) == 4)
            replacement = instances[-1]
            assert replacement is not first and replacement.context_snapshot()["status"] == "idle"
            assert replacement._session.flows == {}
            await first._flow_state(
                first._session, "steep_timer", first_record, replace(first_state, remaining_seconds=2)
            )
            assert replacement.context_snapshot()["active_flows"] == {}
            assert not second_task.done()
            for client in (template._llm, template._decisions, template._rag, template._image_query):
                client.close.assert_not_called()
        finally:
            await router.stop()
            await conversation.close()
            await context.close()


@pytest.mark.asyncio
async def test_bound_sop_rejects_other_participants_and_cannot_reopen_after_close():
    """Direct callers cannot bypass the instance's participant or lifetime boundary."""
    sop, runtime, context = _runner()
    async with runtime:
        try:
            await sop.initialize()
            snapshot = sop.context_snapshot()
            with pytest.raises(ValueError, match="participant does not match"):
                sop.routing_aid("other")
            with pytest.raises(ValueError, match="participant does not match"):
                await sop.release("other")
            admission = await sop.admit(AgentTurn("other", "wrong-admit", "start tea making"), focused=False)
            assert admission.disposition == "decline"
            started = AsyncMock()
            await sop.answer(AgentTurn("other", "wrong-answer", "start tea making"), started=started)
            started.assert_not_awaited()
            assert sop.context_snapshot() == snapshot
            await sop.close()
            with pytest.raises(RuntimeError, match="closed SOP instance"):
                await sop.initialize()
            admission = await sop.admit(AgentTurn("p", "late", "start tea making"), focused=False)
            assert admission.disposition == "decline"
            assert not sop._session.connected and sop._session.procedure is None
        finally:
            await sop.close()
            await context.close()
