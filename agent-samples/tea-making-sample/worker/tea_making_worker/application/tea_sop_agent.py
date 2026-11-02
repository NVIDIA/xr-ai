# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run an explicit Python SOP; route general questions to SDK Conversation.

Read _procedure() for the five-step connection. Independent async flow functions
own recognition/timing and report full state; this agent stores the latest
labeled snapshots, owns controls/tasks/transitions, records evidence, and supplies
one SOP-context source to the separate conversation agent. No YAML step language
or separate workflow engine participates in execution.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Literal
from uuid import uuid4

import nemo_relay
from xr_ai_conversation import ContextManager, ConversationContextUpdate
from xr_ai_hub import FrameUnavailable
from xr_ai_interaction import ResponseOutputs
from xr_ai_models import DecisionQuestion, DecisionService, LLMService
from xr_ai_routing import Admission, AgentTurn, DestinationResult, RoutingAid
from xr_ai_runtime import Agent, AgentRuntime, RuntimeClosedError
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.rag import RAGTools
from xr_ai_tools.vision import ImageQueryTool
from xr_ai_voice import VOICE_OUTPUT_TOPIC

from ..components import (
    VisionRAGRetrievalState,
    run_timer_flow,
    run_vision_rag_retrieval_flow,
    run_visual_verification_flow,
)
from ..components.sop_run_recording import _attempt, _event, _frame, _json
from .tea_data_models import BrewingSettings, TeaIdentity
from .tea_events import GUIDANCE_NOTICE_TOPIC, GUIDANCE_RECORD_TOPIC, GuidanceNotice, GuidanceRecord
from .tea_sop_configuration import SOPConfig

SOP_CONTEXT_SOURCE = "tea-making.sop-context"
_STEP_IDS = ("identify", "fill_water", "heat_water", "start_steeping", "steep_timer")
_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _Session:
    """Runner-owned participant state; flow internals arrive only through snapshots."""

    participant_id: str
    run_id: str | None = None
    status: Literal["idle", "running", "completed", "stopped", "failed"] = "idle"
    revision: int = 0
    step_index: int = -1
    connected: bool = True
    procedure: asyncio.Task[None] | None = None
    active_tasks: dict[str, asyncio.Task[Any]] = field(default_factory=dict)
    flow_states: dict[str, Any] = field(default_factory=dict)
    flow_statuses: dict[str, str] = field(default_factory=dict)
    advance_requested: asyncio.Event = field(default_factory=asyncio.Event)
    ready: bool = False
    skipping: bool = False
    last_run: dict[str, Any] | None = None
    issue: str = ""


class SOPAgent(Agent):
    """Control-only SOP runner and owner of flow state supplied to conversation.

    Args:
        config: Ordinary Python application arguments for the explicit tea procedure.
        llm: Settings extraction/generation service.
        decisions: Semantic control authorization and visual judgment service.
        frames: SDK current-frame tool; the application owns its image lifetime.
        image_query: Omni image tool shared by the independent flow calls.
        rag: Existing typed retrieval tools.
        context: SDK context manager shared with the separate Conversation destination.
        runtime: Started/stopped by application composition.
    """

    def __init__(
        self,
        *,
        config: SOPConfig,
        llm: LLMService,
        decisions: DecisionService,
        frames: CurrentFrameTool,
        image_query: ImageQueryTool,
        rag: RAGTools,
        context: ContextManager,
        runtime: AgentRuntime,
    ) -> None:
        super().__init__()
        self.config = config
        self._llm = llm
        self._decisions = decisions
        self._frames = frames
        self._image_query = image_query
        self._rag = rag
        self._context = context
        self._runtime = runtime
        self._responses = ResponseOutputs(
            runtime,
            VOICE_OUTPUT_TOPIC,
            source="sop",
            closed_errors=(RuntimeClosedError,),
        )
        self._sessions: dict[str, _Session] = {}
        self._pending: dict[str, tuple[str, _Session, int, str | None, str]] = {}
        self._closing = False
        self._participant_locks: dict[str, asyncio.Lock] = {}

    async def participant_joined(self, participant_id: str) -> None:
        """Admit a new participant; late callbacks retain their old session identity."""
        async with self._participant_locks.setdefault(participant_id, asyncio.Lock()):
            if not self._closing and participant_id not in self._sessions:
                session = _Session(participant_id)
                self._sessions[participant_id] = session
                await self._publish_context(session)

    async def participant_left(self, participant_id: str) -> None:
        """Close admission before draining owned flow work and clearing context."""
        async with self._participant_locks.setdefault(participant_id, asyncio.Lock()):
            session = self._sessions.pop(participant_id, None)
            if session is None:
                return
            session.connected = False
            self._pending.pop(participant_id, None)
            await self._stop_run(session)
            await self._context.clear(participant_id)

    def routing_aid(self, participant_id: str) -> RoutingAid:
        """Describe controls without taking focus over general SOP questions."""
        session = self._sessions.get(participant_id)
        return RoutingAid(
            "workflow",
            self.config.workflow_description,
            eligibility=session.status if session else "participant not admitted",
        )

    async def admit(self, turn: AgentTurn, *, focused: bool) -> Admission:
        """Recognize an explicit control and retain a revision-bound authorized proposal."""
        del focused
        session = self._sessions.get(turn.participant_id)
        if self._closing or session is None or not session.connected:
            return Admission("decline")
        revision = session.revision
        admitted_run_id = session.run_id
        response = await self._decisions.decide(
            {"request": turn.query, "status": session.status, "step": self._step_id(session), "ready": session.ready},
            {"control": self.config.control_question},
        )
        control = response.answers["control"].choice
        if control == "none":
            return Admission("decline")
        if control not in self.config.control_question.criteria:
            raise ValueError("invalid SOP control")
        authorization = await self._decisions.decide(
            {
                "request": turn.query,
                "proposed_action": {
                    "operation": control,
                    "target": "tea making",
                    "description": self.config.control_question.criteria[control],
                },
            },
            {"authorization": self.config.authorization_question},
        )
        answer = authorization.answers["authorization"]
        command_probability = answer.probabilities["command"]
        if answer.choice != "command" or command_probability <= sum(
            probability for label, probability in answer.probabilities.items() if label != "command"
        ):
            return Admission("decline")
        if (
            self._sessions.get(turn.participant_id) is not session
            or not session.connected
            or asyncio.current_task().cancelling()
        ):
            return Admission("decline")
        self._pending[turn.participant_id] = (turn.turn_id, session, revision, admitted_run_id, control)
        return Admission("accept")

    async def answer(self, turn: AgentTurn, *, started: Callable[[], Awaitable[None]]) -> DestinationResult:
        """Apply only a still-current admitted control, then publish its native acknowledgment."""
        pending_admission = self._pending.pop(turn.participant_id, None)
        if pending_admission is None or pending_admission[0] != turn.turn_id:
            return DestinationResult("answer")
        _, session, admitted_revision, admitted_run_id, control = pending_admission
        async with self._participant_locks.setdefault(turn.participant_id, asyncio.Lock()):
            if (
                self._sessions.get(turn.participant_id) is not session
                or session.run_id != admitted_run_id
                or (control in {"next", "skip"} and session.revision != admitted_revision)
            ):
                response_text = self.config.messages["changed"]
            else:
                await started()
                # Once admitted, finish this native state transition even if a
                # newer utterance cancels its spoken acknowledgment. The SDK
                # still owns the turn; this task owns only the committed control.
                control_task = asyncio.create_task(
                    self._control(session, control), context=nemo_relay.fork_asyncio_context()
                )
                try:
                    response_text = await asyncio.shield(control_task)
                except asyncio.CancelledError:
                    while not control_task.done():
                        try:
                            await asyncio.shield(control_task)
                        except asyncio.CancelledError:
                            continue
                    control_task.result()
                    raise
            async with self._responses.reply(
                turn.participant_id,
                request_id=turn.turn_id,
                timestamp_us=turn.timestamp_us,
                started=started,
            ) as reply:
                await reply.send(response_text)
        return DestinationResult("answer", response_text)

    async def release(self, _participant_id: str) -> None:
        """Routing focus changes do not cancel the independently running SOP."""

    async def stop(self) -> None:
        """Close admission and drain every participant's owned procedure."""
        self._closing = True
        for participant_id in tuple(self._sessions):
            await self.participant_left(participant_id)
        self._pending.clear()

    async def _control(self, session: _Session, control: str) -> str:
        """Keep next/skip distinct and defaults outside the independent flow functions."""
        if control in {"start", "restart"}:
            if control == "start" and session.status == "running":
                return self.config.messages["already_running"]
            await self._stop_run(session)
            if session.run_id:
                session.last_run = self._run_summary(session)
            session.run_id = uuid4().hex
            session.revision += 1
            session.status = "running"
            session.step_index = -1
            session.flow_states.clear()
            session.flow_statuses.clear()
            session.issue = ""
            session.procedure = asyncio.create_task(
                self._supervise(session),
                name=f"sop:{session.participant_id}",
                context=nemo_relay.fork_asyncio_context(),
            )
            await self._publish_context(session)
            return self.config.messages["start"]
        if control in {"stop", "reset"}:
            await self._stop_run(session)
            session.last_run = self._run_summary(session)
            session.status = "idle" if control == "reset" else "stopped"
            session.revision += 1
            if control == "reset":
                session.run_id = None
                session.step_index = -1
                session.flow_states.clear()
                session.flow_statuses.clear()
            await self._publish_context(session)
            return self.config.messages[control]
        if control in {"next", "skip"}:
            if session.status != "running":
                return self.config.messages["not_running"]
            if control == "next" and not session.ready:
                return self.config.messages["not_ready"]
            if control == "skip":
                session.skipping = True
                # Signal this step before awaiting cancellation; the procedure
                # may already enter its next step while the child drains.
                session.advance_requested.set()
                active_step = self._step_id(session)
                step_task = session.active_tasks.get(active_step)
                if step_task is not None:
                    step_task.cancel()
                    await self._drain(step_task)
            else:
                session.advance_requested.set()
            return self.config.messages[control]
        raise ValueError("unknown SOP control")

    async def _supervise(self, session: _Session) -> None:
        """Surface a procedure failure without abandoning its sibling tasks or latest state."""
        try:
            await self._procedure(session)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # isolation boundary: this participant's background procedure is the
            # unit of work; expose its failure rather than silently restarting it.
            _LOGGER.exception("SOP failed for participant %s", session.participant_id)
            session.status = "failed"
            session.issue = str(error)
            if self._step_id(session) is not None:
                session.flow_statuses[self._step_id(session)] = "failed"
            if self._sessions.get(session.participant_id) is session:
                await self._publish_context(session)
        finally:
            owned_flow_tasks = tuple(session.active_tasks.values())
            for flow_task in owned_flow_tasks:
                flow_task.cancel()
            # Drain every child, including spanning flows, before releasing the
            # run. This is cleanup, not an exception-swallowing execution policy:
            # normal required work is awaited or owned by its TaskGroup.
            drain = asyncio.gather(*owned_flow_tasks, return_exceptions=True)
            cancelled_during_cleanup = False
            while not drain.done():
                try:
                    await asyncio.shield(drain)
                except asyncio.CancelledError:
                    cancelled_during_cleanup = True
            drain.result()
            for flow_id, status in tuple(session.flow_statuses.items()):
                if status == "running":
                    session.flow_statuses[flow_id] = "stopped"
            session.active_tasks.clear()
            if cancelled_during_cleanup:
                raise asyncio.CancelledError

    async def _procedure(self, session: _Session) -> None:
        """Read the actual tea procedure front to back: ordinary Python calls and results."""
        flow_arguments = self._observation_arguments(session)
        await self._run_step(
            session,
            0,
            run_vision_rag_retrieval_flow(
                **flow_arguments,
                llm=self._llm,
                retrieval=self._rag.retrieve,
                visual_model=TeaIdentity,
                result_model=BrewingSettings,
                visual_question=self.config.identification_question,
                extraction_instructions=self.config.extraction_instructions,
                fallback_instructions=self.config.fallback_instructions,
                retrieval_query=lambda identity: identity.tea_name + " " + self.config.retrieval_query_suffix,
                max_retrieved_passages=self.config.max_retrieved_passages,
                retrieved_text_limit=self.config.retrieved_text_limit,
                on_state=partial(self._flow_state, session, session.run_id, "identify"),
            ),
        )

        # Skip defaults are applied here, after preserving validated partial
        # values reported by identification. They are never flow-internal guesses.
        resolved_settings = self._resolved_settings(session)
        await self._run_step(session, 1, self._verify(session, "fill_water", self.config.water))

        # Only this application builds tea-specific wording. The visual function
        # receives an ordinary completed question/DecisionQuestion.
        target_temperature_c = resolved_settings["temperature_c"]
        temperature = self.config.temperature
        replacements = {
            "target_temperature_c": f"{target_temperature_c:g}",
            "target_temperature_f": f"{target_temperature_c * 9 / 5 + 32:g}",
        }
        temperature = replace(
            temperature,
            question=temperature.question.format_map(replacements),
            completion=DecisionQuestion(
                temperature.completion.instructions.format_map(replacements),
                {
                    label: criterion.format_map(replacements)
                    for label, criterion in temperature.completion.criteria.items()
                },
            ),
        )
        await self._run_step(session, 2, self._verify(session, "heat_water", temperature))
        steeping_confirmation = await self._run_step(
            session,
            3,
            self._verify(session, "start_steeping", self.config.tea_bag),
        )
        if steeping_confirmation is not None:
            await self._run_step(
                session,
                4,
                run_timer_flow(
                    duration_seconds=resolved_settings["duration_seconds"],
                    update_interval_seconds=self.config.timer_update_interval_seconds,
                    on_state=partial(self._flow_state, session, session.run_id, "steep_timer"),
                ),
            )
        else:
            # Explicitly skipping tea-water detection does not establish a
            # steeping start time. Preserve the existing associated-timer skip.
            session.flow_statuses["steep_timer"] = "skipped"
        session.status = "completed"
        session.revision += 1
        await self._publish_context(session)
        await self._notice(session, self.config.messages["complete"])

    def _verify(self, session, flow_id, instructions):
        """Supply only this visual check's question, criteria and runtime inputs."""
        return run_visual_verification_flow(
            **self._observation_arguments(session),
            decisions=self._decisions,
            question=instructions.question,
            completion=instructions.completion,
            required_positives=self.config.required_positives,
            window_size=self.config.confirmation_window_size,
            on_state=partial(self._flow_state, session, session.run_id, flow_id),
        )

    def _observation_arguments(self, session: _Session) -> dict[str, Any]:
        """Share ordinary call arguments without introducing a flow configuration interpreter."""
        return {
            "frame_source": partial(self._acquire_frame, session),
            "observer": self._image_query,
            "poll_interval_seconds": self.config.poll_interval_seconds,
            "timeout_seconds": self.config.inference_timeout_seconds,
            "history_limit": self.config.history_limit,
            "observation_text_limit": self.config.observation_text_limit,
            "history_max_age_seconds": self.config.history_max_age_seconds,
        }

    async def _acquire_frame(self, session: _Session):
        """Record acquisition at the runner boundary, keeping flows independent of capture."""
        try:
            frame = await self._frames.execute(CurrentFrameRequest(participant_id=session.participant_id))
        except FrameUnavailable:
            return None
        await _frame(frame)
        return frame

    async def _run_step(self, session: _Session, step_index: int, execution: Coroutine[Any, Any, Any]):
        """Own one step task and close even an execution cancelled before task creation."""
        step_id = _STEP_IDS[step_index]
        flow_task = None
        session.step_index = step_index
        session.revision += 1
        session.ready = False
        session.skipping = False
        session.advance_requested.clear()
        session.flow_statuses[step_id] = "running"
        try:
            await self._publish_context(session)
            await self._notice(session, self.config.messages[step_id].format_map(self._resolved_settings(session)))
            async with _attempt(
                session.participant_id,
                "flow",
                step_id=step_id,
                snapshot=lambda: self.context_snapshot(session.participant_id),
            ):
                flow_task = asyncio.create_task(
                    execution,
                    name=f"sop:{session.participant_id}:{step_id}",
                    context=nemo_relay.fork_asyncio_context(),
                )
                session.active_tasks[step_id] = flow_task
                try:
                    result = await flow_task
                except asyncio.CancelledError:
                    current_task = asyncio.current_task()
                    if current_task is not None and current_task.cancelling():
                        raise
                    if not session.skipping:
                        raise
                    result = None
                session.flow_statuses[step_id] = "skipped" if session.skipping else "completed"
                session.ready = True
                await self._publish_context(session)
                await _event("step.finished", skipped=session.skipping, result=result)
                if self.config.advancement == "manual" and not session.skipping:
                    await session.advance_requested.wait()
                return result
        finally:
            # Entry publication/capture can fail before create_task takes ownership.
            if flow_task is None:
                execution.close()
            elif not flow_task.done():
                flow_task.cancel()
                await self._drain(flow_task)
            session.active_tasks.pop(step_id, None)

    async def _flow_state(self, session: _Session, run_id: str | None, flow_id: str, state: Any) -> None:
        """Store complete labeled state only while this exact participant/run still owns it."""
        if (
            self._sessions.get(session.participant_id) is not session
            or not session.connected
            or session.run_id != run_id
            or session.status != "running"
        ):
            return
        reporting_task = asyncio.current_task()
        if reporting_task is None or reporting_task.cancelling():
            return
        owned_task = session.active_tasks.get(flow_id)
        if owned_task is not None and owned_task is not reporting_task:
            return
        if session.flow_statuses.get(flow_id) in {"skipped", "stopped"}:
            return
        if reporting_task is not session.procedure:
            session.active_tasks[flow_id] = reporting_task
        session.flow_statuses[flow_id] = state.status
        session.flow_states[flow_id] = state
        await _event("flow.state", flow_id=flow_id, state=state)
        if self._sessions.get(session.participant_id) is not session or session.run_id != run_id:
            return
        await self._publish_context(session)
        if (
            self._sessions.get(session.participant_id) is not session
            or not session.connected
            or session.run_id != run_id
            or reporting_task.cancelling()
        ):
            return
        await self._runtime.publish(
            GUIDANCE_RECORD_TOPIC,
            GuidanceRecord(
                run_id=run_id,
                timestamp_us=time.time_ns() // 1000,
                event="flow.state",
                step_id=flow_id,
                message="",
                state=_json(state),
            ),
            participant_id=session.participant_id,
        )

    def _resolved_settings(self, session: _Session) -> dict[str, Any]:
        """Use valid partial results and label only missing values as runner defaults."""
        identification = session.flow_states.get("identify")
        settings = (
            identification.resolved_data if isinstance(identification, VisionRAGRetrievalState) else BrewingSettings()
        )
        sources = identification.sources if isinstance(identification, VisionRAGRetrievalState) else {}
        identity = identification.visual_data if isinstance(identification, VisionRAGRetrievalState) else None
        return {
            "tea_name": identity.tea_name if identity is not None else self.config.default_tea_name,
            "temperature_c": (
                settings.temperature.celsius if settings.temperature is not None else self.config.default_temperature_c
            ),
            "duration_seconds": (
                settings.steep_duration.seconds
                if settings.steep_duration is not None
                else self.config.default_duration_seconds
            ),
            "sources": {
                "temperature": _json(sources["temperature"]) if "temperature" in sources else {"kind": "default"},
                "steep_duration": _json(sources["steep_duration"])
                if "steep_duration" in sources
                else {"kind": "default"},
            },
        }

    def context_snapshot(self, participant_id: str) -> dict[str, Any]:
        """Build read-only SOP context with active and completed flows clearly separated."""
        session = self._sessions.get(participant_id)
        if session is None:
            return {"status": "idle", "active_flows": {}}
        return {
            "run_id": session.run_id,
            "status": session.status,
            "current_step": self._step_id(session),
            "step_ready": session.ready,
            "advancement": self.config.advancement,
            "procedure_steps": {step_id: self.config.messages[step_id] for step_id in _STEP_IDS},
            "resolved_settings": self._resolved_settings(session),
            "flow_outcomes": dict(session.flow_statuses),
            "active_flows": {
                flow_id: _json(state)
                for flow_id, state in session.flow_states.items()
                if session.flow_statuses.get(flow_id) == "running" and session.status == "running"
            },
            "completed_or_skipped_flows": {
                flow_id: {"outcome": session.flow_statuses.get(flow_id), "state": _json(state)}
                for flow_id, state in session.flow_states.items()
                if session.flow_statuses.get(flow_id) != "running"
            },
            "issue": session.issue,
            "historical_previous_run": session.last_run,
        }

    async def _publish_context(self, session: _Session) -> None:
        """Replace the SDK SOP-status source; the conversation agent owns its dialogue."""
        if self._sessions.get(session.participant_id) is not session or not session.connected:
            return
        await self._context.update(
            session.participant_id,
            SOP_CONTEXT_SOURCE,
            ConversationContextUpdate(
                text=json.dumps(self.context_snapshot(session.participant_id), ensure_ascii=False)
            ),
            mode="summary",
            timestamp_us=time.time_ns() // 1000,
        )

    async def _notice(self, session: _Session, text: str) -> None:
        """Publish native step instructions without involving conversation generation."""
        if self._sessions.get(session.participant_id) is session and session.connected:
            timestamp_us = time.time_ns() // 1000
            await self._responses.update(session.participant_id, text, timestamp_us=timestamp_us)
            await self._runtime.publish(
                GUIDANCE_NOTICE_TOPIC,
                GuidanceNotice(timestamp_us=timestamp_us, text=text),
                participant_id=session.participant_id,
            )

    async def _stop_run(self, session: _Session) -> None:
        """Drain owned work before replacing a run or releasing participant images."""
        if session.status == "running":
            session.status = "stopped"
        if session.procedure is not None and not session.procedure.done():
            session.procedure.cancel()
            await self._drain(session.procedure)
        session.procedure = None
        if session.status == "running":
            session.status = "stopped"
        for flow_id in session.active_tasks:
            session.flow_statuses[flow_id] = "stopped"
        session.active_tasks.clear()

    @staticmethod
    async def _drain(task: asyncio.Task[Any]) -> None:
        """Wait for child cleanup while preserving cancellation directed at this owner."""
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            owner = asyncio.current_task()
            if owner is not None and owner.cancelling():
                # Cleanup must finish even when the owner receives cancellation.
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue
                raise
            if not task.cancelled():
                raise

    @staticmethod
    def _step_id(session: _Session) -> str | None:
        return _STEP_IDS[session.step_index] if 0 <= session.step_index < len(_STEP_IDS) else None

    def _run_summary(self, session: _Session) -> dict[str, Any]:
        """Keep the prior run explicitly historical without retaining every frame twice."""
        return {
            "run_id": session.run_id,
            "status": session.status,
            "settings": self._resolved_settings(session),
            "last_step": self._step_id(session),
        }


__all__ = ["SOPAgent"]
