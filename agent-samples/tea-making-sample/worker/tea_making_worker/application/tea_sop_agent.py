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
from datetime import UTC, datetime
from functools import partial
from typing import Any, Literal
from uuid import uuid4

import nemo_relay
from xr_ai_conversation import ContextManager, ConversationContextUpdate
from xr_ai_hub import FrameUnavailable
from xr_ai_interaction import ResponseOutputs
from xr_ai_models import DecisionQuestion, DecisionService, LLMService
from xr_ai_routing import Admission, AgentTurn, DestinationResult, RoutingAid
from xr_ai_runtime import AgentRuntime, RuntimeClosedError
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
class _AcceptedControl:
    """One authorized proposal, bound to its original turn and execution boundary."""

    turn_id: str
    revision: int
    run_id: str | None
    operation: str


@dataclass(slots=True)
class _FlowRecord:
    """One runner-owned execution; callbacks cannot acquire or transfer ownership."""

    run_id: str | None
    status: str = "running"
    state: Any = None
    task: asyncio.Task[Any] | None = None


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
    flows: dict[str, _FlowRecord] = field(default_factory=dict)
    accepted_control: _AcceptedControl | None = None
    advance_requested: asyncio.Event = field(default_factory=asyncio.Event)
    ready: bool = False
    skipping: bool = False
    last_run: dict[str, Any] | None = None
    issue: str = ""
    identification_pending: tuple[str, ...] | None = None


class SOPAgent:
    """One participant's SOP runner and owner of flow state supplied to conversation.

    Args:
        participant_id: Participant bound to this instance for its complete lifetime.
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
        participant_id: str,
        config: SOPConfig,
        llm: LLMService,
        decisions: DecisionService,
        frames: CurrentFrameTool,
        image_query: ImageQueryTool,
        rag: RAGTools,
        context: ContextManager,
        runtime: AgentRuntime,
    ) -> None:
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
        self._session = _Session(participant_id)
        self._control_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Publish initial state before the factory makes this instance routable."""
        if not self._session.connected:
            raise RuntimeError("a closed SOP instance cannot be initialized again")
        await self._publish_context(self._session)

    async def close(self) -> None:
        """Close admission, then drain this instance's work before shared cleanup."""
        session = self._session
        session.connected = False
        async with self._control_lock:
            session.accepted_control = None
            await self._stop_run(session)

    def routing_aid(self, participant_id: str) -> RoutingAid:
        """Describe controls without taking focus over general SOP questions."""
        session = self._session
        if participant_id != session.participant_id:
            raise ValueError("participant does not match this SOP instance")
        return RoutingAid(
            "workflow",
            self.config.workflow_description,
            eligibility=session.status if session.connected else "participant not admitted",
        )

    async def admit(self, turn: AgentTurn, *, focused: bool) -> Admission:
        """Recognize an explicit control and retain a revision-bound authorized proposal."""
        del focused
        session = self._session
        if turn.participant_id != session.participant_id or not session.connected:
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
        if not session.connected or asyncio.current_task().cancelling():
            return Admission("decline")
        session.accepted_control = _AcceptedControl(turn.turn_id, revision, admitted_run_id, control)
        return Admission("accept")

    async def answer(self, turn: AgentTurn, *, started: Callable[[], Awaitable[None]]) -> DestinationResult:
        """Apply only a still-current admitted control, then publish its native acknowledgment."""
        session = self._session
        if turn.participant_id != session.participant_id or not session.connected or session.accepted_control is None:
            return DestinationResult("answer")
        proposal = session.accepted_control
        if proposal.turn_id != turn.turn_id:
            return DestinationResult("answer")
        session.accepted_control = None
        control = proposal.operation
        async with self._control_lock:
            if (
                not session.connected
                or session.run_id != proposal.run_id
                or (control in {"next", "skip"} and session.revision != proposal.revision)
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

    async def release(self, participant_id: str) -> None:
        """Routing focus changes do not cancel the independently running SOP."""
        if participant_id != self._session.participant_id:
            raise ValueError("participant does not match this SOP instance")

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
            session.flows.clear()
            session.issue = ""
            session.identification_pending = None
            session.procedure = asyncio.create_task(
                self._supervise(session),
                name=f"sop:{session.participant_id}",
                context=nemo_relay.fork_asyncio_context(),
            )
            await self._publish_context(session)
            return self.config.messages["start"]
        if control in {"stop", "reset"}:
            await self._stop_run(session)
            session.last_run = self._run_summary(session) if session.run_id else None
            session.status = "idle" if control == "reset" else "stopped"
            session.revision += 1
            if control == "reset":
                session.run_id = None
                session.step_index = -1
                session.flows.clear()
                session.identification_pending = None
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
                step_record = session.flows.get(active_step)
                step_task = step_record.task if step_record else None
                if step_task is not None:
                    if not step_task.cancelling():
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
                session.flows[self._step_id(session)].status = "failed"
            if session.connected:
                await self._publish_context(session)
        finally:
            owned_flow_tasks = tuple(record.task for record in session.flows.values() if record.task is not None)
            for flow_task in owned_flow_tasks:
                if not flow_task.cancelling():
                    flow_task.cancel()
            # Drain every child, including spanning flows, before releasing the
            # run. This is cleanup, not an exception-swallowing execution policy:
            # normal required work is awaited or owned by its TaskGroup.
            drain = asyncio.gather(*owned_flow_tasks, return_exceptions=True)
            try:
                await self._drain(drain)
            finally:
                for record in session.flows.values():
                    if record.status == "running":
                        record.status = "stopped"
                    record.task = None

    async def _procedure(self, session: _Session) -> None:
        """Read the actual tea procedure front to back: ordinary Python calls and results."""
        flow_arguments = self._observation_arguments(session)
        await self._run_step(
            session,
            0,
            lambda on_state: run_vision_rag_retrieval_flow(
                **flow_arguments,
                llm=self._llm,
                retrieval=self._rag.retrieve,
                visual_model=TeaIdentity,
                result_model=BrewingSettings,
                visual_question=self.config.identification_question,
                extraction_instructions=self.config.extraction_instructions,
                fallback_instructions=self.config.fallback_instructions,
                retrieval_query=lambda identity: " ".join(
                    filter(None, (identity.tea_name, identity.tea_type, self.config.retrieval_query_suffix))
                ),
                max_retrieved_passages=self.config.max_retrieved_passages,
                retrieved_text_limit=self.config.retrieved_text_limit,
                on_state=on_state,
            ),
        )

        # Skip defaults are applied here, after preserving validated partial
        # values reported by identification. They are never flow-internal guesses.
        resolved_settings = self._resolved_settings(session)
        await self._run_step(session, 1, lambda on_state: self._verify(session, self.config.water, on_state))

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
        await self._run_step(session, 2, lambda on_state: self._verify(session, temperature, on_state))
        steeping_confirmation = await self._run_step(
            session,
            3,
            lambda on_state: self._verify(session, self.config.tea_bag, on_state),
        )
        if steeping_confirmation is not None:
            await self._run_step(
                session,
                4,
                lambda on_state: run_timer_flow(
                    duration_seconds=resolved_settings["duration_seconds"],
                    update_interval_seconds=self.config.timer_update_interval_seconds,
                    on_state=on_state,
                ),
            )
        else:
            # Explicitly skipping tea-water detection does not establish a
            # steeping start time. Preserve the existing associated-timer skip.
            session.flows["steep_timer"] = _FlowRecord(session.run_id, status="skipped")
        session.status = "completed"
        session.revision += 1
        await self._publish_context(session)
        await self._notice(session, self.config.messages["complete"])

    def _verify(self, session, instructions, on_state):
        """Supply only this visual check's question, criteria and runtime inputs."""
        return run_visual_verification_flow(
            **self._observation_arguments(session),
            decisions=self._decisions,
            question=instructions.question,
            completion=instructions.completion,
            required_positives=self.config.required_positives,
            window_size=self.config.confirmation_window_size,
            on_state=on_state,
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

    def _launch_flow(
        self,
        session: _Session,
        flow_id: str,
        create_flow: Callable[[Callable[[Any], Awaitable[None]]], Coroutine[Any, Any, Any]],
        *,
        record: _FlowRecord | None = None,
        task_group: asyncio.TaskGroup | None = None,
    ) -> _FlowRecord:
        """Register an execution explicitly before its lazy factory can report state.

        Required parallel work belongs to the caller's TaskGroup or must be
        awaited by the procedure. The registry owns cleanup, not error policy.
        """
        if record is None:
            previous = session.flows.get(flow_id)
            if previous is not None and previous.task is not None and not previous.task.done():
                raise RuntimeError("flow is still running; drain it before replacement")
            record = _FlowRecord(session.run_id)
            session.flows[flow_id] = record
        elif session.flows.get(flow_id) is not record:
            raise RuntimeError("flow ownership changed before launch")

        async def execute():
            return await create_flow(partial(self._flow_state, session, flow_id, record))

        launch = task_group.create_task if task_group is not None else asyncio.create_task
        execution = execute()
        try:
            record.task = launch(
                execution,
                name=f"sop:{session.participant_id}:{flow_id}",
                context=nemo_relay.fork_asyncio_context(),
            )
        except BaseException:
            # Task creation has not taken ownership of this wrapper coroutine.
            execution.close()
            record.status = "failed"
            raise
        return record

    async def _run_step(self, session: _Session, step_index: int, create_flow):
        """Enter one step, lazily launch it, then apply readiness/skip policy."""
        step_id = _STEP_IDS[step_index]
        record = _FlowRecord(session.run_id)
        session.flows[step_id] = record
        session.step_index = step_index
        session.revision += 1
        session.ready = False
        session.skipping = False
        session.advance_requested.clear()
        try:
            await self._publish_context(session)
            await self._notice(session, self.config.messages[step_id].format_map(self._resolved_settings(session)))
            async with _attempt(
                session.participant_id,
                "flow",
                step_id=step_id,
                snapshot=lambda: self.context_snapshot(),
            ):
                if (
                    not session.connected
                    or session.status != "running"
                    or session.run_id != record.run_id
                    or session.flows.get(step_id) is not record
                    or asyncio.current_task().cancelling()
                ):
                    raise asyncio.CancelledError
                if session.skipping:
                    # A control can arrive while entry publication is suspended.
                    # Honor it before constructing or starting any flow work.
                    result = None
                else:
                    self._launch_flow(session, step_id, create_flow, record=record)
                    try:
                        # The runner cancels owned children once, then drains them.
                        # Repeated owner cancellation must not interrupt their cleanup.
                        result = await asyncio.shield(record.task)
                    except asyncio.CancelledError:
                        current_task = asyncio.current_task()
                        if current_task is not None and current_task.cancelling():
                            raise
                        if not session.skipping:
                            raise
                        result = None
                record.status = "skipped" if session.skipping else "completed"
                session.ready = True
                await self._publish_context(session)
                await _event("step.finished", skipped=session.skipping, result=result)
                if self.config.advancement == "manual" and not session.skipping:
                    await session.advance_requested.wait()
                return result
        finally:
            if record.task is not None and not record.task.done():
                if not record.task.cancelling():
                    record.task.cancel()
                await self._drain(record.task)
            record.task = None

    async def _flow_state(self, session: _Session, flow_id: str, record: _FlowRecord, state: Any) -> None:
        """Report state only for the exact execution already registered by the runner."""

        def owns_execution():
            return (
                session.connected
                and session.run_id == record.run_id
                and session.status == "running"
                and session.flows.get(flow_id) is record
                and record.status not in {"skipped", "stopped"}
                and record.task is not None
                and not record.task.done()
                and not record.task.cancelling()
            )

        if not owns_execution():
            return
        record.status = state.status
        record.state = state
        await _event("flow.state", flow_id=flow_id, state=state)
        if not owns_execution():
            return
        await self._publish_context(session)
        if not owns_execution():
            return
        await self._runtime.publish(
            GUIDANCE_RECORD_TOPIC,
            GuidanceRecord(
                run_id=record.run_id,
                timestamp_us=time.time_ns() // 1000,
                event="flow.state",
                step_id=flow_id,
                message="",
                state=_json(state),
            ),
            participant_id=session.participant_id,
        )

        if (
            flow_id == "identify"
            and isinstance(state, VisionRAGRetrievalState)
            and state.phase == "observing"
            and state.visual_data is not None
            and state.issue
            and session.run_id == record.run_id
        ):
            missing_fields = tuple(name for name, value in state.resolved_data.model_dump().items() if value is None)
            if missing_fields and missing_fields != session.identification_pending:
                # Deduplicate the actual missing-setting reason, not OCR wording
                # or product-name variants. A new run resets this bounded marker.
                session.identification_pending = missing_fields
                setting_labels = {"temperature": "water temperature", "steep_duration": "steeping time"}
                await self._notice(
                    session,
                    self.config.messages["identify_pending"].format(
                        tea_name=state.visual_data.tea_name or state.visual_data.tea_type,
                        missing_settings=" and ".join(setting_labels[name] for name in missing_fields),
                    ),
                )

    def _resolved_settings(self, session: _Session) -> dict[str, Any]:
        """Use valid partial results and label only missing values as runner defaults."""
        identification_record = session.flows.get("identify")
        identification = identification_record.state if identification_record else None
        settings = (
            identification.resolved_data if isinstance(identification, VisionRAGRetrievalState) else BrewingSettings()
        )
        sources = identification.sources if isinstance(identification, VisionRAGRetrievalState) else {}
        identity = identification.visual_data if isinstance(identification, VisionRAGRetrievalState) else None
        return {
            "tea_name": (identity.tea_name or identity.tea_type)
            if identity is not None
            else self.config.default_tea_name,
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

    def context_snapshot(self) -> dict[str, Any]:
        """Expose entered steps, never turn future configuration into observed state."""
        session = self._session
        if session.status == "idle":
            return {
                "status": "idle",
                "active_flows": {},
                "historical_previous_run": session.last_run,
            }
        snapshot = {
            "run_id": session.run_id,
            "status": session.status,
            "current_step": self._step_id(session),
            "step_ready": session.ready,
            "advancement": self.config.advancement,
            "entered_step_instructions": {
                step_id: self.config.messages[step_id].format_map(self._resolved_settings(session))
                for step_id in _STEP_IDS
                if step_id in session.flows
            },
            "flow_outcomes": {flow_id: record.status for flow_id, record in session.flows.items()},
            "active_flows": {
                flow_id: {
                    **_json(record.state),
                    "sampled_at_utc": datetime.fromtimestamp(record.state.updated_at_us / 1_000_000, UTC).isoformat(
                        timespec="seconds"
                    ),
                }
                for flow_id, record in session.flows.items()
                if record.state is not None and record.status == "running" and session.status == "running"
            },
            "completed_or_skipped_flows": {
                flow_id: {"outcome": record.status, "state": _json(record.state)}
                for flow_id, record in session.flows.items()
                if record.state is not None and record.status != "running"
            },
            "issue": session.issue,
            "historical_previous_run": session.last_run,
        }
        # Defaults become execution settings only after identification completes or
        # is explicitly skipped. Partial model values remain in the flow state.
        if "identify" in session.flows and session.flows["identify"].status in {"completed", "skipped"}:
            snapshot["resolved_settings"] = self._resolved_settings(session)
        return snapshot

    async def _publish_context(self, session: _Session) -> None:
        """Replace the SDK SOP-status source; the conversation agent owns its dialogue."""
        if not session.connected:
            return
        await self._context.update(
            session.participant_id,
            SOP_CONTEXT_SOURCE,
            ConversationContextUpdate(text=json.dumps(self.context_snapshot(), ensure_ascii=False)),
            mode="summary",
            timestamp_us=time.time_ns() // 1000,
        )

    async def _notice(self, session: _Session, text: str) -> None:
        """Publish native step instructions without involving conversation generation."""
        if session.connected:
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
            if not session.procedure.cancelling():
                session.procedure.cancel()
            await self._drain(session.procedure)
        session.procedure = None
        if session.status == "running":
            session.status = "stopped"
        for record in session.flows.values():
            if record.task is not None:
                record.status = "stopped"
                record.task = None

    @staticmethod
    async def _drain(task: asyncio.Future[Any]) -> None:
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
            "settings": self._resolved_settings(session)
            if "identify" in session.flows and session.flows["identify"].status in {"completed", "skipped"}
            else {},
            "last_step": self._step_id(session),
        }


__all__ = ["SOPAgent"]
