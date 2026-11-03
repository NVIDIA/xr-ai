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
import math
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
from xr_ai_models import DecisionAnswer, DecisionQuestion, DecisionService, LLMService
from xr_ai_routing import Admission, AgentTurn, DestinationResult, RoutingAid, current_turn_context
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
from .tea_data_models import BrewingSettings, TeaIdentity, _TemperatureObservation
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
    last_interaction: dict[str, Any] | None = None
    display_unit: str | None = None
    display_unit_source: dict[str, Any] | None = None
    temperature_context_revision: int = 0
    unit_gap_asked: tuple[str, int] | None = None
    native_reply_entry: tuple[int, asyncio.Future[str]] | None = None
    timer_start_source: str | None = None


@dataclass(frozen=True, slots=True)
class _InteractionSnapshot:
    """Native identity plus the small immutable view used to interpret a turn."""

    session: _Session
    run_id: str | None
    revision: int
    state: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _InteractionProposal:
    """Model choices remain proposals; native effects require positive support."""

    operation: DecisionAnswer
    referent: DecisionAnswer | None = None
    instrument_unit: DecisionAnswer | None = None

    @staticmethod
    def supported(answer: DecisionAnswer) -> bool:
        """Keep the existing positive-versus-all-other-mass authorization rule."""
        probabilities = answer.probabilities
        return (
            answer.choice in probabilities
            and all(math.isfinite(value) and value >= 0 for value in probabilities.values())
            and probabilities[answer.choice] > math.fsum(
                value for label, value in probabilities.items() if label != answer.choice
            )
        )


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
        identification_image_query: Optional identification-only structured reader.
        temperature_image_query: Optional temperature-only structured reader; other checks use image_query.
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
        identification_image_query: ImageQueryTool | None = None,
        temperature_image_query: ImageQueryTool | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self._llm = llm
        self._decisions = decisions
        self._frames = frames
        self._image_query = image_query
        self._identification_image_query = identification_image_query or image_query
        self._temperature_image_query = temperature_image_query or image_query
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
            await self._stop_run(session)
            await self._context.clear(participant_id)

    def _interaction_snapshot(self, participant_id: str) -> _InteractionSnapshot | None:
        """Expose native goals/outcomes, without treating the cursor as user intent."""
        session = self._sessions.get(participant_id)
        if session is None or not session.connected or self._closing:
            return None
        settings = self._resolved_settings(session)
        steps = {
            step_id: {
                "goal": self.config.messages[step_id].format_map(settings),
                "outcome": outcome,
                "ready": step_id == self._step_id(session) and session.ready,
            }
            for step_id, outcome in session.flow_statuses.items()
            if step_id in _STEP_IDS
        }
        state = {"status": session.status, "current_step": self._step_id(session), "steps": steps}
        if self._step_id(session) == "heat_water":
            state["instrument_unit"] = {"value": session.display_unit, "source": session.display_unit_source}
        return _InteractionSnapshot(session, session.run_id, session.revision, state)

    async def _apply_interaction(
        self, snapshot: _InteractionSnapshot, turn: AgentTurn, proposal: _InteractionProposal | None,
        *, started: Callable[[], Awaitable[None]],
    ) -> str | None:
        """Claim the response before committing, then return native reply text."""
        session = snapshot.session
        async with self._participant_locks.setdefault(turn.participant_id, asyncio.Lock()):
            if proposal is None:
                if self._sessions.get(turn.participant_id) is session and session.connected and not self._closing:
                    session.last_interaction = {
                        "turn_id": turn.turn_id, "operation": "undetermined",
                        "outcome": "not_interpreted_input_limit", "native_effect": False,
                        "recorded_at_us": time.time_ns() // 1000,
                    }
                    await self._publish_context(session)
                return
            operation = proposal.operation.choice
            target = proposal.referent.choice if proposal.referent is not None else "not_requested"
            unit = proposal.instrument_unit
            scope_matches = self._interaction_is_current(
                snapshot, turn.participant_id, require_revision=operation not in {"start", "stop", "restart"}
            )
            action_supported = (proposal.referent is not None and proposal.supported(proposal.operation) and proposal.supported(proposal.referent))
            step_effect = (
                operation in {"next", "skip"} and target == self._step_id(session)
                and target in session.flow_statuses and session.status == "running"
                and ((session.ready and self.config.advancement == "manual") or (operation == "skip" and not session.ready))
            )
            workflow_effect = operation in {"start", "stop", "restart"} and target == "workflow"
            unit_effect = (
                unit is not None and proposal.supported(unit) and unit.choice in {"celsius", "fahrenheit"}
                and self._step_id(session) == "heat_water" and session.status == "running"
                and unit.choice != session.display_unit
            )
            if (scope_matches and action_supported and (step_effect or workflow_effect)) or (
                self._interaction_is_current(snapshot, turn.participant_id) and unit_effect
            ):
                # Await in the caller's cancellation scope. A newer turn may
                # cancel this claim or the workflow may advance while it waits.
                await started()
                if asyncio.current_task().cancelling():
                    raise asyncio.CancelledError
                committed = asyncio.create_task(
                    self._finish_interaction(snapshot, turn, proposal),
                    context=nemo_relay.fork_asyncio_context(),
                )
                try:
                    return await asyncio.shield(committed)
                except asyncio.CancelledError:
                    while not committed.done():
                        try:
                            await asyncio.shield(committed)
                        except asyncio.CancelledError:
                            continue
                    committed.result()
                    raise
            else:
                return await self._finish_interaction(snapshot, turn, proposal)

    def _interaction_is_current(
        self, snapshot: _InteractionSnapshot, participant_id: str, *, require_revision: bool = True
    ) -> bool:
        """Check exact session/run/revision ownership, including fact updates."""
        return (
            not self._closing and self._sessions.get(participant_id) is snapshot.session
            and snapshot.session.connected and snapshot.session.run_id == snapshot.run_id
            and (not require_revision or snapshot.session.revision == snapshot.revision)
        )

    async def _finish_interaction(self, snapshot: _InteractionSnapshot, turn: AgentTurn, proposal: _InteractionProposal) -> str | None:
        """Revalidate at commit and publish what actually happened for Conversation."""
        session = snapshot.session
        action_supported = (proposal.referent is not None and proposal.supported(proposal.operation) and proposal.supported(proposal.referent))
        operation = proposal.operation.choice if action_supported or proposal.operation.choice == "none" else "unclear"
        target = proposal.referent.choice if proposal.referent is not None else "not_requested"
        unit = proposal.instrument_unit
        instrument_unit = unit.choice if unit is not None and proposal.supported(unit) else None
        outcome = "no_workflow_change"
        announcement: tuple[int, asyncio.Future[str]] | None = None
        response_text = None
        current_scope = self._interaction_is_current(snapshot, turn.participant_id)
        request_scope = self._interaction_is_current(
            snapshot, turn.participant_id, require_revision=operation not in {"start", "stop", "restart"}
        )
        if operation in {"none", "unclear"} and instrument_unit not in {"celsius", "fahrenheit"}:
            # Informational turns have no stale action to repeat. Conversation
            # should see the latest state even if the workflow advanced/reset.
            request_scope = self._sessions.get(turn.participant_id) is session and session.connected and not self._closing
        completed_target = (
            operation in {"next", "skip"} and target != self._step_id(session)
            and session.flow_statuses.get(target) in {"completed", "skipped"}
            and self._interaction_is_current(snapshot, turn.participant_id, require_revision=False)
        )
        if completed_target:
            outcome = "referenced_step_already_finished"
        elif not request_scope:
            outcome = "request_became_stale"
        elif operation in {"next", "skip"}:
            if session.status != "running":
                outcome = "workflow_not_running"
            elif target in session.flow_statuses and target != self._step_id(session):
                outcome = "referenced_step_already_finished"
            elif target == self._step_id(session) and target in session.flow_statuses:
                if operation == "next" and not session.ready:
                    outcome = "current_step_not_confirmed"
                elif session.ready:
                    if self.config.advancement == "manual":
                        announcement = (session.step_index + 1, asyncio.get_running_loop().create_future())
                        await self._control(session, "next", native_reply_entry=announcement)
                        outcome = "advanced_confirmed_step"
                    else:
                        outcome = "referenced_step_already_finished"
                else:
                    next_index = session.step_index + 1
                    announcement = (next_index, asyncio.get_running_loop().create_future())
                    await self._control(session, operation, native_reply_entry=announcement)
                    outcome = "requested_step_skipped"
            else:
                outcome = "requested_step_not_established"
        elif operation in {"start", "stop", "restart"} and target == "workflow":
            previous_run = session.run_id
            already_running = operation == "start" and session.status == "running"
            if operation in {"start", "restart"} and not already_running:
                announcement = (0, asyncio.get_running_loop().create_future())
            response_text = await self._control(session, operation, native_reply_entry=announcement)
            if already_running and self._step_id(session) is not None:
                response_text += " " + self.config.messages[self._step_id(session)].format_map(self._resolved_settings(session))
            outcome = "already_running" if already_running else f"workflow_{operation}_applied"
            if session.run_id is not None and previous_run != session.run_id:
                outcome = f"workflow_{operation}_new_run"
        if (
            current_scope
            and instrument_unit in {"celsius", "fahrenheit"}
            and self._sessions.get(turn.participant_id) is session
            and session.connected
            and session.run_id == snapshot.run_id
            and self._step_id(session) == "heat_water"
            and session.status == "running"
            and instrument_unit != session.display_unit
        ):
            session.display_unit = instrument_unit
            session.display_unit_source = {"kind": "user_declared", "turn_id": turn.turn_id, "text": turn.query}
            session.temperature_context_revision += 1
            session.revision += 1
            outcome = "instrument_unit_updated"
            # Replace this check's input context, not the SOP step. Its
            # confirmation window restarts after old inference has drained.
            current_check = session.active_tasks.get("heat_water")
            if current_check is not None and not current_check.done():
                if not current_check.cancelling():
                    current_check.cancel()
                await self._drain(current_check)
        if announcement is not None:
            # Entry publishes the actual new goal and supplies one authoritative
            # native reply. It is never regenerated by a language model.
            response_text = await announcement[1]
        session.last_interaction = {
            "turn_id": turn.turn_id,
            "operation": operation,
            "requested_target": target,
            "outcome": outcome,
            "recorded_at_us": time.time_ns() // 1000,
        }
        await self._publish_context(session)
        return response_text

    async def stop(self) -> None:
        """Close admission and drain every participant's owned procedure."""
        self._closing = True
        for participant_id in tuple(self._sessions):
            await self.participant_left(participant_id)

    async def _control(
        self, session: _Session, control: str, *, native_reply_entry: tuple[int, asyncio.Future[str]] | None = None
    ) -> str:
        """End/reset share one drained idle transition; restart then starts fresh."""
        # Keep the former private spelling as a boundary alias, never as a
        # second semantic choice or a separate cleanup implementation.
        if control == "reset":
            control = "stop"
        if control in {"start", "restart"}:
            if control == "start" and session.status == "running":
                return self.config.messages["already_running"]
            await self._control(session, "stop")
            session.run_id = uuid4().hex
            session.revision += 1
            session.status = "running"
            session.issue = ""
            session.native_reply_entry = native_reply_entry
            session.procedure = asyncio.create_task(
                self._supervise(session),
                name=f"sop:{session.participant_id}",
                context=nemo_relay.fork_asyncio_context(),
            )
            await self._publish_context(session)
            return self.config.messages["start"]
        if control == "stop":
            await self._stop_run(session)
            if session.run_id:
                session.last_run = self._run_summary(session)
            session.status = "idle"
            session.run_id = None
            session.revision += 1
            session.step_index = -1
            session.flow_states.clear()
            session.flow_statuses.clear()
            session.ready = False
            session.skipping = False
            session.advance_requested.clear()
            session.display_unit = None
            session.display_unit_source = None
            session.temperature_context_revision = 0
            session.unit_gap_asked = None
            session.timer_start_source = None
            if session.native_reply_entry is not None:
                if not session.native_reply_entry[1].done():
                    session.native_reply_entry[1].set_result(self.config.messages["not_running"])
                session.native_reply_entry = None
            await self._publish_context(session)
            return self.config.messages["stop"]
        if control in {"next", "skip"}:
            if session.status != "running":
                return self.config.messages["not_running"]
            if control == "next" and not session.ready:
                return self.config.messages["not_ready"]
            session.native_reply_entry = native_reply_entry
            if control == "skip":
                session.skipping = True
                # Signal this step before awaiting cancellation; the procedure
                # may already enter its next step while the child drains.
                session.advance_requested.set()
                active_step = self._step_id(session)
                step_task = session.active_tasks.get(active_step)
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
                session.flow_statuses[self._step_id(session)] = "failed"
            if self._sessions.get(session.participant_id) is session:
                await self._publish_context(session)
        finally:
            if session.native_reply_entry is not None:
                if not session.native_reply_entry[1].done():
                    session.native_reply_entry[1].set_result(self.config.messages["not_running"])
                session.native_reply_entry = None
            owned_flow_tasks = tuple(session.active_tasks.values())
            for flow_task in owned_flow_tasks:
                if not flow_task.cancelling():
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
        flow_arguments = {**self._observation_arguments(session), "observer": self._identification_image_query}
        await self._run_step(
            session,
            0,
            lambda: run_vision_rag_retrieval_flow(
                **flow_arguments,
                llm=self._llm,
                retrieval=self._rag.retrieve,
                visual_model=TeaIdentity,
                result_model=BrewingSettings,
                can_generate=lambda identity: identity.tea_type is not None,
                visual_question=self.config.identification_question,
                extraction_instructions=self.config.extraction_instructions,
                fallback_instructions=self.config.fallback_instructions,
                retrieval_query=lambda identity: " ".join(
                    filter(None, (identity.tea_name, identity.tea_type, self.config.retrieval_query_suffix))
                ),
                max_retrieved_passages=self.config.max_retrieved_passages,
                retrieved_text_limit=self.config.retrieved_text_limit,
                on_state=partial(self._flow_state, session, session.run_id, "identify"),
            ),
        )

        # Skip defaults are applied here, after preserving validated partial
        # values reported by identification. They are never flow-internal guesses.
        resolved_settings = self._resolved_settings(session)
        await self._run_step(session, 1, lambda: self._verify(session, "fill_water", self.config.water))

        # Only this application builds tea-specific wording. The visual function
        # receives an ordinary completed question/DecisionQuestion.
        await self._run_step(session, 2, lambda: self._temperature_check(session, resolved_settings["temperature_c"]))
        steeping_confirmation = await self._run_step(
            session,
            3,
            lambda: self._verify(session, "start_steeping", self.config.tea_bag),
        )
        # Bypassing visual confirmation is a manual progression, not a request
        # to omit steeping. The independent timer starts now in both cases.
        session.timer_start_source = "visual_confirmation" if steeping_confirmation is not None else "user_skipped_visual_check"
        await self._run_step(
            session,
            4,
            lambda: run_timer_flow(
                duration_seconds=resolved_settings["duration_seconds"],
                update_interval_seconds=self.config.timer_update_interval_seconds,
                on_state=partial(self._flow_state, session, session.run_id, "steep_timer"),
            ),
        )
        session.status = "completed"
        session.revision += 1
        await self._publish_context(session)
        await self._announce_entry(session, len(_STEP_IDS), self.config.messages["complete"])

    def _temperature_check(self, session: _Session, target_celsius: float):
        """Construct this execution from explicit reading/unit provenance."""
        declared_unit = session.display_unit
        replacements = {
            "target_temperature_c": f"{target_celsius:g}",
            "target_temperature_f": f"{target_celsius * 9 / 5 + 32:g}",
            "instrument_unit": declared_unit or "not established",
        }
        instructions = self.config.temperature
        completion = DecisionQuestion(
            instructions.completion.instructions.format_map(replacements),
            {label: criterion.format_map(replacements) for label, criterion in instructions.completion.criteria.items()},
        )

        def can_assess(reading: _TemperatureObservation) -> bool:
            effective_unit = reading.unit or declared_unit
            conflict = reading.unit is not None and declared_unit is not None and reading.unit != declared_unit
            return (reading.value is not None and reading.role == "current"
                    and bool(reading.role_evidence and reading.role_evidence.strip())
                    and effective_unit is not None and not conflict)

        return run_visual_verification_flow(
            **{**self._observation_arguments(session), "observer": self._temperature_image_query},
            decisions=self._decisions, question=instructions.question, completion=completion,
            observation_model=_TemperatureObservation, can_assess=can_assess,
            required_positives=self.config.required_positives, window_size=self.config.confirmation_window_size,
            on_state=partial(self._flow_state, session, session.run_id, "heat_water"),
        )

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

    async def _run_step(self, session: _Session, step_index: int, create_flow: Callable[[], Coroutine[Any, Any, Any]]):
        """Own one lazily constructed step, including parameter-driven replacement."""
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
            settings = self._resolved_settings(session)
            instruction = self.config.messages[step_id].format_map(settings)
            if step_index == 1 and session.flow_statuses.get("identify") == "completed":
                # One update combines resolution and next guidance; consecutive
                # same-source updates can replace each other in the voice queue.
                instruction = self.config.messages["identification_complete"].format_map({
                    **settings, "duration_minutes": settings["duration_seconds"] / 60,
                    "temperature_source": settings["sources"]["temperature"]["kind"],
                    "duration_source": settings["sources"]["steep_duration"]["kind"],
                    "next_instruction": instruction,
                })
            elif step_index == 2:
                water_outcome = session.flow_statuses.get("fill_water")
                if water_outcome in {"completed", "skipped"}:
                    message = "water_verified" if water_outcome == "completed" else "water_skipped"
                    instruction = self.config.messages[message].format_map({
                        **settings, "next_instruction": instruction,
                    })
            await self._announce_entry(session, step_index, instruction)
            async with _attempt(
                session.participant_id,
                "flow",
                step_id=step_id,
                snapshot=lambda: self.context_snapshot(session.participant_id),
            ):
                while True:
                    if session.skipping:
                        result = None
                        break
                    parameter_revision = session.temperature_context_revision
                    flow_task = asyncio.create_task(
                        create_flow(), name=f"sop:{session.participant_id}:{step_id}",
                        context=nemo_relay.fork_asyncio_context(),
                    )
                    session.active_tasks[step_id] = flow_task
                    try:
                        result = await asyncio.shield(flow_task)
                        break
                    except asyncio.CancelledError:
                        owner = asyncio.current_task()
                        if owner is not None and owner.cancelling():
                            raise
                        if session.skipping:
                            result = None
                            break
                        if step_id == "heat_water" and session.temperature_context_revision != parameter_revision:
                            session.flow_states.pop(step_id, None)
                            session.flow_statuses[step_id] = "running"
                            continue
                        raise
                session.flow_statuses[step_id] = "skipped" if session.skipping else "completed"
                session.ready = True
                await self._publish_context(session)
                await _event("step.finished", skipped=session.skipping, result=result)
                if self.config.advancement == "manual" and not session.skipping:
                    await session.advance_requested.wait()
                return result
        finally:
            if flow_task is not None and not flow_task.done():
                if not flow_task.cancelling():
                    flow_task.cancel()
                await self._drain(flow_task)
            if session.active_tasks.get(step_id) is flow_task:
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

        if flow_id == "heat_water" and state.observations:
            reading = state.observations[-1].parsed
            if isinstance(reading, _TemperatureObservation) and reading.value is not None and reading.role == "current":
                conflict = reading.unit is not None and session.display_unit is not None and reading.unit != session.display_unit
                gap = "conflicting_units" if conflict else "missing_unit" if reading.unit is None and session.display_unit is None else None
                marker = (gap, session.temperature_context_revision)
                if gap is not None and marker != session.unit_gap_asked and "unit_needed" in self.config.messages:
                    session.unit_gap_asked = marker
                    await self._notice(session, self.config.messages["unit_needed"])

    def _resolved_settings(self, session: _Session) -> dict[str, Any]:
        """Use valid partial results and label only missing values as runner defaults."""
        identification = session.flow_states.get("identify")
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

    def context_snapshot(self, participant_id: str) -> dict[str, Any]:
        """Expose entered steps, never turn future configuration into observed state."""
        session = self._sessions.get(participant_id)
        if session is None:
            return {"status": "idle", "active_flows": {}}
        if session.status == "idle":
            return {
                "status": "idle",
                "active_flows": {},
                "historical_previous_run": session.last_run,
                "last_interaction": session.last_interaction,
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
                if step_id in session.flow_statuses
            },
            "flow_outcomes": dict(session.flow_statuses),
            "active_flows": {
                flow_id: {
                    **_json(state),
                    "sampled_at_utc": datetime.fromtimestamp(state.updated_at_us / 1_000_000, UTC).isoformat(
                        timespec="seconds"
                    ),
                }
                for flow_id, state in session.flow_states.items()
                if session.flow_statuses.get(flow_id) == "running" and session.status == "running"
            },
            "completed_or_skipped_flows": {
                flow_id: {"outcome": session.flow_statuses.get(flow_id), "state": _json(state)}
                for flow_id, state in session.flow_states.items()
                if session.flow_statuses.get(flow_id) != "running"
            },
            "instrument_unit": {"value": session.display_unit, "source": session.display_unit_source},
            "timer_start_source": session.timer_start_source,
            "issue": session.issue,
            "last_interaction": session.last_interaction,
            "historical_previous_run": session.last_run,
        }
        # Defaults become execution settings only after identification completes or
        # is explicitly skipped. Partial model values remain in the flow state.
        if session.flow_statuses.get("identify") in {"completed", "skipped"}:
            snapshot["resolved_settings"] = self._resolved_settings(session)
        return snapshot

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

    async def _announce_entry(self, session: _Session, step_index: int, text: str) -> None:
        """Give user-caused guidance to the native reply; speak autonomous progress."""
        reservation = session.native_reply_entry
        if reservation is not None and reservation[0] == step_index:
            session.native_reply_entry = None
            if not reservation[1].done():
                reservation[1].set_result(text)
            return
        await self._notice(session, text)

    async def _reply_to_control(
        self, turn: AgentTurn, text: str, *, started: Callable[[], Awaitable[None]]
    ) -> DestinationResult:
        """Deliver native workflow facts once and retain the real spoken exchange."""
        async with self._responses.reply(
            turn.participant_id, request_id=turn.turn_id, timestamp_us=turn.timestamp_us, started=started
        ) as reply:
            await reply.send(text)
        if reply.delivered and not asyncio.current_task().cancelling():
            self._context.record_exchange(
                turn.participant_id, turn.query, text, timestamp_us=time.time_ns() // 1000,
                exchange_id=current_turn_context().exchange_id,
            )
        return DestinationResult("answer", text)

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
            if not session.procedure.cancelling():
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
            "settings": self._resolved_settings(session)
            if session.flow_statuses.get("identify") in {"completed", "skipped"}
            else {},
            "last_step": self._step_id(session),
        }


__all__ = ["SOPAgent"]
