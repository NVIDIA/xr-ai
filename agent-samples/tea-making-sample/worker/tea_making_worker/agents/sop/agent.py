# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose one focused SOP destination with SDK-managed routing and conversation.

Read ``app.run_app`` first for service wiring. The router calls ``admit`` to
classify a turn, then ``answer`` to execute an authorized control or generate a
read-only reply. ``agents.sop._engine`` owns observation scheduling and native state;
``flows`` only returns evidence. Conversation context is a projection of that
state, never an alternative source of workflow authority.

The SDK router owns user-turn cancellation. This module owns the SOP resources,
request-correlated recording, and the handoff between run context and dialogue.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from xr_ai_conversation import ContextManager
from xr_ai_interaction import ResponseOutputs
from xr_ai_models import DecisionService, LLMService, VLMService
from xr_ai_routing import Admission, AgentTurn, DestinationResult, RoutingAid
from xr_ai_runtime import Agent, AgentRuntime, RuntimeClosedError, RuntimeContext, subscribe
from xr_ai_tools import ToolSet
from xr_ai_tools.rag import RAGTools
from xr_ai_tools.vision import ImageQueryTool, StreamingImageQueryTool
from xr_ai_voice import VOICE_OUTPUT_TOPIC

from ..._capture import _attempt, _event
from ...conversation._answer import _answer
from ...conversation._capture import _CompactionLanguage, _context_owner
from ...conversation._routing import _authorized, _choose
from ...events import FOREGROUND_RECORD_TOPIC, GUIDANCE_NOTICE_TOPIC, ForegroundRecord, GuidanceNotice
from ..images import ParticipantImageAgent
from ._context import _HANDOFF, _RunContexts
from ._engine import _WorkflowEngine
from ._spec import Workflow
from ._state import WorkflowEvent, WorkflowSession

_CONTEXT_EVENTS = {
    "step.enter",
    "step.commit",
    "step.ready",
    "workflow.reset",
    "workflow.complete",
    "workflow.skipped",
    "workflow.run_closed",
}
_CONTROLS = {
    "start": ("workflow__start", {}),
    "status": ("workflow__status", {}),
    "advance": ("workflow__advance", {"skip": False}),
    "skip": ("workflow__advance", {"skip": True}),
    "restart": ("workflow__restart", {}),
    "reset": ("workflow__reset", {}),
}


class SOPAgent(Agent):
    """A RoutableAgent-compatible procedure with native state and managed context.

    Load ``workflow`` from YAML and inject existing typed services/tools. Register
    this agent for guidance notices, bind its runtime, and use it as the workflow
    destination of an SDK AgentRouter. The router calls participant initialization
    and departure hooks and owns turn cancellation; stop the router before this
    agent. No conversation agent or independent user-turn supervisor is created.
    """

    def __init__(
        self,
        *,
        workflow: Workflow,
        llm: LLMService,
        decisions: DecisionService,
        images: ParticipantImageAgent,
        vlm: VLMService,
        rag: RAGTools,
        image_query: ImageQueryTool,
        application_tools: Callable[[str], ToolSet],
        prompt: str,
        current_view_prompt: str,
        context_budget_bytes: int = 12000,
        compaction_input_budget_bytes: int = 12000,
        vlm_timeout_s: float = 15,
    ) -> None:
        """Initialize SOP execution, persistent dialogue and run-scoped context.

        Args:
            workflow: Validated external task and step policy, including foreground policy.
            llm: Language service for answers, extraction and context compaction.
            decisions: Bounded semantic-choice service for admission and evidence.
            images: Participant frame source shared with the visual tools.
            vlm: Vision service for streaming current-view answers.
            rag: Retrieval tools used by configured workflow steps.
            image_query: Visual observer with its configured system prompt.
            application_tools: Participant-specific optional background tool catalog.
            prompt: Application-level conversation instructions from worker YAML.
            current_view_prompt: Instructions for streaming current-view answers.
            context_budget_bytes: Budget divided between dialogue and context topics.
            compaction_input_budget_bytes: Input bound for SDK context compaction.
            vlm_timeout_s: Timeout used for visual inference.
        """
        if workflow.foreground is None:
            raise ValueError("SOP routing requires task.foreground")
        self._llm = llm
        self._decisions = decisions
        self._policy = workflow.foreground
        self._images = images
        self._vlm = vlm
        self._rag = rag
        self._prompt = prompt.strip()
        self._view_prompt = current_view_prompt.strip()
        if not self._view_prompt:
            raise ValueError("current_view_prompt must not be empty")
        self._background_tools = application_tools
        self._vlm_timeout_s = vlm_timeout_s
        self._vision = StreamingImageQueryTool(images=images.images, vlm=vlm, system_prompt=self._view_prompt)
        super().__init__((self._vision,))
        # Dialogue outlives a workflow run. Execution history and current readings
        # use separate run-owned managers so a reset cannot retain live evidence.
        self._context = ContextManager(
            _CompactionLanguage(llm),
            incremental_context_topics=(_HANDOFF,),
            dialogue_budget_bytes=context_budget_bytes // 2,
            topic_budget_bytes=max(1, context_budget_bytes // 8),
            compaction_input_budget_bytes=compaction_input_budget_bytes,
        )
        self._workflow_engine = _WorkflowEngine(
            workflow=workflow,
            llm=llm,
            decisions=decisions,
            current_frame=images.get_current_frame,
            image_query=image_query,
            rag=rag,
            vlm_timeout_s=vlm_timeout_s,
            record_event=self._remember_event,
        )
        self._run_contexts = _RunContexts(
            llm,
            self._context,
            budget=max(1, context_budget_bytes // 8),
            compaction_budget=compaction_input_budget_bytes,
        )
        self._pending = {}
        self._closing = False
        self._runtime = None

    def bind_runtime(self, runtime: AgentRuntime) -> None:
        """Bind output publication and native observation tools before routing starts."""
        self._runtime = runtime
        self._responses = ResponseOutputs(
            runtime, VOICE_OUTPUT_TOPIC, source="sop", closed_errors=(RuntimeClosedError,),
        )
        self._workflow_engine.bind_runtime(runtime)

    async def participant_joined(self, participant_id: str) -> None:
        """Initialize native participant state and its observation monitor."""
        await self._workflow_engine._join_participant(participant_id)

    async def participant_left(self, participant_id: str) -> None:
        """Drain native work, then close run context and persistent dialogue."""
        self._pending.pop(participant_id, None)
        try:
            await self._workflow_engine._leave_participant(participant_id)
        finally:
            await self._run_contexts.clear(participant_id)
            await self._context.clear(participant_id)

    def routing_aid(self, participant_id: str) -> RoutingAid:
        """Describe current workflow eligibility without creating participant state."""
        session = self._workflow_engine.store.find(participant_id)
        return RoutingAid(
            "workflow", self._policy.workflow_description,
            eligibility="active procedure" if session and session.active else "idle; explicit start required",
        )

    async def admit(self, turn: AgentTurn, *, focused: bool) -> Admission:
        """Classify and authorize a turn without executing its proposed control.

        The saved admission includes session identity and revision. ``answer`` rechecks
        those values after any intervening observer work or reset.

        Args:
            turn: SDK-owned request and recent interaction history.
            focused: Whether the router currently focuses this workflow destination.

        Returns:
            An admission decision; accepted turns retain a request-correlated proposal.
        """
        self._pending.pop(turn.participant_id, None)
        workflow_session = self._workflow_engine.store.get(turn.participant_id)
        admitted_revision = workflow_session.revision
        routing_state = {
            "active": workflow_session.active,
            "workflow": self._workflow_engine.workflow.name,
            "destination": {"name": "workflow", "focused": focused},
        }
        admission_question = self._policy.idle_question
        if workflow_session.active and workflow_session.step_id is not None:
            active_step = self._workflow_engine.workflow.step(workflow_session.step_id)
            routing_state.update(
                step={"title": active_step.title, "complete": active_step.is_complete(workflow_session.state)},
                known=self._workflow_engine.workflow.project(active_step, workflow_session.state),
            )
            admission_question = self._policy.active_question
        selected_action = await _choose(self._decisions, admission_question, turn, routing_state)
        if selected_action in {"handoff", "other"}:
            return Admission("decline")
        if selected_action in _CONTROLS and selected_action != "status":
            operation_name, operation_arguments = _CONTROLS[selected_action]
            if not await _authorized(
                self._decisions,
                self._policy,
                turn,
                operation=operation_name,
                description=admission_question.criteria[selected_action],
                arguments=operation_arguments,
                target=self._workflow_engine.workflow.name,
            ):
                return Admission("decline")
        self._pending[turn.participant_id] = (turn.turn_id, workflow_session, admitted_revision, selected_action)
        return Admission("accept")

    async def answer(self, turn: AgentTurn, *, started) -> DestinationResult:
        """Execute a still-current admission or answer through read-only conversation.

        Args:
            turn: The same SDK turn previously admitted by this destination.
            started: SDK callback marking response execution as started.

        Returns:
            Focus while the native procedure remains active, otherwise release.
        """
        pending_admission = self._pending.pop(turn.participant_id, None)

        async def produce(reply):
            # Recording/output setup can await before this callback runs. Read the
            # session here, rather than trusting state captured during admission.
            workflow_session = self._workflow_engine.store.find(turn.participant_id)
            if pending_admission is None:
                response_text, executed_tool_names = self._policy.stale_message, []
            else:
                admitted_turn_id, admitted_session, admitted_revision, admitted_action = pending_admission
                if (
                    admitted_turn_id != turn.turn_id
                    or workflow_session is not admitted_session
                    or (admitted_action in _CONTROLS and workflow_session.revision != admitted_revision)
                ):
                    response_text, executed_tool_names = self._policy.stale_message, []
                elif admitted_action in _CONTROLS:
                    operation_name, operation_arguments = _CONTROLS[admitted_action]
                    available_tools = (
                        self._workflow_engine.active_tools(turn.participant_id)
                        or self._workflow_engine.root_tools(turn.participant_id)
                    )
                    selected_tool = available_tools.get(operation_name)
                    if selected_tool is None:
                        response_text, executed_tool_names = self._policy.stale_message, []
                    else:
                        await started()
                        await _event("tool.started", name=operation_name, arguments=operation_arguments)
                        control_result = await selected_tool.execute(
                            selected_tool.request_model.model_validate(operation_arguments)
                        )
                        await _event("tool.finished", name=operation_name, result=control_result)
                        response_text, executed_tool_names = control_result.message, [operation_name]
                else:
                    response_text, executed_tool_names, _ = await self._generate_answer(
                        turn.query, turn.participant_id, workflow_context=True
                    )
            await reply.send(response_text)
            return response_text, executed_tool_names

        response_text = await self._respond(turn, started, produce)
        workflow_session = self._workflow_engine.store.find(turn.participant_id)
        return DestinationResult(
            "focus" if workflow_session and workflow_session.active else "release",
            text=response_text,
        )

    async def release(self, participant_id: str) -> None:
        """Discard unconsumed admission without stopping an active procedure."""
        self._pending.pop(participant_id, None)

    def _snapshot(self, participant_id):
        """Copy native execution state for recording without projecting dialogue."""
        workflow_session = self._workflow_engine.store.find(participant_id)
        return {
            "state": dict(workflow_session.state),
            "step_id": workflow_session.step_id,
            "revision": workflow_session.revision,
            "run_id": workflow_session.run_id,
            "active": workflow_session.active,
            "evidence_hits": workflow_session.evidence_hits,
        } if workflow_session else {}

    async def _respond(self, turn, started, produce):
        """Publish one SDK reply and record only an exchange actually delivered.

        The reply context owns completion/cancellation signaling. History and foreground
        records are appended after its delivery check, so interrupted output cannot become
        a completed assistant exchange.
        """
        workflow_session = self._workflow_engine.store.find(turn.participant_id)
        async with _attempt(
            turn.participant_id,
            "foreground",
            step_id=workflow_session.step_id if workflow_session else None,
            snapshot=lambda: self._snapshot(turn.participant_id),
            request_id=turn.turn_id,
        ):
            await _event("query.received", text=turn.query, timestamp_us=turn.timestamp_us)
            async with self._responses.reply(
                turn.participant_id,
                request_id=turn.turn_id,
                timestamp_us=turn.timestamp_us,
                started=started,
            ) as reply:
                response_text, executed_tool_names = await produce(reply)
            # Delivery is established by the SDK reply context, including its
            # terminator. Do not remember a cancelled/undelivered answer as history.
            if reply.delivered:
                self._remember_exchange(turn.participant_id, turn.query, response_text, turn.turn_id)
                await _event("answer.produced", response=response_text, tools=executed_tool_names)
                await self._runtime.publish(
                    FOREGROUND_RECORD_TOPIC,
                    ForegroundRecord(
                        timestamp_us=turn.timestamp_us if turn.timestamp_us is not None else time.time_ns() // 1000,
                        query=turn.query,
                        response=response_text,
                        tools=executed_tool_names,
                    ),
                    participant_id=turn.participant_id,
                    source="sop",
                )
            return response_text

    def _remember_exchange(self, participant_id, query, response, exchange_id):
        """Attribute the delivered dialogue exchange to its participant for compaction."""
        with _context_owner(participant_id):
            self._context.record_exchange(
                participant_id,
                query,
                response,
                timestamp_us=time.time_ns() // 1000,
                exchange_id=exchange_id,
            )

    def _question_tools(self, participant_id):
        """Expose only the current step's configured read-only voice tools."""
        workflow_session = self._workflow_engine.store.find(participant_id)
        if workflow_session is None or not workflow_session.active or workflow_session.step_id is None:
            return ToolSet(())
        allowed_tool_names = self._workflow_engine.workflow.step(workflow_session.step_id).voice.tools
        available_tools = self._workflow_engine.active_tools(participant_id)
        return ToolSet({name: tool for name, tool in available_tools.items() if name in allowed_tool_names})

    async def _generate_answer(self, query, participant_id, *, workflow_context=False, observation=None):
        """Refresh source-owned context and assemble the common live answer request.

        Workflow questions use the active step's policy and read-only tool allowlist.
        Other questions use application conversation policy; both use the same SDK tool
        loop and the same separation of historical and current context.
        """
        await self._refresh_conversation_context(participant_id)
        workflow_session = self._workflow_engine.store.find(participant_id)
        workflow_specification = self._workflow_engine.workflow
        has_active_step = (
            workflow_context
            and workflow_session is not None
            and workflow_session.active
            and workflow_session.step_id is not None
        )
        projected_state = {}
        instructions = self._prompt + "\n\n" + self._policy.conversation_prompt
        if has_active_step:
            active_step = workflow_specification.step(workflow_session.step_id)
            step_sequence = ", then ".join(item.title for item in workflow_specification.steps.values())
            next_step_title = (
                workflow_specification.step(active_step.next_step).title
                if active_step.next_step else "finish the guide"
            )
            instructions = "\n".join((
                self._prompt,
                workflow_specification.foreground_prompt,
                f"Guide order: {step_sequence}. Current step: {active_step.title}. Next: {next_step_title}.",
                active_step.voice.prompt,
                self._policy.active_prompt,
            ))
            projected_state = workflow_specification.project(active_step, workflow_session.state)
        response_text, executed_tool_names = await _answer(
            self._llm,
            query,
            instructions=instructions,
            history=self._context_blocks(participant_id),
            current=self._current_context_blocks(participant_id),
            state=projected_state,
            execution={
                "active": bool(workflow_session and workflow_session.active),
                "run_id": workflow_session.run_id if workflow_session else None,
            },
            tools=self._question_tools(participant_id) if has_active_step else ToolSet(()),
            observation=observation,
            capabilities=None if has_active_step else {
                "workflow": self._policy.workflow_description,
                **{name: item.description for name, item in self._policy.destinations.items()},
            },
        )
        await _event("conversation.tools", calls=executed_tool_names)
        return response_text, executed_tool_names, False

    @subscribe(GUIDANCE_NOTICE_TOPIC)
    async def notify(self, notice: GuidanceNotice, ctx: RuntimeContext) -> None:
        """Forward guidance only while the participant and this agent remain live."""
        participant_id = ctx.metadata.participant_id
        if not self._closing and participant_id in self._workflow_engine._connected:
            # Guidance updates are not replies to an admitted query. Bind outputs
            # to the notice's runtime context, retaining its original routing metadata.
            outputs = ResponseOutputs(ctx, VOICE_OUTPUT_TOPIC)
            await outputs.update(participant_id, notice.text, timestamp_us=notice.timestamp_us)

    async def _remember_event(self, session: WorkflowSession, event: WorkflowEvent) -> None:
        """Project relevant events only from the current native participant session."""
        if (
            self._closing
            or self._workflow_engine.store.find(session.participant_id) is not session
            or event.event not in _CONTEXT_EVENTS
        ):
            return
        await self._run_contexts.event(session.participant_id, event, current_run=session.run_id)

    async def _refresh_conversation_context(self, participant_id: str) -> None:
        """Refresh native context, retrying snapshots rejected by the run-context guard.

        Snapshot acquisition and manager rollover can race an observation or reset.
        The callback checks session identity and revision before the run-context update;
        if rejected, the loop obtains a new authoritative snapshot.
        """
        while not self._closing:
            snapshot = await self._workflow_engine.context_snapshot(participant_id)

            def valid():
                current_session = self._workflow_engine.store.find(participant_id)
                return (
                    not self._closing
                    and current_session is not None
                    and current_session.run_id == snapshot.get("run_id")
                    and current_session.revision == snapshot.get("revision")
                )

            if self._workflow_engine.store.find(participant_id) is None:
                return
            if await self._run_contexts.refresh(participant_id, snapshot, valid=valid):
                return

    def _context_blocks(self, participant_id: str):
        """Combine persistent dialogue with the active run's historical event context."""
        return (*self._context.blocks(participant_id), *self._run_contexts.history_blocks(participant_id))

    def _current_context_blocks(self, participant_id: str):
        """Return only replaceable live execution/evidence context."""
        return self._run_contexts.current_blocks(participant_id)

    async def stop(self) -> None:
        """Close SOP resources after the caller has stopped the SDK router."""
        self._closing = True
        await self._workflow_engine.stop()
        await self._run_contexts.close()
        await self._context.close()


__all__ = ["SOPAgent"]
