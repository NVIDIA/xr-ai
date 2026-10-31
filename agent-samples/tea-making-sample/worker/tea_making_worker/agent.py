# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The focused SOP destination; SDK routing owns user-turn lifecycle."""

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

from ._capture import _attempt, _event
from .conversation._answer import _answer
from .conversation._capture import _CompactionLanguage, _context_owner
from .conversation._context import _HANDOFF, _RunContexts
from .conversation._routing import _authorized, _choose
from .events import FOREGROUND_RECORD_TOPIC, GUIDANCE_NOTICE_TOPIC, ForegroundRecord, GuidanceNotice
from .guidance_voice import GuidanceVoiceAgent
from .images import ParticipantImageAgent
from .sop._engine import GuidanceAgent
from .sop._spec import Workflow
from .sop._state import WorkflowEvent, WorkflowSession

_CONTEXT_EVENTS = {
    "step.enter", "step.commit", "step.ready", "workflow.reset",
    "workflow.complete", "workflow.skipped", "workflow.run_closed",
}
_CONTROLS = {
    "start": ("workflow__start", {}), "status": ("workflow__status", {}),
    "advance": ("workflow__advance", {"skip": False}), "skip": ("workflow__advance", {"skip": True}),
    "restart": ("workflow__restart", {}), "reset": ("workflow__reset", {}),
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
        self, *, workflow: Workflow, llm: LLMService, decisions: DecisionService,
        images: ParticipantImageAgent, vlm: VLMService, rag: RAGTools,
        image_query: ImageQueryTool, application_tools: Callable[[str], ToolSet],
        prompt: str, current_view_prompt: str, context_budget_bytes: int = 12000,
        compaction_input_budget_bytes: int = 12000, vlm_timeout_s: float = 15,
    ) -> None:
        if workflow.foreground is None:
            raise ValueError("SOP routing requires task.foreground")
        self._llm, self._decisions, self._policy = llm, decisions, workflow.foreground
        self._images, self._vlm, self._rag = images, vlm, rag
        self._prompt, self._view_prompt = prompt.strip(), current_view_prompt.strip()
        if not self._view_prompt:
            raise ValueError("current_view_prompt must not be empty")
        self._background_tools, self._vlm_timeout_s = application_tools, vlm_timeout_s
        self._vision = StreamingImageQueryTool(images=images.images, vlm=vlm, system_prompt=self._view_prompt)
        super().__init__((self._vision,))
        self._context = ContextManager(
            _CompactionLanguage(llm), incremental_context_topics=(_HANDOFF,),
            dialogue_budget_bytes=context_budget_bytes // 2, topic_budget_bytes=max(1, context_budget_bytes // 8),
            compaction_input_budget_bytes=compaction_input_budget_bytes,
        )
        self._guidance = GuidanceAgent(
            workflow=workflow, llm=llm, decisions=decisions,
            current_frame=images.get_current_frame, image_query=image_query, rag=rag,
            vlm_timeout_s=vlm_timeout_s, record_event=self._remember_event,
        )
        self._run_contexts = _RunContexts(
            llm, self._context, budget=max(1, context_budget_bytes // 8),
            compaction_budget=compaction_input_budget_bytes,
        )
        self._pending = {}
        self._notice_voice = GuidanceVoiceAgent()
        self._closing = False
        self._runtime = None

    def bind_runtime(self, runtime: AgentRuntime) -> None:
        self._runtime = runtime
        self._responses = ResponseOutputs(
            runtime, VOICE_OUTPUT_TOPIC, source="sop", closed_errors=(RuntimeClosedError,),
        )
        self._guidance.bind_runtime(runtime)

    async def participant_joined(self, participant_id: str) -> None:
        await self._guidance._join_participant(participant_id)

    async def participant_left(self, participant_id: str) -> None:
        self._pending.pop(participant_id, None)
        try:
            await self._guidance._leave_participant(participant_id)
        finally:
            await self._run_contexts.clear(participant_id)
            await self._context.clear(participant_id)

    def routing_aid(self, participant_id: str) -> RoutingAid:
        session = self._guidance.store.find(participant_id)
        return RoutingAid(
            "workflow", self._policy.workflow_description,
            eligibility="active procedure" if session and session.active else "idle; explicit start required",
        )

    async def admit(self, turn: AgentTurn, *, focused: bool) -> Admission:
        self._pending.pop(turn.participant_id, None)
        session = self._guidance.store.get(turn.participant_id)
        revision = session.revision
        state = {"active": session.active, "workflow": self._guidance.workflow.name,
                 "destination": {"name": "workflow", "focused": focused}}
        question = self._policy.idle_question
        if session.active and session.step_id is not None:
            step = self._guidance.workflow.step(session.step_id)
            state.update(step={"title": step.title, "complete": step.is_complete(session.state)},
                         known=self._guidance.workflow.project(step, session.state))
            question = self._policy.active_question
        action = await _choose(self._decisions, question, turn, state)
        if action in {"handoff", "other"}:
            return Admission("decline")
        if action in _CONTROLS and action != "status":
            name, arguments = _CONTROLS[action]
            if not await _authorized(self._decisions, self._policy, turn, operation=name,
                                     description=question.criteria[action], arguments=arguments,
                                     target=self._guidance.workflow.name):
                return Admission("decline")
        self._pending[turn.participant_id] = (turn.turn_id, session, revision, action)
        return Admission("accept")

    async def answer(self, turn: AgentTurn, *, started) -> DestinationResult:
        pending = self._pending.pop(turn.participant_id, None)

        async def produce(reply):
            session = self._guidance.store.find(turn.participant_id)
            if (pending is None or pending[0] != turn.turn_id or session is not pending[1]
                    or (pending[3] in _CONTROLS and session.revision != pending[2])):
                text, calls = self._policy.stale_message, []
            elif pending[3] in _CONTROLS:
                name, arguments = _CONTROLS[pending[3]]
                catalog = (self._guidance.active_tools(turn.participant_id)
                           or self._guidance.root_tools(turn.participant_id))
                tool = catalog.get(name)
                if tool is None:
                    text, calls = self._policy.stale_message, []
                else:
                    await started()
                    await _event("tool.started", name=name, arguments=arguments)
                    result = await tool.execute(tool.request_model.model_validate(arguments))
                    await _event("tool.finished", name=name, result=result)
                    text, calls = result.message, [name]
            else:
                text, calls, _ = await self._generate_answer(turn.query, turn.participant_id, workflow_context=True)
            await reply.send(text)
            return text, calls

        text = await self._respond(turn, started, produce)
        session = self._guidance.store.find(turn.participant_id)
        return DestinationResult("focus" if session and session.active else "release", text=text)

    async def release(self, participant_id: str) -> None:
        self._pending.pop(participant_id, None)

    def _snapshot(self, participant_id):
        session = self._guidance.store.find(participant_id)
        return {"state": dict(session.state), "step_id": session.step_id, "revision": session.revision,
                "run_id": session.run_id, "active": session.active,
                "evidence_hits": session.evidence_hits} if session else {}

    async def _respond(self, turn, started, produce):
        session = self._guidance.store.find(turn.participant_id)
        async with _attempt(turn.participant_id, "foreground", step_id=session.step_id if session else None,
                            snapshot=lambda: self._snapshot(turn.participant_id), request_id=turn.turn_id):
            await _event("query.received", text=turn.query, timestamp_us=turn.timestamp_us)
            async with self._responses.reply(turn.participant_id, request_id=turn.turn_id,
                                             timestamp_us=turn.timestamp_us, started=started) as reply:
                text, calls = await produce(reply)
            if reply.delivered:
                self._remember_exchange(turn.participant_id, turn.query, text, turn.turn_id)
                await _event("answer.produced", response=text, tools=calls)
                await self._runtime.publish(FOREGROUND_RECORD_TOPIC, ForegroundRecord(
                    timestamp_us=turn.timestamp_us if turn.timestamp_us is not None else time.time_ns() // 1000,
                    query=turn.query, response=text, tools=calls,
                ), participant_id=turn.participant_id, source="sop")
            return text

    def _remember_exchange(self, participant_id, query, response, exchange_id):
        with _context_owner(participant_id):
            self._context.record_exchange(participant_id, query, response,
                                          timestamp_us=time.time_ns() // 1000, exchange_id=exchange_id)

    def _question_tools(self, participant_id):
        session = self._guidance.store.find(participant_id)
        if session is None or not session.active or session.step_id is None:
            return ToolSet(())
        names = self._guidance.workflow.step(session.step_id).voice.tools
        catalog = self._guidance.active_tools(participant_id)
        return ToolSet({name: tool for name, tool in catalog.items() if name in names})

    async def _generate_answer(self, query, participant_id, *, workflow_context=False, observation=None):
        await self._refresh_conversation_context(participant_id)
        session = self._guidance.store.find(participant_id)
        workflow = self._guidance.workflow
        active = workflow_context and session is not None and session.active and session.step_id is not None
        state = {}
        instructions = self._prompt + "\n\n" + self._policy.conversation_prompt
        if active:
            step = workflow.step(session.step_id)
            sequence = ", then ".join(item.title for item in workflow.steps.values())
            next_title = workflow.step(step.next_step).title if step.next_step else "finish the guide"
            instructions = "\n".join((self._prompt, workflow.foreground_prompt,
                f"Guide order: {sequence}. Current step: {step.title}. Next: {next_title}.",
                step.voice.prompt, self._policy.active_prompt))
            state = workflow.project(step, session.state)
        text, calls = await _answer(
            self._llm, query, instructions=instructions, history=self._context_blocks(participant_id),
            current=self._current_context_blocks(participant_id), state=state,
            execution={"active": bool(session and session.active), "run_id": session.run_id if session else None},
            tools=self._question_tools(participant_id) if active else ToolSet(()), observation=observation,
            capabilities=None if active else {"workflow": self._policy.workflow_description,
                **{name: item.description for name, item in self._policy.destinations.items()}},
        )
        await _event("conversation.tools", calls=calls)
        return text, calls, False

    @subscribe(GUIDANCE_NOTICE_TOPIC)
    async def notify(self, notice: GuidanceNotice, ctx: RuntimeContext) -> None:
        if not self._closing and ctx.metadata.participant_id in self._guidance._connected:
            await self._notice_voice.notify(notice, ctx)

    async def _remember_event(self, session: WorkflowSession, event: WorkflowEvent) -> None:
        if (
            self._closing
            or self._guidance.store.find(session.participant_id) is not session
            or event.event not in _CONTEXT_EVENTS
        ):
            return
        await self._run_contexts.event(session.participant_id, event, current_run=session.run_id)


    async def _refresh_conversation_context(self, participant_id: str) -> None:
        while not self._closing:
            snapshot = await self._guidance.context_snapshot(participant_id)
            def valid():
                current = self._guidance.store.find(participant_id)
                return (
                    not self._closing and current is not None
                    and current.run_id == snapshot.get("run_id")
                    and current.revision == snapshot.get("revision")
                )
            if self._guidance.store.find(participant_id) is None:
                return
            if await self._run_contexts.refresh(participant_id, snapshot, valid=valid):
                return


    def _context_blocks(self, participant_id: str):
        return (*self._context.blocks(participant_id), *self._run_contexts.history_blocks(participant_id))


    def _current_context_blocks(self, participant_id: str):
        return self._run_contexts.current_blocks(participant_id)


    async def stop(self) -> None:
        """Close SOP resources after the caller has stopped the SDK router."""
        self._closing = True
        await self._guidance.stop()
        await self._run_contexts.close()
        await self._context.close()


__all__ = ["SOPAgent"]
