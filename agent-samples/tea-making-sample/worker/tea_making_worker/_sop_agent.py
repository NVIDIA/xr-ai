# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One sample-local agent owns SOP execution, conversation context and responses."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AsyncExitStack

from xr_ai_conversation import ContextManager
from xr_ai_interaction import ParticipantJoined, ParticipantLeft, UserQuery
from xr_ai_models import DecisionService, LLMService, VLMService
from xr_ai_runtime import AgentRuntime, RuntimeContext, subscribe
from xr_ai_tools import ToolSet
from xr_ai_tools.rag import RAGTools
from xr_ai_tools.vision import ImageQueryTool

from ._conversation_context import _CompactionLanguage
from ._run_context import _HANDOFF, _RunContexts
from .events import (
    GUIDANCE_NOTICE_TOPIC,
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    USER_QUERY_TOPIC,
    GuidanceNotice,
)
from .foreground import ForegroundAgent
from .guidance_voice import GuidanceVoiceAgent
from .images import ParticipantImageAgent
from .spec import Workflow
from .workflow import GuidanceAgent
from .workflow_state import WorkflowEvent, WorkflowSession

_CONTEXT_EVENTS = {
    "step.enter", "step.commit", "step.ready", "workflow.reset",
    "workflow.complete", "workflow.skipped", "workflow.run_closed",
}


class _SOPAgent(ForegroundAgent):
    """Own one focused SOP and its independent execution and context lifecycles.

    Components remain sample-local. Register only this agent, bind its runtime,
    and stop it before closing model services. Context is historical reference;
    the execution engine's snapshot and live tools remain authoritative.
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
        context_budget_bytes: int = 12000,
        compaction_input_budget_bytes: int = 12000,
        vlm_timeout_s: float = 15,
    ) -> None:
        context = ContextManager(
            _CompactionLanguage(llm),
            incremental_context_topics=(_HANDOFF,),
            dialogue_budget_bytes=context_budget_bytes // 2,
            topic_budget_bytes=max(1, context_budget_bytes // 8),
            compaction_input_budget_bytes=compaction_input_budget_bytes,
        )
        engine = GuidanceAgent(
            workflow=workflow, llm=llm, decisions=decisions,
            current_frame=images.get_current_frame, image_query=image_query, rag=rag,
            vlm_timeout_s=vlm_timeout_s, record_event=self._remember_event,
        )
        super().__init__(
            llm=llm, decisions=decisions, images=images, vlm=vlm, rag=rag,
            guidance=engine, application_tools=application_tools, prompt=prompt,
            vlm_timeout_s=vlm_timeout_s, context=context,
        )
        self._run_contexts = _RunContexts(
            llm, context, budget=max(1, context_budget_bytes // 8),
            compaction_budget=compaction_input_budget_bytes,
        )
        self._notice_voice = GuidanceVoiceAgent()
        self._closing = False
        self._lifecycle_locks: dict[str, asyncio.Lock] = {}

    def bind_runtime(self, runtime: AgentRuntime) -> None:
        self._guidance.bind_runtime(runtime)

    @subscribe(USER_QUERY_TOPIC)
    async def answer(self, query: UserQuery, ctx: RuntimeContext) -> None:
        if not self._closing and self._participant(ctx) in self._guidance._connected:
            await super().answer(query, ctx)

    @subscribe(PARTICIPANT_JOINED_TOPIC)
    async def participant_joined(self, event: ParticipantJoined, ctx: RuntimeContext) -> None:
        if self._closing:
            return
        participant = self._participant(ctx)
        async with self._lifecycle_locks.setdefault(participant, asyncio.Lock()):
            if not self._closing:
                await self._guidance.participant_joined(event, ctx)

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(self, event: ParticipantLeft, ctx: RuntimeContext) -> None:
        if self._closing:
            return
        participant = self._participant(ctx)
        async with self._lifecycle_locks.setdefault(participant, asyncio.Lock()):
            if self._closing:
                return
            # Close admission before draining either producer. A reconnect waits
            # for old cleanup before creating a new execution/context lifetime.
            self._guidance._connected.discard(participant)
            try:
                await super().participant_left(event, ctx)
            finally:
                try:
                    await self._guidance.participant_left(event, ctx)
                finally:
                    await self._run_contexts.clear(participant)
                    await self._context.clear(participant)

    @subscribe(GUIDANCE_NOTICE_TOPIC)
    async def notify(self, notice: GuidanceNotice, ctx: RuntimeContext) -> None:
        if not self._closing and self._participant(ctx) in self._guidance._connected:
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
        self._closing = True
        # Finish lifecycle callbacks before draining producers: an in-flight join
        # must not launch a new monitor after the engine has stopped.
        async with AsyncExitStack() as lifecycle:
            for lock in tuple(self._lifecycle_locks.values()):
                await lifecycle.enter_async_context(lock)
            try:
                results = await asyncio.gather(super().stop(), self._guidance.stop(), return_exceptions=True)
            finally:
                await self._run_contexts.close()
                await self._context.close()
                self._lifecycle_locks.clear()
        for result in results:
            if isinstance(result, BaseException):
                raise result
