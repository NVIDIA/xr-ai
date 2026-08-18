# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic foreground routing with one participant-scoped tool loop."""

from __future__ import annotations

import asyncio
from contextlib import suppress

import nemo_relay
from loguru import logger
from xr_ai_models import ChatMessage, ChatResponse, LLMService, ToolDef, VLMService
from xr_ai_runtime import Agent, RuntimeClosedError, RuntimeContext, subscribe
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.rag import RAGTools
from xr_ai_tools.tool_calling import ToolLoopIterationLimitError, run_tool_loop
from xr_ai_tools.vision import ImageQueryTool
from xr_ai_voice import (
    VOICE_OUTPUT_TOPIC,
    UserQuery,
    VoiceInterrupted,
    VoiceOutput,
    VoiceParticipantLeft,
)

from .background_context import BackgroundContextAgent
from .change_watch import ChangeWatchAgent
from .events import (
    FOREGROUND_RECORD_TOPIC,
    INTERRUPTED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    USER_QUERY_TOPIC,
    ForegroundRecord,
)
from .images import ParticipantImageAgent
from .transcript import TranscriptAgent
from .video_log import VideoLogAgent
from .workflow import GuidanceAgent
from .workflow_tools import participant_current_view_tool, rag_lookup_tool

_MAX_TOOL_ROUNDS = 4


class ForegroundAgent(Agent):
    """Select the idle root or active tea tool set before invoking the model."""

    def __init__(
        self,
        *,
        llm: LLMService,
        images: ParticipantImageAgent,
        vlm: VLMService,
        rag: RAGTools,
        guidance: GuidanceAgent,
        background_context: BackgroundContextAgent,
        change_watch: ChangeWatchAgent,
        transcript: TranscriptAgent,
        video_log: VideoLogAgent,
        prompt: str,
        vlm_timeout_s: float,
    ) -> None:
        self.query_image = ImageQueryTool(images=images.images, vlm=vlm)
        super().__init__((self.query_image,))
        self._llm = llm
        self._images = images
        self._rag = rag
        self._guidance = guidance
        self._background_context = background_context
        self._change_watch = change_watch
        self._transcript = transcript
        self._video_log = video_log
        self._prompt = prompt.strip()
        self._vlm_timeout_s = vlm_timeout_s
        self._tasks: dict[str, asyncio.Task[None]] = {}

    @subscribe(USER_QUERY_TOPIC)
    async def answer(self, query: UserQuery, ctx: RuntimeContext) -> None:
        participant_id = self._participant(ctx)
        await self._cancel(participant_id)
        task = asyncio.create_task(
            self._run_turn(query, ctx),
            name=f"tea-foreground:{participant_id}",
            context=nemo_relay.fork_asyncio_context(),
        )
        self._tasks[participant_id] = task
        task.add_done_callback(
            lambda completed, pid=participant_id: self._discard(pid, completed)
        )

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self,
        _event: VoiceParticipantLeft,
        ctx: RuntimeContext,
    ) -> None:
        await self._cancel(self._participant(ctx))

    @subscribe(INTERRUPTED_TOPIC)
    async def interrupted(
        self,
        _event: VoiceInterrupted,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            await self.stop()
        else:
            await self._cancel(participant_id)

    async def stop(self) -> None:
        """Cancel every participant turn owned by this agent."""

        tasks = tuple(self._tasks.values())
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_turn(self, query: UserQuery, ctx: RuntimeContext) -> None:
        participant_id = self._participant(ctx)
        with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
            try:
                response, tools = await self._answer(query.text, participant_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.opt(exception=True).error(
                    "tea foreground query failed pid={!r}", participant_id
                )
                response = "I couldn't complete that request. Please try again."
                tools = []
            try:
                await ctx.publish(
                    FOREGROUND_RECORD_TOPIC,
                    ForegroundRecord(
                        timestamp_us=query.timestamp_us,
                        query=query.text,
                        response=response,
                        tools=tools,
                    ),
                )
                await ctx.publish(
                    VOICE_OUTPUT_TOPIC,
                    VoiceOutput(
                        text=response,
                        interrupt=True,
                        timestamp_us=query.timestamp_us,
                    ),
                )
            except RuntimeClosedError:
                return

    async def _answer(self, query: str, participant_id: str) -> tuple[str, list[str]]:
        active_context = self._guidance.active_context(participant_id)
        if active_context is None:
            tools = self._root_tools(participant_id)
            system_prompt = self._prompt
            route = "root"
        else:
            active_tools = self._guidance.active_tools(participant_id)
            if active_tools is None:
                raise RuntimeError("active tea context has no active tool set")
            tools = _merge_tool_sets(
                active_tools,
                self._background_context.participant_tools(participant_id),
            )
            system_prompt = f"{self._prompt}\n\nActive tea guide:\n{active_context}"
            route = "tea"

        round_index = 0

        async def call_model(
            messages: tuple[ChatMessage, ...],
            definitions: tuple[ToolDef, ...],
        ) -> ChatResponse:
            nonlocal round_index
            round_index += 1
            response = await self._llm.chat(
                messages,
                tools=definitions,
                max_tokens=512,
                temperature=0.0,
                enable_thinking=False,
            )
            logger.info(
                "tea foreground route pid={!r} route={} round={} tools={}",
                participant_id,
                route,
                round_index,
                [call.name for call in response.tool_calls or ()],
            )
            return response

        try:
            result = await run_tool_loop(
                (
                    ChatMessage(role="system", content=system_prompt),
                    ChatMessage(role="user", content=query),
                ),
                tools,
                call_model,
                max_iterations=_MAX_TOOL_ROUNDS,
            )
        except ToolLoopIterationLimitError as exc:
            return (
                "I couldn't finish that request within the tool limit.",
                [record.call.name for record in exc.tool_calls],
            )
        fallback = "Done." if result.return_direct else "I don't have an answer."
        return (
            result.content.strip() or fallback,
            [record.call.name for record in result.tool_calls],
        )

    def _root_tools(self, participant_id: str) -> ToolSet:
        current_view = participant_current_view_tool(
            participant_id,
            self._images.get_current_frame,
            self.query_image,
            timeout_s=self._vlm_timeout_s,
        )
        return _merge_tool_sets(
            ToolSet((current_view, rag_lookup_tool(self._rag))),
            self._guidance.root_tools(participant_id),
            self._background_context.participant_tools(participant_id),
            self._change_watch.participant_tools(participant_id),
            self._transcript.participant_tools(participant_id),
            self._video_log.participant_tools(participant_id),
        )

    async def _cancel(self, participant_id: str) -> None:
        task = self._tasks.pop(participant_id, None)
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def _discard(self, participant_id: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(participant_id) is task:
            self._tasks.pop(participant_id, None)
        if task.cancelled():
            return
        with suppress(asyncio.CancelledError):
            if error := task.exception():
                logger.error(
                    "tea foreground stopped pid={!r}: {!r}", participant_id, error
                )

    @staticmethod
    def _participant(ctx: RuntimeContext) -> str:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("foreground queries require a participant")
        return participant_id


def _merge_tool_sets(*catalogs: ToolSet) -> ToolSet:
    tools: dict[str, Tool] = {}
    for catalog in catalogs:
        for name, tool in catalog.items():
            if name in tools:
                raise ValueError(f"duplicate participant tool: {name}")
            tools[name] = tool
    return ToolSet(tools)


__all__ = ["ForegroundAgent"]
