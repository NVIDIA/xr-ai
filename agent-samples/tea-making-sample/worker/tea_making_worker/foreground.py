# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic foreground routing with one participant-scoped tool loop."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path

import nemo_relay
from loguru import logger
from xr_ai_hub import FrameUnavailable
from xr_ai_models import ChatMessage, ChatResponse, LLMService, ToolDef, VLMService
from xr_ai_runtime import Agent, RuntimeClosedError, RuntimeContext, subscribe
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.current_frame import CurrentFrameRequest
from xr_ai_tools.rag import RAGTools
from xr_ai_tools.tool_calling import ToolLoopIterationLimitError, run_tool_loop
from xr_ai_tools.vision import (
    ImageQueryRequest,
    ImageQueryResult,
    StreamingImageQueryTool,
)
from xr_ai_voice import (
    VOICE_CONTRIBUTION_TOPIC,
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
    PARTICIPANT_CLEANUP_COMPLETE_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    USER_QUERY_TOPIC,
    ForegroundRecord,
    ParticipantCleanupComplete,
)
from .images import ParticipantImageAgent
from .transcript import TranscriptAgent
from .video_log import VideoLogAgent
from .workflow import GuidanceAgent
from .workflow_tools import CurrentViewRequest, rag_lookup_tool

_MAX_TOOL_ROUNDS = 4
_PROMPTS = Path(__file__).resolve().parent / "prompts"
_IDLE_PROMPT = (_PROMPTS / "foreground_idle.txt").read_text(encoding="utf-8").strip()
_ACTIVE_PROMPT = (_PROMPTS / "foreground_active.txt").read_text(encoding="utf-8").strip()


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
        vlm_timeout_s: float = 15.0,
    ) -> None:
        if vlm_timeout_s <= 0:
            raise ValueError("vlm_timeout_s must be positive")
        self._vision = StreamingImageQueryTool(images=images.images, vlm=vlm)
        super().__init__((self._vision,))
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
        await ctx.publish(
            PARTICIPANT_CLEANUP_COMPLETE_TOPIC,
            ParticipantCleanupComplete(
                generation=ctx.metadata.message_id,
                producer="foreground",
            ),
        )

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
                response, tools, spoken = await self._answer(
                    query.text,
                    participant_id,
                    ctx,
                    timestamp_us=query.timestamp_us,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.opt(exception=True).error(
                    "tea foreground query failed pid={!r}", participant_id
                )
                response = "I couldn't complete that request. Please try again."
                tools = []
                spoken = False
            try:
                if not spoken:
                    await ctx.publish(
                        VOICE_CONTRIBUTION_TOPIC,
                        VoiceOutput(
                            text=response,
                            interrupt=True,
                            timestamp_us=query.timestamp_us,
                        ),
                    )
            except RuntimeClosedError:
                return
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
            except RuntimeClosedError:
                return
            except Exception:
                logger.opt(exception=True).error(
                    "tea foreground record failed pid={!r}", participant_id
                )

    async def _answer(
        self,
        query: str,
        participant_id: str,
        ctx: RuntimeContext | None = None,
        *,
        timestamp_us: int | None = None,
    ) -> tuple[str, list[str], bool]:
        system_prompt, tools, route = self._prepare_route(
            participant_id,
            ctx=ctx,
            timestamp_us=timestamp_us,
        )

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
                False,
            )
        fallback = "Done." if result.return_direct else "I don't have an answer."
        calls = [record.call.name for record in result.tool_calls]
        return (
            result.content.strip() or fallback,
            calls,
            result.return_direct
            and bool(result.tool_calls)
            and result.tool_calls[-1].call.name == "current_view",
        )

    def _prepare_route(
        self,
        participant_id: str,
        *,
        ctx: RuntimeContext | None,
        timestamp_us: int | None,
    ) -> tuple[str, ToolSet, str]:
        """Return the production prompt, tools, and route for one participant."""

        active_context = self._guidance.active_context(participant_id)
        if active_context is None:
            tools = self._root_tools(
                participant_id,
                ctx=ctx,
                timestamp_us=timestamp_us,
            )
            system_prompt = self._with_route_policy(_IDLE_PROMPT)
            route = "root"
        else:
            active_tools = self._guidance.active_tools(participant_id)
            if active_tools is None:
                raise RuntimeError("active tea context has no active tool set")
            tools = _merge_tool_sets(
                active_tools,
                self._background_tools(participant_id),
            )
            system_prompt = (
                f"{self._with_route_policy(_ACTIVE_PROMPT)}"
                f"\n\nActive tea guide:\n{active_context}"
            )
            route = "tea"
        return system_prompt, tools, route

    def _with_route_policy(self, route_prompt: str) -> str:
        return f"{self._prompt}\n\n{route_prompt}"

    def _root_tools(
        self,
        participant_id: str,
        *,
        ctx: RuntimeContext | None,
        timestamp_us: int | None,
    ) -> ToolSet:
        current_view = self._current_view_tool(
            participant_id,
            ctx=ctx,
            timestamp_us=timestamp_us,
        )
        return _merge_tool_sets(
            ToolSet((current_view, rag_lookup_tool(self._rag))),
            self._guidance.root_tools(participant_id),
            self._background_tools(participant_id),
        )

    def _current_view_tool(
        self,
        participant_id: str,
        *,
        ctx: RuntimeContext | None,
        timestamp_us: int | None,
    ) -> Tool[CurrentViewRequest, ImageQueryResult]:
        async def inspect(request: CurrentViewRequest) -> ImageQueryResult:
            if ctx is None:
                raise RuntimeError("current-view streaming requires a runtime context")
            return await self._stream_current_view(
                request.question,
                participant_id,
                ctx,
                timestamp_us=timestamp_us,
            )

        return Tool(
            "current_view",
            "Inspect this participant's current camera frame to answer a question about the current scene.",
            CurrentViewRequest,
            ImageQueryResult,
            inspect,
            return_direct=True,
            render_result=lambda result: result.text,
        )

    async def _stream_current_view(
        self,
        query: str,
        participant_id: str,
        ctx: RuntimeContext,
        *,
        timestamp_us: int | None,
    ) -> ImageQueryResult:
        response_id = ctx.metadata.message_id
        first = True
        opened = False
        cancelled = False
        chunks: list[str] = []
        try:
            async with asyncio.timeout(self._vlm_timeout_s):
                try:
                    frame = await self._images.get_current_frame.execute(
                        CurrentFrameRequest(participant_id=participant_id)
                    )
                except (FrameUnavailable, RuntimeError) as exc:
                    unavailable = _frame_unavailable_message(exc)
                    if unavailable is None:
                        raise
                    await ctx.publish(
                        VOICE_CONTRIBUTION_TOPIC,
                        VoiceOutput(
                            text=unavailable,
                            response_id=response_id,
                            final=False,
                            interrupt=True,
                            timestamp_us=timestamp_us,
                        ),
                    )
                    opened = True
                    return ImageQueryResult(text=unavailable, available=False)
                stream = self._vision.stream(
                    ImageQueryRequest(image=frame.image, query=query)
                )
                try:
                    async for chunk in stream:
                        if first and not chunk.text.strip():
                            continue
                        chunks.append(chunk.text)
                        await ctx.publish(
                            VOICE_CONTRIBUTION_TOPIC,
                            VoiceOutput(
                                text=chunk.text,
                                response_id=response_id,
                                final=False,
                                interrupt=first,
                                timestamp_us=timestamp_us,
                            ),
                        )
                        first = False
                        opened = True
                finally:
                    close = getattr(stream, "aclose", None)
                    if close is not None:
                        await close()
                if not chunks:
                    unavailable = (
                        "Unable to inspect the current frame because the vision model "
                        "returned no description."
                    )
                    await ctx.publish(
                        VOICE_CONTRIBUTION_TOPIC,
                        VoiceOutput(
                            text=unavailable,
                            response_id=response_id,
                            final=False,
                            interrupt=True,
                            timestamp_us=timestamp_us,
                        ),
                    )
                    opened = True
                    return ImageQueryResult(text=unavailable, available=False)
            return ImageQueryResult(text="".join(chunks))
        except TimeoutError:
            unavailable = (
                "Unable to inspect the current frame before the vision timeout."
            )
            await ctx.publish(
                VOICE_CONTRIBUTION_TOPIC,
                VoiceOutput(
                    text=unavailable,
                    response_id=response_id,
                    final=False,
                    interrupt=first,
                    timestamp_us=timestamp_us,
                ),
            )
            opened = True
            return ImageQueryResult(text=unavailable, available=False)
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            if opened and not cancelled:
                with suppress(RuntimeClosedError):
                    await ctx.publish(
                        VOICE_CONTRIBUTION_TOPIC,
                        VoiceOutput(
                            response_id=response_id,
                            timestamp_us=timestamp_us,
                        ),
                    )

    def _background_tools(self, participant_id: str) -> ToolSet:
        return _merge_tool_sets(
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


def _frame_unavailable_message(error: BaseException) -> str | None:
    """Recover a camera error from native or Relay-scrubbed exceptions."""

    relay_prefix = "internal error: FrameUnavailable:"
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, FrameUnavailable):
            return str(current)
        if isinstance(current, RuntimeError):
            message = str(current)
            if message.startswith(relay_prefix):
                return message.removeprefix(relay_prefix).strip()
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


def _merge_tool_sets(*catalogs: ToolSet) -> ToolSet:
    tools: dict[str, Tool] = {}
    for catalog in catalogs:
        for name, tool in catalog.items():
            if name in tools:
                raise ValueError(f"duplicate participant tool: {name}")
            tools[name] = tool
    return ToolSet(tools)


__all__ = ["ForegroundAgent"]
