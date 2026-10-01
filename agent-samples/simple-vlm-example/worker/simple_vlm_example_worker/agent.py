# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-scoped streaming orchestration for the simple VLM sample."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import suppress

import nemo_relay
from xr_ai_models import LLMService
from xr_ai_runtime import (
    Agent,
    RuntimeClosedError,
    RuntimeContext,
    Topic,
    subscribe,
)
from xr_ai_sample_agents import ConversationExchange, QuickConversation
from xr_ai_tools.current_frame import CurrentFrameTool
from xr_ai_tools.vision import StreamingImageQueryTool
from xr_ai_voice import (
    VOICE_OUTPUT_TOPIC,
    UserQuery,
    VoiceInterrupted,
    VoiceOutput,
    VoiceParticipantLeft,
)

USER_QUERY_TOPIC = Topic("simple-vlm.user-query", UserQuery)
PARTICIPANT_LEFT_TOPIC = Topic(
    "simple-vlm.participant-left",
    VoiceParticipantLeft,
)
INTERRUPTED_TOPIC = Topic("simple-vlm.interrupted", VoiceInterrupted)


class SimpleVlmAgent(Agent):
    """Own streamed conversation, recent history, and participant cancellation."""

    def __init__(
        self,
        vision_factory: Callable[
            [],
            tuple[CurrentFrameTool, StreamingImageQueryTool],
        ],
        set_status: Callable[[str, str], Awaitable[None]],
        *,
        llm: LLMService | None = None,
    ) -> None:
        super().__init__()
        self._vision_factory = vision_factory
        self._set_status = set_status
        self._llm = llm
        self._frames: CurrentFrameTool | None = None
        self._conversation: QuickConversation | None = None
        self._history: dict[str, deque[ConversationExchange]] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}

    @subscribe(USER_QUERY_TOPIC)
    async def answer_user(self, request: UserQuery, ctx: RuntimeContext) -> None:
        """Supersede and start one participant's streamed response."""

        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("simple VLM queries require a participant")
        await self._cancel(participant_id)
        task = asyncio.create_task(
            self._stream(request, ctx),
            name=f"simple-vlm-query:{participant_id}",
            context=nemo_relay.fork_asyncio_context(),
        )
        self._tasks[participant_id] = task
        task.add_done_callback(lambda completed, pid=participant_id: self._discard(pid, completed))

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self,
        _event: VoiceParticipantLeft,
        ctx: RuntimeContext,
    ) -> None:
        """Release this agent's work and frame state for a departed participant."""

        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("participant-left events require a participant")
        await self._cancel(participant_id)
        self._history.pop(participant_id, None)
        if self._frames is not None:
            self._frames.release(participant_id)

    @subscribe(INTERRUPTED_TOPIC)
    async def interrupted(
        self,
        _event: VoiceInterrupted,
        ctx: RuntimeContext,
    ) -> None:
        """Cancel participant-scoped or global vision work."""

        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            await self._cancel_all()
            return
        await self._cancel(participant_id)

    async def _stream(self, request: UserQuery, ctx: RuntimeContext) -> None:
        with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
            with nemo_relay.scope.scope(
                "simple-vlm.turn",
                nemo_relay.ScopeType.Agent,
                input=request.model_dump(mode="json"),
                metadata={
                    "agent": ctx.agent_name,
                    "message_id": ctx.metadata.message_id,
                    "correlation_id": ctx.metadata.correlation_id,
                    "participant_id": ctx.metadata.participant_id,
                },
            ):
                await self._stream_response(request, ctx)

    async def _stream_response(
        self,
        request: UserQuery,
        ctx: RuntimeContext,
    ) -> None:
        response_id = ctx.metadata.message_id
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("simple VLM queries require a participant")
        first = True
        opened = False
        cancelled = False
        processing = False
        chunks: list[str] = []
        try:
            if self._conversation is None:
                self._frames, vision = self._vision_factory()
                self._conversation = QuickConversation(
                    self._frames,
                    vision,
                    llm=self._llm,
                )
            await self._set_status("processing", participant_id)
            processing = True
            stream = self._conversation.stream(
                request.text,
                participant_id,
                history=tuple(self._history.get(participant_id, ())),
            )
            try:
                async for chunk in stream:
                    chunks.append(chunk)
                    await ctx.publish(
                        VOICE_OUTPUT_TOPIC,
                        VoiceOutput(
                            text=chunk,
                            response_id=response_id,
                            final=False,
                            interrupt=first,
                            timestamp_us=request.timestamp_us,
                        ),
                    )
                    first = False
                    opened = True
            finally:
                close = getattr(stream, "aclose", None)
                if close is not None:
                    await close()
            if answer := "".join(chunks).strip():
                self._history.setdefault(
                    participant_id,
                    deque(maxlen=4),
                ).append(ConversationExchange(user=request.text[:240], assistant=answer[:240]))
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            if processing:
                await self._set_status("idle", participant_id)
            if opened and not cancelled:
                with suppress(RuntimeClosedError):
                    await ctx.publish(
                        VOICE_OUTPUT_TOPIC,
                        VoiceOutput(
                            response_id=response_id,
                            timestamp_us=request.timestamp_us,
                        ),
                    )

    async def _cancel(self, participant_id: str) -> None:
        task = self._tasks.pop(participant_id, None)
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _cancel_all(self) -> None:
        tasks = tuple(self._tasks.values())
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self) -> None:
        """Cancel all image-query turns owned by this agent."""

        await self._cancel_all()

    def _discard(self, participant_id: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(participant_id) is task:
            self._tasks.pop(participant_id, None)


__all__ = [
    "INTERRUPTED_TOPIC",
    "PARTICIPANT_LEFT_TOPIC",
    "SimpleVlmAgent",
    "USER_QUERY_TOPIC",
]
