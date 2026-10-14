# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-scoped streaming orchestration for the simple VLM sample."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path

import nemo_relay
from loguru import logger
from xr_ai_conversation import Conversation, ConversationFrontEnd, ConversationTurn
from xr_ai_runtime import (
    Agent,
    RuntimeClosedError,
    RuntimeContext,
    Topic,
    subscribe,
)
from xr_ai_tools.current_frame import CurrentFrameTool
from xr_ai_voice import (
    VOICE_OUTPUT_TOPIC,
    UserQuery,
    VoiceInterrupted,
    VoiceOutput,
    VoiceParticipantJoined,
    VoiceParticipantLeft,
)

USER_QUERY_TOPIC = Topic("simple-vlm.user-query", UserQuery)
PARTICIPANT_LEFT_TOPIC = Topic(
    "simple-vlm.participant-left",
    VoiceParticipantLeft,
)
INTERRUPTED_TOPIC = Topic("simple-vlm.interrupted", VoiceInterrupted)
PARTICIPANT_JOINED_TOPIC = Topic("simple-vlm.participant-joined", VoiceParticipantJoined)
_PROMPTS = Path(__file__).parent / "prompts"
_CONVERSATION_INSTRUCTIONS = (_PROMPTS / "conversation.txt").read_text(encoding="utf-8").strip()
_VISION_INSTRUCTIONS = (_PROMPTS / "vision.txt").read_text(encoding="utf-8").strip()
_ENVIRONMENT_INSTRUCTIONS = (_PROMPTS / "environment_reasoning.txt").read_text(encoding="utf-8").strip()


class SimpleVlmAgent(Agent):
    """Own streamed user turns and cancellation around a vision tool."""

    def __init__(
        self,
        set_status: Callable[[str, str], Awaitable[None]],
        *,
        front_end: ConversationFrontEnd,
        conversations: tuple[Conversation, ...],
        frames: CurrentFrameTool | None = None,
    ) -> None:
        super().__init__()
        self._set_status = set_status
        self._front_end = front_end
        self._conversations = conversations
        self._frames = frames
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._participants: set[str] = set()

    @subscribe(USER_QUERY_TOPIC)
    async def answer_user(self, request: UserQuery, ctx: RuntimeContext) -> None:
        """Supersede and start one participant's streamed response."""

        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("simple VLM queries require a participant")
        self._participants.add(participant_id)
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
        await self._front_end.participant_left(participant_id)
        for conversation in self._conversations:
            await conversation.participant_left(participant_id)
        self._participants.discard(participant_id)
        if self._frames is not None:
            self._frames.release(participant_id)

    @subscribe(PARTICIPANT_JOINED_TOPIC)
    async def participant_joined(
        self,
        _event: VoiceParticipantJoined,
        ctx: RuntimeContext,
    ) -> None:
        """Open participant-scoped context when the voice session joins."""

        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("participant-joined events require a participant")
        for conversation in self._conversations:
            conversation.activate_participant(participant_id)
        self._participants.add(participant_id)

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
        response_id = ctx.metadata.message_id or ctx.metadata.correlation_id
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("simple VLM queries require a participant")
        first = True
        opened = False
        cancelled = False
        processing = False
        output_failure: Exception | None = None
        try:
            await self._set_status("processing", participant_id)
            processing = True
            fragments: list[str] = []

            async def emit(source: str, text: str) -> None:
                nonlocal first, opened, output_failure
                fragments.append(text)
                try:
                    with nemo_relay.scope.scope(
                        "simple-vlm.destination-output",
                        nemo_relay.ScopeType.Agent,
                        metadata={"destination": source},
                    ):
                        await ctx.publish(
                            VOICE_OUTPUT_TOPIC,
                            VoiceOutput(
                                text=text,
                                response_id=response_id,
                                final=False,
                                interrupt=first,
                                timestamp_us=request.timestamp_us,
                            ),
                        )
                except Exception as error:
                    output_failure = error
                    raise
                first = False
                opened = True

            try:
                await self._front_end.handle(
                    ConversationTurn(
                        participant_id=participant_id,
                        turn_id=response_id,
                        query=request.text,
                    ),
                    emit=emit,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                if output_failure is not None:
                    raise output_failure
                logger.exception("conversation routing or generation failed participant={!r}", participant_id)
                if not fragments:
                    with suppress(RuntimeClosedError):
                        await ctx.publish(
                            VOICE_OUTPUT_TOPIC,
                            VoiceOutput(
                                text="I couldn't complete that request. Please try again.",
                                timestamp_us=request.timestamp_us,
                            ),
                        )
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
        await self._front_end.interrupt(participant_id)
        task = self._tasks.pop(participant_id, None)
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _cancel_all(self) -> None:
        await self._front_end.interrupt()
        tasks = tuple(self._tasks.values())
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self) -> None:
        """Cancel turns and release participant history owned by this agent."""

        await self._cancel_all()
        await self._front_end.stop()
        participants = tuple(self._participants)
        self._participants.clear()
        for conversation in self._conversations:
            await asyncio.gather(*(conversation.participant_left(pid) for pid in participants))
        if self._frames is not None:
            for participant_id in participants:
                self._frames.release(participant_id)

    def _discard(self, participant_id: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(participant_id) is task:
            self._tasks.pop(participant_id, None)
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("simple VLM turn failed pid={!r}: {!r}", participant_id, error)


__all__ = [
    "INTERRUPTED_TOPIC",
    "PARTICIPANT_LEFT_TOPIC",
    "SimpleVlmAgent",
    "USER_QUERY_TOPIC",
]
