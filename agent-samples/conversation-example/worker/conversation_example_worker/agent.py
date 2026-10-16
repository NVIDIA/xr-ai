# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt runtime voice queries and destination output to voice topics."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress

import nemo_relay
from loguru import logger
from xr_ai_conversation import Conversation
from xr_ai_routing import ConversationFrontEnd, ConversationTurn
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

USER_QUERY_TOPIC = Topic("conversation-example.user-query", UserQuery)
PARTICIPANT_LEFT_TOPIC = Topic(
    "conversation-example.participant-left",
    VoiceParticipantLeft,
)
INTERRUPTED_TOPIC = Topic("conversation-example.interrupted", VoiceInterrupted)
PARTICIPANT_JOINED_TOPIC = Topic("conversation-example.participant-joined", VoiceParticipantJoined)


class ConversationExampleAgent(Agent):
    """Bridge voice runtime topics to the shared conversation front end.

    This sample adapter owns its streaming response tasks. The injected
    ``ConversationFrontEnd`` owns route selection, destination work, and focus;
    this class converts accepted voice queries to turns and attributed output
    back into ``VoiceOutput`` messages.
    """

    def __init__(
        self,
        set_status: Callable[[str, str], Awaitable[None]],
        *,
        front_end: ConversationFrontEnd,
        conversations: tuple[Conversation, ...],
        frames: CurrentFrameTool | None = None,
        participant_left_handlers: tuple[Callable[[str], Awaitable[None]], ...] = (),
    ) -> None:
        super().__init__()
        self._set_status = set_status
        self._front_end = front_end
        self._conversations = conversations
        self._participant_left_handlers = participant_left_handlers
        self._frames = frames
        # The adapter owns response tasks; the front end owns its route tasks.
        self._response_tasks: dict[str, asyncio.Task[None]] = {}
        self._participants: set[str] = set()

    @subscribe(USER_QUERY_TOPIC)
    async def answer_user(self, request: UserQuery, ctx: RuntimeContext) -> None:
        """Supersede and start one participant's streamed response."""

        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("conversation example queries require a participant")
        self._participants.add(participant_id)
        await self._cancel_response(participant_id)
        # Runtime publication awaits subscribers; detach model streaming so a
        # query delivery can finish while this response remains active.
        task = asyncio.create_task(
            self._stream(request, ctx),
            name=f"conversation-example-query:{participant_id}",
            context=nemo_relay.fork_asyncio_context(),
        )
        self._response_tasks[participant_id] = task
        task.add_done_callback(lambda completed, pid=participant_id: self._discard_response(pid, completed))

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self,
        _event: VoiceParticipantLeft,
        ctx: RuntimeContext,
    ) -> None:
        """Release participant state owned across the sample adapter.

        The front end cleans destinations it used for turns. The explicit
        handlers and conversation tuple also clean join-activated resources for
        participants who have not used a route. Cleanup can therefore reach a
        used destination twice; its departure hooks are idempotent.
        """

        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("participant-left events require a participant")
        await self._cancel_response(participant_id)
        # Departure still releases every owner when an earlier hook fails.
        errors: list[Exception] = []
        try:
            await self._front_end.participant_left(participant_id)
        except Exception as error:
            errors.append(error)
        for handler in self._participant_left_handlers:
            try:
                await handler(participant_id)
            except Exception as error:
                errors.append(error)
        for conversation in self._conversations:
            try:
                await conversation.participant_left(participant_id)
            except Exception as error:
                errors.append(error)
        self._participants.discard(participant_id)
        if self._frames is not None:
            try:
                self._frames.release(participant_id)
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup(f"failed to clean up participant {participant_id!r}", errors)

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
        """Cancel participant-scoped or global response work."""

        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            await self._cancel_responses()
            return
        await self._cancel_response(participant_id)

    async def _stream(self, request: UserQuery, ctx: RuntimeContext) -> None:
        response_id = ctx.metadata.message_id or ctx.metadata.correlation_id
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("conversation example queries require a participant")

        with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
            with nemo_relay.scope.scope(
                "conversation-example.turn",
                nemo_relay.ScopeType.Agent,
                input=request.model_dump(mode="json"),
                metadata={
                    "agent": ctx.agent_name,
                    "message_id": ctx.metadata.message_id,
                    "correlation_id": ctx.metadata.correlation_id,
                    "participant_id": ctx.metadata.participant_id,
                },
            ):
                stream_opened = False
                cancelled = False
                output_failure: Exception | None = None

                async def emit(source: str, text: str) -> None:
                    nonlocal stream_opened, output_failure
                    try:
                        with nemo_relay.scope.scope(
                            "conversation-example.destination-output",
                            nemo_relay.ScopeType.Agent,
                            metadata={"destination": source},
                        ):
                            await ctx.publish(
                                VOICE_OUTPUT_TOPIC,
                                VoiceOutput(
                                    text=text,
                                    response_id=response_id,
                                    final=False,
                                    interrupt=not stream_opened,
                                    timestamp_us=request.timestamp_us,
                                ),
                            )
                    except Exception as error:
                        output_failure = error
                        raise
                    # A response starts only after its first chunk is accepted.
                    stream_opened = True

                await self._set_status("processing", participant_id)
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
                    cancelled = True
                    raise
                except Exception:
                    if output_failure is not None:
                        raise output_failure
                    logger.exception(
                        "conversation routing or generation failed participant={!r}",
                        participant_id,
                    )
                    if not stream_opened:
                        with suppress(RuntimeClosedError):
                            await ctx.publish(
                                VOICE_OUTPUT_TOPIC,
                                VoiceOutput(
                                    text="I couldn't complete that request. Please try again.",
                                    timestamp_us=request.timestamp_us,
                                ),
                            )
                finally:
                    await self._set_status("idle", participant_id)
                    if stream_opened and not cancelled:
                        with suppress(RuntimeClosedError):
                            await ctx.publish(
                                VOICE_OUTPUT_TOPIC,
                                VoiceOutput(
                                    response_id=response_id,
                                    timestamp_us=request.timestamp_us,
                                ),
                            )

    async def _cancel_response(self, participant_id: str) -> None:
        await self._front_end.interrupt(participant_id)
        task = self._response_tasks.pop(participant_id, None)
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _cancel_responses(self) -> None:
        await self._front_end.interrupt()
        tasks = tuple(self._response_tasks.values())
        self._response_tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self) -> None:
        """Cancel turns and release participant history owned by this agent."""

        await self._cancel_responses()
        await self._front_end.stop()
        participants = tuple(self._participants)
        self._participants.clear()
        # Shutdown cleans each resource across participants concurrently and
        # stops at the first failing cleanup stage.
        for handler in self._participant_left_handlers:
            await asyncio.gather(*(handler(pid) for pid in participants))
        for conversation in self._conversations:
            await asyncio.gather(*(conversation.participant_left(pid) for pid in participants))
        if self._frames is not None:
            for participant_id in participants:
                self._frames.release(participant_id)

    def _discard_response(self, participant_id: str, task: asyncio.Task[None]) -> None:
        if self._response_tasks.get(participant_id) is task:
            self._response_tasks.pop(participant_id, None)
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("conversation example turn failed pid={!r}: {!r}", participant_id, error)


__all__ = [
    "INTERRUPTED_TOPIC",
    "PARTICIPANT_LEFT_TOPIC",
    "ConversationExampleAgent",
    "USER_QUERY_TOPIC",
]
