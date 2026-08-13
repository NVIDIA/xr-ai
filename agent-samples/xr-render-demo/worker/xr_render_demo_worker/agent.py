# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed topics and the resident xr-render runtime agent."""

from __future__ import annotations

import asyncio
from builtins import ExceptionGroup
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass

import nemo_relay
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from xr_ai_runtime import (
    Agent,
    RuntimeClosedError,
    RuntimeContext,
    Topic,
    subscribe,
)
from xr_ai_voice import (
    VOICE_OUTPUT_TOPIC,
    UserQuery,
    VoiceInterrupted,
    VoiceOutput,
    VoiceParticipantLeft,
    VoiceStreamClosedError,
)

from .scene_loop import SceneModelLoop


class RenderNotice(BaseModel):
    """One agent-authored lifecycle notice."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    interrupt_output: bool = False


USER_QUERY_TOPIC = Topic("xr-render.user-query", UserQuery)
RENDER_NOTICE_TOPIC = Topic("xr-render.notice", RenderNotice)
PARTICIPANT_LEFT_TOPIC = Topic("xr-render.participant-left", VoiceParticipantLeft)
INTERRUPTED_TOPIC = Topic("xr-render.interrupted", VoiceInterrupted)


@dataclass(slots=True)
class _TurnControl:
    finish_stream: bool = True


@dataclass(slots=True)
class _Turn:
    task: asyncio.Task[None]
    control: _TurnControl


def _expected_stream_close(error: Exception) -> bool:
    if isinstance(error, (RuntimeClosedError, VoiceStreamClosedError)):
        return True
    if isinstance(error, ExceptionGroup):
        return all(_expected_stream_close(child) for child in error.exceptions)
    return False


class RenderAgent(Agent):
    """Run participant-scoped render turns and publish their spoken chunks."""

    def __init__(self, scene: SceneModelLoop) -> None:
        super().__init__()
        self.scene = scene
        self._tasks: dict[str, _Turn] = {}
        self._stopped = False

    @subscribe(USER_QUERY_TOPIC)
    async def answer_user(self, query: UserQuery, ctx: RuntimeContext) -> None:
        """Start a render turn from a user-facing input path."""

        await self._start_turn(
            query.text,
            ctx,
            is_notice=False,
            interrupt_output=False,
            timestamp_us=query.timestamp_us,
        )

    @subscribe(RENDER_NOTICE_TOPIC)
    async def answer_notice(self, notice: RenderNotice, ctx: RuntimeContext) -> None:
        """Start a render turn from an application-authored notice."""

        await self._start_turn(
            notice.text,
            ctx,
            is_notice=True,
            interrupt_output=notice.interrupt_output,
            timestamp_us=None,
        )

    async def _start_turn(
        self,
        text: str,
        ctx: RuntimeContext,
        *,
        is_notice: bool,
        interrupt_output: bool,
        timestamp_us: int | None,
    ) -> None:
        """Supersede and start one participant's render turn."""

        if self._stopped:
            return
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("render turns require a participant")
        await self._cancel(participant_id, finish_stream=True)
        control = _TurnControl()
        task = asyncio.create_task(
            self._run_turn(
                text,
                ctx,
                control,
                is_notice=is_notice,
                interrupt_output=interrupt_output,
                timestamp_us=timestamp_us,
            ),
            name=f"xr-render:{participant_id}",
            context=nemo_relay.fork_asyncio_context(),
        )
        self._tasks[participant_id] = _Turn(task=task, control=control)
        task.add_done_callback(lambda completed, pid=participant_id: self._discard(pid, completed))

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self,
        _event: VoiceParticipantLeft,
        ctx: RuntimeContext,
    ) -> None:
        """Cancel work and release state for a departed participant."""

        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("participant-left events require a participant")
        await self._cancel(participant_id, finish_stream=False)
        await self.scene.on_participant_left(participant_id)

    @subscribe(INTERRUPTED_TOPIC)
    async def interrupted(
        self,
        _event: VoiceInterrupted,
        ctx: RuntimeContext,
    ) -> None:
        """Cancel participant-scoped or global render work."""

        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            await self._cancel_all(finish_stream=False)
            return
        await self._cancel(participant_id, finish_stream=False)

    async def stop(self) -> None:
        """Cancel active turns before runtime shutdown."""

        self._stopped = True
        await self._cancel_all(finish_stream=False)

    async def _run_turn(
        self,
        text: str,
        ctx: RuntimeContext,
        control: _TurnControl,
        *,
        is_notice: bool,
        interrupt_output: bool,
        timestamp_us: int | None,
    ) -> None:
        with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
            with nemo_relay.scope.scope(
                "xr-render.turn",
                nemo_relay.ScopeType.Agent,
                input={"text": text, "is_notice": is_notice},
                metadata={
                    "agent": ctx.agent_name,
                    "message_id": ctx.metadata.message_id,
                    "correlation_id": ctx.metadata.correlation_id,
                    "participant_id": ctx.metadata.participant_id,
                },
            ):
                await self._run_turn_scoped(
                    text,
                    ctx,
                    control,
                    is_notice=is_notice,
                    interrupt_output=interrupt_output,
                    timestamp_us=timestamp_us,
                )

    async def _run_turn_scoped(
        self,
        text: str,
        ctx: RuntimeContext,
        control: _TurnControl,
        *,
        is_notice: bool,
        interrupt_output: bool,
        timestamp_us: int | None,
    ) -> None:
        metadata = ctx.metadata
        participant_id = metadata.participant_id
        if participant_id is None:
            raise ValueError("render turns require a participant")
        response_id = metadata.message_id
        opened = False
        response: AsyncIterator[str] | None = None
        try:
            if is_notice:
                response = self.scene.handle_notice(participant_id, text)
            else:
                response = await self.scene.handle_query(
                    participant_id,
                    text,
                )
            first = True
            async for chunk in response:
                opened = True
                try:
                    await ctx.publish(
                        VOICE_OUTPUT_TOPIC,
                        VoiceOutput(
                            text=chunk,
                            response_id=response_id,
                            final=False,
                            interrupt=interrupt_output and first,
                            timestamp_us=timestamp_us,
                        ),
                    )
                except RuntimeClosedError:
                    return
                first = False
        finally:
            if response is not None:
                with suppress(Exception, asyncio.CancelledError):
                    await response.aclose()
            if opened and control.finish_stream:
                try:
                    await ctx.publish(
                        VOICE_OUTPUT_TOPIC,
                        VoiceOutput(
                            response_id=response_id,
                            timestamp_us=timestamp_us,
                        ),
                    )
                except Exception as exc:
                    if not _expected_stream_close(exc):
                        raise

    async def _cancel(self, participant_id: str, *, finish_stream: bool) -> None:
        turn = self._tasks.pop(participant_id, None)
        if turn is None:
            return
        turn.control.finish_stream = finish_stream
        turn.task.cancel()
        await asyncio.gather(turn.task, return_exceptions=True)

    async def _cancel_all(self, *, finish_stream: bool) -> None:
        turns = tuple(self._tasks.values())
        self._tasks.clear()
        for turn in turns:
            turn.control.finish_stream = finish_stream
            turn.task.cancel()
        if turns:
            await asyncio.gather(
                *(turn.task for turn in turns),
                return_exceptions=True,
            )

    def _discard(self, participant_id: str, task: asyncio.Task[None]) -> None:
        turn = self._tasks.get(participant_id)
        if turn is not None and turn.task is task:
            self._tasks.pop(participant_id, None)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error("xr-render turn failed for {}: {!r}", participant_id, error)


__all__ = [
    "RENDER_NOTICE_TOPIC",
    "USER_QUERY_TOPIC",
    "INTERRUPTED_TOPIC",
    "PARTICIPANT_LEFT_TOPIC",
    "RenderAgent",
    "RenderNotice",
]
