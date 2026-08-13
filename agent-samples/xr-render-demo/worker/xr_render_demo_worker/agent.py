# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed topics and the resident xr-render runtime agent."""

from __future__ import annotations

import asyncio
from contextlib import suppress

import nemo_relay
from pydantic import BaseModel, ConfigDict, Field
from xr_ai_runtime import (
    Agent,
    RuntimeClosedError,
    RuntimeContext,
    Topic,
    subscribe,
)
from xr_ai_tools import Tool
from xr_ai_voice import (
    VOICE_OUTPUT_TOPIC,
    UserQuery,
    VoiceInterrupted,
    VoiceOutput,
    VoiceParticipantLeft,
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


class RenderAgent(Agent):
    """Run participant-scoped render turns and publish their spoken chunks."""

    def __init__(self, scene: SceneModelLoop, tools: tuple[Tool, ...]) -> None:
        super().__init__(tools)
        self.scene = scene
        self._tasks: dict[str, asyncio.Task[None]] = {}

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

        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("render turns require a participant")
        await self._cancel(participant_id)
        task = asyncio.create_task(
            self._run_turn(
                text,
                ctx,
                is_notice=is_notice,
                interrupt_output=interrupt_output,
                timestamp_us=timestamp_us,
            ),
            name=f"xr-render:{participant_id}",
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
        """Cancel work and release state for a departed participant."""

        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("participant-left events require a participant")
        await self._cancel(participant_id)
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
            await self._cancel_all()
            return
        await self._cancel(participant_id)

    async def stop(self) -> None:
        """Cancel active turns before runtime shutdown."""

        await self._cancel_all()

    async def _run_turn(
        self,
        text: str,
        ctx: RuntimeContext,
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
                    is_notice=is_notice,
                    interrupt_output=interrupt_output,
                    timestamp_us=timestamp_us,
                )

    async def _run_turn_scoped(
        self,
        text: str,
        ctx: RuntimeContext,
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
                first = False
                opened = True
        finally:
            if opened:
                with suppress(RuntimeClosedError):
                    await ctx.publish(
                        VOICE_OUTPUT_TOPIC,
                        VoiceOutput(
                            response_id=response_id,
                            timestamp_us=timestamp_us,
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

    def _discard(self, participant_id: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(participant_id) is task:
            self._tasks.pop(participant_id, None)


__all__ = [
    "RENDER_NOTICE_TOPIC",
    "USER_QUERY_TOPIC",
    "INTERRUPTED_TOPIC",
    "PARTICIPANT_LEFT_TOPIC",
    "RenderAgent",
    "RenderNotice",
]
