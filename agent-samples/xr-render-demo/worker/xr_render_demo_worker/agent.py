# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed topics and the resident xr-render runtime agent."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from loguru import logger
from xr_ai_runtime import Agent, RuntimeContext, Topic, subscribe
from xr_ai_voice import (
    VOICE_OUTPUT_TOPIC,
    UserQuery,
    VoiceInterrupted,
    VoiceOutput,
    VoiceParticipantLeft,
)

from .models import SceneRequest

USER_QUERY_TOPIC: Topic[UserQuery] = Topic("xr-render.user-query", UserQuery)
PARTICIPANT_LEFT_TOPIC: Topic[VoiceParticipantLeft] = Topic(
    "xr-render.participant-left", VoiceParticipantLeft
)
INTERRUPTED_TOPIC: Topic[VoiceInterrupted] = Topic("xr-render.interrupted", VoiceInterrupted)


class RenderAgent(Agent):
    """Route participant-scoped render turns from voice to the scene supervisor."""

    def __init__(
        self,
        supervisor,
        *,
        on_participant_left: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self._supervisor = supervisor
        self._on_participant_left = on_participant_left
        self._tasks: dict[str, asyncio.Task] = {}
        self._stopped = False

    @subscribe(USER_QUERY_TOPIC)
    async def answer_user(self, query: UserQuery, ctx: RuntimeContext) -> None:
        if self._stopped:
            return
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("render turns require a participant")
        await self._cancel(participant_id)
        task = asyncio.create_task(
            self._run_turn(query, ctx),
            name=f"xr-render:{participant_id}",
        )
        self._tasks[participant_id] = task
        task.add_done_callback(lambda t, pid=participant_id: self._discard(pid, t))

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self, _event: VoiceParticipantLeft, ctx: RuntimeContext
    ) -> None:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            return
        await self._cancel(participant_id)
        if self._on_participant_left is not None:
            self._on_participant_left(participant_id)

    @subscribe(INTERRUPTED_TOPIC)
    async def interrupted(self, _event: VoiceInterrupted, ctx: RuntimeContext) -> None:
        participant_id = ctx.metadata.participant_id
        if participant_id is not None:
            await self._cancel(participant_id)
        else:
            for pid in list(self._tasks):
                await self._cancel(pid)

    async def stop(self) -> None:
        self._stopped = True
        for pid in list(self._tasks):
            await self._cancel(pid)

    async def _run_turn(self, query: UserQuery, ctx: RuntimeContext) -> None:
        participant_id = ctx.metadata.participant_id
        response_id = ctx.metadata.message_id
        opened = False
        try:
            reply = await self._supervisor.handle(
                SceneRequest(
                    transcript=query.text,
                    participant_id=participant_id,
                    timestamp_us=query.timestamp_us,
                    trace_id=response_id or "",
                )
            )
            await ctx.publish(
                VOICE_OUTPUT_TOPIC,
                VoiceOutput(
                    text=reply.response,
                    response_id=response_id,
                    final=False,
                    timestamp_us=query.timestamp_us,
                ),
            )
            opened = True
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.exception("xr-render turn failed for {}: {!r}", participant_id, error)
            try:
                await ctx.publish(
                    VOICE_OUTPUT_TOPIC,
                    VoiceOutput(
                        text="Something went wrong. Please try again.",
                        response_id=response_id,
                        final=False,
                        timestamp_us=query.timestamp_us,
                    ),
                )
                opened = True
            except Exception:
                # Best-effort failure notice; a publish failure here has no
                # further fallback and must not mask the original error.
                pass
        finally:
            if opened:
                try:
                    await ctx.publish(
                        VOICE_OUTPUT_TOPIC,
                        VoiceOutput(response_id=response_id, timestamp_us=query.timestamp_us),
                    )
                except Exception:
                    pass

    async def _cancel(self, participant_id: str) -> None:
        task = self._tasks.pop(participant_id, None)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def _discard(self, participant_id: str, task: asyncio.Task) -> None:
        current = self._tasks.get(participant_id)
        if current is task:
            self._tasks.pop(participant_id, None)
        if task.cancelled():
            return
        if (error := task.exception()) is not None:
            logger.error("xr-render turn failed for {}: {!r}", participant_id, error)


__all__ = [
    "INTERRUPTED_TOPIC",
    "PARTICIPANT_LEFT_TOPIC",
    "USER_QUERY_TOPIC",
    "RenderAgent",
]
