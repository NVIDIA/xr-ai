# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Voice delivery for actionable tea-guidance notices."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque

from loguru import logger
from xr_ai_runtime import (
    Agent,
    AgentRuntime,
    RuntimeClosedError,
    RuntimeContext,
    subscribe,
)
from xr_ai_voice import (
    VOICE_CONTRIBUTION_TOPIC,
    UserQuery,
    VoiceInterrupted,
    VoiceOutput,
    VoiceParticipantLeft,
    VoiceSpeechStarted,
    VoiceSpeechStopped,
)

from .events import (
    FOREGROUND_RECORD_TOPIC,
    GUIDANCE_NOTICE_TOPIC,
    INTERRUPTED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    SPEECH_STARTED_TOPIC,
    SPEECH_STOPPED_TOPIC,
    USER_QUERY_TOPIC,
    ForegroundRecord,
    GuidanceNotice,
)

_RELEASE_DELAY_S = 1.0


class GuidanceVoiceAgent(Agent):
    """Speak workflow notices while background applications remain file-only."""

    def __init__(self, *, release_delay_s: float = _RELEASE_DELAY_S) -> None:
        if release_delay_s < 0:
            raise ValueError("guidance voice release delay must not be negative")
        super().__init__()
        self._release_delay_s = release_delay_s
        self._runtime: AgentRuntime | None = None
        self._speaking: set[str] = set()
        self._foreground: set[str] = set()
        self._pending: defaultdict[str, deque[GuidanceNotice]] = defaultdict(deque)
        self._release_tasks: dict[str, asyncio.Task[None]] = {}

    def bind_runtime(self, runtime: AgentRuntime) -> None:
        """Bind the runtime used by delayed notice delivery."""

        self._runtime = runtime

    @subscribe(GUIDANCE_NOTICE_TOPIC)
    async def notify(self, notice: GuidanceNotice, ctx: RuntimeContext) -> None:
        participant_id = self._participant(ctx)
        if self._busy(participant_id):
            self._pending[participant_id].append(notice)
            logger.debug(
                "guidance notice deferred pid={!r} speaking={} foreground={}",
                participant_id,
                participant_id in self._speaking,
                participant_id in self._foreground,
            )
            return
        await ctx.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text=notice.text, timestamp_us=notice.timestamp_us),
        )

    @subscribe(SPEECH_STARTED_TOPIC)
    async def speech_started(
        self,
        _event: VoiceSpeechStarted,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = self._participant(ctx)
        self._speaking.add(participant_id)
        self._cancel_release(participant_id)

    @subscribe(SPEECH_STOPPED_TOPIC)
    async def speech_stopped(
        self,
        _event: VoiceSpeechStopped,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = self._participant(ctx)
        self._speaking.discard(participant_id)
        self._schedule_release(participant_id)

    @subscribe(USER_QUERY_TOPIC)
    async def foreground_started(
        self,
        _query: UserQuery,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = self._participant(ctx)
        self._foreground.add(participant_id)
        self._cancel_release(participant_id)

    @subscribe(FOREGROUND_RECORD_TOPIC)
    async def foreground_finished(
        self,
        _record: ForegroundRecord,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = self._participant(ctx)
        self._foreground.discard(participant_id)
        await self._flush(participant_id)

    @subscribe(INTERRUPTED_TOPIC)
    async def interrupted(
        self,
        _event: VoiceInterrupted,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            for active_id in tuple(self._foreground):
                self._foreground.discard(active_id)
                self._schedule_release(active_id)
            return
        self._foreground.discard(participant_id)
        self._schedule_release(participant_id)

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self,
        _event: VoiceParticipantLeft,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = self._participant(ctx)
        self._speaking.discard(participant_id)
        self._foreground.discard(participant_id)
        self._pending.pop(participant_id, None)
        self._cancel_release(participant_id)

    async def stop(self) -> None:
        """Cancel delayed delivery and discard participant-local notices."""

        tasks = tuple(self._release_tasks.values())
        self._release_tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._speaking.clear()
        self._foreground.clear()
        self._pending.clear()
        self._runtime = None

    def _schedule_release(self, participant_id: str) -> None:
        self._cancel_release(participant_id)
        if participant_id not in self._pending:
            return
        task = asyncio.create_task(
            self._release_after_delay(participant_id),
            name=f"guidance-notice-release:{participant_id}",
        )
        self._release_tasks[participant_id] = task
        task.add_done_callback(
            lambda completed, pid=participant_id: self._release_done(
                pid,
                completed,
            )
        )

    async def _release_after_delay(self, participant_id: str) -> None:
        await asyncio.sleep(self._release_delay_s)
        await self._flush(participant_id)

    async def _flush(self, participant_id: str) -> None:
        runtime = self._runtime
        if runtime is None or not runtime.running or self._busy(participant_id):
            return
        pending = self._pending.pop(participant_id, deque())
        while pending:
            if self._busy(participant_id):
                self._pending[participant_id].extendleft(reversed(pending))
                return
            notice = pending.popleft()
            try:
                await runtime.publish(
                    VOICE_CONTRIBUTION_TOPIC,
                    VoiceOutput(text=notice.text, timestamp_us=notice.timestamp_us),
                    participant_id=participant_id,
                    source="guidance-voice",
                )
            except RuntimeClosedError:
                return
        logger.debug("deferred guidance notices released pid={!r}", participant_id)

    def _busy(self, participant_id: str) -> bool:
        return (
            participant_id in self._speaking
            or participant_id in self._foreground
        )

    def _cancel_release(self, participant_id: str) -> None:
        task = self._release_tasks.pop(participant_id, None)
        if task is not None:
            task.cancel()

    def _release_done(
        self,
        participant_id: str,
        task: asyncio.Task[None],
    ) -> None:
        if self._release_tasks.get(participant_id) is task:
            self._release_tasks.pop(participant_id, None)
        if task.cancelled():
            return
        if error := task.exception():
            logger.error(
                "deferred guidance notice failed pid={!r}: {}",
                participant_id,
                error,
            )

    @staticmethod
    def _participant(ctx: RuntimeContext) -> str:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("guidance voice requires a participant")
        return participant_id


__all__ = ["GuidanceVoiceAgent"]
