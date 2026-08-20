# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in transcript persistence with summaries of unsummarized speech."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import suppress
from dataclasses import dataclass, field

import nemo_relay
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator
from xr_ai_models import ChatMessage, ChatResponse, LLMService, ToolDef
from xr_ai_runtime import Agent, AgentRuntime, RuntimeClosedError, RuntimeContext, subscribe
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.tool_calling import run_tool_loop
from xr_ai_voice import (
    VOICE_TRANSCRIPT_TOPIC,
    VoiceParticipantLeft,
    VoiceTranscript,
)

from .events import (
    BACKGROUND_FACT_TOPIC,
    PARTICIPANT_CLEANUP_COMPLETE_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    TRANSCRIPT_RECORD_TOPIC,
    BackgroundFact,
    ParticipantCleanupComplete,
    TranscriptRecord,
)


class TranscriptControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TranscriptState(BaseModel):
    active: bool
    message: str = Field(min_length=1)


class TranscriptSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=500)

    @field_validator("summary")
    @classmethod
    def strip_summary(cls, value: str) -> str:
        summary = value.strip()
        if not summary:
            raise ValueError("summary must not be blank")
        return summary


@dataclass(slots=True)
class _TranscriptState:
    turns: list[str] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class TranscriptAgent(Agent):
    """Record final voice transcripts only while explicitly active."""

    def __init__(
        self,
        *,
        llm: LLMService,
        summary_prompt: str,
        summary_interval_s: float = 120,
    ) -> None:
        if summary_interval_s <= 0:
            raise ValueError("summary_interval_s must be positive")
        if not summary_prompt.strip():
            raise ValueError("summary_prompt must not be empty")
        self._llm = llm
        self._summary_prompt = summary_prompt.strip()
        self._summary_interval_s = summary_interval_s
        self._runtime: AgentRuntime | None = None
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._states: dict[str, _TranscriptState] = {}
        self._stopped = False
        super().__init__()

    def bind_runtime(self, runtime: AgentRuntime) -> None:
        if not runtime.running:
            raise RuntimeError("transcript recorder requires a running agent runtime")
        if self._runtime is not None:
            raise RuntimeError("transcript recorder is already bound")
        self._runtime = runtime

    def participant_tools(self, participant_id: str) -> ToolSet:
        """Return transcript lifecycle controls bound to one participant."""

        async def start(_request: TranscriptControlRequest) -> TranscriptState:
            return await self._start(participant_id)

        async def stop(_request: TranscriptControlRequest) -> TranscriptState:
            return await self._stop(participant_id)

        async def status(_request: TranscriptControlRequest) -> TranscriptState:
            return self._status(participant_id)

        def render(result: TranscriptState) -> str:
            return result.message

        return ToolSet(
            (
                Tool(
                    "transcript__start",
                    "Start recording final speech transcripts in the background.",
                    TranscriptControlRequest,
                    TranscriptState,
                    start,
                    return_direct=True,
                    render_result=render,
                ),
                Tool(
                    "transcript__stop",
                    "Stop background transcript recording.",
                    TranscriptControlRequest,
                    TranscriptState,
                    stop,
                    return_direct=True,
                    render_result=render,
                ),
                Tool(
                    "transcript__status",
                    "Report whether transcript recording is running.",
                    TranscriptControlRequest,
                    TranscriptState,
                    status,
                    return_direct=True,
                    render_result=render,
                ),
            )
        )

    @subscribe(VOICE_TRANSCRIPT_TOPIC)
    async def record_transcript(
        self,
        transcript: VoiceTranscript,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = self._participant(ctx)
        state = self._states.get(participant_id)
        if state is None:
            return
        text = transcript.text.strip()
        if not text:
            return
        async with state.lock:
            if self._states.get(participant_id) is not state:
                return
            state.turns.append(text)
        await ctx.publish(
            TRANSCRIPT_RECORD_TOPIC,
            TranscriptRecord(
                timestamp_us=transcript.timestamp_us,
                record_type="utterance",
                text=text,
            ),
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
                producer="transcript",
            ),
        )

    async def stop(self) -> None:
        """Cancel all summary tasks before runtime shutdown."""

        self._stopped = True
        tasks = tuple(self._tasks.values())
        self._tasks.clear()
        self._states.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._runtime = None

    async def _start(self, participant_id: str) -> TranscriptState:
        runtime = self._running_runtime()
        task = self._tasks.get(participant_id)
        if task is not None and not task.done():
            return TranscriptState(
                active=True,
                message="Transcript recording is already running.",
            )
        self._states[participant_id] = _TranscriptState()
        await runtime.publish(
            TRANSCRIPT_RECORD_TOPIC,
            TranscriptRecord(
                timestamp_us=time.time_ns() // 1_000,
                record_type="started",
            ),
            participant_id=participant_id,
            source="transcript",
        )
        task = asyncio.create_task(
            self._summary_loop(participant_id),
            name=f"transcript-summary:{participant_id}",
            context=nemo_relay.fork_asyncio_context(),
        )
        self._tasks[participant_id] = task
        task.add_done_callback(
            lambda completed, pid=participant_id: self._discard(pid, completed)
        )
        return TranscriptState(
            active=True,
            message="Transcript recording started.",
        )

    async def _stop(self, participant_id: str) -> TranscriptState:
        active = await self._cancel(participant_id)
        if not active:
            return TranscriptState(
                active=False,
                message="Transcript recording is not running.",
            )
        await self._publish_record(
            participant_id,
            TranscriptRecord(
                timestamp_us=time.time_ns() // 1_000,
                record_type="stopped",
            ),
        )
        return TranscriptState(active=False, message="Transcript recording stopped.")

    def _status(self, participant_id: str) -> TranscriptState:
        task = self._tasks.get(participant_id)
        active = task is not None and not task.done()
        return TranscriptState(
            active=active,
            message=(
                "Transcript recording is running."
                if active
                else "Transcript recording is stopped."
            ),
        )

    async def _summary_loop(self, participant_id: str) -> None:
        with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
            while True:
                await asyncio.sleep(self._summary_interval_s)
                await self._summarize(participant_id)

    async def _summarize(self, participant_id: str) -> None:
        state = self._states.get(participant_id)
        if state is None:
            return
        async with state.lock:
            turns = tuple(state.turns)
        if not turns:
            return
        try:
            summary = await self._generate_summary(turns)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.opt(exception=True).warning(
                "transcript summary failed pid={!r}", participant_id
            )
            await self._publish_record(
                participant_id,
                TranscriptRecord(
                    timestamp_us=time.time_ns() // 1_000,
                    record_type="error",
                    turn_count=len(turns),
                    error=str(exc),
                ),
            )
            return

        async with state.lock:
            if tuple(state.turns[: len(turns)]) != turns:
                logger.warning(
                    "transcript prefix changed before commit pid={!r}", participant_id
                )
                return
            del state.turns[: len(turns)]
        now_us = time.time_ns() // 1_000
        await self._publish_record(
            participant_id,
            TranscriptRecord(
                timestamp_us=now_us,
                record_type="summary",
                text=summary.summary,
                turn_count=len(turns),
            ),
        )
        await self._publish_fact(participant_id, now_us, summary.summary)

    async def _generate_summary(
        self,
        turns: tuple[str, ...],
    ) -> TranscriptSummary:
        async def commit(request: TranscriptSummary) -> TranscriptSummary:
            return request

        tools = ToolSet(
            (
                Tool(
                    "transcript__commit_summary",
                    "Commit one concise summary of the supplied utterances.",
                    TranscriptSummary,
                    TranscriptSummary,
                    commit,
                    return_direct=True,
                ),
            )
        )

        async def call_model(
            messages: tuple[ChatMessage, ...],
            definitions: tuple[ToolDef, ...],
        ) -> ChatResponse:
            return await self._llm.chat(
                messages,
                tools=definitions,
                max_tokens=384,
                temperature=0.0,
                enable_thinking=False,
            )

        result = await run_tool_loop(
            (
                ChatMessage(role="system", content=self._summary_prompt),
                ChatMessage(
                    role="user",
                    content=json.dumps(
                        {"utterances": turns},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ),
            ),
            tools,
            call_model,
            max_iterations=3,
            max_tool_calls=2,
        )
        return TranscriptSummary.model_validate_json(result.content)

    async def _publish_record(
        self,
        participant_id: str,
        record: TranscriptRecord,
    ) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        try:
            await runtime.publish(
                TRANSCRIPT_RECORD_TOPIC,
                record,
                participant_id=participant_id,
                source="transcript",
            )
        except RuntimeClosedError:
            return

    async def _publish_fact(
        self,
        participant_id: str,
        timestamp_us: int,
        text: str,
    ) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        try:
            await runtime.publish(
                BACKGROUND_FACT_TOPIC,
                BackgroundFact(
                    timestamp_us=timestamp_us,
                    application="transcript",
                    text=text,
                ),
                participant_id=participant_id,
                source="transcript",
            )
        except RuntimeClosedError:
            return

    async def _cancel(self, participant_id: str) -> bool:
        task = self._tasks.pop(participant_id, None)
        self._states.pop(participant_id, None)
        if task is None:
            return False
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return True

    def _discard(self, participant_id: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(participant_id) is task:
            self._tasks.pop(participant_id, None)
            self._states.pop(participant_id, None)
        if task.cancelled():
            return
        with suppress(asyncio.CancelledError):
            if error := task.exception():
                logger.error(
                    "transcript summary stopped pid={!r}: {!r}",
                    participant_id,
                    error,
                )

    def _running_runtime(self) -> AgentRuntime:
        if self._stopped:
            raise RuntimeError("transcript recorder is stopping")
        if self._runtime is None or not self._runtime.running:
            raise RuntimeError("transcript recorder requires a running agent runtime")
        return self._runtime

    @staticmethod
    def _participant(ctx: RuntimeContext) -> str:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("transcript lifecycle requires a participant")
        return participant_id


__all__ = [
    "TranscriptAgent",
    "TranscriptControlRequest",
    "TranscriptState",
    "TranscriptSummary",
]
