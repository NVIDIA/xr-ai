# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in visual change monitoring with two-caption deduplication."""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import nemo_relay
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator
from xr_ai_hub import FrameUnavailable
from xr_ai_models import ChatMessage, ChatResponse, LLMService, ToolDef, VLMService
from xr_ai_runtime import Agent, AgentRuntime, RuntimeClosedError, RuntimeContext, subscribe
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.current_frame import CurrentFrameRequest
from xr_ai_tools.tool_calling import run_tool_loop
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryTool
from xr_ai_voice import VoiceParticipantLeft

from .events import (
    BACKGROUND_FACT_TOPIC,
    CHANGE_WATCH_RECORD_TOPIC,
    PARTICIPANT_CLEANUP_COMPLETE_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    BackgroundFact,
    ChangeWatchRecord,
    ParticipantCleanupComplete,
)

if TYPE_CHECKING:
    from .images import ParticipantImageAgent


class ChangeWatchStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(default="", max_length=240)


class ChangeWatchControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChangeWatchState(BaseModel):
    active: bool
    instruction: str = ""
    message: str = Field(min_length=1)


class ChangeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    important: bool
    summary: str = Field(default="", max_length=300)

    @model_validator(mode="after")
    def validate_summary(self) -> ChangeDecision:
        self.summary = self.summary.strip()
        if self.important and not self.summary:
            raise ValueError("important changes require a summary")
        if not self.important:
            self.summary = ""
        return self


@dataclass(slots=True)
class _WatchState:
    instruction: str
    captions: deque[str] = field(default_factory=lambda: deque(maxlen=2))
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _same_caption(left: str, right: str) -> bool:
    return left.strip().casefold() == right.strip().casefold()


class ChangeWatchAgent(Agent):
    """Own participant-scoped change-watch tasks and model decisions."""

    def __init__(
        self,
        *,
        images: ParticipantImageAgent,
        vlm: VLMService,
        llm: LLMService,
        caption_prompt: str,
        event_prompt: str,
        default_instruction: str,
        interval_s: float,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        if not caption_prompt.strip():
            raise ValueError("caption_prompt must not be empty")
        if not event_prompt.strip():
            raise ValueError("event_prompt must not be empty")
        if not default_instruction.strip():
            raise ValueError("default_instruction must not be empty")
        self._images = images
        self._query_image = ImageQueryTool(images=images.images, vlm=vlm)
        self._llm = llm
        self._caption_prompt = caption_prompt.strip()
        self._event_prompt = event_prompt.strip()
        self._default_instruction = default_instruction.strip().rstrip(".!? ")
        self._interval_s = interval_s
        self._runtime: AgentRuntime | None = None
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._states: dict[str, _WatchState] = {}
        self._stopped = False
        super().__init__()

    def bind_runtime(self, runtime: AgentRuntime) -> None:
        if not runtime.running:
            raise RuntimeError("change watch requires a running agent runtime")
        if self._runtime is not None:
            raise RuntimeError("change watch is already bound")
        self._runtime = runtime

    def participant_tools(self, participant_id: str) -> ToolSet:
        """Return lifecycle controls with participant identity bound by the caller."""

        async def start(request: ChangeWatchStartRequest) -> ChangeWatchState:
            return await self._start(participant_id, request.instruction)

        async def stop(_request: ChangeWatchControlRequest) -> ChangeWatchState:
            return await self._stop(participant_id)

        async def status(_request: ChangeWatchControlRequest) -> ChangeWatchState:
            return self._status(participant_id)

        def render(result: ChangeWatchState) -> str:
            return result.message

        return ToolSet(
            (
                Tool(
                    "change_watch__start",
                    "Start visual change monitoring in the background.",
                    ChangeWatchStartRequest,
                    ChangeWatchState,
                    start,
                    return_direct=True,
                    render_result=render,
                ),
                Tool(
                    "change_watch__stop",
                    "Stop visual change monitoring.",
                    ChangeWatchControlRequest,
                    ChangeWatchState,
                    stop,
                    return_direct=True,
                    render_result=render,
                ),
                Tool(
                    "change_watch__status",
                    "Report whether visual change monitoring is running.",
                    ChangeWatchControlRequest,
                    ChangeWatchState,
                    status,
                    return_direct=True,
                    render_result=render,
                ),
            )
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
                producer="change_watch",
            ),
        )

    async def stop(self) -> None:
        """Cancel all monitoring tasks before runtime shutdown."""

        self._stopped = True
        tasks = tuple(self._tasks.values())
        self._tasks.clear()
        self._states.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._runtime = None

    async def _start(self, participant_id: str, instruction: str) -> ChangeWatchState:
        runtime = self._running_runtime()
        task = self._tasks.get(participant_id)
        if task is not None and not task.done():
            current = self._states[participant_id].instruction
            return ChangeWatchState(
                active=True,
                instruction=current,
                message=f"Visual change monitoring is already running. Monitoring: {current}.",
            )
        focus = instruction.strip().rstrip(".!? ") or self._default_instruction
        self._states[participant_id] = _WatchState(instruction=focus)
        await runtime.publish(
            CHANGE_WATCH_RECORD_TOPIC,
            ChangeWatchRecord(
                timestamp_us=time.time_ns() // 1_000,
                record_type="started",
                instruction=focus,
            ),
            participant_id=participant_id,
            source="change-watch",
        )
        task = asyncio.create_task(
            self._run(participant_id),
            name=f"change-watch:{participant_id}",
            context=nemo_relay.fork_asyncio_context(),
        )
        self._tasks[participant_id] = task
        task.add_done_callback(
            lambda completed, pid=participant_id: self._discard(pid, completed)
        )
        return ChangeWatchState(
            active=True,
            instruction=focus,
            message=f"Visual change monitoring started. Monitoring: {focus}.",
        )

    async def _stop(self, participant_id: str) -> ChangeWatchState:
        state = self._states.get(participant_id)
        instruction = state.instruction if state is not None else ""
        active = await self._cancel(participant_id)
        if not active:
            return ChangeWatchState(
                active=False,
                message="Visual change monitoring is not running.",
            )
        await self._publish_record(
            participant_id,
            ChangeWatchRecord(
                timestamp_us=time.time_ns() // 1_000,
                record_type="stopped",
                instruction=instruction,
            ),
        )
        return ChangeWatchState(
            active=False,
            instruction=instruction,
            message="Visual change monitoring stopped.",
        )

    def _status(self, participant_id: str) -> ChangeWatchState:
        task = self._tasks.get(participant_id)
        active = task is not None and not task.done()
        state = self._states.get(participant_id)
        instruction = state.instruction if active and state is not None else ""
        message = (
            f"Visual change monitoring is running. Monitoring: {instruction}."
            if active
            else "Visual change monitoring is stopped."
        )
        return ChangeWatchState(
            active=active,
            instruction=instruction,
            message=message,
        )

    async def _run(self, participant_id: str) -> None:
        with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
            while True:
                await self._observe(participant_id)
                await asyncio.sleep(self._interval_s)

    async def _observe(self, participant_id: str) -> None:
        state = self._states.get(participant_id)
        if state is None or state.lock.locked():
            return
        async with state.lock:
            now_us = time.time_ns() // 1_000
            try:
                frame = await self._images.get_current_frame.execute(
                    CurrentFrameRequest(participant_id=participant_id)
                )
                caption_result = await self._query_image.execute(
                    ImageQueryRequest(
                        image=frame.image,
                        query=(
                            f"{self._caption_prompt}\n"
                            f"Focus: {state.instruction}"
                        ),
                    )
                )
            except asyncio.CancelledError:
                raise
            except FrameUnavailable as exc:
                await self._publish_record(
                    participant_id,
                    ChangeWatchRecord(
                        timestamp_us=now_us,
                        record_type="unavailable",
                        instruction=state.instruction,
                        error=str(exc),
                    ),
                )
                return
            except Exception as exc:
                logger.opt(exception=True).warning(
                    "change watch vision failed pid={!r}", participant_id
                )
                await self._publish_error(participant_id, state, now_us, str(exc))
                return

            caption = caption_result.text.strip()
            if not caption_result.available or not caption:
                await self._publish_record(
                    participant_id,
                    ChangeWatchRecord(
                        timestamp_us=now_us,
                        record_type="unavailable",
                        instruction=state.instruction,
                        error=caption or "visual caption unavailable",
                    ),
                )
                return
            if not state.captions:
                state.captions.append(caption)
                await self._publish_record(
                    participant_id,
                    ChangeWatchRecord(
                        timestamp_us=now_us,
                        record_type="baseline",
                        instruction=state.instruction,
                        caption=caption,
                    ),
                )
                return

            duplicate = _same_caption(state.captions[-1], caption)
            if duplicate:
                state.captions.append(caption)
                await self._publish_record(
                    participant_id,
                    ChangeWatchRecord(
                        timestamp_us=now_us,
                        record_type="observation",
                        instruction=state.instruction,
                        caption=caption,
                        duplicate=True,
                    ),
                )
                return

            try:
                decision = await self._decide(state, caption)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.opt(exception=True).warning(
                    "change watch decision failed pid={!r}", participant_id
                )
                await self._publish_error(
                    participant_id,
                    state,
                    now_us,
                    str(exc),
                    caption=caption,
                )
                return
            state.captions.append(caption)
            record = ChangeWatchRecord(
                timestamp_us=now_us,
                record_type="observation",
                instruction=state.instruction,
                caption=caption,
                important=decision.important,
                summary=decision.summary,
            )
            await self._publish_record(participant_id, record)
            if decision.important:
                await self._publish_fact(participant_id, now_us, decision.summary)

    async def _decide(
        self,
        state: _WatchState,
        caption: str,
    ) -> ChangeDecision:
        async def commit(request: ChangeDecision) -> ChangeDecision:
            return request

        tools = ToolSet(
            (
                Tool(
                    "change_watch__commit",
                    "Commit whether the current visual change is important.",
                    ChangeDecision,
                    ChangeDecision,
                    commit,
                    return_direct=True,
                ),
            )
        )
        payload = json.dumps(
            {
                "watch_for": state.instruction,
                "previous": list(state.captions),
                "current": caption,
            },
            ensure_ascii=False,
            separators=(",", ":"),
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
                ChatMessage(role="system", content=self._event_prompt),
                ChatMessage(role="user", content=payload),
            ),
            tools,
            call_model,
            max_iterations=3,
            max_tool_calls=2,
        )
        return ChangeDecision.model_validate_json(result.content)

    async def _publish_error(
        self,
        participant_id: str,
        state: _WatchState,
        timestamp_us: int,
        error: str,
        *,
        caption: str = "",
    ) -> None:
        await self._publish_record(
            participant_id,
            ChangeWatchRecord(
                timestamp_us=timestamp_us,
                record_type="error",
                instruction=state.instruction,
                caption=caption,
                error=error,
            ),
        )

    async def _publish_record(
        self,
        participant_id: str,
        record: ChangeWatchRecord,
    ) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        try:
            await runtime.publish(
                CHANGE_WATCH_RECORD_TOPIC,
                record,
                participant_id=participant_id,
                source="change-watch",
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
                    application="change_watch",
                    text=text,
                ),
                participant_id=participant_id,
                source="change-watch",
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
                    "change watch stopped pid={!r}: {!r}", participant_id, error
                )

    def _running_runtime(self) -> AgentRuntime:
        if self._stopped:
            raise RuntimeError("change watch is stopping")
        if self._runtime is None or not self._runtime.running:
            raise RuntimeError("change watch requires a running agent runtime")
        return self._runtime

    @staticmethod
    def _participant(ctx: RuntimeContext) -> str:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("change watch lifecycle requires a participant")
        return participant_id


__all__ = [
    "ChangeDecision",
    "ChangeWatchAgent",
    "ChangeWatchControlRequest",
    "ChangeWatchStartRequest",
    "ChangeWatchState",
]
