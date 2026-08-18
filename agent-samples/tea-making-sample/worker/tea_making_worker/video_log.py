# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in broad visual captions with five-caption delta context."""

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
from pydantic import BaseModel, ConfigDict, Field, field_validator
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
    PARTICIPANT_LEFT_TOPIC,
    VIDEO_LOG_RECORD_TOPIC,
    BackgroundFact,
    VideoLogRecord,
)

if TYPE_CHECKING:
    from .images import ParticipantImageAgent

_NO_CHANGE = "no meaningful visual change"


class VideoLogControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VideoLogState(BaseModel):
    active: bool
    message: str = Field(min_length=1)


class VideoDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delta: str = Field(min_length=1, max_length=400)

    @field_validator("delta")
    @classmethod
    def strip_delta(cls, value: str) -> str:
        delta = value.strip()
        if not delta:
            raise ValueError("delta must not be blank")
        return delta


@dataclass(slots=True)
class _VideoState:
    captions: deque[str]
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _is_no_change(delta: str) -> bool:
    return delta.rstrip(". ").casefold() == _NO_CHANGE


class VideoLogAgent(Agent):
    """Own participant-scoped broad visual activity logs."""

    def __init__(
        self,
        *,
        images: ParticipantImageAgent,
        vlm: VLMService,
        llm: LLMService,
        caption_prompt: str,
        delta_prompt: str,
        interval_s: float,
        history_size: int = 5,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        if history_size < 2:
            raise ValueError("history_size must be at least two")
        if not caption_prompt.strip():
            raise ValueError("caption_prompt must not be empty")
        if not delta_prompt.strip():
            raise ValueError("delta_prompt must not be empty")
        self._images = images
        self._query_image = ImageQueryTool(images=images.images, vlm=vlm)
        self._llm = llm
        self._caption_prompt = caption_prompt.strip()
        self._delta_prompt = delta_prompt.strip()
        self._interval_s = interval_s
        self._history_size = history_size
        self._runtime: AgentRuntime | None = None
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._states: dict[str, _VideoState] = {}
        self._stopped = False
        super().__init__()

    def bind_runtime(self, runtime: AgentRuntime) -> None:
        if not runtime.running:
            raise RuntimeError("video log requires a running agent runtime")
        if self._runtime is not None:
            raise RuntimeError("video log is already bound")
        self._runtime = runtime

    def participant_tools(self, participant_id: str) -> ToolSet:
        """Return video-log controls bound to one participant."""

        async def start(_request: VideoLogControlRequest) -> VideoLogState:
            return await self._start(participant_id)

        async def stop(_request: VideoLogControlRequest) -> VideoLogState:
            return await self._stop(participant_id)

        async def status(_request: VideoLogControlRequest) -> VideoLogState:
            return self._status(participant_id)

        def render(result: VideoLogState) -> str:
            return result.message

        return ToolSet(
            (
                Tool(
                    "video_log__start",
                    "Start broad visual activity logging in the background.",
                    VideoLogControlRequest,
                    VideoLogState,
                    start,
                    return_direct=True,
                    render_result=render,
                ),
                Tool(
                    "video_log__stop",
                    "Stop broad visual activity logging.",
                    VideoLogControlRequest,
                    VideoLogState,
                    stop,
                    return_direct=True,
                    render_result=render,
                ),
                Tool(
                    "video_log__status",
                    "Report whether broad visual activity logging is running.",
                    VideoLogControlRequest,
                    VideoLogState,
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

    async def stop(self) -> None:
        """Cancel all video logging tasks before runtime shutdown."""

        self._stopped = True
        tasks = tuple(self._tasks.values())
        self._tasks.clear()
        self._states.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._runtime = None

    async def _start(self, participant_id: str) -> VideoLogState:
        runtime = self._running_runtime()
        task = self._tasks.get(participant_id)
        if task is not None and not task.done():
            return VideoLogState(
                active=True,
                message="Video activity logging is already running.",
            )
        self._states[participant_id] = _VideoState(
            captions=deque(maxlen=self._history_size)
        )
        await runtime.publish(
            VIDEO_LOG_RECORD_TOPIC,
            VideoLogRecord(
                timestamp_us=time.time_ns() // 1_000,
                record_type="started",
            ),
            participant_id=participant_id,
            source="video-log",
        )
        task = asyncio.create_task(
            self._run(participant_id),
            name=f"video-log:{participant_id}",
            context=nemo_relay.fork_asyncio_context(),
        )
        self._tasks[participant_id] = task
        task.add_done_callback(
            lambda completed, pid=participant_id: self._discard(pid, completed)
        )
        return VideoLogState(active=True, message="Video activity logging started.")

    async def _stop(self, participant_id: str) -> VideoLogState:
        active = await self._cancel(participant_id)
        if not active:
            return VideoLogState(
                active=False,
                message="Video activity logging is not running.",
            )
        await self._publish_record(
            participant_id,
            VideoLogRecord(
                timestamp_us=time.time_ns() // 1_000,
                record_type="stopped",
            ),
        )
        return VideoLogState(active=False, message="Video activity logging stopped.")

    def _status(self, participant_id: str) -> VideoLogState:
        task = self._tasks.get(participant_id)
        active = task is not None and not task.done()
        return VideoLogState(
            active=active,
            message=(
                "Video activity logging is running."
                if active
                else "Video activity logging is stopped."
            ),
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
                        query=self._caption_prompt,
                    )
                )
            except asyncio.CancelledError:
                raise
            except FrameUnavailable as exc:
                await self._publish_record(
                    participant_id,
                    VideoLogRecord(
                        timestamp_us=now_us,
                        record_type="unavailable",
                        error=str(exc),
                    ),
                )
                return
            except Exception as exc:
                logger.opt(exception=True).warning(
                    "video caption failed pid={!r}", participant_id
                )
                await self._publish_error(participant_id, now_us, str(exc))
                return

            caption = caption_result.text.strip()
            if not caption_result.available or not caption:
                await self._publish_record(
                    participant_id,
                    VideoLogRecord(
                        timestamp_us=now_us,
                        record_type="unavailable",
                        error=caption or "visual caption unavailable",
                    ),
                )
                return
            try:
                delta = await self._generate_delta(state, caption)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.opt(exception=True).warning(
                    "video delta failed pid={!r}", participant_id
                )
                await self._publish_error(
                    participant_id,
                    now_us,
                    str(exc),
                    caption=caption,
                )
                return

            state.captions.append(caption)
            await self._publish_record(
                participant_id,
                VideoLogRecord(
                    timestamp_us=now_us,
                    record_type="observation",
                    caption=caption,
                    delta=delta.delta,
                ),
            )
            if not _is_no_change(delta.delta):
                await self._publish_fact(participant_id, now_us, delta.delta)

    async def _generate_delta(
        self,
        state: _VideoState,
        caption: str,
    ) -> VideoDelta:
        async def commit(request: VideoDelta) -> VideoDelta:
            return request

        tools = ToolSet(
            (
                Tool(
                    "video_log__commit",
                    "Commit the material visible delta for the current caption.",
                    VideoDelta,
                    VideoDelta,
                    commit,
                    return_direct=True,
                ),
            )
        )
        previous = list(state.captions)[-(self._history_size - 1) :]
        payload = json.dumps(
            {"previous": previous, "current": caption},
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
                ChatMessage(role="system", content=self._delta_prompt),
                ChatMessage(role="user", content=payload),
            ),
            tools,
            call_model,
            max_iterations=3,
            max_tool_calls=2,
        )
        return VideoDelta.model_validate_json(result.content)

    async def _publish_error(
        self,
        participant_id: str,
        timestamp_us: int,
        error: str,
        *,
        caption: str = "",
    ) -> None:
        await self._publish_record(
            participant_id,
            VideoLogRecord(
                timestamp_us=timestamp_us,
                record_type="error",
                caption=caption,
                error=error,
            ),
        )

    async def _publish_record(
        self,
        participant_id: str,
        record: VideoLogRecord,
    ) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        try:
            await runtime.publish(
                VIDEO_LOG_RECORD_TOPIC,
                record,
                participant_id=participant_id,
                source="video-log",
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
                    application="video_log",
                    text=text,
                ),
                participant_id=participant_id,
                source="video-log",
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
                    "video log stopped pid={!r}: {!r}", participant_id, error
                )

    def _running_runtime(self) -> AgentRuntime:
        if self._stopped:
            raise RuntimeError("video log is stopping")
        if self._runtime is None or not self._runtime.running:
            raise RuntimeError("video log requires a running agent runtime")
        return self._runtime

    @staticmethod
    def _participant(ctx: RuntimeContext) -> str:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("video log lifecycle requires a participant")
        return participant_id


__all__ = [
    "VideoDelta",
    "VideoLogAgent",
    "VideoLogControlRequest",
    "VideoLogState",
]
