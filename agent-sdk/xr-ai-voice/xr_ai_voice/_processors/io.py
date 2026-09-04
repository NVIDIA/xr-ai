# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private Pipecat processor for runtime voice input and output."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    TextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .._frames import (
    AssistantResponseEndFrame,
    GatedQueryFrame,
    ParticipantJoinedFrame,
    ParticipantLeftFrame,
)
from .._types import VoiceInputSink, VoiceQuery, VoiceResponse

if TYPE_CHECKING:
    from .._transport import HubVoiceTransport


@dataclass(frozen=True, slots=True)
class _QueuedResponse:
    participant_id: str
    response: VoiceResponse
    pts_us: int
    interrupt: bool = False


class _VoiceIOProcessor(FrameProcessor):
    """Publish accepted input and serialize participant-scoped voice output."""

    def __init__(
        self,
        input_sink: VoiceInputSink,
        *,
        transport: HubVoiceTransport | None = None,
        on_participant_joined: Callable[[str], Awaitable[None] | None] | None = None,
        on_participant_left: Callable[[str], Awaitable[None] | None] | None = None,
        on_speech_started: Callable[[str], Awaitable[None] | None] | None = None,
        on_speech_stopped: Callable[[str], Awaitable[None] | None] | None = None,
        on_interrupted: Callable[[str | None], Awaitable[None] | None] | None = None,
        interrupt_on_supersede: bool = False,
    ) -> None:
        super().__init__()
        self._input_sink = input_sink
        self._transport = transport
        self._on_participant_joined = on_participant_joined
        self._on_participant_left = on_participant_left
        self._on_speech_started = on_speech_started
        self._on_speech_stopped = on_speech_stopped
        self._on_interrupted = on_interrupted
        self._interrupt_on_supersede = interrupt_on_supersede
        self._inflight: dict[str, asyncio.Task[None]] = {}
        self._input_tasks: set[asyncio.Task[None]] = set()
        self._queued: dict[str, deque[_QueuedResponse]] = {}
        self._turn_tokens: dict[str, object] = {}
        # Completed output may still be draining through TTS when new input arrives.
        self._seen_output: set[str] = set()

    async def enqueue_query(
        self,
        participant_id: str,
        text: str,
        *,
        pts_us: int | None = None,
    ) -> None:
        """Submit typed text through the same path as accepted speech."""

        await self._spawn_query(
            GatedQueryFrame(
                participant_id=participant_id,
                text=text,
                fresh_match=False,
                pts_us=pts_us if pts_us is not None else time.time_ns() // 1_000,
            )
        )

    async def enqueue_response(
        self,
        participant_id: str,
        response: VoiceResponse,
        *,
        interrupt: bool = False,
        pts_us: int | None = None,
    ) -> None:
        """Submit finite or incremental assistant output."""

        await self._spawn_response(
            _QueuedResponse(
                participant_id=participant_id,
                response=response,
                interrupt=interrupt,
                pts_us=pts_us if pts_us is not None else time.time_ns() // 1_000,
            )
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            pid = frame.transport_source
            if pid:
                logger.info("voice cancel pid={!r} reason=interruption", pid)
                await self._cancel_pid(pid)
            else:
                await self._cancel_all()
            await self.push_frame(frame, direction)
            await self._notify_interrupted(pid)
            return

        if isinstance(frame, UserStartedSpeakingFrame):
            if frame.transport_source:
                await self._notify_speech_started(frame.transport_source)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, UserStoppedSpeakingFrame):
            if frame.transport_source:
                await self._notify_speech_stopped(frame.transport_source)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (EndFrame, CancelFrame)):
            await self._shutdown()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, GatedQueryFrame):
            await self._spawn_query(frame)
            return

        if isinstance(frame, ParticipantJoinedFrame):
            logger.info("voice participant joined pid={!r}", frame.participant_id)
            if self._transport is not None:
                self._transport.set_target_participant(frame.participant_id)
            await self._notify_joined(frame.participant_id)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, ParticipantLeftFrame):
            self._seen_output.discard(frame.participant_id)
            logger.info("voice participant left pid={!r}", frame.participant_id)
            if self._transport is not None:
                self._transport.cleanup_participant(frame.participant_id)
            await self._notify_left(frame.participant_id)
            await self._cancel_pid(frame.participant_id)
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    async def _spawn_query(self, frame: GatedQueryFrame) -> None:
        pid = frame.participant_id
        logger.info("voice input pid={!r}", pid)
        interrupted_response: asyncio.Task[None] | None = None
        if current := self._inflight.get(pid):
            if not current.done():
                interrupted_response = await self._cancel_pid(pid)
        interrupt = self._interrupt_on_supersede and pid in self._seen_output
        if interrupt:
            frame_to_push = InterruptionFrame()
            frame_to_push.transport_source = pid
            await self.push_frame(frame_to_push)
        task = asyncio.create_task(
            self._run_query(frame, interrupted_response=interrupted_response),
            name=f"voice-input-{pid}",
        )
        self._input_tasks.add(task)
        task.add_done_callback(self._input_tasks.discard)

    async def _run_query(
        self,
        frame: GatedQueryFrame,
        *,
        interrupted_response: asyncio.Task[None] | None,
    ) -> None:
        pid = frame.participant_id
        try:
            if interrupted_response is not None:
                await asyncio.gather(interrupted_response, return_exceptions=True)
            await self._input_sink(
                VoiceQuery(
                    participant_id=pid,
                    text=frame.text,
                    timestamp_us=frame.pts_us,
                    interrupted_output=interrupted_response is not None,
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("voice input sink raised pid={!r}", pid)

    async def _spawn_response(self, item: _QueuedResponse) -> None:
        pid = item.participant_id
        if isinstance(item.response, str) and not item.response.strip():
            return
        current = self._inflight.get(pid)
        if not item.interrupt and (
            (current is not None and not current.done()) or self._queued.get(pid)
        ):
            pending = self._queued.setdefault(pid, deque())
            pending.append(item)
            logger.info("voice response queued pid={!r} depth={}", pid, len(pending))
            return
        if item.interrupt:
            await self._cancel_pid(pid)
            interruption = InterruptionFrame()
            interruption.transport_source = pid
            await self.push_frame(interruption)
        await self._start_response(item)

    async def _start_response(self, item: _QueuedResponse) -> None:
        pid = item.participant_id
        self._seen_output.add(pid)
        token = object()
        self._turn_tokens[pid] = token
        task = asyncio.create_task(
            self._run_response(item, token),
            name=f"voice-response-{pid}",
        )
        self._inflight[pid] = task

    async def _run_response(self, item: _QueuedResponse, token: object) -> None:
        pid = item.participant_id
        chunks: list[str] = []
        cancelled = False
        try:
            await self._consume_response(
                item.response,
                pid=pid,
                token=token,
                accumulated=chunks,
            )
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception:
            logger.exception("voice response failed pid={!r}", pid)
        finally:
            await self._close_response(item.response)
            is_current = self._turn_tokens.get(pid) is token
            if not cancelled and is_current:
                try:
                    await self.push_frame(
                        AssistantResponseEndFrame(
                            pid=pid,
                            text="".join(chunks),
                            pts_us=item.pts_us,
                        )
                    )
                except Exception:
                    logger.exception("emit voice response end failed pid={!r}", pid)
            if is_current:
                self._turn_tokens.pop(pid, None)
                self._inflight.pop(pid, None)
                await self._start_next(pid)

    async def _consume_response(
        self,
        response: VoiceResponse,
        *,
        pid: str,
        token: object,
        accumulated: list[str],
    ) -> None:
        if isinstance(response, str):
            if response and self._turn_tokens.get(pid) is token:
                accumulated.append(response)
                await self._push_text(response, pid=pid)
            return
        async for chunk in response:
            if self._turn_tokens.get(pid) is not token:
                return
            if not chunk:
                continue
            accumulated.append(chunk)
            await self._push_text(chunk, pid=pid)

    async def _start_next(self, pid: str) -> None:
        pending = self._queued.get(pid)
        if not pending:
            return
        item = pending.popleft()
        if not pending:
            self._queued.pop(pid, None)
        await self._start_response(item)

    async def _push_text(self, text: str, *, pid: str) -> None:
        frame = TextFrame(text=text)
        frame.transport_destination = pid
        await self.push_frame(frame)

    async def _notify_joined(self, pid: str) -> None:
        if self._on_participant_joined is None:
            return
        try:
            result = self._on_participant_joined(pid)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("voice participant-joined callback raised pid={!r}", pid)

    async def _notify_left(self, pid: str) -> None:
        if self._on_participant_left is None:
            return
        try:
            result = self._on_participant_left(pid)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("voice participant-left callback raised pid={!r}", pid)

    async def _notify_speech_started(self, pid: str) -> None:
        if self._on_speech_started is None:
            return
        try:
            result = self._on_speech_started(pid)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("voice speech-started callback raised pid={!r}", pid)

    async def _notify_speech_stopped(self, pid: str) -> None:
        if self._on_speech_stopped is None:
            return
        try:
            result = self._on_speech_stopped(pid)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("voice speech-stopped callback raised pid={!r}", pid)

    async def _notify_interrupted(self, pid: str | None) -> None:
        if self._on_interrupted is None:
            return
        try:
            result = self._on_interrupted(pid)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("voice interruption callback raised pid={!r}", pid)

    async def _cancel_pid(self, pid: str) -> asyncio.Task[None] | None:
        self._turn_tokens.pop(pid, None)
        for item in self._queued.pop(pid, ()):
            await self._close_response(item.response)
        task = self._inflight.pop(pid, None)
        if task is not None and not task.done():
            task.cancel()
        return task

    @staticmethod
    async def _close_response(response: VoiceResponse) -> None:
        if isinstance(response, str):
            return
        close = getattr(response, "aclose", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            _ = await result

    async def _shutdown(self) -> None:
        tasks = [
            task
            for task in (*self._inflight.values(), *self._input_tasks)
            if not task.done()
        ]
        await self._cancel_all()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _cancel_all(self) -> None:
        for task in self._inflight.values():
            if not task.done():
                task.cancel()
        for pending in self._queued.values():
            for item in pending:
                await self._close_response(item.response)
        self._queued.clear()
        self._inflight.clear()
        self._turn_tokens.clear()
        for task in self._input_tasks:
            if not task.done():
                task.cancel()
        self._input_tasks.clear()


__all__ = ["_VoiceIOProcessor"]
