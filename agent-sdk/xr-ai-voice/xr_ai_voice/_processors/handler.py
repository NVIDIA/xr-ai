# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private Pipecat processor that executes a public voice handler."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    TextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .._frames import (
    AssistantResponseEndFrame,
    GatedQueryFrame,
    ParticipantJoinedFrame,
    ParticipantLeftFrame,
)
from .._handler import VoiceHandler, VoiceQuery, VoiceTurn

if TYPE_CHECKING:
    from .._transport import HubVoiceTransport


class _VoiceHandlerProcessor(FrameProcessor):
    """Run a handler while preserving participant-aware pipeline semantics."""

    def __init__(
        self,
        handler: VoiceHandler,
        *,
        transport: "HubVoiceTransport | None" = None,
        observer: Callable[[VoiceTurn], Awaitable[None]] | None = None,
        on_participant_joined: Callable[[str], Awaitable[None] | None] | None = None,
        on_participant_left: Callable[[str], Awaitable[None] | None] | None = None,
        on_user_started_speaking: Callable[[str], Awaitable[None] | None] | None = None,
        on_query_superseded: Callable[[str], Awaitable[None] | None] | None = None,
        interrupt_on_supersede: bool = False,
        queue_queries: bool = False,
    ) -> None:
        super().__init__()
        self._handler = handler
        self._turn_observer = observer
        self._on_participant_joined = on_participant_joined
        self._on_participant_left = on_participant_left
        self._on_user_started_speaking = on_user_started_speaking
        self._on_query_superseded = on_query_superseded
        self._interrupt_on_supersede = interrupt_on_supersede
        self._queue_queries = queue_queries
        self._inflight: dict[str, asyncio.Task[None]] = {}
        self._queued: dict[str, deque[GatedQueryFrame]] = {}
        self._turn_tokens: dict[str, object] = {}
        # "Has this participant spoken a turn before." A finished turn may still
        # have TTS draining downstream, so a follow-up turn interrupts to clear
        # that lingering audio — tracked by history, not live task state. NOTE:
        # this drives the downstream interrupt only; the on_query_superseded
        # callback fires solely on an actual in-flight replacement (see
        # _spawn_query), never merely because a turn was seen before.
        self._seen_query: set[str] = set()
        # Joined participants receive speech hooks even before their first turn.
        self._joined: set[str] = set()
        # A supplied transport enables automatic single-participant routing.
        self._transport = transport

    async def enqueue_query(
        self,
        participant_id: str,
        text: str,
        *,
        fresh_match: bool = False,
        pts_us: int | None = None,
    ) -> None:
        """Submit text through the same participant-aware path as voice input."""
        await self._spawn_query(
            GatedQueryFrame(
                participant_id=participant_id,
                text=text,
                fresh_match=fresh_match,
                pts_us=pts_us if pts_us is not None else time.time_ns() // 1_000,
            )
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        # Pipecat owns interruption metrics, but cannot cancel handler tasks.
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            # Speech onset may be acoustic echo; only InterruptionFrame cancels.
            await self._dispatch_user_started_speaking(frame.transport_source)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, InterruptionFrame):
            pid = frame.transport_source
            if pid:
                logger.info("voice handler cancel pid={!r} reason=interruption", pid)
                self._cancel_pid(pid)
            else:
                if self._inflight:
                    for p in list(self._inflight):
                        logger.info("voice handler cancel pid={!r} reason=interruption", p)
                self._cancel_all_inflight()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (EndFrame, CancelFrame)):
            # Pipeline shutdown: cancel and await every in-flight handler task so
            # a turn cannot keep emitting text — or writing transcripts through a
            # turn observer — after the session has ended.
            await self._shutdown_inflight()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, GatedQueryFrame):
            await self._spawn_query(frame)
            return

        if isinstance(frame, ParticipantJoinedFrame):
            self._joined.add(frame.participant_id)
            logger.info("voice participant joined pid={!r}", frame.participant_id)
            if self._transport is not None:
                self._transport.set_target_participant(frame.participant_id)
            await self._notify(self._on_participant_joined, frame.participant_id)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, ParticipantLeftFrame):
            self._joined.discard(frame.participant_id)
            self._seen_query.discard(frame.participant_id)
            logger.info("voice participant left pid={!r}", frame.participant_id)
            if self._transport is not None:
                self._transport.cleanup_participant(frame.participant_id)
            await self._notify(self._on_participant_left, frame.participant_id)
            self._cancel_pid(frame.participant_id)
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    async def _dispatch_user_started_speaking(self, pid: str | None) -> None:
        # Third-party frames may omit transport_source; notify all joined users.
        targets = [pid] if (pid and pid in self._joined) else list(self._joined)
        for p in targets:
            await self._notify(self._on_user_started_speaking, p)

    async def _notify(self, callback: Callable[[str], Awaitable[None] | None] | None, pid: str) -> None:
        if callback is None:
            return None
        try:
            result = callback(pid)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("voice session callback raised pid={!r}", pid)

    async def _observe(self, turn: VoiceTurn) -> None:
        if self._turn_observer is None:
            return
        try:
            await self._turn_observer(turn)
        except Exception:
            logger.exception("voice session observer raised pid={!r} role={}", turn.participant_id, turn.role)

    async def _spawn_query(self, frame: GatedQueryFrame) -> None:
        pid = frame.participant_id
        logger.info(
            "voice handler dispatch pid={!r} fresh_match={}",
            pid,
            frame.fresh_match,
        )
        current = self._inflight.get(pid)
        is_active = current is not None and not current.done()

        if is_active and self._queue_queries:
            # A turn is still running: queue this one as a follow-up. This is
            # NOT a supersede — the running turn completes and this runs after
            # it — so on_query_superseded does not fire.
            pending = self._queued.setdefault(pid, deque())
            pending.append(frame)
            logger.info("voice handler queued pid={!r} depth={}", pid, len(pending))
            return

        if is_active:
            # Non-queue mode: the new query actually replaces the in-flight turn.
            # That replacement is the only real supersede.
            logger.info("voice handler superseded pid={!r}", pid)
            await self._notify(self._on_query_superseded, pid)
            self._cancel_pid(pid)

        # A prior turn — even a finished one whose TTS may still be draining —
        # means the new turn interrupts downstream audio so it starts clean.
        had_prior = pid in self._seen_query
        self._seen_query.add(pid)
        await self._start_query(frame, interrupt=had_prior)

    async def _start_query(self, frame: GatedQueryFrame, *, interrupt: bool) -> None:
        pid = frame.participant_id
        if interrupt and self._interrupt_on_supersede:
            # Tag the pid so the downstream TTS drain/flush scopes to this
            # participant instead of every participant's audio.
            interruption = InterruptionFrame()
            interruption.transport_source = pid
            await self.push_frame(interruption)
        token = object()
        self._turn_tokens[pid] = token
        task = asyncio.create_task(
            self._run_query(frame, token),
            name=f"voice-query-{pid}",
        )
        self._inflight[pid] = task

    async def _run_query(self, frame: GatedQueryFrame, token: object) -> None:
        pid = frame.participant_id
        # The end frame carries one assembled data-channel response per turn.
        accumulated: list[str] = []
        cancelled = False
        try:
            query = VoiceQuery(
                participant_id=pid,
                text=frame.text,
                fresh_match=frame.fresh_match,
                timestamp_us=frame.pts_us,
            )
            await self._observe(
                VoiceTurn(participant_id=pid, role="user", timestamp_us=frame.pts_us, text=frame.text)
            )
            result = await self._handler(query)
            if isinstance(result, str):
                if result and self._turn_tokens.get(pid) is token:
                    accumulated.append(result)
                    await self._push_text(result, pid=pid)
                return
            try:
                async for chunk in result:
                    if not chunk or self._turn_tokens.get(pid) is not token:
                        continue
                    accumulated.append(chunk)
                    await self._push_text(chunk, pid=pid)
            finally:
                close = getattr(result, "aclose", None)
                if close is not None:
                    await close()
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception:
            logger.exception("voice handler raised pid={!r}", pid)
        finally:
            # A cancelled turn must not emit a partial response as final data.
            is_current = self._turn_tokens.get(pid) is token
            if not cancelled and is_current:
                logger.info("voice handler query complete pid={!r}", pid)
                try:
                    response = "".join(accumulated)
                    await self._observe(
                        VoiceTurn(participant_id=pid, role="agent", timestamp_us=frame.pts_us, text=response)
                    )
                    await self.push_frame(
                        AssistantResponseEndFrame(
                            pid=pid,
                            text=response,
                            pts_us=frame.pts_us,
                        )
                    )
                except Exception:
                    logger.exception("emit AssistantResponseEndFrame failed pid={!r}", pid)

            if is_current:
                self._turn_tokens.pop(pid, None)
                self._inflight.pop(pid, None)
                pending = self._queued.get(pid)
                if pending:
                    next_frame = pending.popleft()
                    if not pending:
                        self._queued.pop(pid, None)
                    await self._start_query(next_frame, interrupt=True)

    async def _push_text(self, text: str, *, pid: str) -> None:
        """Push a ``TextFrame`` tagged with the participant id.

        ``transport_destination`` flows through the pipeline to the
        ``StreamingTtsProcessor``, which copies it onto the
        ``OutputAudioRawFrame``s it emits so the output transport knows
        which participant to address. Without this tag, the empty
        string ends up on every downstream send and the hub drops the
        audio.
        """
        f = TextFrame(text=text)
        f.transport_destination = pid
        await self.push_frame(f)

    def _cancel_pid(self, pid: str) -> None:
        self._turn_tokens.pop(pid, None)
        self._queued.pop(pid, None)
        task = self._inflight.pop(pid, None)
        if task is not None and not task.done():
            task.cancel()

    async def _shutdown_inflight(self) -> None:
        """Cancel every in-flight handler task and await its teardown.

        Awaiting matters: a cancelled turn still runs its ``finally`` (turn
        observer, end frame), so returning before that lands would let a
        transcript write outlive the session.
        """
        tasks = [t for t in self._inflight.values() if not t.done()]
        self._cancel_all_inflight()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass  # expected — we cancelled it ourselves
            except Exception:
                logger.exception("voice handler task raised during shutdown")

    def _cancel_all_inflight(self) -> None:
        for task in self._inflight.values():
            if not task.done():
                task.cancel()
        self._queued.clear()
        self._inflight.clear()
        self._turn_tokens.clear()


__all__ = ["_VoiceHandlerProcessor"]
