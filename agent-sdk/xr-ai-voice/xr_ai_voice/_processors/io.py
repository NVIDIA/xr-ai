# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private Pipecat processor for runtime voice input and output."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from xr_ai_hub import DataMessage
from zmq import ZMQError

from .._frames import (
    AssistantResponseEndFrame,
    GatedQueryFrame,
    ParticipantJoinedFrame,
    ParticipantLeftFrame,
    _OutputDoneFrame,
    _SpeakerDiagnosticFrame,
    _SpeakerGatedQueryFrame,
)
from .._types import UpdateCombiner, VoiceInputSink, VoiceQuery, VoiceResponse

if TYPE_CHECKING:
    from .._transport import HubVoiceTransport


_PENDING_UPDATE_CAPACITY = 64
_SUPERSEDED_CAPACITY = 256


@dataclass(frozen=True, slots=True)
class _QueuedResponse:
    participant_id: str
    response: VoiceResponse
    pts_us: int
    # None marks an update; a reply names the request it answers.
    request_id: str | None = None
    urgent: bool = False
    source: str | None = None


@dataclass(frozen=True, slots=True)
class _PendingUpdate:
    text: str
    pts_us: int
    source: str


class _VoiceIOProcessor(FrameProcessor):
    """Publish accepted input and arbitrate each participant's voice floor.

    Replies answer the active request and supersede output for other requests.
    Urgent updates cut current speech. Other updates wait for silence and are
    combined into one utterance.
    """

    def __init__(
        self,
        input_sink: VoiceInputSink,
        *,
        transport: HubVoiceTransport | None = None,
        on_participant_joined: Callable[[str], Awaitable[None] | None] | None = None,
        on_participant_left: Callable[[str], Awaitable[None] | None] | None = None,
        on_interrupted: Callable[[str | None], Awaitable[None] | None] | None = None,
        interrupt_on_supersede: bool = False,
        combiner: UpdateCombiner | None = None,
    ) -> None:
        super().__init__()
        self._input_sink = input_sink
        self._transport = transport
        self._on_participant_joined = on_participant_joined
        self._on_participant_left = on_participant_left
        self._on_interrupted = on_interrupted
        self._interrupt_on_supersede = interrupt_on_supersede
        self._combiner = combiner
        self._inflight: dict[str, asyncio.Task[None]] = {}
        self._flight: dict[str, _QueuedResponse] = {}
        # Holds the floor from its start until the output transport returns its
        # done marker or its audio is interrupted. The next output starts only
        # then, so an interruption only ever flushes the owner's audio.
        self._owner: dict[str, _QueuedResponse] = {}
        self._input_tasks: set[asyncio.Task[None]] = set()
        self._queued_replies: dict[str, deque[_QueuedResponse]] = {}
        self._queued_urgent: dict[str, deque[_QueuedResponse]] = {}
        self._pending_updates: dict[str, deque[_PendingUpdate]] = {}
        self._active_request: dict[str, str] = {}
        # Insertion-ordered sets, bounded like the voice agent's closed streams.
        self._superseded: dict[str, dict[str, None]] = {}
        self._speaking: set[str] = set()
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

    async def enqueue_reply(
        self,
        participant_id: str,
        response: VoiceResponse,
        *,
        request_id: str,
        pts_us: int | None = None,
    ) -> None:
        """Submit a finite or incremental reply to one request."""

        if isinstance(response, str) and not response.strip():
            return
        await self._spawn_reply(
            _QueuedResponse(
                participant_id=participant_id,
                response=response,
                request_id=request_id,
                pts_us=pts_us if pts_us is not None else time.time_ns() // 1_000,
            ),
            request_id,
        )

    async def enqueue_update(
        self,
        participant_id: str,
        text: str,
        *,
        source: str,
        urgent: bool = False,
        pts_us: int | None = None,
    ) -> None:
        """Submit unprompted output; an urgent update cuts current speech.

        ``source`` names the producer so :meth:`withdraw_updates` can drop its updates.
        """

        text = text.strip()
        if not text:
            return
        pts_us = pts_us if pts_us is not None else time.time_ns() // 1_000
        if urgent:
            await self._spawn_urgent(
                _QueuedResponse(
                    participant_id=participant_id, response=text, pts_us=pts_us, urgent=True, source=source,
                )
            )
            return
        pending = self._pending_updates.setdefault(participant_id, deque())
        if len(pending) >= _PENDING_UPDATE_CAPACITY:
            _ = pending.popleft()
            logger.warning("voice update dropped pid={!r} capacity={}", participant_id, _PENDING_UPDATE_CAPACITY)
        pending.append(_PendingUpdate(text=text, pts_us=pts_us, source=source))
        await self._start_next(participant_id)

    async def withdraw_updates(self, participant_id: str, *, source: str) -> None:
        """Drop one producer's updates that have not started playing.

        The response in flight and other producers' updates are kept.
        """

        if pending := self._pending_updates.get(participant_id):
            kept = deque(update for update in pending if update.source != source)
            if kept:
                self._pending_updates[participant_id] = kept
            else:
                del self._pending_updates[participant_id]
        if urgent := self._queued_urgent.get(participant_id):
            kept_urgent = deque(item for item in urgent if item.source != source)
            for item in urgent:
                if item.source == source:
                    await self._close_response(item.response)
            if kept_urgent:
                self._queued_urgent[participant_id] = kept_urgent
            else:
                del self._queued_urgent[participant_id]

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            pid = frame.transport_source
            if pid:
                logger.info("voice cancel pid={!r} reason=interruption", pid)
                _ = await self._barge_in(pid)
                self._owner.pop(pid, None)
            else:
                await self._barge_in_all()
                self._owner.clear()
            await self.push_frame(frame, direction)
            await self._notify_interrupted(pid)
            for waiting in [pid] if pid else [*self._queued_urgent, *self._pending_updates]:
                await self._start_next(waiting)
            return

        if isinstance(frame, _OutputDoneFrame):
            if self._owner.get(frame.pid) is frame.output:
                del self._owner[frame.pid]
                await self._start_next(frame.pid)
            return

        if isinstance(frame, (EndFrame, CancelFrame)):
            await self._shutdown()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
            await self.push_frame(frame, direction)
            pid = frame.transport_destination
            if pid:
                if isinstance(frame, BotStartedSpeakingFrame):
                    self._speaking.add(pid)
                else:
                    self._speaking.discard(pid)
                    await self._start_next(pid)
            return

        if isinstance(frame, GatedQueryFrame):
            await self._spawn_query(frame)
            return

        if isinstance(frame, _SpeakerDiagnosticFrame):
            label = f"Speaker {frame.speaker_id} [{frame.status}]"
            text = f"{label}: {frame.text}" if frame.text else f"{label} detected"
            await self._display(DataMessage(
                topic="voice.speaker-diagnostic", participant_id=frame.participant_id,
                pts_us=frame.pts_us, data=text.encode("utf-8"),
            ))
            return

        if isinstance(frame, ParticipantJoinedFrame):
            logger.info("voice participant joined pid={!r}", frame.participant_id)
            if self._transport is not None:
                self._transport.set_target_participant(frame.participant_id)
            await self._notify_joined(frame.participant_id)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, ParticipantLeftFrame):
            logger.info("voice participant left pid={!r}", frame.participant_id)
            if self._transport is not None:
                self._transport.cleanup_participant(frame.participant_id)
            await self._notify_left(frame.participant_id)
            await self._forget(frame.participant_id)
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    async def _spawn_query(self, frame: GatedQueryFrame) -> None:
        pid = frame.participant_id
        logger.info("voice input pid={!r}", pid)
        interrupted_response = await self._barge_in(pid)
        if interrupted_response is not None and interrupted_response.done():
            interrupted_response = None
        if self._interrupt_on_supersede and pid in self._seen_output:
            await self._interrupt_audio(pid)
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
            query = VoiceQuery(
                participant_id=pid,
                text=frame.text,
                timestamp_us=frame.pts_us,
                interrupted_output=interrupted_response is not None,
            )
            speaker = f"Speaker {frame.speaker_id}" if isinstance(frame, _SpeakerGatedQueryFrame) else "User"
            # Both deliveries belong to this input task, so shutdown drains them.
            results = await asyncio.gather(
                self._input_sink(query),
                self._display(DataMessage(
                    participant_id=pid, topic="voice.user-query",
                    pts_us=frame.pts_us, data=f"{speaker}: {frame.text}".encode("utf-8"),
                )),
                return_exceptions=True,
            )
            failures = [result for result in results if isinstance(result, BaseException)]
            for failure in failures:
                if isinstance(failure, asyncio.CancelledError):
                    raise failure
            if failures:
                raise BaseExceptionGroup("voice input delivery failed", failures)
        except asyncio.CancelledError:
            raise
        except Exception:
            # isolation boundary: this task supervises both input deliveries.
            logger.exception("voice input delivery failed pid={!r}", pid)

    async def _display(self, message: DataMessage) -> None:
        if self._transport is None:
            return
        try:
            await self._transport.send_return_data(message)
        except (OSError, ZMQError):
            logger.exception("voice input display failed pid={!r}", message.participant_id)

    async def _spawn_reply(self, item: _QueuedResponse, request_id: str) -> None:
        pid = item.participant_id
        if request_id in self._superseded.get(pid, ()):
            logger.debug("voice reply dropped pid={!r} request_id={!r} reason=superseded", pid, request_id)
            await self._close_response(item.response)
            return
        await self._drop_queued_replies(pid, keep=request_id)
        previous = self._active_request.get(pid)
        if previous is not None and previous != request_id:
            self._supersede(pid, previous)
        owner = self._owner.get(pid)
        if owner is not None and not owner.urgent and owner.request_id != request_id:
            await self._cut(pid, owner, keep=request_id)
        self._active_request[pid] = request_id
        pending = self._queued_replies.setdefault(pid, deque())
        pending.append(item)
        logger.info("voice reply queued pid={!r} depth={}", pid, len(pending))
        await self._start_next(pid)

    async def _spawn_urgent(self, item: _QueuedResponse) -> None:
        pid = item.participant_id
        self._queued_urgent.setdefault(pid, deque()).append(item)
        owner = self._owner.get(pid)
        if owner is not None and owner.urgent:
            # Starts once the urgent output holding the floor finishes playing.
            logger.info("voice urgent update queued pid={!r}", pid)
        elif owner is not None:
            await self._cut(pid, owner)
        else:
            await self._interrupt_audio(pid)
        await self._start_next(pid)

    async def _cut(self, pid: str, owner: _QueuedResponse, *, keep: str | None = None) -> None:
        """Stop the owner's output, supersede its request, and flush its audio."""

        _ = self._stop_flight(pid)
        if owner.request_id is not None:
            self._supersede(pid, owner.request_id)
            await self._drop_queued_replies(pid, keep=keep)
        await self._interrupt_audio(pid)

    async def _interrupt_audio(self, pid: str) -> None:
        """Flush ``pid``'s audio, releasing the floor; no output may be in flight."""

        self._owner.pop(pid, None)
        interruption = InterruptionFrame()
        interruption.transport_source = pid
        await self.push_frame(interruption)

    async def _start_response(self, item: _QueuedResponse) -> None:
        pid = item.participant_id
        self._seen_output.add(pid)
        token = object()
        self._turn_tokens[pid] = token
        task = asyncio.create_task(
            self._run_response(item, token),
            name=f"voice-response-{pid}",
        )
        self._flight[pid] = item
        self._owner[pid] = item
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
                self._flight.pop(pid, None)
            if self._owner.get(pid) is item:
                # Its audio may still be synthesizing or playing; the transport
                # returns this marker once that audio is written, and the next
                # output starts then.
                done = _OutputDoneFrame(pid=pid, output=item)
                done.transport_destination = pid
                try:
                    await self.push_frame(done)
                except Exception:
                    logger.exception("emit voice output done failed pid={!r}", pid)

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
        if pid in self._owner:
            return
        if urgent := self._queued_urgent.get(pid):
            item = urgent.popleft()
            if not urgent:
                self._queued_urgent.pop(pid, None)
        elif replies := self._queued_replies.get(pid):
            item = replies.popleft()
            if not replies:
                self._queued_replies.pop(pid, None)
        elif pid not in self._speaking and (updates := self._pending_updates.pop(pid, None)):
            item = _QueuedResponse(
                participant_id=pid,
                response=self._update_response([update.text for update in updates]),
                pts_us=updates[0].pts_us,
            )
        else:
            return
        await self._start_response(item)

    def _update_response(self, texts: list[str]) -> VoiceResponse:
        if len(texts) == 1 or self._combiner is None:
            return " ".join(texts)
        return self._combine(self._combiner, texts)

    @staticmethod
    async def _combine(combiner: UpdateCombiner, texts: Sequence[str]) -> AsyncIterator[str]:
        # One-item stream so combining runs inside, and is cancelled with, the response task.
        try:
            text = await combiner(texts)
        except Exception:  # noqa: BLE001
            # isolation boundary: a failing user-supplied combiner must not drop the updates.
            logger.exception("voice update combiner failed")
            text = " ".join(texts)
        yield text

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

    async def _notify_interrupted(self, pid: str | None) -> None:
        if self._on_interrupted is None:
            return
        try:
            result = self._on_interrupted(pid)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("voice interruption callback raised pid={!r}", pid)

    async def _barge_in(self, pid: str) -> asyncio.Task[None] | None:
        """Cancel replies and the in-flight response, keeping updates for later."""

        task = await self._cancel_output(pid)
        if request_id := self._active_request.pop(pid, None):
            self._supersede(pid, request_id)
        return task

    async def _barge_in_all(self) -> None:
        pids = {*self._flight, *self._queued_replies, *self._active_request}
        for pid in pids:
            _ = await self._barge_in(pid)
        self._cancel_input_tasks()

    async def _cancel_output(self, pid: str) -> asyncio.Task[None] | None:
        current = self._flight.get(pid)
        if current is not None and current.request_id is not None:
            self._supersede(pid, current.request_id)
        await self._drop_queued_replies(pid)
        return self._stop_flight(pid)

    def _stop_flight(self, pid: str) -> asyncio.Task[None] | None:
        self._flight.pop(pid, None)
        self._turn_tokens.pop(pid, None)
        task = self._inflight.pop(pid, None)
        if task is not None and not task.done():
            task.cancel()
        return task

    async def _drop_queued_replies(self, pid: str, *, keep: str | None = None) -> None:
        pending = self._queued_replies.get(pid)
        if not pending:
            return
        kept: deque[_QueuedResponse] = deque()
        for item in pending:
            if item.request_id == keep:
                kept.append(item)
                continue
            if item.request_id is not None:
                self._supersede(pid, item.request_id)
            await self._close_response(item.response)
        if kept:
            self._queued_replies[pid] = kept
        else:
            self._queued_replies.pop(pid, None)

    def _supersede(self, pid: str, request_id: str) -> None:
        superseded = self._superseded.setdefault(pid, {})
        superseded.pop(request_id, None)
        superseded[request_id] = None
        if len(superseded) > _SUPERSEDED_CAPACITY:
            del superseded[next(iter(superseded))]

    async def _forget(self, pid: str) -> None:
        _ = await self._cancel_output(pid)
        for queued in self._queued_urgent.pop(pid, ()):
            await self._close_response(queued.response)
        self._pending_updates.pop(pid, None)
        self._active_request.pop(pid, None)
        self._superseded.pop(pid, None)
        self._speaking.discard(pid)
        self._owner.pop(pid, None)
        self._seen_output.discard(pid)

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
        for queues in (*self._queued_replies.values(), *self._queued_urgent.values()):
            for item in queues:
                await self._close_response(item.response)
        self._queued_replies.clear()
        self._queued_urgent.clear()
        self._pending_updates.clear()
        self._active_request.clear()
        self._superseded.clear()
        self._speaking.clear()
        self._inflight.clear()
        self._flight.clear()
        self._owner.clear()
        self._turn_tokens.clear()
        self._cancel_input_tasks()

    def _cancel_input_tasks(self) -> None:
        for task in self._input_tasks:
            if not task.done():
                task.cancel()
        self._input_tasks.clear()


__all__ = ["_VoiceIOProcessor"]
