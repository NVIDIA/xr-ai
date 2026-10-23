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
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
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


@dataclass(frozen=True, slots=True, eq=False)
class _PendingUpdate:
    text: str
    pts_us: int
    source: str


@dataclass(frozen=True, slots=True)
class _QueuedResponse:
    participant_id: str
    response: VoiceResponse
    pts_us: int
    # None marks an update; a reply names the request it answers.
    request_id: str | None = None
    urgent: bool = False
    source: str | None = None
    # The pending updates this output speaks; they leave the pending set once it starts.
    updates: tuple[_PendingUpdate, ...] = ()


@dataclass(slots=True, eq=False)
class _Output:
    """The output holding one participant's floor.

    It is *preparing* until its first chunk is committed to TTS and can still be
    dropped without a trace; after that it is *playing* and keeps the floor until
    the transport returns its done marker or its audio is interrupted. Playing
    output is never replayed.
    """

    item: _QueuedResponse
    task: asyncio.Task[None] | None = None
    playing: bool = False


class _VoiceIOProcessor(FrameProcessor):
    """Publish accepted input and arbitrate each participant's voice floor.

    Replies answer the active request and supersede output for other requests.
    Urgent updates cut current speech. Other updates wait until neither the bot
    nor the participant is speaking and are combined into one utterance. Output
    that has started speaking is never replayed, even when it is cut.
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
        # One output per participant; the next starts only once it gives up the
        # floor, so an interruption only ever flushes that output's audio.
        self._floor: dict[str, _Output] = {}
        self._input_tasks: set[asyncio.Task[None]] = set()
        self._queued_replies: dict[str, deque[_QueuedResponse]] = {}
        self._queued_urgent: dict[str, deque[_QueuedResponse]] = {}
        self._pending_updates: dict[str, deque[_PendingUpdate]] = {}
        self._active_request: dict[str, str] = {}
        # Insertion-ordered sets, bounded like the voice agent's closed streams.
        self._superseded: dict[str, dict[str, None]] = {}
        self._speaking: set[str] = set()
        self._user_speaking: set[str] = set()
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
        """Drop one producer's updates that have not been heard.

        Output already playing is not cut, and other producers' updates are kept.
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
        await self._recheck(participant_id)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            pid = frame.transport_source
            if pid:
                logger.info("voice cancel pid={!r} reason=interruption", pid)
                _ = await self._barge_in(pid)
                self._release(pid)
            else:
                await self._barge_in_all()
                for owner in [*self._floor]:
                    self._release(owner)
            await self.push_frame(frame, direction)
            await self._notify_interrupted(pid)
            for waiting in [pid] if pid else [*self._queued_urgent, *self._pending_updates]:
                await self._start_next(waiting)
            return

        if isinstance(frame, _OutputDoneFrame):
            output = self._floor.get(frame.pid)
            if output is not None and output is frame.output:
                del self._floor[frame.pid]
                await self._start_next(frame.pid)
            return

        if isinstance(frame, (UserStartedSpeakingFrame, UserStoppedSpeakingFrame)):
            await self.push_frame(frame, direction)
            if pid := frame.transport_source:
                if isinstance(frame, UserStartedSpeakingFrame):
                    self._user_speaking.add(pid)
                    await self._recheck(pid)
                else:
                    self._user_speaking.discard(pid)
                    await self._start_next(pid)
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
                    await self._recheck(pid)
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
        # A query ends the utterance even if its stop frame was not forwarded.
        self._user_speaking.discard(pid)
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
        await self._start_next(pid)

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
        owner = self._floor.get(pid)
        if owner is not None and owner.item.request_id not in {None, request_id}:
            await self._cut(pid, owner.item, keep=request_id)
        elif owner is not None and owner.item.request_id is None and not owner.item.urgent and not owner.playing:
            # An update not yet heard yields to the reply and stays pending;
            # one already playing finishes first.
            _ = self._stop(pid)
        self._active_request[pid] = request_id
        pending = self._queued_replies.setdefault(pid, deque())
        pending.append(item)
        logger.info("voice reply queued pid={!r} depth={}", pid, len(pending))
        await self._start_next(pid)

    async def _spawn_urgent(self, item: _QueuedResponse) -> None:
        pid = item.participant_id
        self._queued_urgent.setdefault(pid, deque()).append(item)
        owner = self._floor.get(pid)
        if owner is not None and owner.item.urgent:
            # Starts once the urgent output holding the floor finishes playing.
            logger.info("voice urgent update queued pid={!r}", pid)
        elif owner is not None:
            await self._cut(pid, owner.item)
        else:
            await self._interrupt_audio(pid)
        await self._start_next(pid)

    async def _cut(self, pid: str, owner: _QueuedResponse, *, keep: str | None = None) -> None:
        """Stop the owner's output, supersede its request, and flush its audio."""

        _ = self._stop(pid)
        if owner.request_id is not None:
            self._supersede(pid, owner.request_id)
            await self._drop_queued_replies(pid, keep=keep)
        await self._interrupt_audio(pid)

    async def _interrupt_audio(self, pid: str) -> None:
        """Flush ``pid``'s audio, releasing the floor."""

        self._release(pid)
        interruption = InterruptionFrame()
        interruption.transport_source = pid
        await self.push_frame(interruption)

    async def _start_response(self, item: _QueuedResponse) -> None:
        pid = item.participant_id
        self._seen_output.add(pid)
        output = _Output(item)
        self._floor[pid] = output
        output.task = asyncio.create_task(self._run_response(output), name=f"voice-response-{pid}")

    async def _run_response(self, output: _Output) -> None:
        item = output.item
        pid = item.participant_id
        chunks: list[str] = []
        cancelled = False
        dropped = False
        try:
            dropped = not await self._consume_response(output, accumulated=chunks)
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception:
            logger.exception("voice response failed pid={!r}", pid)
        finally:
            # Settle the floor before closing the stream: a cancel landing in
            # aclose() would otherwise skip it and hold the floor forever.
            current = self._floor.get(pid) is output
            if current and not cancelled and not dropped:
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
            if current and output.playing:
                # Its audio may still be synthesizing or playing; the transport
                # returns this marker once that audio is written, and the next
                # output starts then.
                done = _OutputDoneFrame(pid=pid, output=output)
                done.transport_destination = pid
                await self.push_frame(done)
            elif current:
                # Nothing reached TTS, so there is no audio to wait for.
                del self._floor[pid]
                await self._start_next(pid)
            await self._close_response(item.response)

    async def _consume_response(self, output: _Output, *, accumulated: list[str]) -> bool:
        """Push ``output``'s chunks; False if it was dropped before its first chunk."""

        response = output.item.response
        if isinstance(response, str):
            if not response:
                return True
            if not self._commit(output):
                return False
            accumulated.append(response)
            await self._push_text(response, pid=output.item.participant_id)
            return True
        async for chunk in response:
            if not chunk:
                continue
            if not output.playing and not self._commit(output):
                return False
            accumulated.append(chunk)
            await self._push_text(chunk, pid=output.item.participant_id)
        return True

    def _commit(self, output: _Output) -> bool:
        """Turn ``output`` from preparing to playing, if it may still speak.

        Preparing awaits (task startup, the combiner, a model stream) let the
        state change after scheduling, so this repeats the scheduling check at
        the last moment the output can still be dropped cleanly.
        """

        pid = output.item.participant_id
        if self._floor.get(pid) is not output or not self._may_speak(output.item):
            return False
        output.playing = True
        self._discard_updates(pid, output.item.updates)
        return True

    async def _recheck(self, pid: str) -> None:
        """Drop ``pid``'s preparing output once it may no longer speak, without
        waiting for its combiner or stream, and start whatever may speak now."""

        output = self._floor.get(pid)
        if output is not None and not output.playing and not self._may_speak(output.item):
            _ = self._stop(pid)
            await self._start_next(pid)

    def _may_speak(self, item: _QueuedResponse) -> bool:
        """Replies and urgent updates speak once they hold the floor; other
        updates only while both sides are quiet and none of them was withdrawn."""

        if item.request_id is not None or item.urgent:
            return True
        pid = item.participant_id
        return self._quiet(pid) and self._still_pending(pid, item.updates)

    def _quiet(self, pid: str) -> bool:
        return pid not in self._speaking and pid not in self._user_speaking

    async def _start_next(self, pid: str) -> None:
        if pid in self._floor:
            return
        if urgent := self._queued_urgent.get(pid):
            item = urgent.popleft()
            if not urgent:
                self._queued_urgent.pop(pid, None)
        elif replies := self._queued_replies.get(pid):
            item = replies.popleft()
            if not replies:
                self._queued_replies.pop(pid, None)
        elif self._quiet(pid) and (updates := tuple(self._pending_updates.get(pid, ()))):
            item = _QueuedResponse(
                participant_id=pid,
                response=self._update_response([update.text for update in updates]),
                pts_us=updates[0].pts_us,
                updates=updates,
            )
        else:
            return
        await self._start_response(item)

    def _still_pending(self, pid: str, updates: tuple[_PendingUpdate, ...]) -> bool:
        pending = {id(update) for update in self._pending_updates.get(pid, ())}
        return all(id(update) in pending for update in updates)

    def _discard_updates(self, pid: str, updates: tuple[_PendingUpdate, ...]) -> None:
        """Drop updates that have started speaking; later ones stay pending."""

        if not updates or not (pending := self._pending_updates.get(pid)):
            return
        heard = {id(update) for update in updates}
        kept = deque(update for update in pending if id(update) not in heard)
        if kept:
            self._pending_updates[pid] = kept
        else:
            del self._pending_updates[pid]

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
        if not text.strip():
            # An empty rewrite would leave the batch pending and rebuild it at once.
            logger.warning("voice update combiner returned no text")
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
        pids = {*self._floor, *self._queued_replies, *self._active_request}
        for pid in pids:
            _ = await self._barge_in(pid)
        self._cancel_input_tasks()

    async def _cancel_output(self, pid: str) -> asyncio.Task[None] | None:
        current = self._floor.get(pid)
        if current is not None and current.item.request_id is not None:
            self._supersede(pid, current.item.request_id)
        await self._drop_queued_replies(pid)
        return self._stop(pid)

    def _stop(self, pid: str) -> asyncio.Task[None] | None:
        """Cancel ``pid``'s output task; output that is not playing yet also gives up the floor.

        The floor is released here rather than in the task, because a task
        cancelled before it first runs never reaches its cleanup. An urgent
        update that has not been heard goes back to the front of its queue.
        Playing output keeps the floor until its done marker or an interruption.
        """

        output = self._floor.get(pid)
        if output is None:
            return None
        task = output.task
        if task is not None and not task.done():
            task.cancel()
        if not output.playing:
            del self._floor[pid]
            if output.item.urgent:
                self._queued_urgent.setdefault(pid, deque()).appendleft(output.item)
        return task

    def _release(self, pid: str) -> None:
        """Give up ``pid``'s floor; its task never outlives it."""

        output = self._floor.pop(pid, None)
        if output is not None and output.task is not None and not output.task.done():
            output.task.cancel()

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
        self._user_speaking.discard(pid)
        self._release(pid)
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
            for task in (*(output.task for output in self._floor.values()), *self._input_tasks)
            if task is not None and not task.done()
        ]
        await self._cancel_all()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _cancel_all(self) -> None:
        for pid in [*self._floor]:
            self._release(pid)
        for queues in (*self._queued_replies.values(), *self._queued_urgent.values()):
            for item in queues:
                await self._close_response(item.response)
        self._queued_replies.clear()
        self._queued_urgent.clear()
        self._pending_updates.clear()
        self._active_request.clear()
        self._superseded.clear()
        self._speaking.clear()
        self._user_speaking.clear()
        self._cancel_input_tasks()

    def _cancel_input_tasks(self) -> None:
        for task in self._input_tasks:
            if not task.done():
                task.cancel()
        self._input_tasks.clear()


__all__ = ["_VoiceIOProcessor"]
