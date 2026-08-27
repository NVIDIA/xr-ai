# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``StreamingTtsProcessor`` — per-participant sentence-batched parallel TTS.

Consumes ``TextFrame``s (assistant output, one per chunk/token) and emits
``OutputAudioRawFrame``s. Buffers text per participant until a sentence
boundary, synthesizes each sentence in parallel, then streams the WAVs out in
order so each participant's playback stays monotonic.

All streaming state — pending text, the synth/order queue, and the sender task
— is keyed by participant id, so concurrent participants never share a buffer
(which would interleave their words) or a sender (which would misroute audio).

An ``InterruptionFrame`` cancels only the interrupting participant's in-flight
synthesis and flushes only their hub audio (``transport_source`` carries the
pid); a frame with no pid falls back to draining every participant and flushing
each of their hub audio. A ``ParticipantLeftFrame`` releases just that
participant's state, and a pipeline ``EndFrame``/``CancelFrame`` tears down all
sender tasks.

Every synthesized WAV is offered to ``VoiceGate.observe_tts_wav`` so the gate's
listening chime can lazily build at the right sample rate. When constructed with
a non-empty ``text_topic`` and a ``transport``, the processor also echoes each
assistant turn's full assembled response on the data channel under that topic.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import nemo_relay
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    TextFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from xr_ai_hub import DataMessage
from xr_ai_hub._capture import CAPTURE_TTS_TOPIC
from xr_ai_models import TTSService
from xr_ai_voicegate import VoiceGate

from .._audio import wav_to_output_frames
from .._frames import (
    AssistantResponseEndFrame,
    ParticipantLeftFrame,
    TextResponseEndFrame,
)

if TYPE_CHECKING:
    from .._transport import HubVoiceTransport


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# A sentence-final char optionally followed by closing punctuation (quote,
# paren, bracket) and trailing whitespace, e.g. ``... looking at?"``. A plain
# ``endswith((".", "!", "?"))`` would miss the trailing quote and leave the tail
# in the buffer to concatenate onto the next turn.
_TRAILING_SENTENCE_END = re.compile(r"""[.!?]["')\]]*\s*$""")


@dataclass(frozen=True, slots=True)
class _TtsResponseBoundary:
    """Ordered marker placed behind every sentence in one assistant response."""

    pid: str


class _TtsPidState:
    """Per-participant streaming state: pending text + its ordered sender."""

    __slots__ = (
        "active_responses",
        "pending",
        "response_sentences",
        "sender_task",
        "sender_queue",
        "synth_seq",
    )

    def __init__(self) -> None:
        self.active_responses: int = 0
        self.pending: str = ""
        self.response_sentences: int = 0
        self.sender_task: asyncio.Task | None = None
        self.sender_queue: asyncio.Queue | None = None
        self.synth_seq: int = 0


class StreamingTtsProcessor(FrameProcessor):
    """Per-participant sentence-batched parallel TTS at the pipeline tail.

    ``transport`` and ``text_topic`` are optional; when both are supplied (and
    the topic is non-empty), the processor emits one ``send_return_data`` per
    :class:`AssistantResponseEndFrame` so the client receives the full assembled
    reply on the data channel. Leaving them out (or passing an empty topic)
    disables the echo — used by samples whose assistant already sends its own
    per-turn data response.
    """

    def __init__(
        self,
        *,
        tts: TTSService,
        voice_gate: VoiceGate,
        transport: "HubVoiceTransport | None" = None,
        text_topic: str = "",
    ) -> None:
        super().__init__()
        self._tts        = tts
        self._voice_gate = voice_gate
        self._transport  = transport
        self._text_topic = text_topic
        # Per-participant streaming state, keyed by pid.
        self._by_pid: dict[str, _TtsPidState] = {}
        self._capture_tasks: set[asyncio.Task[None]] = set()

    # ── pipecat frame entrypoint ──────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            pid = frame.transport_source
            if pid:
                await self._drain_on_interrupt(pid)
            else:
                # No pid on the frame — drain every participant (legacy fallback).
                await self._drain_all_on_interrupt()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, ParticipantLeftFrame):
            # Release the departing participant's synthesis state before the
            # frame reaches the output transport (which drops that pid's media
            # sender). Without this, a lingering synth task could still emit
            # audio for the departed pid afterwards, and the transport's lazy
            # routing would recreate the sender it had just released.
            await self._drain_on_interrupt(frame.participant_id, flush=False)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (EndFrame, CancelFrame)):
            # Normal pipeline shutdown: cancel/await every participant's sender
            # task so no TTS background work is retained.
            await self._shutdown_all()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TextResponseEndFrame):
            await self._handle_text_response_end(frame)
            if isinstance(frame, AssistantResponseEndFrame):
                await self._echo_assistant_response(frame)
            # Forward the marker so any tail processor / sink that tracks turn
            # boundaries still sees it.
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TextFrame):
            await self._handle_text(frame)
            return

        await self.push_frame(frame, direction)

    # ── private ───────────────────────────────────────────────────────────────

    def _state(self, pid: str) -> _TtsPidState:
        st = self._by_pid.get(pid)
        if st is None:
            st = _TtsPidState()
            self._by_pid[pid] = st
        return st

    def _ensure_sender(self, pid: str) -> asyncio.Queue:
        """Spin up ``pid``'s ordered-sender task lazily on first sentence."""
        st = self._state(pid)
        if st.sender_queue is None or st.sender_task is None or st.sender_task.done():
            st.sender_queue = asyncio.Queue()
            st.sender_task  = asyncio.create_task(
                self._sender_loop(st.sender_queue),
                name=f"tts-sender-{pid}",
            )
        return st.sender_queue

    def has_active_response(self, pid: str) -> bool:
        """Return whether ordered response audio is still open for ``pid``."""
        st = self._by_pid.get(pid)
        return bool(st is not None and st.active_responses)

    async def _handle_text(self, frame: TextFrame) -> None:
        if not frame.text:
            return
        # transport_destination carries the addressed participant; text with no
        # destination buffers under "" (single-participant / fallback routing).
        pid = frame.transport_destination or ""
        self._state(pid).pending += frame.text
        await self._flush_complete_sentences(pid)

    async def _handle_text_response_end(self, frame: TextResponseEndFrame) -> None:
        """Flush trailing text and queue its participant-scoped audio boundary.

        A producer may finish an utterance with text that has no sentence-final
        punctuation. The boundary regex would leave that fragment buffered
        forever, so end-of-response flushes it before placing the boundary on
        the same ordered queue as synthesis.
        """
        st = self._by_pid.get(frame.pid)
        if st is not None and st.pending.strip():
            sentence = st.pending.strip()
            st.pending = ""
            await self._dispatch_sentence(sentence, pid=frame.pid)

        # The output transport aggregates small frames into 40 ms chunks. Put
        # the boundary on the same FIFO as synthesis so it arrives only after
        # every WAV in this response; Pipecat then flushes and silence-pads the
        # final partial chunk instead of carrying it into the next response.
        if st is not None and st.response_sentences:
            queue = self._ensure_sender(frame.pid)
            await queue.put(_TtsResponseBoundary(pid=frame.pid))
            st.response_sentences = 0

    async def _echo_assistant_response(self, frame: AssistantResponseEndFrame) -> None:
        """Send one completed assistant response on the configured data topic."""
        if not self._text_topic or self._transport is None:
            return
        if not frame.text:
            return
        logger.info(
            "data text echo pid={!r} topic={!r} len={}",
            frame.pid, self._text_topic, len(frame.text),
        )
        try:
            await self._transport.send_return_data(DataMessage(
                participant_id = frame.pid,
                topic          = self._text_topic,
                pts_us         = frame.pts_us,
                data           = frame.text.encode(),
            ))
        except Exception:
            logger.exception(
                "send_return_data failed pid={!r} topic={!r}",
                frame.pid, self._text_topic,
            )

    async def _flush_complete_sentences(self, pid: str) -> None:
        """Drain every complete sentence in ``pid``'s pending buffer, leaving a
        trailing fragment in place until more text arrives. A buffer already
        ending in sentence-final punctuation is flushed in one shot — covers the
        assistant-returns-a-complete-string case with no trailing whitespace."""
        st = self._state(pid)
        while True:
            m = _SENTENCE_END.search(st.pending)
            if m is None:
                break
            sentence  = st.pending[: m.end()].strip()
            st.pending = st.pending[m.end() :]
            if not sentence:
                continue
            await self._dispatch_sentence(sentence, pid=pid)
        if st.pending and _TRAILING_SENTENCE_END.search(st.pending):
            sentence  = st.pending.strip()
            st.pending = ""
            if sentence:
                await self._dispatch_sentence(sentence, pid=pid)

    async def _dispatch_sentence(self, sentence: str, *, pid: str) -> None:
        logger.info("tts sentence dispatch pid={!r} len={}", pid, len(sentence))
        queue = self._ensure_sender(pid)
        st = self._state(pid)
        if not st.response_sentences:
            st.active_responses += 1
        st.response_sentences += 1
        st.synth_seq += 1
        task  = asyncio.create_task(
            self._synthesize_with_caption(sentence, pid=pid),
            name=f"tts-synth-{pid}-{st.synth_seq}",
            context=nemo_relay.fork_asyncio_context(),
        )
        await queue.put((task, pid))

    async def _synthesize_with_caption(
        self,
        text: str,
        *,
        pid: str,
    ) -> tuple[bytes, str]:
        return await self._synthesize(text, pid=pid), text

    async def _synthesize(self, text: str, *, pid: str) -> bytes:
        with nemo_relay.scope.scope(
            "voice.tts",
            nemo_relay.ScopeType.Function,
            input={"text": text},
            metadata={"participant_id": pid or None},
        ):
            return await self._tts.synthesize(text)

    async def _sender_loop(self, queue: asyncio.Queue) -> None:
        """Await each synth task in FIFO order, observe the WAV, and push the
        decoded audio downstream as ``OutputAudioRawFrame``s."""
        try:
            while True:
                item = await queue.get()
                if item is None:
                    return
                if isinstance(item, _TtsResponseBoundary):
                    stopped = TTSStoppedFrame()
                    stopped.transport_destination = item.pid
                    await self.push_frame(stopped)
                    st = self._by_pid.get(item.pid)
                    if st is not None and st.active_responses:
                        st.active_responses -= 1
                    continue
                task, pid = item
                try:
                    wav, text = await task
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("tts synth failed pid={!r}", pid)
                    continue
                if not wav:
                    continue
                # Let the gate's lazy chime build pick up the sample rate from
                # real TTS output exactly once.
                try:
                    self._voice_gate.observe_tts_wav(wav)
                except Exception:
                    logger.exception("observe_tts_wav raised pid={!r}", pid)
                self._schedule_capture_caption(pid, text)
                await self._push_wav(wav, pid=pid)
        except asyncio.CancelledError:
            return

    async def _push_wav(self, wav_bytes: bytes, *, pid: str) -> None:
        try:
            frames = wav_to_output_frames(wav_bytes, pid)
        except Exception:
            logger.exception("tts WAV decode failed pid={!r}", pid)
            return
        for out in frames:
            await self.push_frame(out)

    async def _publish_capture_caption(self, pid: str, text: str) -> None:
        if not pid or self._transport is None:
            return
        sender = getattr(self._transport, "send_return_data", None)
        if sender is None:
            return
        try:
            await sender(DataMessage(
                participant_id=pid,
                topic=CAPTURE_TTS_TOPIC,
                pts_us=time.time_ns() // 1_000,
                data=text.encode(),
            ))
        except Exception:
            logger.opt(exception=True).debug(
                "capture TTS caption failed pid={!r}", pid,
            )

    def _schedule_capture_caption(self, pid: str, text: str) -> None:
        if not pid or self._transport is None:
            return
        task = asyncio.create_task(
            self._publish_capture_caption(pid, text),
            name=f"capture-tts-caption-{pid}",
        )
        self._capture_tasks.add(task)
        task.add_done_callback(self._capture_tasks.discard)

    async def _stop_capture_tasks(self) -> None:
        tasks = tuple(self._capture_tasks)
        self._capture_tasks.clear()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _teardown_sender(self, st: _TtsPidState) -> None:
        """Cancel one participant's sender task and drop any parked synth tasks."""
        task = st.sender_task
        st.sender_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass  # expected — we cancelled it ourselves
        queue = st.sender_queue
        st.sender_queue = None
        if queue is not None:
            while not queue.empty():
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    continue
                if isinstance(item, _TtsResponseBoundary):
                    continue
                synth_task, _ = item
                synth_task.cancel()

    async def _flush_hub(self, pid: str) -> None:
        """Drop already-paced hub audio for ``pid`` so a stop is immediate.

        Cancelling synth/sender tasks stops *new* audio, but anything already
        queued downstream (the hub's pacing pipe, the LiveKit jitter buffer)
        keeps playing; flushing the hub return-audio buffer drops it at source.
        """
        if self._transport is None or not pid:
            return
        logger.info("hub return-audio flushed pid={!r}", pid)
        try:
            await self._transport.endpoint.flush_return_audio(pid)
        except Exception:
            logger.opt(exception=True).debug(
                "flush_return_audio failed on interrupt pid={!r}", pid,
            )

    async def _drain_on_interrupt(self, pid: str, *, flush: bool = True) -> None:
        """Cancel one participant's in-flight TTS without touching any other's.

        ``flush`` also drops that participant's already-paced hub audio — right
        for an interruption, pointless for a participant who has already left.
        """
        st = self._by_pid.pop(pid, None)
        if st is not None:
            st.pending = ""
            queue_len = st.sender_queue.qsize() if st.sender_queue is not None else 0
            await self._teardown_sender(st)
            logger.info("tts sender drained pid={!r} queue_len={}", pid, queue_len)
        if flush:
            await self._flush_hub(pid)

    async def _drain_all_on_interrupt(self) -> None:
        """No-pid fallback: drain every participant's synthesis and flush the
        hub audio of each of them.

        Flushing only the fallback ``target_participant`` would leave audio that
        is already paced for the other active participants playing out after a
        global interruption.
        """
        pids = list(self._by_pid)
        for pid in pids:
            st = self._by_pid.pop(pid)
            st.pending = ""
            await self._teardown_sender(st)
        if self._transport is None:
            return
        # Include the fallback target: audio pushed with no destination is paced
        # under that pid and has no entry of its own in ``_by_pid``.
        targets = list(dict.fromkeys([*pids, self._transport.target_participant]))
        for pid in targets:
            await self._flush_hub(pid)

    async def _shutdown_all(self) -> None:
        """Cancel every participant's sender task on pipeline EndFrame/CancelFrame."""
        for pid in list(self._by_pid):
            st = self._by_pid.pop(pid)
            await self._teardown_sender(st)
        await self._stop_capture_tasks()
